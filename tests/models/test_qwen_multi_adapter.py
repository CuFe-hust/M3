"""Offline contracts for the shared Qwen3.5 multi-LoRA engine.

共享 Qwen3.5 多 LoRA engine 的离线契约测试。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from models.base import RequestMeta, build_request_hash
from models.cache import JsonResponseCache
from models.qwen3_5.multi_adapter import (
    BoundQwenAdapterClient,
    MultiAdapterQwenEngine,
    QwenAdapterError,
)
from models.settings import QwenAdapterSettings, QwenSettings


_SOURCE_KEY = "base_model.model.model.layer.lora_A.weight"


class _Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


class _Tensor:
    shape = (1, 2)

    def __getitem__(self, key: Any) -> "_Tensor":
        return self


class _Processor:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        return "prompt"

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, _Tensor]:
        return {"input_ids": _Tensor()}

    def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
        return [self.responses.pop(0)]


class _BaseModel:
    config = SimpleNamespace(model_type="qwen3_5")
    device = "cpu"

    def named_modules(self):
        return iter((("model.layer", object()),))


class _PeftModel(_BaseModel):
    def __init__(self, adapter_names: tuple[str, ...]) -> None:
        self.adapter_names = adapter_names
        self.activations: list[str] = []
        self.frozen = False
        self.eval_called = False
        self.active_adapter: str | None = None

    def state_dict(self) -> dict[str, object]:
        return {
            _SOURCE_KEY.replace(
                ".lora_A.weight", f".lora_A.{name}.weight"
            ): object()
            for name in self.adapter_names
        }

    def requires_grad_(self, value: bool) -> "_PeftModel":
        self.frozen = value is False
        return self

    def eval(self) -> "_PeftModel":
        self.eval_called = True
        return self

    def enable_adapter_layers(self) -> None:
        return None

    def disable_adapter_layers(self) -> None:
        self.activations.append("base")
        self.active_adapter = None

    def set_adapter(self, name: str) -> None:
        self.activations.append(name)
        self.active_adapter = name

    def generate(self, **kwargs: Any) -> list[_Tensor]:
        return [_Tensor()]


def _write_adapter(
    root: Path,
    name: str,
    *,
    modules_to_save: object = None,
    source_key: str = _SOURCE_KEY,
) -> QwenAdapterSettings:
    path = root / name
    path.mkdir(parents=True)
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "/remote/M3/models/Qwen3.5-9B",
        "target_modules": "model\\.layer",
        "modules_to_save": modules_to_save,
        "peft_version": "0.18.1",
    }
    (path / "adapter_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    header = json.dumps(
        {source_key: {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}},
        separators=(",", ":"),
    ).encode("utf-8")
    weights = len(header).to_bytes(8, "little") + header
    weight_path = path / "adapter_model.safetensors"
    weight_path.write_bytes(weights)
    return QwenAdapterSettings(
        path=Path(name),
        logical_id=f"{name}-logical-v1",
        revision=hashlib.sha256(weights).hexdigest(),
    )


def _engine(
    tmp_path: Path,
    *,
    names: tuple[str, ...] = ("adapter-a", "adapter-b"),
    processor: _Processor | None = None,
) -> tuple[MultiAdapterQwenEngine, _PeftModel, list[tuple[str, ...]]]:
    adapters = {name: _write_adapter(tmp_path, name) for name in names}
    loaded: list[tuple[str, ...]] = []
    wrapped = _PeftModel(names)

    def loader(base: object, specs: tuple[Any, ...]) -> _PeftModel:
        assert isinstance(base, _BaseModel)
        loaded.append(tuple(spec.name for spec in specs))
        return wrapped

    engine = MultiAdapterQwenEngine(
        QwenSettings(
            model="models/Qwen3.5-9B",
            cache_model_id="Qwen/Qwen3.5-9B:local",
            max_tokens=8,
        ),
        adapters=adapters,
        project_root=tmp_path,
        repair_prompt="repair",
        model=_BaseModel(),
        processor=processor or _Processor(['{"answer":"ok"}']),
        adapter_loader=loader,
    )
    return engine, wrapped, loaded


def _meta(path: Path, request_hash: str) -> RequestMeta:
    return RequestMeta(
        request_id="adapter-test",
        request_hash=request_hash,
        prompt_version="v1",
        artifact_dir=path,
    )


def test_engine_loads_adapter_inventory_once_and_freezes_model(tmp_path: Path) -> None:
    engine, model, loaded = _engine(tmp_path)

    assert loaded == [("adapter-a", "adapter-b")]
    assert model.frozen is True
    assert model.eval_called is True
    assert set(engine.adapter_inventory) == {"adapter-a", "adapter-b"}
    assert "path" not in json.dumps(engine.runtime_identity)


def test_bound_clients_have_distinct_path_free_cache_identities(tmp_path: Path) -> None:
    engine, _, _ = _engine(tmp_path)
    first = engine.bind("adapter-a")
    second = engine.bind("adapter-b")

    assert isinstance(first, BoundQwenAdapterClient)
    assert first.cache_identity != second.cache_identity
    payload = first.cache_identity.generation_payload()
    assert payload["adapter"] == {
        "logical_id": "adapter-a-logical-v1",
        "revision": engine.adapter_inventory["adapter-a"]["revision"],
        "peft_version": "0.18.1",
    }
    assert str(tmp_path) not in json.dumps(payload)
    messages = [{"role": "user", "content": "x"}]
    first_hash = build_request_hash(
        model=first.cache_identity.model,
        generation=first.cache_identity.generation_payload(),
        prompt_version="v1",
        messages=messages,
        image_sha256=None,
    )
    second_hash = build_request_hash(
        model=second.cache_identity.model,
        generation=second.cache_identity.generation_payload(),
        prompt_version="v1",
        messages=messages,
        image_sha256=None,
    )
    assert first_hash != second_hash


def test_switch_and_repair_stay_under_one_activation(tmp_path: Path) -> None:
    processor = _Processor(["not-json", '{"answer":"repaired"}'])
    engine, model, _ = _engine(
        tmp_path, names=("adapter-a",), processor=processor
    )
    client = engine.bind("adapter-a")

    result = asyncio.run(
        client.complete_json(
            messages=[{"role": "user", "content": "x"}],
            response_model=_Result,
            request_meta=_meta(tmp_path / "artifacts", "a" * 64),
        )
    )

    assert result.answer == "repaired"
    assert model.activations == ["adapter-a"]


def test_cache_hit_does_not_switch_adapter(tmp_path: Path) -> None:
    processor = _Processor(['{"answer":"cached"}'])
    adapters = {"adapter-a": _write_adapter(tmp_path, "adapter-a")}
    wrapped = _PeftModel(("adapter-a",))
    cache = JsonResponseCache(tmp_path / "cache")
    engine = MultiAdapterQwenEngine(
        QwenSettings(
            model="models/Qwen3.5-9B",
            cache_model_id="Qwen/Qwen3.5-9B:local",
            max_tokens=8,
        ),
        adapters=adapters,
        project_root=tmp_path,
        cache=cache,
        model=_BaseModel(),
        processor=processor,
        adapter_loader=lambda base, specs: wrapped,
    )
    client = engine.bind("adapter-a")
    meta = _meta(tmp_path / "artifacts", "b" * 64)
    asyncio.run(
        client.complete_json(
            messages=[{"role": "user", "content": "x"}],
            response_model=_Result,
            request_meta=meta,
        )
    )
    asyncio.run(
        client.complete_json(
            messages=[{"role": "user", "content": "x"}],
            response_model=_Result,
            request_meta=meta,
        )
    )
    assert wrapped.activations == ["adapter-a"]


def test_concurrent_bound_requests_cannot_cross_adapter_state(tmp_path: Path) -> None:
    engine, model, _ = _engine(tmp_path)
    first, second = engine.bind("adapter-a"), engine.bind("adapter-b")

    def generation(expected: str):
        def run(*args: Any, **kwargs: Any) -> tuple[str, dict[str, int]]:
            assert model.active_adapter == expected
            time.sleep(0.01)
            assert model.active_adapter == expected
            return json.dumps({"answer": expected}), {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }

        return run

    first._client._generate = generation("adapter-a")  # type: ignore[method-assign]
    second._client._generate = generation("adapter-b")  # type: ignore[method-assign]

    async def run_both() -> tuple[_Result, _Result]:
        return await asyncio.gather(
            first.complete_json(
                messages=[{"role": "user", "content": "a"}],
                response_model=_Result,
                request_meta=_meta(tmp_path / "a", "c" * 64),
            ),
            second.complete_json(
                messages=[{"role": "user", "content": "b"}],
                response_model=_Result,
                request_meta=_meta(tmp_path / "b", "d" * 64),
            ),
        )

    results = asyncio.run(run_both())
    assert [result.answer for result in results] == ["adapter-a", "adapter-b"]
    assert model.activations == ["adapter-a", "adapter-b"]


def test_invalid_assets_fail_before_adapter_loader(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path, "adapter-a", modules_to_save=["head"])
    calls: list[object] = []
    with pytest.raises(
        QwenAdapterError, match="QWEN_ADAPTER_MODULES_TO_SAVE_UNSUPPORTED"
    ):
        MultiAdapterQwenEngine(
            QwenSettings(
                model="models/Qwen3.5-9B",
                cache_model_id="Qwen/Qwen3.5-9B:local",
            ),
            adapters={"adapter-a": adapter},
            project_root=tmp_path,
            model=_BaseModel(),
            processor=_Processor([]),
            adapter_loader=lambda base, specs: calls.append(specs),
        )
    assert calls == []


def test_lfs_pointer_and_digest_mismatch_fail_stably(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path, "adapter-a")
    weight_path = tmp_path / "adapter-a" / "adapter_model.safetensors"
    weight_path.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"a" * 64 + b"\nsize 123\n"
    )
    with pytest.raises(QwenAdapterError, match="ModelAssetPointerError"):
        MultiAdapterQwenEngine(
            QwenSettings(
                model="models/Qwen3.5-9B",
                cache_model_id="Qwen/Qwen3.5-9B:local",
            ),
            adapters={"adapter-a": adapter},
            project_root=tmp_path,
            model=_BaseModel(),
            processor=_Processor([]),
        )

    adapter = _write_adapter(tmp_path, "adapter-b")
    drifted = adapter.model_copy(update={"revision": "0" * 64})
    with pytest.raises(QwenAdapterError, match="ModelAssetHashMismatchError"):
        MultiAdapterQwenEngine(
            QwenSettings(
                model="models/Qwen3.5-9B",
                cache_model_id="Qwen/Qwen3.5-9B:local",
            ),
            adapters={"adapter-b": drifted},
            project_root=tmp_path,
            model=_BaseModel(),
            processor=_Processor([]),
        )


def test_missing_corrupt_and_unconsumed_weights_fail_closed(tmp_path: Path) -> None:
    missing = _write_adapter(tmp_path, "missing")
    (tmp_path / "missing" / "adapter_model.safetensors").unlink()
    with pytest.raises(QwenAdapterError, match="ModelAssetMissingError"):
        MultiAdapterQwenEngine(
            QwenSettings(
                model="models/Qwen3.5-9B",
                cache_model_id="Qwen/Qwen3.5-9B:local",
            ),
            adapters={"missing": missing},
            project_root=tmp_path,
            model=_BaseModel(),
            processor=_Processor([]),
        )

    corrupt = _write_adapter(tmp_path, "corrupt")
    corrupt_path = tmp_path / "corrupt" / "adapter_model.safetensors"
    corrupt_path.write_bytes(b"not-a-safetensors-file")
    corrupt = corrupt.model_copy(
        update={"revision": hashlib.sha256(corrupt_path.read_bytes()).hexdigest()}
    )
    with pytest.raises(QwenAdapterError, match="QWEN_ADAPTER_WEIGHTS_INVALID"):
        MultiAdapterQwenEngine(
            QwenSettings(
                model="models/Qwen3.5-9B",
                cache_model_id="Qwen/Qwen3.5-9B:local",
            ),
            adapters={"corrupt": corrupt},
            project_root=tmp_path,
            model=_BaseModel(),
            processor=_Processor([]),
        )

    unconsumed = _write_adapter(tmp_path, "unconsumed")
    with pytest.raises(
        QwenAdapterError, match="QWEN_ADAPTER_WEIGHT_CONSUMPTION_INCOMPLETE"
    ):
        MultiAdapterQwenEngine(
            QwenSettings(
                model="models/Qwen3.5-9B",
                cache_model_id="Qwen/Qwen3.5-9B:local",
            ),
            adapters={"unconsumed": unconsumed},
            project_root=tmp_path,
            model=_BaseModel(),
            processor=_Processor([]),
            adapter_loader=lambda base, specs: _PeftModel(()),
        )


def test_base_type_target_and_legacy_head_fail_closed(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path, "adapter-a")
    incompatible = _BaseModel()
    incompatible.config = SimpleNamespace(model_type="qwen3_vl")
    with pytest.raises(QwenAdapterError, match="BASE_MODEL_TYPE_INCOMPATIBLE"):
        MultiAdapterQwenEngine(
            QwenSettings(
                model="models/Qwen3.5-9B",
                cache_model_id="Qwen/Qwen3.5-9B:local",
            ),
            adapters={"adapter-a": adapter},
            project_root=tmp_path,
            model=incompatible,
            processor=_Processor([]),
        )

    incompatible_targets = _BaseModel()
    incompatible_targets.named_modules = lambda: iter((("model.other", object()),))
    with pytest.raises(QwenAdapterError, match="TARGET_MODULES_INCOMPATIBLE"):
        MultiAdapterQwenEngine(
            QwenSettings(
                model="models/Qwen3.5-9B",
                cache_model_id="Qwen/Qwen3.5-9B:local",
            ),
            adapters={"adapter-a": adapter},
            project_root=tmp_path,
            model=incompatible_targets,
            processor=_Processor([]),
        )

    (tmp_path / "adapter-a" / "visual_planner_roi_head.safetensors").write_bytes(
        b"obsolete"
    )
    with pytest.raises(QwenAdapterError, match="AUXILIARY_HEAD_UNSUPPORTED"):
        MultiAdapterQwenEngine(
            QwenSettings(
                model="models/Qwen3.5-9B",
                cache_model_id="Qwen/Qwen3.5-9B:local",
            ),
            adapters={"adapter-a": adapter},
            project_root=tmp_path,
            model=_BaseModel(),
            processor=_Processor([]),
        )


def test_unknown_binding_fails_closed(tmp_path: Path) -> None:
    engine, _, _ = _engine(tmp_path, names=("adapter-a",))
    with pytest.raises(QwenAdapterError, match="QWEN_ADAPTER_BINDING_UNKNOWN"):
        engine.bind("missing")
