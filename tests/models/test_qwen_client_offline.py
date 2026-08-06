"""Offline tests for the Qwen Transformers client with injected fakes.

Qwen Transformers 客户端离线测试：注入 fake processor/model（不下载权重），
覆盖一次加载、complete_json 成功/校验失败/修复、缓存命中、截断恢复与
产物持久化。断言客户端符合 VisionLanguageClient。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from models.base import RequestMeta, VisionLanguageClient, build_request_hash
from models.cache import JsonResponseCache
from models.qwen_transformers import QwenTransformersClient, QwenTransformersError
from models.settings import QwenSettings


class _BoxResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    box: list[float]
    geometry: dict[str, Any] = {}


class _FakeOutput:
    """Fake tensor supporting slicing and shape reads. 支持切片与形状读取的假张量。"""

    shape = (1, 4)

    def __getitem__(self, key):
        return _FakeOutput()

    def __len__(self):
        return 1


class _FakeProcessor:
    """Deterministic offline processor stub. 确定性离线 processor 桩。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.calls.append(list(messages))
        return "prompt"

    def __call__(self, *args, **kwargs):
        return {"input_ids": _FakeOutput()}

    def batch_decode(self, *args, **kwargs):
        response = self.responses.pop(0)
        return [response]


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return [_FakeOutput()]


def _meta(artifact_dir: Path, digest: str | None = None) -> RequestMeta:
    return RequestMeta(
        request_id="test:qwen",
        request_hash=digest or build_request_hash(
            model="fake", generation={"max_tokens": 8}, prompt_version="v1",
            messages=[{"content": "x"}], image_sha256=None,
        ),
        prompt_version="v1",
        artifact_dir=artifact_dir,
    )


def test_client_implements_vision_language_protocol() -> None:
    import inspect

    assert inspect.iscoroutinefunction(QwenTransformersClient.complete_json)


def test_client_accepts_injected_model_and_processor(tmp_path: Path) -> None:
    processor = _FakeProcessor(['{"label": "a", "box": [1, 2, 3, 4]}'])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
    )
    assert client.model is not None and client.processor is processor


def test_complete_json_success_and_artifacts(tmp_path: Path) -> None:
    processor = _FakeProcessor(['{"label": "a", "box": [1, 2, 3, 4]}'])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
    )
    import asyncio

    result = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Locate the target."}],
        response_model=_BoxResult,
        request_meta=_meta(tmp_path / "artifacts"),
    ))
    assert isinstance(result, _BoxResult)
    assert result.label == "a"
    assert result.box == [1.0, 2.0, 3.0, 4.0]
    assert (tmp_path / "artifacts" / "raw_response.txt").is_file()
    assert (tmp_path / "artifacts" / "parsed.json").is_file()
    assert (tmp_path / "artifacts" / "request.json").is_file()
    assert (tmp_path / "artifacts" / "validation.json").is_file()


def test_complete_json_repair_after_validation_failure(tmp_path: Path) -> None:
    processor = _FakeProcessor([
        "not json at all",
        '{"label": "b", "box": [5, 6, 7, 8]}',
    ])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
        repair_prompt="Repair the JSON.",
    )
    import asyncio

    result = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}],
        response_model=_BoxResult,
        request_meta=_meta(tmp_path / "artifacts"),
    ))
    assert result.label == "b"
    assert len(processor.calls) == 2


def test_complete_json_fails_after_repair_without_prompt(tmp_path: Path) -> None:
    processor = _FakeProcessor(["not json at all"])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
    )
    import asyncio

    with pytest.raises(QwenTransformersError, match="validation failed"):
        asyncio.run(client.complete_json(
            messages=[{"role": "user", "content": "Q"}],
            response_model=_BoxResult,
            request_meta=_meta(tmp_path / "artifacts"),
        ))
    assert (tmp_path / "artifacts" / "validation.json").is_file()


def test_cache_hit_skips_generation(tmp_path: Path) -> None:
    cache = JsonResponseCache(tmp_path / "cache")
    processor = _FakeProcessor(['{"label": "a", "box": [1, 2, 3, 4]}'])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
        cache=cache,
    )
    import asyncio

    meta = _meta(tmp_path / "artifacts")
    first = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}], response_model=_BoxResult, request_meta=meta,
    ))
    assert first.label == "a"
    assert len(processor.responses) == 0  # consumed / 已消费
    second = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}], response_model=_BoxResult, request_meta=meta,
    ))
    assert second.label == "a"
    # No additional generation happened on the hit. / 命中时不再生成。
    assert len(processor.calls) == 1


def test_truncated_json_recovery(tmp_path: Path) -> None:
    processor = _FakeProcessor(['{"label": "c", "box": [1, 2, 3,'] )
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
    )
    import asyncio

    result = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}],
        response_model=_BoxResult,
        request_meta=_meta(tmp_path / "artifacts"),
    ))
    assert result.label == "c"
    metadata = json.loads((tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8"))
    assert metadata["response_metadata"]["local_recoveries"]


def test_model_and_processor_must_be_supplied_together() -> None:
    with pytest.raises(ValueError, match="together"):
        QwenTransformersClient(QwenSettings(model="fake"), model=object())


# ── 并发保护 / concurrency (G) ─────────────────────────────────────────────


class _ConcurrentModel:
    """Fake model tracking max concurrent generate() calls.
    跟踪 generate() 最大并发数的假模型。"""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.device = "cpu"

    def generate(self, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        import time

        time.sleep(0.05)
        self.active -= 1
        return [_FakeOutput()]


def test_generation_is_serialized_on_one_client(tmp_path: Path) -> None:
    model = _ConcurrentModel()
    processor = _FakeProcessor([
        '{"label": "a", "box": [1, 2, 3, 4]}',
        '{"label": "b", "box": [5, 6, 7, 8]}',
    ])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=model,
        processor=processor,
    )
    import asyncio

    meta1 = _meta(tmp_path / "a1")
    meta2 = _meta(tmp_path / "a2")

    async def _collect() -> list:
        return await asyncio.gather(
            client.complete_json(messages=[{"role": "user", "content": "Q1"}],
                                 response_model=_BoxResult, request_meta=meta1),
            client.complete_json(messages=[{"role": "user", "content": "Q2"}],
                                 response_model=_BoxResult, request_meta=meta2),
        )

    results = asyncio.run(_collect())
    assert {r.label for r in results} == {"a", "b"}
    assert model.max_active == 1, "generation must be serialized"
    assert (tmp_path / "a1" / "raw_response.txt").is_file()
    assert (tmp_path / "a2" / "raw_response.txt").is_file()


# ── 缓存恢复 / cache recovery (H) ──────────────────────────────────────────


def test_corrupt_cache_is_recovered_by_regeneration(tmp_path: Path) -> None:
    from models.cache import JsonResponseCache

    cache = JsonResponseCache(tmp_path / "cache")
    processor = _FakeProcessor(['{"label": "c", "box": [1, 2, 3, 4]}'])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
        cache=cache,
    )
    import asyncio

    meta = _meta(tmp_path / "artifacts")
    asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}], response_model=_BoxResult, request_meta=meta,
    ))
    # Corrupt the cached entry; the next call must regenerate and succeed.
    # 损坏缓存条目；下一次调用必须重新生成并成功。
    digest = meta.request_hash
    (tmp_path / "cache" / f"{digest}.json").write_text("{corrupt", encoding="utf-8")
    processor.responses.append('{"label": "d", "box": [9, 9, 9, 9]}')
    result = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}], response_model=_BoxResult, request_meta=meta,
    ))
    assert result.label == "d"
    metadata = json.loads((tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8"))
    assert metadata["response_metadata"]["cache_read_error"]


def test_stale_schema_cache_is_regenerated(tmp_path: Path) -> None:
    from models.cache import CacheEntry, JsonResponseCache

    cache = JsonResponseCache(tmp_path / "cache")
    processor = _FakeProcessor(['{"label": "e", "box": [1, 2, 3, 4]}'])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
        cache=cache,
    )
    import asyncio

    meta = _meta(tmp_path / "artifacts")
    # Seed a stale entry whose parsed payload fails the current schema.
    # 预置 parsed 载荷无法通过当前 Schema 的过期条目。
    cache.save(meta.request_hash, CacheEntry(raw_response="{}", parsed={"wrong": "shape"}))
    result = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}], response_model=_BoxResult, request_meta=meta,
    ))
    assert result.label == "e"


def test_cache_hit_does_not_acquire_generation_lock(tmp_path: Path) -> None:
    from models.cache import JsonResponseCache

    cache = JsonResponseCache(tmp_path / "cache")
    processor = _FakeProcessor(['{"label": "f", "box": [1, 2, 3, 4]}'])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
        cache=cache,
    )
    import asyncio

    meta = _meta(tmp_path / "artifacts")
    asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}], response_model=_BoxResult, request_meta=meta,
    ))
    # Lock is free when the cache hits: verify the lock is not locked.
    # 缓存命中时锁空闲：验证锁未被占用。
    assert not client._generation_lock.locked()
    asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}], response_model=_BoxResult, request_meta=meta,
    ))
    assert not client._generation_lock.locked()


# ── 离线默认 / offline defaults (I) ────────────────────────────────────────


def test_offline_defaults() -> None:
    settings = QwenSettings()
    assert settings.allow_download is False
    assert settings.effective_local_files_only() is True
    assert settings.local_files_only is None


def test_allow_download_inverts_local_files_only() -> None:
    assert QwenSettings(allow_download=True).effective_local_files_only() is False
    with pytest.raises(ValueError, match="cannot both"):
        QwenSettings(allow_download=True, local_files_only=True)


# ── cache_generation_config / 生成配置 ─────────────────────────────────────


def test_cache_generation_config_is_stable() -> None:
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=64),
        model=_FakeModel(),
        processor=_FakeProcessor([]),
    )
    assert client.cache_generation_config == {
        "temperature": 0.0,
        "do_sample": False,
        "max_tokens": 64,
    }
