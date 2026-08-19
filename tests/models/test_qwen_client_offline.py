"""Offline tests for the Qwen Transformers client with injected fakes.

Qwen Transformers 客户端离线测试：注入 fake processor/model（不下载权重），
覆盖一次加载、complete_json 成功/校验失败/修复、缓存命中、截断恢复与
产物持久化。断言客户端符合 VisionLanguageClient。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from models.base import (
    JsonDecodingPolicy,
    OUTLINES_ADAPTER_VERSION,
    PINNED_OUTLINES_VERSION,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
    json_schema_sha256,
)
from models.cache import JsonResponseCache
from models.qwen_transformers import QwenTransformersClient, QwenTransformersError
from models.settings import QwenSettings


class _BoxResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    box: list[float]
    geometry: dict[str, Any] = {}


class _BoxWithAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    box: list[float]
    answer: str


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


class _RecordingModel(_FakeModel):
    """Record generation kwargs for native/constrained seam assertions.
    记录生成参数，用于 native/约束解码 seam 断言。"""

    def __init__(self) -> None:
        self.generate_kwargs: list[dict[str, Any]] = []

    def generate(self, **kwargs):
        self.generate_kwargs.append(kwargs)
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


def _constrained_meta(artifact_dir: Path, digest: str | None = None) -> RequestMeta:
    schema_sha256 = json_schema_sha256(_BoxResult.model_json_schema())
    request_hash = digest or build_request_hash(
        model="fake",
        generation={"max_tokens": 8},
        prompt_version="v1",
        messages=[{"content": "x"}],
        image_sha256=None,
        response_schema=_BoxResult.model_json_schema(),
        structured_decoding=JsonDecodingPolicy.OUTLINES_JSON_SCHEMA.value,
        outlines_adapter_version=OUTLINES_ADAPTER_VERSION,
        pinned_outlines_version=PINNED_OUTLINES_VERSION,
        schema_sha256=schema_sha256,
    )
    return RequestMeta(
        request_id="test:qwen:constrained",
        request_hash=request_hash,
        prompt_version="v1",
        artifact_dir=artifact_dir,
        decoding_policy=JsonDecodingPolicy.OUTLINES_JSON_SCHEMA,
        outlines_adapter_version=OUTLINES_ADAPTER_VERSION,
        pinned_outlines_version=PINNED_OUTLINES_VERSION,
        schema_sha256=schema_sha256,
    )


def _install_fake_outlines(monkeypatch, model: Any, processor: Any) -> None:
    """Install a tiny API-compatible Outlines seam without importing the package.
    注入微型 API 兼容 Outlines seam，且不导入真实依赖。"""
    outlines = types.ModuleType("outlines")
    outlines.__version__ = PINNED_OUTLINES_VERSION

    def from_transformers(existing_model, existing_processor):
        existing_processor.padding_side = "left"
        existing_processor.pad_token = "[PAD]"
        return types.SimpleNamespace(model=existing_model, processor=existing_processor)

    outlines.from_transformers = from_transformers
    backends = types.ModuleType("outlines.backends")

    def get_json_schema_logits_processor(backend_name, adapter, schema_text):
        assert backend_name is None
        assert adapter.model is model
        assert adapter.processor is processor
        assert "properties" in schema_text
        return object()

    backends.get_json_schema_logits_processor = get_json_schema_logits_processor
    monkeypatch.setitem(sys.modules, "outlines", outlines)
    monkeypatch.setitem(sys.modules, "outlines.backends", backends)


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


def test_constrained_generation_fails_closed_without_outlines(tmp_path: Path) -> None:
    """Missing Outlines must not silently fall back or invoke native generation.
    缺少 Outlines 时不得静默回退，也不得调用 native 生成。"""
    processor = _FakeProcessor(['{"label": "a", "box": [1, 2, 3, 4]}'])
    model = _RecordingModel()
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=model,
        processor=processor,
        repair_prompt="Repair the JSON.",
    )
    import asyncio

    with pytest.raises(QwenTransformersError) as raised:
        asyncio.run(
            client.complete_json(
                messages=[{"role": "user", "content": "Q"}],
                response_model=_BoxResult,
                request_meta=_constrained_meta(tmp_path / "artifacts"),
            )
        )
    assert raised.value.code == "OUTLINES_UNAVAILABLE"
    assert model.generate_kwargs == []
    metadata = json.loads(
        (tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8")
    )
    assert metadata["validation_error"] == "OUTLINES_UNAVAILABLE"
    assert all(
        "error" not in attempt
        for attempt in metadata["response_metadata"]["attempt_errors"]
    )


def test_direct_decoding_policy_cannot_change_request_identity(tmp_path: Path) -> None:
    """An explicit policy must agree with durable request metadata.
    显式策略必须与持久化请求元数据一致。"""
    model = _RecordingModel()
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=model,
        processor=_FakeProcessor(['{"label": "a", "box": [1, 2, 3, 4]}']),
    )
    import asyncio

    with pytest.raises(QwenTransformersError) as raised:
        asyncio.run(
            client.complete_json(
                messages=[{"role": "user", "content": "planner"}],
                response_model=_BoxResult,
                request_meta=_meta(tmp_path / "artifacts"),
                decoding_policy=JsonDecodingPolicy.OUTLINES_JSON_SCHEMA,
            )
        )
    assert raised.value.code == "DECODING_POLICY_MISMATCH"
    assert model.generate_kwargs == []


def test_outlines_adapter_reuses_loaded_objects_and_restores_processor_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The optional adapter is lazy, identity-preserving, and isolated.
    可选适配器必须惰性创建、保持对象身份并隔离处理器状态。"""
    processor = _FakeProcessor([])
    processor.padding_side = "right"
    processor.pad_token = "<old>"
    model = _RecordingModel()
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=model,
        processor=processor,
    )
    _install_fake_outlines(monkeypatch, model, processor)

    logits_processor, metadata = client._build_outlines_logits_processor(_BoxResult)

    assert logits_processor is not None
    assert metadata["schema_compilation"] == "created"
    assert client._outlines_model.model is model
    assert client._outlines_model.processor is processor
    assert processor.padding_side == "right"
    assert processor.pad_token == "<old>"


def test_constrained_generation_is_single_pass_and_native_is_unconstrained(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Planner constraints apply once while a native child call stays native.
    规划器约束只应用一次，而 native 子 Agent 调用保持原路径。"""
    model = _RecordingModel()
    processor = _FakeProcessor([
        '{"label": "a", "box": [1, 2, 3, 4]}',
        '{"label": "b", "box": [5, 6, 7, 8]}',
    ])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=model,
        processor=processor,
        repair_prompt="Repair the JSON.",
    )
    _install_fake_outlines(monkeypatch, model, processor)
    import asyncio

    constrained = asyncio.run(
        client.complete_json(
            messages=[{"role": "user", "content": "planner"}],
            response_model=_BoxResult,
            request_meta=_constrained_meta(tmp_path / "constrained"),
        )
    )
    native = asyncio.run(
        client.complete_json(
            messages=[{"role": "user", "content": "child"}],
            response_model=_BoxResult,
            request_meta=_meta(tmp_path / "native"),
        )
    )

    assert constrained.label == "a"
    assert native.label == "b"
    assert len(model.generate_kwargs) == 2
    assert "logits_processor" in model.generate_kwargs[0]
    assert "logits_processor" not in model.generate_kwargs[1]
    constrained_metadata = json.loads(
        (tmp_path / "constrained" / "validation.json").read_text(encoding="utf-8")
    )
    decoding = constrained_metadata["response_metadata"]["decoding"]
    assert decoding["structured_decoding"] == "outlines-json-schema"
    assert decoding["schema_sha256"] == json_schema_sha256(_BoxResult.model_json_schema())


def test_constrained_validation_never_uses_repair_generation(monkeypatch, tmp_path: Path) -> None:
    """A constrained validation failure is terminal after one generation.
    约束解码后的校验失败在一次生成后终止，不触发 repair。"""
    model = _RecordingModel()
    processor = _FakeProcessor(["not json"])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=model,
        processor=processor,
        repair_prompt="Repair the JSON.",
    )
    _install_fake_outlines(monkeypatch, model, processor)
    import asyncio

    with pytest.raises(QwenTransformersError) as raised:
        asyncio.run(
            client.complete_json(
                messages=[{"role": "user", "content": "planner"}],
                response_model=_BoxResult,
                request_meta=_constrained_meta(tmp_path / "artifacts"),
            )
        )
    assert raised.value.code == "JSON_VALIDATION_FAILED"
    assert len(model.generate_kwargs) == 1


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


# ── cache identity / 缓存身份 (A) ──────────────────────────────────────────


def test_cache_identity_is_stable() -> None:
    from models.qwen_transformers import QWEN_CLIENT_VERSION

    client = QwenTransformersClient(
        QwenSettings(model="fake-model", max_tokens=64, revision="rev-1"),
        model=_FakeModel(),
        processor=_FakeProcessor([]),
    )
    identity = client.cache_identity
    assert identity.model == "fake-model"
    assert identity.revision == "rev-1"
    assert identity.client_version == QWEN_CLIENT_VERSION
    assert identity.generation_payload() == {
        "temperature": 0.0,
        "do_sample": False,
        "max_tokens": 64,
    }


def test_cache_identity_reflects_max_tokens_and_revision() -> None:
    base = QwenSettings(model="fake-model", max_tokens=64, revision="rev-1")
    client_a = QwenTransformersClient(
        base, model=_FakeModel(), processor=_FakeProcessor([])
    )
    client_b = QwenTransformersClient(
        base.model_copy(update={"revision": "rev-2"}),
        model=_FakeModel(),
        processor=_FakeProcessor([]),
    )
    client_c = QwenTransformersClient(
        base.model_copy(update={"max_tokens": 128}),
        model=_FakeModel(),
        processor=_FakeProcessor([]),
    )
    assert client_a.cache_identity != client_b.cache_identity
    assert client_a.cache_identity != client_c.cache_identity


# ── 缓存写失败 / cache write failures (C) ──────────────────────────────────


def test_cache_write_failure_does_not_drop_result(tmp_path: Path, monkeypatch) -> None:
    from models.cache import CacheWriteError

    cache = JsonResponseCache(tmp_path / "cache")

    def _broken_save(request_hash, entry):
        raise CacheWriteError("cache write failed for deadbeef (write_text): OSError")

    monkeypatch.setattr(cache, "save", _broken_save)
    processor = _FakeProcessor(['{"label": "a", "box": [1, 2, 3, 4]}'])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
        cache=cache,
    )
    import asyncio

    meta = _meta(tmp_path / "artifacts")
    result = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}], response_model=_BoxResult, request_meta=meta,
    ))
    assert result.label == "a"
    metadata = json.loads((tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8"))
    assert metadata["response_metadata"]["cache_write_error"]
    assert metadata["response_metadata"]["cache_write_recovered"] is True


def test_successful_cache_write_records_no_error(tmp_path: Path) -> None:
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
    asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}], response_model=_BoxResult, request_meta=meta,
    ))
    metadata = json.loads((tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8"))
    assert metadata["response_metadata"]["cache_write_error"] is None
    assert metadata["response_metadata"]["cache_write_recovered"] is False


def test_cache_entry_rejection_does_not_drop_result(tmp_path: Path) -> None:
    """A schema-valid result whose cached content is unsafe must still be
    returned; the artifact records the stable rejection label.
    Schema 校验通过但缓存内容不安全的合法结果仍必须返回；产物记录稳定的
    拒绝标签。"""
    cache = JsonResponseCache(tmp_path / "cache")
    processor = _FakeProcessor([
        '{"label": "a", "box": [1, 2, 3, 4], "answer": "Bearer abc"}'
    ])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
        cache=cache,
    )
    import asyncio

    meta = _meta(tmp_path / "artifacts")
    result = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}],
        response_model=_BoxWithAnswer,
        request_meta=meta,
    ))
    assert result.label == "a"
    assert result.answer == "Bearer abc"
    # No unsafe cache entry was written. / 未写入不安全缓存条目。
    assert list((tmp_path / "cache").glob("*.json")) == []
    metadata = json.loads((tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8"))
    assert metadata["response_metadata"]["cache_write_error"] == (
        "cache entry rejected: ValidationError"
    )
    assert metadata["response_metadata"]["cache_write_recovered"] is True


def test_cache_entry_rejection_data_url_does_not_drop_result(tmp_path: Path) -> None:
    cache = JsonResponseCache(tmp_path / "cache")
    processor = _FakeProcessor([
        '{"label": "b", "box": [1, 2, 3, 4], "answer": "data:image/png;base64,AAAA"}'
    ])
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=8),
        model=_FakeModel(),
        processor=processor,
        cache=cache,
    )
    import asyncio

    meta = _meta(tmp_path / "artifacts")
    result = asyncio.run(client.complete_json(
        messages=[{"role": "user", "content": "Q"}],
        response_model=_BoxWithAnswer,
        request_meta=meta,
    ))
    assert result.label == "b"
    assert list((tmp_path / "cache").glob("*.json")) == []
    metadata = json.loads((tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8"))
    assert metadata["response_metadata"]["cache_write_error"] == (
        "cache entry rejected: ValidationError"
    )


# ── 离线默认 / offline defaults (I) ────────────────────────────────────────


@pytest.mark.parametrize("path_like", [
    "/models/Qwen3-VL-4B",
    r"C:\models\Qwen3-VL-4B",
    "C:/models/Qwen3-VL-4B",
    r"\\server\share\Qwen3-VL-4B",
    "//server/share/Qwen3-VL-4B",
    "file:///models/Qwen3-VL-4B",
])
def test_path_like_model_requires_cache_model_id(path_like: str) -> None:
    """A local checkpoint path of any platform shape must carry an explicit
    logical id. 任何平台形态的本地 checkpoint 路径都必须携带显式逻辑 ID。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="cache_model_id is required"):
        QwenSettings(model=path_like)
    settings = QwenSettings(model=path_like, cache_model_id="qwen3-vl-4b-local")
    assert settings.effective_cache_model_id == "qwen3-vl-4b-local"


@pytest.mark.parametrize("bad_id", [
    "/models/Qwen",
    r"C:\models\Qwen",
    "C:/models/Qwen",
    r"\\server\share\Qwen",
    "//server/share/Qwen",
    "file:///models/Qwen",
    "",
    "   ",
    "line\nbreak",
    "nul\x00byte",
])
def test_path_like_cache_model_id_rejected(bad_id: str) -> None:
    """cache_model_id itself must be a logical identifier, never a path or
    empty/control-character text. cache_model_id 本身必须是逻辑标识符，
    绝不能是路径、空文本或含控制字符的文本。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QwenSettings(model="/models/Qwen3-VL-4B", cache_model_id=bad_id)


def test_logical_cache_model_ids_allowed() -> None:
    for value in ("qwen3-vl-4b-local", "Qwen/Qwen3-VL-4B-Instruct", "org:model@rev"):
        settings = QwenSettings(model="/models/Qwen3-VL-4B", cache_model_id=value)
        assert settings.effective_cache_model_id == value
    # Remote model names default to the declared name. / 远程模型名默认用声明名。
    remote = QwenSettings(model="Qwen/Qwen3-VL-4B-Instruct")
    assert remote.effective_cache_model_id == "Qwen/Qwen3-VL-4B-Instruct"


def test_offline_defaults() -> None:
    settings = QwenSettings()
    assert settings.allow_download is False
    # The dual-field config is gone: allow_download is the single source.
    # 双字段配置已移除：allow_download 是唯一来源。
    assert not hasattr(settings, "local_files_only")


def test_allow_download_is_single_source() -> None:
    assert QwenSettings(allow_download=True).allow_download is True
    assert QwenSettings(allow_download=False).allow_download is False


# ── cache identity / 缓存身份 ───────────────────────────────────────────────


def test_cache_identity_generation_matches_max_tokens() -> None:
    client = QwenTransformersClient(
        QwenSettings(model="fake", max_tokens=64),
        model=_FakeModel(),
        processor=_FakeProcessor([]),
    )
    assert client.cache_identity.generation_payload() == {
        "temperature": 0.0,
        "do_sample": False,
        "max_tokens": 64,
    }
