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
