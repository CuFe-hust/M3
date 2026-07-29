import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from spacers_agent.clients.base import JsonResponseCache, RequestMeta, build_request_hash, image_to_data_url, sanitize_messages
from spacers_agent.clients.mock import MockVisionClient
from spacers_agent.clients.qwen_vllm import QwenVLLMClient
from spacers_agent.settings import QwenSettings


class PointResponse(BaseModel):
    value: int = Field(ge=0)


def _response(content: str) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=4, total_tokens=16)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def _meta(tmp_path: Path, request_id: str = "request-1") -> RequestMeta:
    return RequestMeta(
        request_id=request_id,
        request_hash="a" * 64,
        prompt_version="test-v1",
        sample_id="sample-1",
        tile_id="r000_c000",
        artifact_dir=tmp_path / "artifacts",
    )


@pytest.mark.asyncio
async def test_mock_client_validates_configured_response(tmp_path: Path) -> None:
    client = MockVisionClient({"request-1": {"value": 3}})

    result = await client.complete_json(messages=[], response_model=PointResponse, request_meta=_meta(tmp_path))

    assert result.value == 3
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_qwen_client_repairs_fenced_invalid_json_and_persists_artifacts(tmp_path: Path) -> None:
    responses = iter([_response("```json\n{\"value\": -1}\n```"), _response('{"value": 4}')])

    async def complete(**_: object) -> SimpleNamespace:
        return next(responses)

    client = QwenVLLMClient(
        QwenSettings(max_retries=0),
        repair_prompt="repair-json",
        completion_create=complete,
        retry_base_seconds=0,
    )
    result = await client.complete_json(
        messages=[{"role": "user", "content": "count"}], response_model=PointResponse, request_meta=_meta(tmp_path)
    )

    assert result.value == 4
    artifact_dir = tmp_path / "artifacts"
    assert (artifact_dir / "parsed.json").is_file()
    assert "ValidationError" in (artifact_dir / "validation.json").read_text(encoding="utf-8")
    assert "total_tokens" in (artifact_dir / "validation.json").read_text(encoding="utf-8")
    assert "response_attempt=1" in (artifact_dir / "raw_response.txt").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_qwen_client_uses_cache_without_a_second_completion(tmp_path: Path) -> None:
    calls = 0

    async def complete(**_: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return _response('{"value": 8}')

    cache = JsonResponseCache(tmp_path / "cache")
    client = QwenVLLMClient(QwenSettings(max_retries=0), repair_prompt="repair", cache=cache, completion_create=complete)
    meta = _meta(tmp_path)

    first = await client.complete_json(messages=[], response_model=PointResponse, request_meta=meta)
    second = await client.complete_json(messages=[], response_model=PointResponse, request_meta=meta)

    assert first.value == second.value == 8
    assert calls == 1


@pytest.mark.asyncio
async def test_qwen_client_retries_a_transient_status_code(tmp_path: Path) -> None:
    class TransientError(RuntimeError):
        status_code = 429

    responses = iter([TransientError("busy"), _response('{"value": 9}')])

    async def complete(**_: object) -> SimpleNamespace:
        next_value = next(responses)
        if isinstance(next_value, Exception):
            raise next_value
        return next_value

    client = QwenVLLMClient(
        QwenSettings(max_retries=1), repair_prompt="repair", completion_create=complete, retry_base_seconds=0
    )

    result = await client.complete_json(messages=[], response_model=PointResponse, request_meta=_meta(tmp_path))

    assert result.value == 9


def test_data_url_hashing_and_sanitizing_do_not_retain_base64() -> None:
    encoded = image_to_data_url(b"image-bytes", "image/png")
    assert base64.b64encode(b"image-bytes").decode("ascii") in encoded
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": encoded}}]}]

    sanitized = sanitize_messages(messages)
    request_hash = build_request_hash(
        model="qwen-test",
        generation={"temperature": 0.0},
        prompt_version="v1",
        messages=messages,
        image_sha256="b" * 64,
    )

    assert "image-bytes" not in str(sanitized)
    assert "aW1hZ2UtYnl0ZXM=" not in str(sanitized)
    assert len(request_hash) == 64
