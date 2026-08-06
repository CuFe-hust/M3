import base64
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from models.base import JsonResponseCache, RequestMeta, build_request_hash, image_to_data_url, sanitize_messages
from spacers_agent.clients.mock import MockVisionClient


class PointResponse(BaseModel):
    value: int = Field(ge=0)


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
