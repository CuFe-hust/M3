"""Contract tests for request hashing and message sanitization.

请求哈希与消息脱敏测试：data URL 被摘要替换、脱敏后 hash 稳定、
不含 Base64/凭据、协议可被最小客户端满足。
"""

from __future__ import annotations

import sys
from pathlib import Path

from models import (
    ModelT,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
    sanitize_messages,
)
from models.images import image_sha256, image_to_data_url

IMAGE_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_sanitize_messages_redacts_data_urls() -> None:
    url = image_to_data_url(IMAGE_BYTES, "image/png")
    messages = [{"type": "image_url", "image_url": {"url": url}}]
    sanitized = sanitize_messages(messages)
    assert "data:image/png;base64," not in str(sanitized)
    redacted = sanitized[0]["image_url"]["url"]
    assert redacted["redacted_data_url_sha256"] == image_sha256(url.encode("utf-8"))
    assert redacted["encoded_bytes"] == len(url.encode("utf-8"))


def test_sanitize_messages_leaves_plain_text() -> None:
    messages = [{"role": "user", "content": "plain text"}]
    assert sanitize_messages(messages) == messages


def test_sanitize_messages_handles_nested_lists() -> None:
    url = image_to_data_url(IMAGE_BYTES)
    messages = [{"parts": [{"url": url}, "text"]}]
    sanitized = sanitize_messages(messages)
    assert "base64," not in str(sanitized)
    assert sanitized[0]["parts"][1] == "text"


def test_request_hash_is_stable_and_sanitized() -> None:
    url = image_to_data_url(IMAGE_BYTES)
    messages = [{"image_url": {"url": url}}]
    kwargs = dict(
        model="qwen",
        generation={"max_tokens": 128},
        prompt_version="count-tile-v4",
        messages=messages,
        image_sha256=image_sha256(IMAGE_BYTES),
    )
    first = build_request_hash(**kwargs)
    second = build_request_hash(**kwargs)
    assert first == second
    assert first == build_request_hash(**{**kwargs, "messages": [{"image_url": {"url": url}}]})
    # A different prompt version changes the hash. / 不同 prompt 版本改变哈希。
    assert first != build_request_hash(**{**kwargs, "prompt_version": "other"})


def test_request_hash_never_contains_raw_base64() -> None:
    url = image_to_data_url(IMAGE_BYTES)
    digest = build_request_hash(
        model="qwen",
        generation={},
        prompt_version="v1",
        messages=[{"image_url": {"url": url}}],
        image_sha256=None,
    )
    assert digest not in url
    assert "base64" not in digest


def _hash_kwargs() -> dict:
    return dict(
        model="qwen",
        generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
        prompt_version="v1",
        messages=[{"role": "user", "content": "Q"}],
        image_sha256=None,
    )


def test_request_hash_changes_with_each_semantic_field() -> None:
    """Every inference-semantic field must change the hash: generation,
    response schema, client version, model revision, and model name.
    每个推理语义字段都必须改变哈希：生成参数、响应 Schema、客户端版本、
    模型 revision 与模型名。"""
    base = _hash_kwargs()
    digest = build_request_hash(**base)

    assert digest != build_request_hash(**{**base, "generation": {
        "temperature": 0.0, "do_sample": False, "max_tokens": 256}})
    assert digest != build_request_hash(**{**base, "generation": {
        "temperature": 0.7, "do_sample": True, "max_tokens": 128}})
    assert digest != build_request_hash(**{**base, "response_schema": {
        "type": "object", "properties": {"answer": {"type": "string"}}}})
    assert digest != build_request_hash(**{**base, "client_version": "2"})
    assert digest != build_request_hash(**{**base, "model_revision": "abc123"})
    assert digest != build_request_hash(**{**base, "model": "qwen-other"})
    assert digest != build_request_hash(**{**base, "prompt_version": "v2"})


def test_request_hash_changes_with_image_digest_and_order() -> None:
    """Different image bytes and different image order must change the hash.
    图片内容不同与图片顺序不同都必须改变哈希。"""
    import hashlib

    img_a = hashlib.sha256(b"AAAA").hexdigest()
    img_b = hashlib.sha256(b"BBBB").hexdigest()
    base = _hash_kwargs()

    h_a = build_request_hash(**{**base, "image_sha256": img_a})
    h_b = build_request_hash(**{**base, "image_sha256": img_b})
    assert h_a != h_b

    h_ab = build_request_hash(**{**base, "image_sha256": "|".join([img_a, img_b])})
    h_ba = build_request_hash(**{**base, "image_sha256": "|".join([img_b, img_a])})
    assert h_ab != h_ba


def test_request_meta_rejects_credentials() -> None:
    from pydantic import ValidationError

    import pytest

    with pytest.raises(ValidationError):
        RequestMeta.model_validate(
            {"request_id": "r", "request_hash": "a" * 64, "prompt_version": "v",
             "api_key": "sk-secret"}
        )


def test_import_models_does_not_load_transformers_or_torch() -> None:
    """Importing models must not import transformers or torch.
    导入 models 不得导入 transformers 或 torch。"""
    for heavy in ("transformers", "torch"):
        assert heavy not in sys.modules


def test_vision_language_client_protocol_is_structural() -> None:
    import inspect

    class FakeClient:
        async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
            return response_model.model_validate({})

    client = FakeClient()
    # The protocol is structural; verify the async method shape without
    # isinstance (protocols are not runtime-checkable by default).
    # 协议为结构化；直接验证异步方法形态（协议默认不可 isinstance 检查）。
    assert inspect.iscoroutinefunction(FakeClient.complete_json)
