"""Shared model-client contracts, request hashing, and sanitization.

模型客户端共用协议、请求哈希与脱敏。本模块不依赖 data / agents /
application，不读取任何 API key，不触发 transformers / torch 导入。
缓存与图像工具分别位于 models/cache.py 与 models/images.py。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

ModelT = TypeVar("ModelT", bound=BaseModel)

# Sensitive key names and high-risk value prefixes that must never appear in a
# cache identity. Keys are matched after normalization; values after
# lstrip().lower(). / 缓存身份中绝不能出现的敏感键名与高风险值前缀；键在
# 归一化后精确匹配，值在 lstrip().lower() 后检查前缀。
_SENSITIVE_IDENTITY_KEYS = frozenset({
    "api_key", "apikey", "authorization", "access_token", "refresh_token",
    "private_key", "password", "credential",
})
_IDENTITY_VALUE_PREFIXES = (
    "sk-",
    "bearer ",
    "data:image/",
    "-----begin private key-----",
)


@dataclass(frozen=True)
class ModelCacheIdentity:
    """Stable, JSON-safe identity of one model client for cache keying.

    The visual agent builds its request hash exclusively from this object so
    the hashed model name, generation parameters, client version, and revision
    can never drift from the client that actually runs the call.
    单个模型客户端用于缓存键的稳定、JSON 安全身份。视觉 Agent 只从该对象
    构建请求哈希，使参与哈希的模型名、生成参数、客户端版本与 revision 永远
    不会与实际执行调用的客户端漂移。"""

    model: str
    generation: Mapping[str, Any]
    client_version: str
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.client_version:
            raise ValueError("client_version must not be empty")
        _validate_identity_value(self.generation, "generation")


class CacheIdentifiedClient(Protocol):
    """A model client exposing its stable cache identity.
    暴露其稳定缓存身份的模型客户端。"""

    @property
    def cache_identity(self) -> ModelCacheIdentity: ...


def _validate_identity_value(value: Any, where: str) -> None:
    """Require JSON-safe, finite, secret-free identity content.
    要求身份内容 JSON 安全、数值有限、不含密钥。"""
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        normalized = value.lstrip().lower()
        for prefix in _IDENTITY_VALUE_PREFIXES:
            if normalized.startswith(prefix):
                raise ValueError(f"{where} contains a sensitive value prefix")
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} contains a non-finite number")
        return
    if isinstance(value, Path):
        raise ValueError(f"{where} contains a Path object")
    if isinstance(value, (set, bytes, bytearray)):
        raise ValueError(f"{where} contains a {type(value).__name__}")
    if callable(value):
        raise ValueError(f"{where} contains a callable")
    if isinstance(value, list):
        for item in value:
            _validate_identity_value(item, where)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized_key in _SENSITIVE_IDENTITY_KEYS:
                raise ValueError(f"{where} contains a sensitive key {key!r}")
            _validate_identity_value(item, where)
        return
    raise ValueError(f"{where} contains unsupported type {type(value).__name__}")


class RequestMeta(BaseModel):
    """Traceable request metadata that deliberately excludes credentials and Base64.
    可追踪的请求元数据，刻意排除凭据和 Base64。
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    request_hash: str
    prompt_version: str
    sample_id: str | None = None
    tile_id: str | None = None
    image_sha256: str | None = None
    artifact_dir: Path | None = None


class VisionLanguageClient(Protocol):
    """Protocol shared by live and offline structured vision-language clients.
    线上与离线结构化视觉语言客户端共用的协议。
    """

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        request_meta: RequestMeta,
        max_tokens: int | None = None,
    ) -> ModelT:
        """Return one schema-validated JSON response.
        返回一条经 Schema 校验的 JSON 响应。
        """


def build_request_hash(
    *,
    model: str,
    generation: Mapping[str, Any],
    prompt_version: str,
    messages: Sequence[Mapping[str, Any]],
    image_sha256: str | None,
    tile_geometry: Mapping[str, Any] | None = None,
    target_spec: Mapping[str, Any] | None = None,
    response_schema: Mapping[str, Any] | None = None,
    client_version: str | None = None,
    model_revision: str | None = None,
) -> str:
    """Hash cache inputs while replacing data URLs with their digest and size.
    The sanitized payload makes the hash stable across machines and runs.
    response_schema / client_version / model_revision are included so cache
    keys cover the full inference semantics.
    对缓存输入计算哈希，同时以摘要和大小替换数据 URL；脱敏载荷使哈希
    跨机器、跨运行稳定。response_schema / client_version / model_revision
    参与哈希，使缓存键覆盖完整推理语义。"""

    payload = {
        "model": model,
        "generation": generation,
        "prompt_version": prompt_version,
        "messages": sanitize_messages(messages),
        "image_sha256": image_sha256,
        "tile_geometry": tile_geometry,
        "target_spec": target_spec,
        "response_schema": response_schema,
        "client_version": client_version,
        "model_revision": model_revision,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sanitize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove raw Base64 data URLs before logs, hashes, or artifacts are written.
    在写入日志、哈希或产物前移除原始 Base64 数据 URL。"""

    return [_sanitize_value(message) for message in messages]


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
        return {
            "redacted_data_url_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "encoded_bytes": len(value.encode("utf-8")),
        }
    if isinstance(value, Mapping):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value
