"""Shared contracts, caching, and safe request metadata for model clients.
模型客户端共用的协议、缓存和安全请求元数据。
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field


ModelT = TypeVar("ModelT", bound=BaseModel)


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
    ) -> ModelT:
        """Return one schema-validated JSON response.
        返回一条经 Schema 校验的 JSON 响应。
        """


class CacheEntry(BaseModel):
    """Cached structured response without image payloads or credentials.
    不包含图像载荷或凭据的缓存结构化响应。
    """

    model_config = ConfigDict(extra="forbid")

    raw_response: str
    parsed: dict[str, Any]


class JsonResponseCache:
    """File cache keyed by caller-supplied stable request hashes.
    以调用方提供的稳定请求哈希为键的文件缓存。
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, request_hash: str) -> CacheEntry | None:
        """Load one cached entry if it exists and remains valid JSON.
        若缓存存在且仍为合法 JSON，则加载一条缓存记录。
        """

        path = self._path(request_hash)
        if not path.is_file():
            return None
        return CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, request_hash: str, entry: CacheEntry) -> None:
        """Persist one cache entry using UTF-8 JSON.
        使用 UTF-8 JSON 持久化一条缓存记录。
        """

        path = self._path(request_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(entry.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _path(self, request_hash: str) -> Path:
        if not request_hash or any(character not in "0123456789abcdef" for character in request_hash.lower()):
            raise ValueError("request_hash must be a hexadecimal digest")
        return self.root / f"{request_hash}.json"


def image_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Encode image bytes as one OpenAI-compatible data URL.
    将图像字节编码为一条 OpenAI 兼容的数据 URL。
    """

    if not image_bytes:
        raise ValueError("image_bytes must not be empty")
    if not mime.startswith("image/"):
        raise ValueError("mime must identify an image type")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_request_hash(
    *,
    model: str,
    generation: Mapping[str, Any],
    prompt_version: str,
    messages: Sequence[Mapping[str, Any]],
    image_sha256: str | None,
    tile_geometry: Mapping[str, Any] | None = None,
    target_spec: Mapping[str, Any] | None = None,
) -> str:
    """Hash cache inputs while replacing data URLs with their digest and size.
    对缓存输入计算哈希，同时以摘要和大小替换数据 URL。
    """

    payload = {
        "model": model,
        "generation": generation,
        "prompt_version": prompt_version,
        "messages": sanitize_messages(messages),
        "image_sha256": image_sha256,
        "tile_geometry": tile_geometry,
        "target_spec": target_spec,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sanitize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove raw Base64 data URLs before logs, hashes, or artifacts are written.
    在写入日志、哈希或产物前移除原始 Base64 数据 URL。
    """

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
