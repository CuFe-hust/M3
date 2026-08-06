"""File-backed structured response cache for model clients.

模型客户端的文件型结构化响应缓存。原子写入；拒绝非十六进制 key；
缓存条目绝不包含图片 Base64 或凭据。损坏/过期条目以稳定错误类型报告。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class ModelCacheError(RuntimeError):
    """Raised when a cache entry cannot be read or validated.
    缓存条目无法读取或校验时抛出。"""


class CorruptCacheEntryError(ModelCacheError):
    """Raised when a cache file is corrupted (bad JSON, bad encoding, or
    schema-invalid content). 缓存文件损坏（坏 JSON/编码或内容不合 Schema）时抛出。"""


class CacheEntry(BaseModel):
    """Cached structured response without image payloads or credentials.
    不包含图像载荷或凭据的缓存结构化响应。"""

    model_config = ConfigDict(extra="forbid")

    raw_response: str
    parsed: dict[str, Any]


class JsonResponseCache:
    """File cache keyed by caller-supplied stable request hashes.
    以调用方提供的稳定请求哈希为键的文件缓存。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, request_hash: str) -> CacheEntry | None:
        """Load one cached entry; corrupt entries raise CorruptCacheEntryError.
        加载一条缓存记录；损坏条目抛出 CorruptCacheEntryError。"""
        path = self._path(request_hash)
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise CorruptCacheEntryError(
                f"cache entry {request_hash[:8]} is unreadable: {type(error).__name__}"
            ) from error
        try:
            return CacheEntry.model_validate_json(text)
        except (ValidationError, ValueError) as error:
            raise CorruptCacheEntryError(
                f"cache entry {request_hash[:8]} is invalid: {type(error).__name__}"
            ) from error

    def save(self, request_hash: str, entry: CacheEntry) -> None:
        """Persist one cache entry using UTF-8 JSON and an atomic replace;
        temporary files are cleaned up on failure.
        使用 UTF-8 JSON 与原子替换持久化一条缓存记录；失败时清理临时文件。"""
        path = self._path(request_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(entry.model_dump_json(indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _path(self, request_hash: str) -> Path:
        if not request_hash or any(
            character not in "0123456789abcdef" for character in request_hash.lower()
        ):
            raise ValueError("request_hash must be a hexadecimal digest")
        return self.root / f"{request_hash}.json"
