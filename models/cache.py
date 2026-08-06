"""File-backed structured response cache for model clients.

模型客户端的文件型结构化响应缓存。原子写入；拒绝非十六进制 key；
缓存条目绝不包含图片 Base64 或凭据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


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
        """Load one cached entry if it exists and remains valid JSON.
        若缓存存在且仍为合法 JSON，则加载一条缓存记录。"""
        path = self._path(request_hash)
        if not path.is_file():
            return None
        return CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, request_hash: str, entry: CacheEntry) -> None:
        """Persist one cache entry using UTF-8 JSON and an atomic replace.
        使用 UTF-8 JSON 与原子替换持久化一条缓存记录。"""
        path = self._path(request_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(entry.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _path(self, request_hash: str) -> Path:
        if not request_hash or any(
            character not in "0123456789abcdef" for character in request_hash.lower()
        ):
            raise ValueError("request_hash must be a hexadecimal digest")
        return self.root / f"{request_hash}.json"
