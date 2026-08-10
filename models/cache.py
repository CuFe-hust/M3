"""File-backed structured response cache for model clients.

模型客户端的文件型结构化响应缓存。原子写入；拒绝非十六进制 key；
缓存条目绝不包含图片 Base64 或凭据。损坏/过期条目以稳定错误类型报告；
写失败以 CacheWriteError 报告且不残留临时文件。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class ModelCacheError(RuntimeError):
    """Raised when a cache entry cannot be read or validated.
    缓存条目无法读取或校验时抛出。"""


class CorruptCacheEntryError(ModelCacheError):
    """Raised when a cache file is corrupted (bad JSON, bad encoding, or
    schema-invalid content). 缓存文件损坏（坏 JSON/编码或内容不合 Schema）时抛出。"""


class CacheWriteError(ModelCacheError):
    """Raised when a cache entry cannot be persisted (mkdir, write, replace,
    or cleanup failure). 缓存条目无法持久化（mkdir/写入/替换/清理失败）时抛出。"""


# Key names and value prefixes that must never appear in a cached entry.
# Key names are matched exactly after normalization; values after
# lstrip().lower(). / 缓存条目中绝不能出现的键名与值前缀；键归一化后精确
# 匹配，值在 lstrip().lower() 后检查前缀。
_BLOCKED_CACHE_KEYS = frozenset({
    "api_key", "apikey", "authorization", "access_token", "refresh_token",
    "private_key", "password", "credential",
})
_BLOCKED_CACHE_VALUE_PREFIXES = (
    "sk-",
    "bearer ",
    "data:image/",
    "-----begin private key-----",
)


class CacheEntry(BaseModel):
    """Cached structured response without image payloads or credentials.
    不包含图像载荷或凭据的缓存结构化响应。"""

    model_config = ConfigDict(extra="forbid")

    raw_response: str
    parsed: dict[str, Any]

    @model_validator(mode="after")
    def validate_no_secrets(self) -> "CacheEntry":
        """Reject Base64 image data and credentials in either field. raw
        response JSON is checked recursively; unparseable text is scanned for
        high-risk markers per repair attempt.
        拒绝 raw_response 与 parsed 中的 Base64 图像数据或凭据。raw response
        的 JSON 递归检查；不可解析文本按修复尝试逐段扫描高风险标记。"""
        _check_raw_response_safe(self.raw_response)
        _check_cache_content_safe(self.parsed, "parsed")
        return self


# Separator between rendered repair attempts, e.g. "[response_attempt=1]".
# 渲染出的各修复尝试之间的分隔行，如 "[response_attempt=1]"。
_ATTEMPT_SEPARATOR = re.compile(r"(?m)^\[response_attempt=\d+\]\s*$")

# High-risk markers scanned in unparseable (non-JSON) raw responses.
# 在不可解析（非 JSON）raw response 中扫描的高风险标记。
_BLOCKED_RAW_MARKERS = (
    "data:image/",
    "-----begin private key-----",
    '"api_key"',
    '"apikey"',
    '"authorization"',
    '"access_token"',
    '"refresh_token"',
    '"private_key"',
    '"password"',
    '"credential"',
)


def _check_raw_response_safe(raw_response: str) -> None:
    """Check the whole raw response; parseable JSON is checked recursively,
    otherwise each repair attempt is checked separately so sensitive history
    in any attempt rejects the entry.
    检查整段 raw response；可解析 JSON 递归检查，否则逐修复尝试检查，任何
    尝试段中的敏感历史都会拒绝该条目。"""
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        for segment in re.split(_ATTEMPT_SEPARATOR, raw_response):
            stripped = segment.strip()
            if stripped:
                _check_raw_response_segment(stripped)
        return
    _check_cache_content_safe(parsed, "raw_response_json")


def _check_raw_response_segment(segment: str) -> None:
    """Check one repair attempt: parse it when possible, otherwise scan markers.
    检查单个修复尝试：可解析时递归检查，否则扫描高风险标记。"""
    try:
        parsed = json.loads(segment)
    except json.JSONDecodeError:
        _scan_unstructured_raw_response(segment)
    else:
        _check_cache_content_safe(parsed, "raw_response_json")


def _scan_unstructured_raw_response(raw_response: str) -> None:
    """Scan unparseable raw text for high-risk markers without flagging
    ordinary natural-language words. 扫描不可解析原始文本中的高风险标记，
    不误报普通自然语言单词。"""
    normalized = raw_response.lower()
    for marker in _BLOCKED_RAW_MARKERS:
        if marker in normalized:
            raise ValueError("cache raw_response contains a blocked marker")
    stripped = raw_response.lstrip().lower()
    if stripped.startswith("sk-") or stripped.startswith("bearer "):
        raise ValueError("cache raw_response starts with a blocked prefix")


def _normalize_cache_key(key: Any) -> str:
    return str(key).lower().replace("-", "_").replace(" ", "_")


def _check_cache_content_safe(value: Any, where: str) -> None:
    """Reject blocked keys and high-risk value prefixes recursively; plain
    words such as "token" or "secret" inside normal text are allowed.
    递归拒绝被禁键与高风险值前缀；普通文本中的 token/secret 等单词不误报。"""
    if isinstance(value, str):
        normalized = value.lstrip().lower()
        for prefix in _BLOCKED_CACHE_VALUE_PREFIXES:
            if normalized.startswith(prefix):
                raise ValueError(f"cache {where} contains a blocked value prefix")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if _normalize_cache_key(key) in _BLOCKED_CACHE_KEYS:
                raise ValueError(f"cache {where} contains blocked key {key!r}")
            _check_cache_content_safe(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_cache_content_safe(item, f"{where}[{index}]")


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
        temporary files are cleaned up on failure. Persistence failures raise
        CacheWriteError and never leak OSError.
        使用 UTF-8 JSON 与原子替换持久化一条缓存记录；失败时清理临时文件。
        持久化失败抛出 CacheWriteError，不泄漏 OSError。"""
        path = self._path(request_hash)
        digest_label = request_hash[:8]
        temporary = path.with_suffix(path.suffix + ".tmp")
        primary_error: Exception | None = None
        stage = "mkdir"
        try:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                stage = "write_text"
                temporary.write_text(entry.model_dump_json(indent=2) + "\n", encoding="utf-8")
                stage = "replace"
                temporary.replace(path)
            except OSError as error:
                primary_error = error
        finally:
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError as error:
                # Cleanup must never mask the original persistence error.
                # 清理错误不得掩盖原始持久化错误。
                if primary_error is None:
                    raise CacheWriteError(
                        f"cache write cleanup failed for {digest_label}: "
                        f"{type(error).__name__}"
                    ) from error
        if primary_error is not None:
            raise CacheWriteError(
                f"cache write failed for {digest_label} ({stage}): "
                f"{type(primary_error).__name__}"
            ) from primary_error

    def _path(self, request_hash: str) -> Path:
        if not request_hash or any(
            character not in "0123456789abcdef" for character in request_hash.lower()
        ):
            raise ValueError("request_hash must be a hexadecimal digest")
        return self.root / f"{request_hash}.json"
