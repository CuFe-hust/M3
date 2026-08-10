"""Structured local event persistence without secret payloads.

不包含密钥载荷的本地结构化事件持久化。EventWriter 以原子替换方式追加
JSONL（每次写入读现有内容 + 新行，写临时文件后 replace），并在进程内
锁保护下执行整个 read-compose-write-replace 序列：

- JSONL writers are safe for concurrent writers within one Python process.
- Cross-process concurrent append is not supported by the current workflow
  layer.

任何写入失败都不会破坏既有事件；details 递归拒绝敏感键与敏感值前缀。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence, Set
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Sensitive key names rejected in details (normalized before matching).
# 详情中拒绝的敏感键名（匹配前先归一化）。
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "secret",
        "token",
        "password",
        "base64",
        "credential",
        "private_key",
        "image_data_url",
    }
)
# High-risk sensitive value prefixes, checked after lstrip().lower().
# 高风险敏感值前缀；在 lstrip().lower() 后检查。
_SENSITIVE_VALUE_PREFIXES = (
    "sk-",
    "bearer ",
    "data:image/",
    "-----begin private key-----",
)

# Per-path locks serialize read-compose-write-replace within one process.
# 按路径的锁使读-组-写-替换在单进程内串行化。
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class RunEvent(BaseModel):
    """One auditable state change or error event.
    一条可审计的状态变更或错误事件。"""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    event: str
    sample_id: str | None = None
    tile_id: str | None = None
    error_code: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EventWriter:
    """Append JSONL events while rejecting accidental secret field names.
    追加 JSONL 事件并拒绝意外的密钥字段名。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        event: str,
        *,
        sample_id: str | None = None,
        tile_id: str | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> RunEvent:
        """Persist an event with safe structured details. The whole
        read-compose-write-replace sequence runs under a per-path lock, so
        concurrent writers inside one process never lose lines.
        使用安全的结构化详情持久化事件。整个读-组-写-替换序列在按路径锁
        内执行，单进程并发写入不会丢行。"""

        safe_details = details or {}
        _reject_secrets(safe_details, "event details")
        record = RunEvent(
            timestamp=datetime.now(timezone.utc),
            event=event,
            sample_id=sample_id,
            tile_id=tile_id,
            error_code=error_code,
            details=safe_details,
        )
        line = (
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            + "\n"
        )
        with _path_lock(self.path):
            existing = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(existing + line, encoding="utf-8")
            _atomic_replace(temporary, self.path)
        return record


def _atomic_replace(source: Path, target: Path) -> None:
    """Replace target with source, retrying transient Windows file locks
    (antivirus/indexers hold short-lived handles). The per-path lock already
    serializes writers; this retry only absorbs filesystem-level transients.
    用 source 替换 target，对 Windows 瞬态文件锁（杀毒/索引器持有短时句柄）
    有限重试。按路径锁已串行化写入者；此重试只吸收文件系统级瞬态。"""

    for attempt in range(5):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))


def _path_lock(path: Path) -> threading.Lock:
    """Return the process-wide lock for one canonical path. The key is the
    resolved path identity (strict=False), so lexical aliases such as
    'a/../events.jsonl' and 'events.jsonl' share one lock; resolution is used
    only for lock identity, never to relocate stored artifacts.
    返回某一路径的进程内共享锁。锁键为解析后的路径身份（strict=False），
    使 'a/../events.jsonl' 与 'events.jsonl' 等词法别名共享一把锁；解析
    仅用于锁身份，绝不改变实际产物存储位置。"""

    key = str(Path(path).resolve(strict=False))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.Lock())


def _reject_secrets(value: Any, where: str) -> None:
    """Recursively reject sensitive keys and high-risk value prefixes in a
    payload of any Mapping/Sequence/Set nesting; error messages never echo
    the offending value. str is handled first so it is never treated as a
    generic container. 递归拒绝任意 Mapping/Sequence/Set 嵌套载荷中的敏感键
    与高风险值前缀；错误消息绝不回显违规值。str 最先处理，绝不当作通用
    容器。"""

    if isinstance(value, str):
        normalized = value.lstrip().lower()
        for prefix in _SENSITIVE_VALUE_PREFIXES:
            if normalized.startswith(prefix):
                raise ValueError(f"{where} contains a sensitive value prefix")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized_key in _SENSITIVE_KEYS:
                raise ValueError(f"{where} contains a sensitive key")
            _reject_secrets(item, f"{where}.{key}")
        return
    if isinstance(value, Set):
        for item in value:
            _reject_secrets(item, where)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{where}[{index}]")
        return
