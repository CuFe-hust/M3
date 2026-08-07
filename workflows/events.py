"""Structured local event persistence without secret payloads.

不包含密钥载荷的本地结构化事件持久化。EventWriter 以原子替换方式追加
JSONL（每次写入读现有内容 + 新行，写临时文件后 replace），单写者场景下
任何写入失败都不会破坏既有事件。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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

    _FORBIDDEN_DETAIL_KEYS = frozenset(
        {"api_key", "authorization", "base64", "image_data_url"}
    )

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
        """Persist an event with safe structured details.
        使用安全的结构化详情持久化事件。"""

        safe_details = details or {}
        forbidden = self._FORBIDDEN_DETAIL_KEYS.intersection(safe_details)
        if forbidden:
            raise ValueError(
                f"Event details must not contain secret fields: {sorted(forbidden)}"
            )
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
        existing = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(existing + line, encoding="utf-8")
        temporary.replace(self.path)
        return record
