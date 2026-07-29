"""Structured local event persistence without secret payloads.
不包含密钥载荷的本地结构化事件持久化。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from spacers_agent.errors import ErrorCode


class RunEvent(BaseModel):
    """One auditable state change or error event.
    一条可审计的状态变更或错误事件。
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    event: str
    sample_id: str | None = None
    tile_id: str | None = None
    error_code: ErrorCode | None = None
    details: dict[str, Any] = {}


class EventWriter:
    """Append JSONL events while rejecting accidental secret field names.
    追加 JSONL 事件并拒绝意外的密钥字段名。
    """

    _FORBIDDEN_DETAIL_KEYS = frozenset({"api_key", "authorization", "base64", "image_data_url"})

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        event: str,
        *,
        sample_id: str | None = None,
        tile_id: str | None = None,
        error_code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
    ) -> RunEvent:
        """Persist an event with safe structured details.
        使用安全的结构化详情持久化事件。
        """

        safe_details = details or {}
        forbidden = self._FORBIDDEN_DETAIL_KEYS.intersection(safe_details)
        if forbidden:
            raise ValueError(f"Event details must not contain secret fields: {sorted(forbidden)}")
        record = RunEvent(
            timestamp=datetime.now(timezone.utc),
            event=event,
            sample_id=sample_id,
            tile_id=tile_id,
            error_code=error_code,
            details=safe_details,
        )
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
        return record
