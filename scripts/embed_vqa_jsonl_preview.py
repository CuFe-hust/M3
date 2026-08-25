#!/usr/bin/env python3
"""Embed the first 500 complete JSONL records into the local preview HTML.

将前 500 条完整 JSONL 记录嵌入本地预览 HTML，生成后不依赖 HTTP 或 fetch。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "20260824_VQA_agent_finetuning"
HTML_PATH = DATA_DIR / "index.html"
JSONL_PATH = DATA_DIR / "train.jsonl"
MARKER = "/*__EMBEDDED_RECORDS__*/[]"


def main() -> int:
    """Validate and embed source records atomically. / 校验并原子嵌入源记录。"""
    records = []
    with JSONL_PATH.open("r", encoding="utf-8") as handle:
        for _ in range(500):
            line = handle.readline()
            if not line:
                raise ValueError("train.jsonl contains fewer than 500 records")
            records.append(json.loads(line))
    html = HTML_PATH.read_text(encoding="utf-8")
    if html.count(MARKER) != 1:
        raise ValueError("HTML embed marker is missing or ambiguous")
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    rendered = html.replace(MARKER, payload)
    with NamedTemporaryFile("w", encoding="utf-8", dir=DATA_DIR, delete=False) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, HTML_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
