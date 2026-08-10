"""Public `inspect-data` CLI command: read-only dataset root audit.

公开 `inspect-data` CLI 命令：只读数据集根审计。委托
data.validation.audit_dataset_root（quick 抽样 / full 全量）；绝不修改
源目录或标注。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.validation import audit_dataset_root

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


def run_inspect_data(args: argparse.Namespace) -> int:
    """Audit one dataset root and print the report as JSON.
    审计一个数据集根并以 JSON 输出报告。"""

    try:
        report = audit_dataset_root(
            Path(args.root),
            scan_mode=args.scan_mode,
        )
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as error:
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}"}),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    payload = {"status": "ok", "report": report.model_dump(mode="json")}
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )
    return EXIT_OK
