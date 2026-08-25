#!/usr/bin/env python3
"""Compare legacy and rebuilt Change SFT corpora by episode id. / 按 episode id 对比语料。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.multimodal_sft.change_target_migration import compare_corpora, write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a legacy Change SFT corpus with a canonical v2 corpus.")
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = compare_corpora(old_dir=args.old_dir, new_dir=args.new_dir)
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
