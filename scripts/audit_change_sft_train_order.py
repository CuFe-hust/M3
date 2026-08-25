#!/usr/bin/env python3
"""Audit exact content invariants and source mixing between two Change corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.multimodal_sft.change_target_migration import write_json_atomic
from training.multimodal_sft.change_train_order_audit import audit_train_order


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit deterministic Change SFT train ordering.")
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_train_order(old_dir=args.old_dir, new_dir=args.new_dir)
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
