#!/usr/bin/env python3
"""Audit legacy/current Change SFT target rows without modifying them. / 只读审计目标。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.multimodal_sft.change_target_migration import audit_target_contract, write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit ChangeAgent SFT target-contract compatibility.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_target_contract(train=args.train, validation=args.validation, manifest=args.manifest)
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["migration_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
