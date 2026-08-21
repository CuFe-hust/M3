"""Normalize a generic CSV/JSONL change dataset into training JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from training.change_head.schema import ChangeTrainingRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.suffix.lower() == ".csv":
        rows = list(csv.DictReader(args.input.open("r", encoding="utf-8", newline="")))
    else:
        rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [ChangeTrainingRecord.model_validate(row) for row in rows]
    if len({record.sample_id for record in records}) != len(records):
        raise SystemExit("duplicate sample_id")
    groups = {record.group_id: record.split for record in records}
    if len(groups) != len({(record.group_id, record.split) for record in records}):
        raise SystemExit("group_id crosses splits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

