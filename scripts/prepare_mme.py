"""MME-RealWorld Remote Sensing preprocessing -> questions.jsonl.

The committee stratified subset (MMERealRS_Stratified_914) provides
core_annotations.jsonl with question/options/answer and image_path;
images live under the subset images/ directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_TASK_MAP = {"color": "color", "count": "count", "position": "position"}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(prog="prepare_mme")
    parser.add_argument("--annotations", required=True, type=Path, help="core_annotations.jsonl")
    parser.add_argument("--images-root", required=True, type=Path, help="subset dir containing images/")
    parser.add_argument("--output", required=True, type=Path, help="questions.jsonl")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="copy referenced images under the output images/ directory so "
        "annotations and assets share one asset_root",
    )
    args = parser.parse_args()
    prep_image_root = args.output.parent / "images"
    if args.copy_images:
        prep_image_root.mkdir(parents=True, exist_ok=True)
        print(f"mme: copying images to {prep_image_root}", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with args.annotations.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            question_type = str(record.get("question_type", "")).lower()
            task = _TASK_MAP.get(question_type)
            if task is None:
                print(f"warn: line {line_number} unsupported question_type {question_type!r}, skipping",
                      file=sys.stderr)
                continue
            question = str(record.get("question", "")).strip()
            options = record.get("options") or {}
            answer = str(record.get("answer", "")).strip()
            if not question or not options or not answer:
                print(f"warn: line {line_number} missing fields, skipping", file=sys.stderr)
                continue
            sample_id = str(record.get("sample_id", f"mme-{line_number}"))
            if sample_id in seen:
                print(f"warn: duplicate sample_id {sample_id}, skipping", file=sys.stderr)
                continue
            seen.add(sample_id)
            image_path = (args.images_root / str(record["image_path"])).resolve()
            if not image_path.is_file():
                print(f"warn: missing image {image_path}, skipping", file=sys.stderr)
                continue
            if args.copy_images:
                destination = prep_image_root / image_path.name
                if not destination.is_file():
                    shutil.copy2(image_path, destination)
                image_value = destination.relative_to(args.output.parent).as_posix()
            else:
                image_value = image_path.as_posix()
            choices = [f"{letter}. {value}" for letter, value in sorted(options.items())]
            rows.append(
                {
                    "id": sample_id,
                    "domain": "Remote_Sensing",
                    "task": task,
                    # relative path under the prep dir (adapter resolves
                    # root / image); asset_root is the prep dir itself
                    "image": image_value,
                    "question": question + " Answer with only the option letter (A, B, C, D, or E).",
                    "choices": choices,
                    "answer": answer,
                }
            )

    _write_jsonl(args.output, rows)
    print(f"mme: wrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
