"""XLRS-Bench preprocessing: HF arrow datasets -> tasks.jsonl.

The committee subset (XLRSBench_Native_690) provides English caption,
English visual grounding (fine-grained) and Lite VQA. Images are embedded
as PIL objects inside the arrow files and are exported under the output
images/ directory (assets stay inside the prep folder).
Chinese caption / grounding are not part of the subset and remain N/A.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000  # XLRS tiles are 4096x4096

_GROUNDING_SUFFIX = (
    " Return only the bounding box as [x1, y1, x2, y2] with coordinates "
    "normalized from 0 to 100."
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _export_image(image: Any, index: int, image_dir: Path) -> str:
    if isinstance(image, (list, tuple)):
        image = image[0]
    image_dir.mkdir(parents=True, exist_ok=True)
    name = f"xlrs_{index:06d}.jpg"
    image.convert("RGB").save(image_dir / name, quality=90)
    return (image_dir / name).resolve().as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(prog="prepare_xlrs")
    parser.add_argument("--root", required=True, type=Path, help="XLRSBench_Native_690 dir")
    parser.add_argument("--output", required=True, type=Path, help="output dir (tasks.jsonl + images/)")
    parser.add_argument("--split", default="public_test")
    args = parser.parse_args()

    from datasets import load_from_disk

    image_dir = args.output / "images"
    rows: list[dict[str, Any]] = []

    caption_ds = load_from_disk(args.root / "XLRS-Bench_caption_en" / "train")
    for index, row in enumerate(caption_ds, start=1):
        image = _export_image(row["image"], 100000 + index, image_dir)
        answer = row["answer"]
        if isinstance(answer, (list, tuple)):
            answer = answer[0]
        rows.append(
            {
                "id": f"full-caption-en-{index:06d}",
                "split": args.split,
                "variant": "full",
                "language": "en",
                "task": "caption",
                "image": image,
                "prompt": "Describe this remote sensing image in detail.",
                "answer": str(answer).strip(),
            }
        )

    grounding_ds = load_from_disk(args.root / "XLRS-Bench_visual_grounding_en" / "test")
    for index, row in enumerate(grounding_ds, start=1):
        image = _export_image(row["image"], 200000 + index, image_dir)
        bbox = row.get("bbox") or row.get("box") or row.get("boxes")
        if not bbox:
            print(f"warn: grounding row {index} has no bbox, skipping", file=sys.stderr)
            continue
        # arrow bbox is 0-1 normalized; the prompt contract is 0-100
        normalized_bbox = [float(value) * 100.0 for value in bbox]
        rows.append(
            {
                "id": f"full-grounding-en-{index:06d}",
                "split": args.split,
                "variant": "full",
                "language": "en",
                "task": "grounding",
                "image": image,
                "prompt": str(row["question"]).strip() + _GROUNDING_SUFFIX,
                "boxes": [normalized_bbox],
                "scope": "fine_grained",
            }
        )

    lite_ds = load_from_disk(args.root / "XLRS-Bench-lite" / "train")
    for index, row in enumerate(lite_ds, start=1):
        image = _export_image(row["image"], 300000 + index, image_dir)
        choices = row.get("multi-choice options") or row.get("choices") or []
        answer = str(row.get("answer", "")).strip()
        if not choices or not answer:
            print(f"warn: lite row {index} missing choices/answer, skipping", file=sys.stderr)
            continue
        # L3 task label: "Land use classification/Overall Land use
        # classification" -> "overall_land_use_classification"
        category = str(row.get("category", "") or "")
        tail = category.split("/")[-1].strip().lower()
        l3 = re.sub(r"[^a-z0-9]+", "_", tail).strip("_")
        rows.append(
            {
                "id": f"lite-vqa-en-{index:06d}",
                "split": args.split,
                "variant": "lite",
                "language": "en",
                "task": "vqa",
                "image": image,
                "prompt": str(row["question"]).strip()
                + " Answer with only the option letter (A, B, C, D, or E).",
                "choices": [str(choice) for choice in choices],
                "answer": answer,
                "l3": l3,
            }
        )

    _write_jsonl(args.output / "tasks.jsonl", rows)
    print(f"xlrs: wrote {len(rows)} rows -> {args.output}/tasks.jsonl (images under {image_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
