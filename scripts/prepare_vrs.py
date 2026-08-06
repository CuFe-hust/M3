"""VRSBench preprocessing: EVAL json files -> caption/grounding/vqa.jsonl.

The committee subset keeps the official VRSBench EVAL file format:
  Caption:    {image_id, ground_truth}
  Referring:  {image_id, question, ground_truth: "{<x1><y1><x2><y2>}...", unique}
  VQA:        {image_id, question, ground_truth, type}
Images live under Images_val/Images_val/<image_id> (nested dir).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

_BOX_GROUPS_RE = re.compile(r"\{<([\d.]+)><([\d.]+)><([\d.]+)><([\d.]+)>\}")

# VRSBench EVAL "type" -> adapter vqa_category enum mapping.
# The official EVAL file uses 12 type labels; the adapter's 10-class
# category enum is derived from the VRSBench paper, so the two extra
# labels (rural-or-urban, image-level) map to the closest categories.
_GROUNDING_SUFFIX = (
    " Return only the bounding box as [x1, y1, x2, y2] with coordinates "
    "normalized from 0 to 100."
)

_VQA_CATEGORY_MAP = {
    "object category": "category",
    "object existence": "presence",
    "object quantity": "quantity",
    "object color": "color",
    "object shape": "shape",
    "object size": "size",
    "object position": "position",
    "object direction": "direction",
    "scene type": "scene",
    "reasoning": "reasoning",
    "rural or urban": "scene",
    "image": "category",
}


def _boxes(ground_truth: str) -> list[list[float]]:
    return [
        [float(match.group(i)) for i in range(1, 5)]
        for match in _BOX_GROUPS_RE.finditer(ground_truth)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(prog="prepare_vrs")
    parser.add_argument("--eval-dir", required=True, type=Path, help="dir with VRSBench_EVAL_*.json")
    parser.add_argument("--images-root", required=True, type=Path, help="dir containing Images_val/")
    parser.add_argument("--output", required=True, type=Path, help="output dir for three jsonl files")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="copy Images_val under the output directory so annotations and "
        "assets share one asset_root",
    )
    args = parser.parse_args()

    image_dir = (args.images_root / "Images_val" / "Images_val").resolve()
    if args.copy_images:
        prep_image_dir = (args.output / "Images_val" / "Images_val").resolve()
        prep_image_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(image_dir, prep_image_dir, dirs_exist_ok=True)
        image_dir = prep_image_dir
        print(f"vrs: copied images to {prep_image_dir}", file=sys.stderr)

    caption_rows: list[dict[str, Any]] = []
    cap = json.loads((args.eval_dir / "VRSBench_EVAL_Cap.json").read_text(encoding="utf-8"))
    for index, record in enumerate(cap, start=1):
        image_id = record["image_id"]
        image = (image_dir / image_id).resolve()
        if not image.is_file():
            print(f"warn: missing image {image_id}", file=sys.stderr)
            continue
        caption_rows.append(
            {
                "id": f"vrs-caption-{index:06d}",
                "split": args.split,
                "image": image.as_posix(),
                "caption": record["ground_truth"].strip(),
            }
        )

    grounding_rows: list[dict[str, Any]] = []
    ref = json.loads((args.eval_dir / "VRSBench_EVAL_referring.json").read_text(encoding="utf-8"))
    for index, record in enumerate(ref, start=1):
        image_id = record["image_id"]
        image = (image_dir / image_id).resolve()
        boxes = _boxes(str(record.get("ground_truth", "")))
        if not image.is_file() or not boxes:
            print(f"warn: missing image/box for {image_id}", file=sys.stderr)
            continue
        grounding_rows.append(
            {
                "id": f"vrs-grounding-{index:06d}",
                "split": args.split,
                "image": image.as_posix(),
                "question": record["question"].strip() + _GROUNDING_SUFFIX,
                "boxes": boxes,
                "grounding_slice": "unique" if record.get("unique") else "non_unique",
            }
        )

    vqa_rows: list[dict[str, Any]] = []
    vqa = json.loads((args.eval_dir / "VRSBench_EVAL_vqa.json").read_text(encoding="utf-8"))
    for index, record in enumerate(vqa, start=1):
        image_id = record["image_id"]
        image = (image_dir / image_id).resolve()
        if not image.is_file():
            print(f"warn: missing image {image_id}", file=sys.stderr)
            continue
        vqa_type = str(record.get("type", "")).strip()
        vqa_rows.append(
            {
                "id": f"vrs-vqa-{index:06d}",
                "split": args.split,
                "image": image.as_posix(),
                "question": record["question"].strip(),
                "choices": None,
                "answer": record["ground_truth"].strip(),
                "vqa_category": _VQA_CATEGORY_MAP.get(vqa_type, "category"),
            }
        )

    _write_jsonl(args.output / "caption.jsonl", caption_rows)
    _write_jsonl(args.output / "grounding.jsonl", grounding_rows)
    _write_jsonl(args.output / "vqa.jsonl", vqa_rows)
    print(
        f"vrs: caption={len(caption_rows)} grounding={len(grounding_rows)} vqa={len(vqa_rows)} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
