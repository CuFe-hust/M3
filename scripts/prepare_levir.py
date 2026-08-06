"""LEVIR-CC preprocessing: LevirCCcaptions.json -> annotations.json.

The committee subset keeps the official COCO-style captions file; the
evaluation images are the A/B time pairs under images/<split>/A|B/.
change/no-change discrimination labels are derived from the reference
caption via the five fixed no-change templates (self-implemented rule).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics.levir import classify_no_change

_NO_CHANGE_RE = re.compile(r"^(no change|no visible change|unchanged|same|nothing changed)", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser(prog="prepare_levir")
    parser.add_argument("--captions", required=True, type=Path, help="LevirCCcaptions.json")
    parser.add_argument("--images-root", required=True, type=Path, help="dataset dir containing images/")
    parser.add_argument("--output", required=True, type=Path, help="annotations.json")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--images-subdir",
        type=str,
        default=None,
        help="image directory name under images/ (defaults to --split)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="copy the A/B image pairs under the output directory so that "
        "annotations and assets share one asset_root",
    )
    args = parser.parse_args()
    images_subdir = args.images_subdir or args.split
    prep_images_root = args.output.parent / "images" / args.split
    if args.copy_images:
        source = args.images_root / "images" / images_subdir
        for side in ("A", "B"):
            destination = prep_images_root / side
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source / side, destination, dirs_exist_ok=True)
        print(f"levir: copied images to {prep_images_root}", file=sys.stderr)

    raw = json.loads(args.captions.read_text(encoding="utf-8"))
    images = raw["images"]
    rows = []
    seen: set[str] = set()
    for index, entry in enumerate(images, start=1):
        filename = entry["filename"]
        sentences = entry.get("sentences") or []
        caption = sentences[0]["raw"].strip() if sentences else ""
        if not caption:
            print(f"warn: {filename} has no caption, skipping", file=sys.stderr)
            continue
        image_a_source = (args.images_root / "images" / images_subdir / "A" / filename).resolve()
        image_b_source = (args.images_root / "images" / images_subdir / "B" / filename).resolve()
        if not image_a_source.is_file() or not image_b_source.is_file():
            print(f"warn: missing images for {filename}, skipping", file=sys.stderr)
            continue
        image_a = (prep_images_root / "A" / filename).resolve() if args.copy_images else image_a_source
        image_b = (prep_images_root / "B" / filename).resolve() if args.copy_images else image_b_source
        sample_id = f"levir-{args.split}-{index:06d}"
        if sample_id in seen:
            raise SystemExit(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        rows.append(
            {
                "id": sample_id,
                "split": args.split,
                "image_a": image_a.as_posix(),
                "image_b": image_b.as_posix(),
                "caption": caption,
                "change": "no-change" if classify_no_change(caption) else "change",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"levir: wrote {len(rows)} samples -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
