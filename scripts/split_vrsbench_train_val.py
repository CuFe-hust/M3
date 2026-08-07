"""Repartition prepared VRSBench annotations into train/val/test.
将已生成的 VRSBench 标注按图片重新划分为 train/val/test。

Reads VRSBench_{train,val}_{caption,vqa}.jsonl, deterministically moves about
one tenth of the train images (by unique image_id) into a new val split, and
renames the old val files to test. The split is image-level so no image appears
in more than one split, and caption/VQA records stay consistent per image.
Only the "split" field and the split prefix of "id" change; image paths and
all other fields are preserved as-is.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path

# Input/output suffixes shared by caption and VQA files.
# caption 与 VQA 文件共用的输入/输出后缀。
FAMILIES = ("caption", "vqa")


def build_parser() -> argparse.ArgumentParser:
    """Build the split CLI. / 构建划分 CLI。"""
    parser = argparse.ArgumentParser(
        description=(
            "Split prepared VRSBench train annotations by image into a new val "
            "split and rename the old val annotations to test."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing VRSBench_{train,val}_{caption,vqa}.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the deterministic image shuffle; default 42.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of train images moved to val; default 0.1.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing VRSBench_test_* files if present.",
    )
    return parser


def _read_rows(path: Path) -> list[dict[str, object]]:
    """Read one JSONL annotation file into parsed records in source order.
    按源顺序读取一个 JSONL 标注文件为解析后的记录。
    """
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return records


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    """Atomically write records as compact JSONL. / 以紧凑 JSONL 原子写出记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in rows:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _rebase(record: dict[str, object], new_split: str) -> dict[str, object]:
    """Update split and the split prefix of id while keeping other fields.
    更新 split 字段与 id 的 split 前缀，保持其他字段不变。
    """
    old_split = record["split"]
    parts = str(record["id"]).split("/", 2)
    if len(parts) != 3 or parts[0] != "vrsbench" or parts[1] != old_split:
        raise RuntimeError(
            f"Unexpected id {record['id']!r} for split={old_split!r}; "
            "expected vrsbench/{split}/... format"
        )
    updated = dict(record)
    updated["split"] = new_split
    updated["id"] = f"vrsbench/{new_split}/{parts[2]}"
    return updated


def _unique_images(rows: list[dict[str, object]]) -> set[str]:
    """Return the set of image_id values. / 返回 image_id 集合。"""
    return {str(record["image_id"]) for record in rows}


def _validate_splits(rows: list[dict[str, object]], expected: str, path: Path) -> None:
    """Refuse to proceed when a file contains unexpected split fields.
    当文件包含非预期 split 字段时拒绝继续。
    """
    bad = sorted({str(record["split"]) for record in rows} - {expected})
    if bad:
        raise SystemExit(f"{path} contains unexpected split values: {bad}")


def split_root(
    root: Path, out_root: Path, seed: int, val_ratio: float, force: bool = False
) -> dict[str, object]:
    """Perform the image-level train/val split and val-to-test rename.
    执行图片级 train/val 划分并将旧 val 重命名为 test。
    """
    if not 0.0 < val_ratio < 1.0:
        raise SystemExit(f"--val-ratio must be between 0 and 1, got {val_ratio}")
    test_outputs = [out_root / f"VRSBench_test_{family}.jsonl" for family in FAMILIES]
    existing = [path for path in test_outputs if path.exists()]
    if existing and not force:
        raise SystemExit(
            "Test output already exists; use --force to overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    train_rows: dict[str, list[dict[str, object]]] = {}
    val_rows: dict[str, list[dict[str, object]]] = {}
    train_images: set[str] = set()
    for family in FAMILIES:
        train_path = root / f"VRSBench_train_{family}.jsonl"
        val_path = root / f"VRSBench_val_{family}.jsonl"
        train_rows[family] = _read_rows(train_path)
        val_rows[family] = _read_rows(val_path)
        _validate_splits(train_rows[family], "train", train_path)
        _validate_splits(val_rows[family], "val", val_path)
        train_images |= _unique_images(train_rows[family])

    # Deterministic image-level selection: sort ids, shuffle with the seed, and
    # take the first ceil(ratio * count) images for the new val split.
    # 确定性图片级选择：排序 id 后用种子洗牌，取前 ceil(ratio * 总数) 张图为新 val。
    ordered_images = sorted(train_images)
    rng = random.Random(seed)
    rng.shuffle(ordered_images)
    val_count = max(1, math.ceil(len(ordered_images) * val_ratio))
    val_images = set(ordered_images[:val_count])

    stats: dict[str, object] = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train_images_total": len(train_images),
        "new_val_images": len(val_images),
        "remaining_train_images": len(train_images) - len(val_images),
        "files": {},
    }
    new_train_rows: dict[str, list[dict[str, object]]] = {}
    new_val_rows: dict[str, list[dict[str, object]]] = {}
    test_rows: dict[str, list[dict[str, object]]] = {}
    final_train_images: set[str] = set()
    final_val_images: set[str] = set()
    final_test_images: set[str] = set()
    for family in FAMILIES:
        test_rows[family] = [_rebase(record, "test") for record in val_rows[family]]
        new_val_rows[family] = [
            _rebase(record, "val")
            for record in train_rows[family]
            if str(record["image_id"]) in val_images
        ]
        new_train_rows[family] = [
            record
            for record in train_rows[family]
            if str(record["image_id"]) not in val_images
        ]
        stats["files"][family] = {
            "train_rows": len(new_train_rows[family]),
            "val_rows": len(new_val_rows[family]),
            "test_rows": len(test_rows[family]),
            "train_images": len(_unique_images(new_train_rows[family])),
            "val_images": len(_unique_images(new_val_rows[family])),
            "test_images": len(_unique_images(test_rows[family])),
        }
        final_train_images |= _unique_images(new_train_rows[family])
        final_val_images |= _unique_images(new_val_rows[family])
        final_test_images |= _unique_images(test_rows[family])

    # Verify the three splits never share an image before touching the disk.
    # 在写盘前校验三个划分之间没有共享图片。
    overlaps = sorted(
        (final_train_images & final_val_images)
        | (final_train_images & final_test_images)
        | (final_val_images & final_test_images)
    )
    if overlaps:
        raise RuntimeError(f"Image overlap across splits before writing: {overlaps[:5]}")
    stats["overlap_images"] = 0
    for family in FAMILIES:
        _write_rows(test_rows[family], test_outputs[FAMILIES.index(family)])
        _write_rows(new_train_rows[family], out_root / f"VRSBench_train_{family}.jsonl")
        _write_rows(new_val_rows[family], out_root / f"VRSBench_val_{family}.jsonl")
    return stats


def main() -> int:
    args = build_parser().parse_args()
    out_root = args.output_dir if args.output_dir is not None else args.root
    stats = split_root(args.root, out_root, args.seed, args.val_ratio, force=args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
