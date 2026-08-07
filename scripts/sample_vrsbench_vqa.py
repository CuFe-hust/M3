"""Sample prepared VRSBench VQA JSONL files proportionally by original QA type.
按原始 QA 类型从已生成的 VRSBench VQA JSONL 标注中按原比例抽样。

Reads VRSBench_{split}_vqa.jsonl and writes VRSBench_{split}_vqa_sampled.jsonl.
Target counts are allocated across the four original QA types with the largest
remainder method so the sampled proportions match the source proportions and
the total is exact. Sampling is deterministic for a fixed seed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Original QA types preserved by prepare_vrsbench_annotations.py.
# prepare_vrsbench_annotations.py 保留的原始 QA 类型。
QA_ORDER = ("object existence", "object category", "scene type", "rural or urban")


def build_parser() -> argparse.ArgumentParser:
    """Build the sampling CLI. / 构建抽样 CLI。"""
    parser = argparse.ArgumentParser(
        description="Sample prepared VRSBench VQA JSONL files by original QA type."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing VRSBench_{split}_vqa.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
        help="Splits to sample.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10000,
        help="Target total for train; default 10000.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="Target total for val; default 5000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed; default 42.",
    )
    return parser


def allocate_target(target: int, counts: dict[str, int]) -> dict[str, int]:
    """Allocate target counts proportionally using largest remainders.
    用最大余数法按原占比分配目标数量，保证总和恰好为目标值。
    """
    total = sum(counts.values())
    if target < 0 or target > total:
        raise SystemExit(f"Target {target} out of range [0, {total}]")
    raw = {kind: counts[kind] * target / total for kind in QA_ORDER}
    allocated = {kind: int(raw[kind]) for kind in QA_ORDER}
    remaining = target - sum(allocated.values())
    for kind in sorted(QA_ORDER, key=lambda k: (raw[k] - int(raw[k]), k), reverse=True):
        if remaining <= 0:
            break
        if allocated[kind] < counts[kind]:
            allocated[kind] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"Cannot allocate target {target} exactly")
    return allocated


def sample_split(
    root: Path, split: str, out_root: Path, target: int, seed: int
) -> dict[str, object]:
    """Sample one split and write the output file. / 抽样一个 split 并写出文件。"""
    in_path = root / f"VRSBench_{split}_vqa.jsonl"
    out_path = out_root / f"VRSBench_{split}_vqa_sampled.jsonl"
    if not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")
    # Keep (line index, raw line) so selected rows can be written in source order.
    # 保存 (行索引, 原始行)，使选中行按源文件顺序写出。
    groups: dict[str, list[tuple[int, str]]] = {kind: [] for kind in QA_ORDER}
    with in_path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            kind = json.loads(line)["source"]["original_type"]
            if kind not in groups:
                raise RuntimeError(f"Unexpected QA type: {kind}")
            groups[kind].append((index, line))
    counts = {kind: len(groups[kind]) for kind in QA_ORDER}
    target_counts = allocate_target(target, counts)
    rng = random.Random(seed)
    selected: list[tuple[int, str]] = []
    for kind in QA_ORDER:
        selected.extend(rng.sample(groups[kind], target_counts[kind]))
    selected.sort(key=lambda item: item[0])
    out_root.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, line in selected:
            f.write(line + "\n")
    total = sum(counts.values())
    return {
        "input_total": total,
        "target_total": target,
        "sampled_total": len(selected),
        "per_type": {
            kind: {
                "source_count": counts[kind],
                "source_ratio": round(counts[kind] / total, 6),
                "sampled_count": target_counts[kind],
                "sampled_ratio": round(target_counts[kind] / target, 6),
            }
            for kind in QA_ORDER
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    out_root = args.output_dir if args.output_dir is not None else args.root
    size_by_split = {"train": args.train_size, "val": args.val_size}
    all_stats: dict[str, object] = {}
    for split in args.splits:
        all_stats[split] = sample_split(
            args.root, split, out_root, size_by_split[split], args.seed
        )
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
