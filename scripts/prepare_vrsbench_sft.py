"""Convert processed VRSBench JSONL annotations into the reference SFT JSON format.
将处理后的 VRSBench JSONL 标注转换为参考 SFT JSON 格式。

The generated records follow the Qwen-VL-Series-Finetune conversation format
``{"id", "image", "conversations": [human, gpt]}``. English records can be
duplicated during training to emphasize English while Chinese records stay
single-copy; the validation split always keeps one copy per record.
生成记录遵循 Qwen-VL-Series-Finetune 的对话格式
``{"id", "image", "conversations": [human, gpt]}``。训练集中英文记录可复制以
侧重英文，中文记录保持单份；验证集每个记录始终只保留一份。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any


# Mapping from (split, task, language) to the local annotation filename.
# (split, task, language) 到本地标注文件名的映射。
FILE_MAP: dict[tuple[str, str, str], str] = {
    ("train", "caption", "en"): "VRSBench_train_caption_cleaned.jsonl",
    ("train", "caption", "zh"): "VRSBench_train_caption_cleaned_zh.jsonl",
    ("train", "vqa", "en"): "VRSBench_train_vqa.jsonl",
    ("train", "vqa", "zh"): "VRSBench_train_vqa_zh.jsonl",
    ("val", "caption", "en"): "VRSBench_val_caption.jsonl",
    ("val", "caption", "zh"): "VRSBench_val_caption_zh.jsonl",
    ("val", "vqa", "en"): "VRSBench_val_vqa.jsonl",
    ("val", "vqa", "zh"): "VRSBench_val_vqa_zh.jsonl",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the conversion CLI. / 构建转换 CLI。"""
    parser = argparse.ArgumentParser(
        description="Convert processed VRSBench JSONL annotations into SFT JSON files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="VRSBench dataset root containing the JSONL annotation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --root.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=("en", "zh"),
        default=("en", "zh"),
        help="Annotation languages to include; default en zh.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("caption", "vqa"),
        default=("caption", "vqa"),
        help="Task families to include; default caption vqa.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
        help="Splits to convert; default train val.",
    )
    parser.add_argument(
        "--english-multiplier",
        type=int,
        default=2,
        help="How many times each English train record is written; default 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic shuffle seed for train records; default 42.",
    )
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL annotation file into parsed records.
    将一个 JSONL 标注文件读取为解析后的记录列表。
    """
    if not path.is_file():
        raise SystemExit(f"Annotation file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return records


def to_sft_record(record: dict[str, Any], task: str, language: str) -> dict[str, Any]:
    """Convert one annotation record into the reference SFT conversation record.
    将一条标注记录转换为参考 SFT 对话记录。
    """
    image = record.get("image")
    if not image:
        raise RuntimeError(f"Record {record.get('id')!r} has no image field.")
    if task == "caption":
        human_text = record.get("instruction")
        gpt_text = record.get("caption")
    elif task == "vqa":
        human_text = record.get("question")
        gpt_text = record.get("answer")
    else:
        raise ValueError(f"Unsupported task: {task}")
    if not human_text or not gpt_text:
        raise RuntimeError(
            f"Record {record.get('id')!r} misses {task} text fields; "
            f"human={human_text!r} gpt={gpt_text!r}"
        )
    return {
        "id": record["id"],
        "image": image,
        "task": task,
        "language": language,
        "conversations": [
            {"from": "human", "value": f"<image>\n{human_text}"},
            {"from": "gpt", "value": gpt_text},
        ],
    }


def collect_split_records(
    root: Path,
    split: str,
    tasks: list[str],
    languages: list[str],
    english_multiplier: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect and deterministically shuffle records for one split.
    收集并按确定性顺序洗牌一个 split 的记录。
    """
    if english_multiplier < 1:
        raise ValueError(f"--english-multiplier must be >= 1, got {english_multiplier}")
    records: list[dict[str, Any]] = []
    stats: dict[str, int] = {"records": 0, "english_multiplier": english_multiplier}
    for task in tasks:
        for language in languages:
            key = (split, task, language)
            filename = FILE_MAP.get(key)
            if filename is None:
                raise ValueError(f"No annotation file mapped for {key}")
            source_records = read_jsonl(root / filename)
            converted = [to_sft_record(record, task, language) for record in source_records]
            repeat = english_multiplier if split == "train" and language == "en" else 1
            for record in converted:
                records.extend([record] * repeat)
            stats[f"{task}.{language}"] = len(source_records)
            stats[f"{task}.{language}.written"] = len(converted) * repeat
    if split == "train":
        rng = random.Random(seed)
        rng.shuffle(records)
    stats["records"] = len(records)
    return records, stats


def write_json_atomic(records: list[dict[str, Any]], path: Path) -> None:
    """Atomically write a JSON array. / 原子写入 JSON 数组。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=1)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def build_sft_datasets(
    root: Path,
    output_dir: Path,
    languages: list[str],
    tasks: list[str],
    splits: list[str],
    english_multiplier: int,
    seed: int,
) -> dict[str, Any]:
    """Build one SFT JSON per split and return generation statistics.
    为每个 split 生成一个 SFT JSON 并返回生成统计。
    """
    all_stats: dict[str, Any] = {
        "root": str(root),
        "output_dir": str(output_dir),
        "languages": languages,
        "tasks": tasks,
        "splits": splits,
        "seed": seed,
        "files": {},
    }
    for split in splits:
        records, stats = collect_split_records(
            root, split, tasks, languages, english_multiplier, seed
        )
        out_path = output_dir / f"vrsbench_sft_{split}.json"
        write_json_atomic(records, out_path)
        all_stats["files"][split] = {"path": str(out_path), **stats}
    return all_stats


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.root
    stats = build_sft_datasets(
        root=args.root,
        output_dir=output_dir,
        languages=list(args.languages),
        tasks=list(args.tasks),
        splits=list(args.splits),
        english_multiplier=args.english_multiplier,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
