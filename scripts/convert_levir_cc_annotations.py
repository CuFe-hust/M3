"""Convert the official LEVIR-CC caption annotation into a readable JSONL format.
将官方 LEVIR-CC caption 标注转换为易读的 JSONL 格式。

The official ``LevirCCcaptions.json`` uses a COCO-style nested layout. This
script flattens each image pair into one JSONL line with explicit A/B image
paths, ``changeflag``, raw captions, global ``sentids``, and optional official
token lists.
官方 ``LevirCCcaptions.json`` 采用 COCO 风格嵌套结构。本脚本将每个图像对
拍平成一行 JSONL，包含显式 A/B 图像路径、``changeflag``、原始 caption、
全局 ``sentids``，并可选择保留官方 token 列表。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build the conversion CLI. / 构建转换 CLI。"""
    parser = argparse.ArgumentParser(
        description="Flatten the official LEVIR-CC annotation into readable JSONL."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/Levir-CC-dataset"),
        help="LEVIR-CC dataset root; used to locate default input/output paths.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="Official annotation path; defaults to <root>/LevirCCcaptions.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path; defaults to <root>/LevirCCcaptions_readable.jsonl.",
    )
    parser.add_argument(
        "--include-tokens",
        action="store_true",
        help="Also keep official tokenized sentences; omitted by default for readability.",
    )
    return parser


def load_annotation(path: Path) -> list[dict[str, Any]]:
    """Load and validate the official COCO-style annotation file.
    加载并校验官方 COCO 风格标注文件。
    """
    if not path.is_file():
        raise SystemExit(f"Annotation file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {path}: {error}") from error
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise SystemExit(f"Expected a top-level 'images' list in {path}")
    return [record for record in images if isinstance(record, dict)]


def to_readable_record(record: dict[str, Any], include_tokens: bool) -> dict[str, Any]:
    """Convert one official image-pair record into a flat JSONL row.
    将一条官方图像对记录转换为扁平 JSONL 行。
    """
    required = {"filepath", "filename", "imgid", "split", "changeflag", "sentences", "sentids"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Record {record.get('imgid')!r} misses required fields: {missing}")

    split = str(record["split"])
    filename = str(record["filename"])
    sentences = record["sentences"]
    sentids = record["sentids"]
    if not isinstance(sentences, list) or not sentences:
        raise ValueError(f"Record {record['imgid']!r} has invalid sentences: {sentences!r}")

    captions: list[str] = []
    tokens: list[list[str]] = []
    for sentence in sentences:
        if not isinstance(sentence, dict):
            raise ValueError(f"Record {record['imgid']!r} contains a non-object sentence")
        raw = sentence.get("raw")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Record {record['imgid']!r} contains an empty raw caption")
        captions.append(raw.strip())
        if include_tokens:
            token_list = sentence.get("tokens")
            if not isinstance(token_list, list):
                raise ValueError(f"Record {record['imgid']!r} contains invalid tokens")
            tokens.append([str(token) for token in token_list])

    if not isinstance(sentids, list) or len(sentids) != len(captions):
        raise ValueError(
            f"Record {record['imgid']!r} has sentids={sentids!r} but "
            f"{len(captions)} captions"
        )

    readable: dict[str, Any] = {
        "imgid": record["imgid"],
        "split": split,
        "filepath": str(record["filepath"]),
        "filename": filename,
        "image_a": f"images/{split}/A/{filename}",
        "image_b": f"images/{split}/B/{filename}",
        "changeflag": record["changeflag"],
        "captions": captions,
        "sentids": [int(sentid) for sentid in sentids],
    }
    if include_tokens:
        readable["tokens"] = tokens
    return readable


def write_jsonl_atomic(records: list[dict[str, Any]], path: Path) -> None:
    """Atomically write one JSON object per line. / 原子写入逐行 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def convert_annotations(
    annotation_path: Path,
    output_path: Path,
    include_tokens: bool,
) -> dict[str, Any]:
    """Convert the whole annotation file and return generation statistics.
    转换整个标注文件并返回生成统计。
    """
    images = load_annotation(annotation_path)
    records: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    changeflag_counts: dict[str, int] = {}
    for record in images:
        readable = to_readable_record(record, include_tokens)
        records.append(readable)
        split_counts[readable["split"]] = split_counts.get(readable["split"], 0) + 1
        changeflag_counts[str(readable["changeflag"])] = (
            changeflag_counts.get(str(readable["changeflag"]), 0) + 1
        )
    write_jsonl_atomic(records, output_path)
    return {
        "annotation": str(annotation_path),
        "output": str(output_path),
        "pairs": len(records),
        "captions": sum(len(record["captions"]) for record in records),
        "split_counts": split_counts,
        "changeflag_counts": changeflag_counts,
    }


def main() -> int:
    """CLI entry point. / CLI 入口。"""
    args = build_parser().parse_args()
    root = args.root
    annotation_path = args.annotation if args.annotation is not None else root / "LevirCCcaptions.json"
    output_path = args.output if args.output is not None else root / "LevirCCcaptions_readable.jsonl"
    stats = convert_annotations(annotation_path, output_path, args.include_tokens)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
