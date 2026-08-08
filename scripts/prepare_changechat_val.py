"""Convert official LEVIR-CC val captions into the ChangeChat-105k train JSON format.
将官方 LEVIR-CC val caption 转换为 ChangeChat-105k train JSON 格式。

ChangeChat-105k uses one JSON row per instruction sample. This script converts
the official five reference captions of each val image pair into five
``Please briefly describe the changes...`` instruction rows with the same
image/changeflag/conversations schema as ``changechat_105k_train.json``.
Only caption rows are produced; count/open/localization/dialogue rows cannot
be reconstructed from the official annotation because they were generated
by a rule+GPT process not shipped with the dataset.
ChangeChat-105k 使用每条指令一行 JSON。本脚本将官方 val 每个图像对的 5 条
参考 caption 转换为 5 条 “Please briefly describe the changes...” 指令行，
字段结构与 ``changechat_105k_train.json`` 一致。由于 count/open/定位/对话
行是数据集未随附的规则+GPT 过程生成的，官方标注无法重建，因此只生成 caption 行。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from convert_levir_cc_annotations import load_annotation


# The exact caption instruction prompt used by ChangeChat-105k train rows.
# ChangeChat-105k train 行使用的 caption 指令模板。
CAPTION_PROMPT = "<image> <image> Please briefly describe the changes in these two images."


def build_parser() -> argparse.ArgumentParser:
    """Build the conversion CLI. / 构建转换 CLI。"""
    parser = argparse.ArgumentParser(
        description="Convert official LEVIR-CC val captions into ChangeChat-105k JSON."
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("datasets/Levir-CC-dataset/LevirCCcaptions.json"),
        help="Official LEVIR-CC annotation file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/Levir-CC-dataset/changechat-105k/changechat_105k_val.json"),
        help="Output JSON file in ChangeChat-105k train format.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Official split to convert; default val.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="First output row id; default 0.",
    )
    return parser


def normalize_caption(raw: str) -> str:
    """Match the caption wording used by ChangeChat-105k train rows.
    复现 ChangeChat-105k train 行使用的 caption 写法。
    """
    text = raw.strip()
    # Collapse whitespace before punctuation, e.g. " ." -> ".".
    # 压缩标点前的空白，例如 “ .” -> “.”。
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


def to_changechat_rows(
    records: list[dict[str, Any]],
    split: str,
    start_id: int,
) -> list[dict[str, Any]]:
    """Convert official records of one split into ChangeChat-style rows.
    将某个官方 split 的记录转换为 ChangeChat 风格行。
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("split")) != split:
            continue
        filename = str(record["filename"])
        sentences = record["sentences"]
        if not isinstance(sentences, list) or len(sentences) != 5:
            raise ValueError(f"Record {record.get('imgid')!r} must contain exactly 5 sentences")
        for sentence in sentences:
            if not isinstance(sentence, dict) or not isinstance(sentence.get("raw"), str):
                raise ValueError(f"Record {record.get('imgid')!r} contains an invalid sentence")
            rows.append(
                {
                    "id": start_id + len(rows),
                    "image": [
                        f"{split}/A/{filename}",
                        f"{split}/B/{filename}",
                    ],
                    "changeflag": record["changeflag"],
                    "conversations": [
                        {"from": "human", "value": CAPTION_PROMPT},
                        {"from": "gpt", "value": normalize_caption(sentence["raw"])},
                    ],
                }
            )
    return rows


def write_json_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    """Atomically write a JSON array. / 原子写入 JSON 数组。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=1)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def convert_val(annotation_path: Path, output_path: Path, split: str, start_id: int) -> dict[str, Any]:
    """Convert the requested split and return generation statistics.
    转换指定 split 并返回生成统计。
    """
    records = load_annotation(annotation_path)
    split_records = [record for record in records if str(record.get("split")) == split]
    rows = to_changechat_rows(records, split, start_id)
    write_json_atomic(rows, output_path)
    return {
        "annotation": str(annotation_path),
        "output": str(output_path),
        "split": split,
        "pairs": len(split_records),
        "rows": len(rows),
        "unique_pairs": len({tuple(row["image"]) for row in rows}),
        "changeflag_counts": {
            str(flag): sum(1 for record in split_records if record["changeflag"] == flag)
            for flag in sorted({record["changeflag"] for record in split_records})
        },
    }


def main() -> int:
    """CLI entry point. / CLI 入口。"""
    args = build_parser().parse_args()
    stats = convert_val(args.annotation, args.output, args.split, args.start_id)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
