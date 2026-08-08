"""Remove ChangeChat-105k 3x3 grid localization rows from a JSON annotation file.
从 ChangeChat-105k JSON 标注中删除 3×3 网格定位行。

Only rows whose first human turn contains the canonical localization template
are removed. Open QA rows that merely mention "location" or "grid" are kept.
仅删除首条 human 消息包含定位模板的行；只提到 “location” 或 “grid” 的开放问答
会被保留。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


# Canonical ChangeChat-105k localization prompt.
# ChangeChat-105k 的定位提示模板。
LOCALIZATION_MARKER = (
    "Please indicate the locations where changes have occurred "
    "in the buildings and roads, using a 3x3 grid"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the filtering CLI. / 构建过滤 CLI。"""
    parser = argparse.ArgumentParser(
        description="Remove 3x3 grid localization rows from ChangeChat-105k JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/Levir-CC-dataset/changechat-105k/changechat_105k_train.json"),
        help="Input ChangeChat-105k JSON annotation file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/Levir-CC-dataset/changechat-105k/changechat_105k_train_no_loc.json"),
        help="Output JSON file; the input file is never overwritten.",
    )
    return parser


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSON array of instruction rows.
    加载并校验指令行 JSON 数组。
    """
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise SystemExit(f"Expected a JSON array of objects in {path}")
    return payload


def is_localization_row(row: dict[str, Any]) -> bool:
    """Return True when the first human turn uses the localization template.
    当首条 human 消息使用定位模板时返回 True。
    """
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"Row {row.get('id')!r} has no conversations")
    first_turn = conversations[0]
    if not isinstance(first_turn, dict):
        raise ValueError(f"Row {row.get('id')!r} has a non-object first conversation turn")
    return LOCALIZATION_MARKER in str(first_turn.get("value", ""))


def filter_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Split rows into kept and localization-removed sets with counts.
    将行分为保留集与定位删除集并返回计数。
    """
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        if is_localization_row(row):
            removed += 1
        else:
            kept.append(row)
    return kept, {
        "input_rows": len(rows),
        "removed_rows": removed,
        "output_rows": len(kept),
    }


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


def filter_json(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Filter the whole file and return statistics.
    过滤整个文件并返回统计信息。
    """
    rows = load_rows(input_path)
    kept, stats = filter_rows(rows)

    def pair_key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(item) for item in row.get("image", []))

    before_pairs = {pair_key(row) for row in rows}
    after_pairs = {pair_key(row) for row in kept}
    if before_pairs != after_pairs:
        raise RuntimeError(
            f"Filtering changed image-pair coverage: before={len(before_pairs)} after={len(after_pairs)}"
        )
    if any(is_localization_row(row) for row in kept):
        raise RuntimeError("Localization rows remain after filtering")

    write_json_atomic(kept, output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        **stats,
        "unique_pairs_before": len(before_pairs),
        "unique_pairs_after": len(after_pairs),
    }


def main() -> int:
    """CLI entry point. / CLI 入口。"""
    args = build_parser().parse_args()
    stats = filter_json(args.input, args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
