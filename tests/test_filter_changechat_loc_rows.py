"""Tests for the ChangeChat-105k localization-row filter script.
ChangeChat-105k 定位行过滤脚本的测试。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import filter_changechat_loc_rows as filter_script  # noqa: E402


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _loc_row(row_id: int) -> dict:
    return {
        "id": row_id,
        "image": ["train/A/train_000001.png", "train/B/train_000001.png"],
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image> <image> Please indicate the locations where changes have occurred "
                    "in the buildings and roads, using a 3x3 grid (namely, 'top', 'center', "
                    "and 'bottom' vertically and 'left', 'center', 'right' horizontally)."
                ),
            },
            {"from": "gpt", "value": "roads: []. buildings: []."},
        ],
    }


def test_removes_only_localization_rows_and_keeps_location_qa(tmp_path: Path) -> None:
    input_path = tmp_path / "train.json"
    output_path = tmp_path / "train_no_loc.json"
    location_qa = {
        "id": 1,
        "image": ["train/A/train_000001.png", "train/B/train_000001.png"],
        "conversations": [
            {"from": "human", "value": "<image> <image> what is the location of the road in the image?"},
            {"from": "gpt", "value": "On the left."},
        ],
    }
    caption_row = {
        "id": 2,
        "image": ["train/A/train_000001.png", "train/B/train_000001.png"],
        "conversations": [
            {"from": "human", "value": "<image> <image> Please briefly describe the changes in these two images."},
            {"from": "gpt", "value": "There is no difference."},
        ],
    }
    _write_json(input_path, [_loc_row(0), location_qa, caption_row])

    stats = filter_script.filter_json(input_path, output_path)
    kept = json.loads(output_path.read_text(encoding="utf-8"))

    assert stats["input_rows"] == 3
    assert stats["removed_rows"] == 1
    assert stats["output_rows"] == 2
    assert stats["unique_pairs_before"] == 1
    assert stats["unique_pairs_after"] == 1
    assert [row["id"] for row in kept] == [1, 2]
    assert all(not filter_script.is_localization_row(row) for row in kept)


def test_missing_input_fails(tmp_path: Path) -> None:
    try:
        filter_script.filter_json(tmp_path / "missing.json", tmp_path / "out.json")
    except SystemExit as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("Expected SystemExit for a missing input file.")


def test_malformed_row_fails(tmp_path: Path) -> None:
    input_path = tmp_path / "train.json"
    _write_json(input_path, [{"id": 0, "conversations": []}])
    try:
        filter_script.filter_json(input_path, tmp_path / "out.json")
    except ValueError as error:
        assert "no conversations" in str(error)
    else:
        raise AssertionError("Expected ValueError for a row without conversations.")
