"""Tests for the official LEVIR-CC val -> ChangeChat JSON converter.
官方 LEVIR-CC val 转 ChangeChat JSON 转换脚本的测试。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_changechat_val as converter  # noqa: E402


def _fixture_record(imgid: int, split: str, filename: str, changeflag: int = 0) -> dict:
    return {
        "filepath": split,
        "filename": filename,
        "imgid": imgid,
        "split": split,
        "changeflag": changeflag,
        "sentences": [
            {
                "tokens": ["caption", str(index)],
                "raw": f" caption {index} .",
                "imgid": imgid,
                "sentid": imgid * 5 + index,
            }
            for index in range(5)
        ],
        "sentids": [imgid * 5 + index for index in range(5)],
    }


def _write_annotation(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"images": records}, ensure_ascii=False), encoding="utf-8")


def test_converts_only_val_split_into_changechat_format(tmp_path: Path) -> None:
    annotation = tmp_path / "LevirCCcaptions.json"
    output = tmp_path / "changechat_105k_val.json"
    _write_annotation(
        annotation,
        [
            _fixture_record(0, "train", "train_000001.png"),
            _fixture_record(1, "val", "val_000001.png", changeflag=1),
        ],
    )

    stats = converter.convert_val(annotation, output, split="val", start_id=0)
    rows = json.loads(output.read_text(encoding="utf-8"))

    assert stats["pairs"] == 1
    assert stats["rows"] == 5
    assert stats["unique_pairs"] == 1
    assert stats["changeflag_counts"] == {"1": 1}
    assert [row["id"] for row in rows] == [0, 1, 2, 3, 4]
    assert rows[0]["image"] == ["val/A/val_000001.png", "val/B/val_000001.png"]
    assert rows[0]["changeflag"] == 1
    assert rows[0]["conversations"][0]["from"] == "human"
    assert rows[0]["conversations"][0]["value"].startswith("<image> <image> Please briefly describe")
    assert rows[0]["conversations"][1] == {"from": "gpt", "value": "Caption 0."}


def test_normalize_caption_matches_changechat_wording() -> None:
    assert converter.normalize_caption(" there is no difference .") == "There is no difference."
    assert converter.normalize_caption(" the two scenes seem identical .") == "The two scenes seem identical."


def test_missing_annotation_fails(tmp_path: Path) -> None:
    try:
        converter.convert_val(tmp_path / "missing.json", tmp_path / "out.json", split="val", start_id=0)
    except SystemExit as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("Expected SystemExit for a missing annotation file.")
