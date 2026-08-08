"""Tests for the LEVIR-CC annotation conversion script.
LEVIR-CC 标注转换脚本的测试。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import convert_levir_cc_annotations as converter  # noqa: E402


def _fixture_record(
    imgid: int,
    split: str,
    filename: str,
    changeflag: int = 0,
) -> dict:
    """Create one minimal official LEVIR-CC record with five captions.
    创建一条包含五条 caption 的最小官方 LEVIR-CC 记录。
    """
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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_convert_flattens_pairs_and_derives_image_paths(tmp_path: Path) -> None:
    annotation = tmp_path / "levir" / "LevirCCcaptions.json"
    output = tmp_path / "levir" / "LevirCCcaptions_readable.jsonl"
    _write_annotation(
        annotation,
        [
            _fixture_record(0, "train", "train_000001.png", changeflag=0),
            _fixture_record(1, "test", "test_000004.png", changeflag=1),
        ],
    )

    stats = converter.convert_annotations(annotation, output, include_tokens=False)
    rows = _read_jsonl(output)

    assert stats["pairs"] == 2
    assert stats["captions"] == 10
    assert stats["split_counts"] == {"train": 1, "test": 1}
    assert stats["changeflag_counts"] == {"0": 1, "1": 1}
    assert len(rows) == 2
    assert rows[0]["image_a"] == "images/train/A/train_000001.png"
    assert rows[0]["image_b"] == "images/train/B/train_000001.png"
    assert rows[0]["captions"] == ["caption 0 .", "caption 1 .", "caption 2 .", "caption 3 .", "caption 4 ."]
    assert rows[0]["sentids"] == [0, 1, 2, 3, 4]
    assert "tokens" not in rows[0]
    assert rows[1]["changeflag"] == 1


def test_include_tokens_keeps_official_tokenization(tmp_path: Path) -> None:
    annotation = tmp_path / "LevirCCcaptions.json"
    output = tmp_path / "readable.jsonl"
    _write_annotation(annotation, [_fixture_record(0, "val", "val_000001.png")])

    converter.convert_annotations(annotation, output, include_tokens=True)
    row = _read_jsonl(output)[0]

    assert row["tokens"] == [["caption", "0"], ["caption", "1"], ["caption", "2"], ["caption", "3"], ["caption", "4"]]


def test_missing_annotation_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "readable.jsonl"
    try:
        converter.convert_annotations(missing, output, include_tokens=False)
    except SystemExit as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("Expected SystemExit for a missing annotation file.")


def test_invalid_record_fails(tmp_path: Path) -> None:
    annotation = tmp_path / "LevirCCcaptions.json"
    output = tmp_path / "readable.jsonl"
    _write_annotation(annotation, [{"imgid": 0}])
    try:
        converter.convert_annotations(annotation, output, include_tokens=False)
    except ValueError as error:
        assert "misses required fields" in str(error)
    else:
        raise AssertionError("Expected ValueError for an incomplete record.")
