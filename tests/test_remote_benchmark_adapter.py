"""Tests for the remote 评测数据集 adapters.
远端评测数据集适配器测试。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import data.loaders as loaders


def _write_image(path: Path, color: tuple[int, int, int] = (120, 80, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _make_vrsbench(root: Path) -> None:
    images = root / "Images_val" / "Images_val"
    _write_image(images / "P0019_0046.png")
    _write_image(images / "P0524_0003.png")
    _write_image(images / "P0060_0010.png")
    (root / "VRSBench_EVAL_Cap.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "P0019_0046.png",
                    "ground_truth": "A harbor scene with greenery.",
                    "question": "Describe the image in detail",
                    "dataset": "RSBench",
                    "question_id": 30,
                    "type": "caption",
                }
            ]
        ),
        encoding="utf-8",
    )
    (root / "VRSBench_EVAL_vqa.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "P0524_0003.png",
                    "question": "Which category does the topmost object belong to?",
                    "ground_truth": "ship",
                    "dataset": "RSBench",
                    "question_id": 1841,
                    "type": "object category",
                }
            ]
        ),
        encoding="utf-8",
    )
    (root / "VRSBench_EVAL_referring.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "P0060_0010.png",
                    "question": "The larger swimming pool.",
                    "ground_truth": "{<85><79><93><91>}",
                    "dataset": "RSBench",
                    "question_id": 209,
                    "type": "ref",
                    "obj_corner": [0.87, 0.80, 0.85, 0.91, 0.94, 0.91, 0.94, 0.83],
                }
            ]
        ),
        encoding="utf-8",
    )


def _make_mme(root: Path) -> None:
    _write_image(root / "images" / "mmerealrs_sample.png", (10, 90, 160))
    rows = [
        {
            "sample_id": "perception/remote_sensing/count/0005",
            "question": "How many red cars are visible?",
            "options": {"A": "2", "B": "1", "C": "3", "D": "4", "E": "not present"},
            "answer": "A",
            "image_path": "images/mmerealrs_sample.png",
            "question_type": "count",
            "difficulty": "easy",
            "evaluation_group": "core",
        }
    ]
    (root / "annotations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_levir(root: Path) -> None:
    _write_image(root / "images" / "val" / "A" / "val_000001.png", (200, 100, 50))
    _write_image(root / "images" / "val" / "B" / "val_000001.png", (50, 180, 90))
    record = {
        "filepath": "val",
        "filename": "val_000001.png",
        "imgid": 6815,
        "split": "val",
        "changeflag": 1,
        "sentences": [
            {"raw": f" reference caption {index} .", "imgid": 6815, "sentid": 34075 + index}
            for index in range(5)
        ],
    }
    (root / "LevirCCcaptions.json").write_text(
        json.dumps({"images": [record]}),
        encoding="utf-8",
    )


class FakeArrowSplit:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __getitem__(self, key: str) -> "FakeArrowSplit":
        return self

    def __iter__(self):
        return iter(self.rows)


def _make_xlrs_rows() -> dict[str, list[dict]]:
    image = Image.new("RGB", (32, 32), (30, 120, 200))
    return {
        "lite": [
            {
                "index": "66",
                "question": "Select all land use types in the image.",
                "multi-choice options": ["(A) Racetrack", "(B) Crop Field", "(C) Water", "(D) Desert"],
                "answer": "B",
                "category": "Land use classification",
                "l2_category": "default",
                "image": [image],
            }
        ],
        "caption": [
            {
                "id": 456,
                "question_id": "456",
                "question": "Describe the image in detail.",
                "answer": ["A rural landscape with farmlands."],
                "image": image,
            }
        ],
        "grounding": [
            {
                "question_id": "Visual grounding/001",
                "question": "Locate the green lake.",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "image": image,
                "image_width": 1000.0,
                "image_height": 1000.0,
            }
        ],
    }


def _make_xlrs(root: Path, monkeypatch) -> None:
    rows = _make_xlrs_rows()
    (root / "XLRS-Bench-lite").mkdir(parents=True, exist_ok=True)
    (root / "XLRS-Bench_caption_en" / "train").mkdir(parents=True, exist_ok=True)
    (root / "XLRS-Bench_visual_grounding_en" / "train").mkdir(parents=True, exist_ok=True)
    (root / "XLRS-Bench_visual_grounding_en" / "test").mkdir(parents=True, exist_ok=True)

    def fake_load(path: Path) -> FakeArrowSplit:
        text = str(path)
        if "XLRS-Bench-lite" in text:
            return FakeArrowSplit(rows["lite"])
        if "XLRS-Bench_caption_en" in text:
            return FakeArrowSplit(rows["caption"])
        return FakeArrowSplit(rows["grounding"])

    monkeypatch.setattr(loaders, "_load_arrow_split", fake_load)


def test_remote_vrsbench_samples(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench(1100)"
    _make_vrsbench(root)
    samples = list(loaders.load_remote_benchmark_samples(tmp_path, "vrsbench"))
    assert len(samples) == 3
    by_task = {sample.task_type: sample for sample in samples}
    assert by_task["caption"].meta["benchmark_task"] == "vrsbench_caption"
    assert by_task["caption"].answers == ["A harbor scene with greenery."]
    assert by_task["vqa"].meta["question_type"] == "object category"
    assert by_task["vqa"].prompt.startswith("Answer the question using only the image.")
    grounding = by_task["grounding"]
    assert grounding.boxes[0][0] == 85.0
    assert grounding.boxes[0][3] == 91.0


def test_remote_mme_samples(tmp_path: Path) -> None:
    root = tmp_path / "MMERealRS_Stratified_914"
    _make_mme(root)
    samples = list(loaders.load_remote_benchmark_samples(tmp_path, "mme_real_rs"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.id == "perception/remote_sensing/count/0005"
    assert sample.task_type == "vqa"
    assert sample.answers == ["A"]
    assert sample.choices[0] == "A. 2"
    assert sample.meta["benchmark_task"] == "mme_real_rs_vqa"
    assert sample.meta["difficulty"] == "easy"
    assert sample.images[0].size == (32, 32)


def test_remote_levir_samples(tmp_path: Path) -> None:
    root = tmp_path / "levir-cc(400)"
    _make_levir(root)
    samples = list(loaders.load_remote_benchmark_samples(tmp_path, "levir_cc"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.task_type == "change_caption"
    assert len(sample.images) == 2
    assert len(sample.answers) == 5
    assert sample.meta["changeflag"] == 1
    assert sample.meta["benchmark_task"] == "levir_cc_change_caption"


def test_remote_xlrs_samples(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "XLRSBench_Native_690"
    _make_xlrs(root, monkeypatch)
    samples = list(loaders.load_remote_benchmark_samples(tmp_path, "xlrs"))
    assert len(samples) == 4
    by_label = {sample.meta["benchmark_task"]: sample for sample in samples}
    vqa = by_label["xlrs_vqa_lite"]
    assert vqa.task_type == "vqa"
    assert vqa.answers == ["B"]
    assert "may be more than one correct option" not in vqa.prompt
    caption = by_label["xlrs_caption_en"]
    assert caption.task_type == "caption"
    assert caption.answers == ["A rural landscape with farmlands."]
    grounding = by_label["xlrs_grounding_condition"]
    assert grounding.task_type == "grounding"
    assert grounding.boxes[0] == [10.0, 20.0, 30.0, 40.0]
    assert grounding.meta["image_width"] == 1000.0
    grounding_fine = by_label["xlrs_grounding_fine"]
    assert grounding_fine.task_type == "grounding"
    assert grounding_fine.boxes[0] == [10.0, 20.0, 30.0, 40.0]
    assert grounding_fine.meta["release_split"] == "test"


def test_remote_benchmark_unknown_dataset(tmp_path: Path) -> None:
    try:
        list(loaders.load_remote_benchmark_samples(tmp_path, "unknown"))
    except ValueError as error:
        assert "Unsupported remote benchmark dataset" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown dataset")


def test_remote_benchmark_missing_directory(tmp_path: Path) -> None:
    try:
        list(loaders.load_remote_benchmark_samples(tmp_path, "vrsbench"))
    except FileNotFoundError as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError for missing directory")
