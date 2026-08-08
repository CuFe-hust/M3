"""Unit tests for pure helpers of the LoRA evaluation script.
LoRA 评测脚本纯辅助函数的单元测试。
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_qwen3vl_merger_lora as evaluator  # noqa: E402


def test_normalize_answer_ignores_case_punctuation_and_space() -> None:
    assert evaluator.normalize_answer("  Rural! ") == evaluator.normalize_answer("rural")
    assert evaluator.normalize_answer("3 艘船") == evaluator.normalize_answer("3艘船")
    assert evaluator.normalize_answer("3艘船") != evaluator.normalize_answer("三艘船")
    assert evaluator.normalize_answer("高速公路收费站") == "高速公路收费站"


def test_record_to_sample_caption() -> None:
    record = {
        "id": "vrsbench/test/00001_0000.png/caption",
        "image": "Images_test/Images_test/00001_0000.png",
        "instruction": "Describe this image.",
        "caption": "A parking lot.",
        "source": {"annotation_file": "Annotations_test/Annotations_test/00001_0000.json"},
    }
    sample = evaluator.record_to_sample(record, "caption")
    assert sample.task_type == "caption"
    assert sample.prompt == "Describe this image."
    assert sample.answers == ["A parking lot."]
    assert sample.images == ["Images_test/Images_test/00001_0000.png"]
    assert sample.meta["split"] == "test"


def test_record_to_sample_vqa() -> None:
    record = {
        "id": "vrsbench/test/00002_0000.png/scene_classification/1",
        "image": "Images_test/Images_test/00002_0000.png",
        "question": "What is the main structure?",
        "answer": "Highway toll station",
    }
    sample = evaluator.record_to_sample(record, "vqa")
    assert sample.task_type == "vqa"
    assert sample.prompt == "What is the main structure?"
    assert sample.answers == ["Highway toll station"]


def test_build_messages_contains_image_and_text() -> None:
    record = {
        "id": "vrsbench/test/00002_0000.png/scene_classification/1",
        "image": "Images_test/Images_test/00002_0000.png",
        "question": "What is the main structure?",
        "answer": "Highway toll station",
    }
    sample = evaluator.record_to_sample(record, "vqa")
    messages = evaluator.build_messages(sample)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][1]["type"] == "text"
    assert messages[0]["content"][1]["text"] == sample.prompt


def test_resolve_image_path_joins_relative_and_keeps_absolute(tmp_path: Path) -> None:
    relative = evaluator.resolve_image_path("Images_test/a.png", tmp_path)
    assert relative == tmp_path / "Images_test" / "a.png"
    absolute = evaluator.resolve_image_path(str(tmp_path / "b.png"), tmp_path)
    assert absolute == tmp_path / "b.png"


def test_record_to_sample_keeps_question_type() -> None:
    record = {
        "id": "vrsbench/test/00002_0000.png/object_existence/1",
        "image": "Images_test/Images_test/00002_0000.png",
        "task": "object_existence",
        "question": "Is there a ship?",
        "answer": "yes",
        "source": {"original_type": "object existence", "ques_id": 1},
    }
    sample = evaluator.record_to_sample(record, "vqa")
    assert sample.meta["question_type"] == "object_existence"
    assert sample.meta["original_type"] == "object existence"


def test_ordered_test_image_ids_caps_and_dedupes(tmp_path: Path) -> None:
    rows = [
        {"id": f"c{i}", "image": f"Images_test/{i}.png", "image_id": f"{i}.png"}
        for i in range(5)
    ]
    rows.append({"id": "dup", "image": "Images_test/0.png", "image_id": "0.png"})
    (tmp_path / "VRSBench_test_caption.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    assert evaluator.ordered_test_image_ids(tmp_path, 2) == ["0.png", "1.png"]
    assert len(evaluator.ordered_test_image_ids(tmp_path, 0)) == 5


def test_load_task_records_filters_images(tmp_path: Path) -> None:
    rows = [
        {"id": f"v{i}", "image": f"Images_test/{i}.png", "image_id": f"{i}.png"}
        for i in range(4)
    ]
    (tmp_path / "VRSBench_test_vqa.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    filtered = evaluator.load_task_records(tmp_path, "vqa", ["0.png", "2.png"])
    assert [row["image_id"] for row in filtered] == ["0.png", "2.png"]
    assert len(evaluator.load_task_records(tmp_path, "vqa", None)) == 4


def test_compute_vqa_accuracy_by_type() -> None:
    pairs = []
    for question_type, exact in (
        ("object_existence", True),
        ("object_existence", False),
        ("scene_classification", True),
        (None, False),
    ):
        sample = evaluator.record_to_sample(
            {
                "id": f"q{len(pairs)}",
                "image": "Images_test/0.png",
                "task": question_type,
                "question": "?",
                "answer": "a",
            },
            "vqa",
        )
        pairs.append((sample, exact))
    result = evaluator.compute_vqa_accuracy_by_type(pairs)
    assert result["object_existence"] == {
        "total": 2,
        "exact_matches": 1,
        "accuracy": 0.5,
    }
    assert result["scene_classification"]["accuracy"] == 1.0
    assert result["unknown"]["total"] == 1


def test_compute_caption_metrics_uses_pycocoevalcap(monkeypatch) -> None:
    pytest.importorskip("pycocoevalcap")
    monkeypatch.setattr(evaluator, "compute_meteor", lambda gts, res: 0.5)
    sample = evaluator.record_to_sample(
        {
            "id": "c1",
            "image": "Images_test/0.png",
            "instruction": "Describe.",
            "caption": "A cat sits on the mat.",
        },
        "caption",
    )
    metrics = evaluator.compute_caption_metrics(
        [(sample, "the cat sits on the mat")]
    )
    assert set(metrics) == {
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "cider",
    }
    assert metrics["meteor"] == 0.5
    assert 0 <= metrics["bleu_1"] <= 1
