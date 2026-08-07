"""Unit tests for pure helpers of the LoRA evaluation script.
LoRA 评测脚本纯辅助函数的单元测试。
"""

from __future__ import annotations

import sys
from pathlib import Path


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
