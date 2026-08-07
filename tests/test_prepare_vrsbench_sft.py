"""Tests for the VRSBench -> SFT JSON conversion script.
VRSBench 转 SFT JSON 转换脚本的测试。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_vrsbench_sft as converter  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fixture_dataset(root: Path) -> None:
    """Create tiny en/zh caption and VQA fixtures for every mapped file.
    为每个映射文件创建小型中英文 caption/VQA 夹具。
    """
    fixtures: dict[str, list[dict]] = {}
    fixtures["VRSBench_train_caption_cleaned.jsonl"] = [
        {
            "id": "vrsbench/train/00001_0000.png/caption",
            "image": "Images_train/Images_train/00001_0000.png",
            "instruction": "Describe this image.",
            "caption": "A parking lot.",
        }
    ]
    fixtures["VRSBench_train_caption_cleaned_zh.jsonl"] = [
        {
            "id": "vrsbench/train/00001_0000.png/caption",
            "image": "Images_train/Images_train/00001_0000.png",
            "instruction": "请描述这张图片的内容。",
            "caption": "一个停车场。",
        }
    ]
    fixtures["VRSBench_train_vqa.jsonl"] = [
        {
            "id": "vrsbench/train/00002_0000.png/scene_classification/1",
            "image": "Images_train/Images_train/00002_0000.png",
            "question": "What is the main structure?",
            "answer": "Highway toll station",
        }
    ]
    fixtures["VRSBench_train_vqa_zh.jsonl"] = [
        {
            "id": "vrsbench/train/00002_0000.png/scene_classification/1",
            "image": "Images_train/Images_train/00002_0000.png",
            "question": "图像中心的主要结构是什么？",
            "answer": "高速公路收费站",
        }
    ]
    fixtures["VRSBench_val_caption.jsonl"] = [
        {
            "id": "vrsbench/val/00003_0000.png/caption",
            "image": "Images_train/Images_train/00003_0000.png",
            "instruction": "Describe this image.",
            "caption": "A highway service area.",
        }
    ]
    fixtures["VRSBench_val_caption_zh.jsonl"] = [
        {
            "id": "vrsbench/val/00003_0000.png/caption",
            "image": "Images_train/Images_train/00003_0000.png",
            "instruction": "请描述这张图片的内容。",
            "caption": "一个高速公路服务区。",
        }
    ]
    fixtures["VRSBench_val_vqa.jsonl"] = [
        {
            "id": "vrsbench/val/00004_0000.png/object_existence/1",
            "image": "Images_train/Images_train/00004_0000.png",
            "question": "Is there a vehicle?",
            "answer": "Yes",
        }
    ]
    fixtures["VRSBench_val_vqa_zh.jsonl"] = [
        {
            "id": "vrsbench/val/00004_0000.png/object_existence/1",
            "image": "Images_train/Images_train/00004_0000.png",
            "question": "是否有车辆？",
            "answer": "是",
        }
    ]
    for filename, records in fixtures.items():
        _write_jsonl(root / filename, records)


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_sft_datasets_train_english_multiplier_and_val_balanced(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench"
    output = tmp_path / "sft"
    _fixture_dataset(root)

    stats = converter.build_sft_datasets(
        root=root,
        output_dir=output,
        languages=["en", "zh"],
        tasks=["caption", "vqa"],
        splits=["train", "val"],
        english_multiplier=2,
        seed=42,
    )

    train_records = _load_json(output / "vrsbench_sft_train.json")
    val_records = _load_json(output / "vrsbench_sft_val.json")

    # 1 English caption + 1 English VQA are written twice; Chinese once.
    # 1 条英文 caption + 1 条英文 VQA 写两份；中文只写一份。
    assert len(train_records) == 6
    assert stats["files"]["train"]["records"] == 6
    assert len(val_records) == 4
    assert stats["files"]["val"]["records"] == 4

    english_ids = [
        record["id"]
        for record in train_records
        if record["language"] == "en"
    ]
    assert english_ids.count(english_ids[0]) == 2
    assert all(record["language"] == "zh" for record in train_records if record["language"] == "zh")


def test_sft_record_conversation_structure_and_image_path(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench"
    output = tmp_path / "sft"
    _fixture_dataset(root)
    converter.build_sft_datasets(
        root=root,
        output_dir=output,
        languages=["zh"],
        tasks=["caption", "vqa"],
        splits=["train"],
        english_multiplier=1,
        seed=42,
    )
    train_records = _load_json(output / "vrsbench_sft_train.json")
    caption_record = next(record for record in train_records if record["task"] == "caption")
    assert caption_record["image"] == "Images_train/Images_train/00001_0000.png"
    assert caption_record["conversations"][0]["from"] == "human"
    assert caption_record["conversations"][0]["value"].startswith("<image>\n请描述")
    assert caption_record["conversations"][1]["from"] == "gpt"
    assert caption_record["conversations"][1]["value"] == "一个停车场。"


def test_deterministic_seed(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench"
    output_a = tmp_path / "sft_a"
    output_b = tmp_path / "sft_b"
    _fixture_dataset(root)
    for output in (output_a, output_b):
        converter.build_sft_datasets(
            root=root,
            output_dir=output,
            languages=["en", "zh"],
            tasks=["vqa"],
            splits=["train"],
            english_multiplier=2,
            seed=42,
        )
    assert _load_json(output_a / "vrsbench_sft_train.json") == _load_json(
        output_b / "vrsbench_sft_train.json"
    )


def test_missing_file_fails(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    try:
        converter.collect_split_records(
            root=root,
            split="train",
            tasks=["vqa"],
            languages=["en"],
            english_multiplier=1,
            seed=42,
        )
    except SystemExit as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("Expected SystemExit for a missing annotation file.")
