"""Offline tests for General VQA Agent structured LoRA supervision.

General VQA Agent 结构化 LoRA 监督的离线测试。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from agents.schema import AgentResult, VisualTaskPlan
from scripts.finetune_qwen35_9b_general_vqa_agent_lora import (
    _load_prepared_images,
    _supervised_target,
)


def _plan(*, assisted: bool = True) -> VisualTaskPlan:
    return VisualTaskPlan(
        version="visual-task-plan-v5",
        task="general_vqa",
        needs_visual_assistance=assisted,
        object_categories=["vehicle"] if assisted else [],
    )


def _source_target() -> AgentResult:
    return AgentResult(agent_name="general_vqa_agent", answer="A vehicle is visible.")


def test_supervised_target_uses_global_yolo_geometry_and_ignores_segformer() -> None:
    bundle = {
        "rois": [{"roi_id": "roi-1", "image_id": "image-1"}],
        "detections": [
            {
                "leaf_category": "small_vehicle",
                "roi_id": "roi-1",
                "local_xyxy": [10.0, 20.0, 30.0, 40.0],
                "local_roi_size": [100, 100],
                "global_xyxy": [100.0, 50.0, 300.0, 250.0],
                "global_image_size": [1000, 500],
            }
        ],
        "segments": [{"leaf_category": "building", "roi_id": "roi-1"}],
        "leaf_states": {"small_vehicle": "hit", "building": "hit"},
    }

    target = _supervised_target(_source_target(), _plan(), bundle)

    assert target.boxes == [[100, 100, 300, 500]]
    assert len(target.evidence_items) == 1
    assert target.evidence_items[0].label == "small_vehicle"
    assert target.evidence_items[0].image_id == "image-1"
    assert target.geometry == {}


def test_supervised_target_keeps_direct_vqa_evidence_empty() -> None:
    target = _supervised_target(
        _source_target(),
        _plan(assisted=False),
        {"rois": [], "detections": [], "leaf_states": {}},
    )

    assert target.boxes == []
    assert target.evidence_items == []
    assert target.answer == "A vehicle is visible."


def test_cached_prepared_image_is_shrunk_for_training_qwen(tmp_path: Path) -> None:
    """Legacy prepared images obey the same shrink-only final-Qwen contract.
    旧 prepared 图像遵守同一只缩不放的 final-Qwen 契约。
    """
    image_path = tmp_path / "large.png"
    Image.new("RGB", (2048, 1024), (1, 2, 3)).save(image_path)
    record = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "image": "large.png"}],
            },
            {"role": "assistant", "content": []},
        ]
    }

    images = _load_prepared_images(record, root=tmp_path)

    assert len(images) == 1
    assert images[0].size == (1080, 540)
