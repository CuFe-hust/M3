"""Offline tests for the VQA Agent SFT v2 migration.

VQA Agent SFT v2 迁移的离线测试。
"""

from __future__ import annotations

import json

import pytest

from scripts.migrate_vqa_agent_sft_v2 import (
    MigrationError,
    _generate_vrsbench_validation,
    _migrate_record,
)


def _record(*, choices: list[str] | None = None) -> dict:
    payload = {
        "question": "Which surface is shown?",
        "task": "multiple_choice_vqa",
        "coordinate_frame": "normalized_0_999_top_left",
        "box_format": "integer_xyxy_json",
        "answer_constraints": {},
        "choices": choices,
        "allow_multiple": False,
    }
    return {
        "schema_version": "vqa-agent-sft-v1",
        "sample_id": "sample-1",
        "input": {
            "visual_task_plan": {
                "version": "visual-task-plan-v5",
                "task": "multiple_choice_vqa",
                "needs_visual_assistance": False,
                "object_categories": [],
                "count_target": None,
                "region_request": {
                    "explicit": False,
                    "image_index": None,
                    "roi_xyxy": None,
                },
                "reason_codes": ["source_multiple_choice"],
            },
            "agent_input": {
                "sample": {
                    "sample_id": "sample-1",
                    "dataset": "demo",
                    "split": "train",
                    "task": "multiple_choice_vqa",
                    "images": [
                        {"image_id": "img", "path": "img.png", "role": "image"}
                    ],
                    "question": "Which surface is shown?",
                    "ground_truth": None,
                    "metadata": {},
                    "normalization": {
                        "source_task": "source_vqa",
                        "normalized_task": "multiple_choice_vqa",
                        "semantic_subtype": "classification",
                        "normalizer": "test",
                        "version": "1",
                        "answer_constraints": {"choices": choices or []},
                    },
                },
                "user_payload": payload,
            },
        },
        "output": {
            "agent_result": {
                "agent_name": "general_vqa_agent",
                "answer": "A",
                "boxes": [],
                "evidence_items": [],
                "geometry": {},
                "status": "completed",
            }
        },
        "supervision": {
            "loss_scope": ["output.agent_result.answer"],
            "evidence_supervised": False,
            "source_answer": "A",
        },
    }


def test_migration_uses_production_base_payload_and_preserves_identity() -> None:
    migrated = _migrate_record(
        _record(choices=["(A) Road", "(B) Water"]), location="test:1"
    )
    agent_input = migrated["input"]["agent_input"]
    assert migrated["schema_version"] == "vqa-agent-sft-v2"
    assert "user_payload" not in agent_input
    assert agent_input["base_user_payload"] == {
        "question": "Which surface is shown?",
        "task": "multiple_choice_vqa",
        "choices": ["(A) Road", "(B) Water"],
        "allow_multiple": False,
        "semantic_subtype": "classification",
        "coordinate_frame": "normalized_0_999_top_left",
        "box_format": "integer_xyxy_json",
    }
    assert agent_input["sample"]["sample_id"] == "sample-1"
    assert migrated["output"]["agent_result"]["answer"] == "A"


def test_migration_never_derives_missing_choices_from_answer() -> None:
    with pytest.raises(MigrationError, match="canonical choices missing"):
        _migrate_record(_record(choices=None), location="test:1")


def test_generate_vrsbench_validation_excludes_counting_and_hides_gt(tmp_path) -> None:
    export_root = tmp_path / "bundle"
    (export_root / "VRSBench").mkdir(parents=True)
    dataset_root = tmp_path / "VRSBench-full"
    (dataset_root / "Images_val").mkdir(parents=True)
    (dataset_root / "Images_val" / "image.png").write_bytes(b"fixture")
    rows = [
        {
            "image_id": "image.png",
            "question": "What is beside the river?",
            "ground_truth": "road",
            "question_id": "q1",
            "type": "relation",
        },
        {
            "image_id": "image.png",
            "question": "How many buildings are visible?",
            "ground_truth": "4",
            "question_id": "q2",
            "type": "counting",
        },
    ]
    (dataset_root / "VRSBench_EVAL_vqa.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )

    assert _generate_vrsbench_validation(export_root) == 1
    records = [
        json.loads(line)
        for line in (export_root / "VRSBench" / "validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["input"]["agent_input"]["sample"]["ground_truth"] is None
    assert records[0]["supervision"]["source_answer"] == "road"
    assert records[0]["input"]["visual_task_plan"]["task"] == (
        records[0]["input"]["agent_input"]["sample"]["task"]
    )
