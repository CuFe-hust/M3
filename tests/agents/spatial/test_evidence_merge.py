"""Contract tests for spatial evidence merging rules.

空间证据合并纯规则契约测试：去重、重叠抑制、标签兼容、角点锚定、修复
严重度、候选复核判定——全部数据集无关。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.schema import AgentResult, VisualEvidence
from agents.spatial.evidence_merge import (
    box_intersection_over_smaller,
    box_iou,
    canonical_answer,
    compatible_evidence_labels,
    extreme_evidence_is_sufficient,
    is_corner_anchored_box,
    is_status_answer_placeholder,
    matches_target_label,
    maximum_repair_severity,
    merge_visual_evidence,
    needs_candidate_review,
    normalized_box_center_distance,
    point_distance,
    position_review_evidence,
    same_box_observation,
    same_visual_observation,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _box(label: str, box: list[int], confidence: float = 0.9) -> VisualEvidence:
    return VisualEvidence(label=label, box=box, confidence=confidence)


def _result(items: list[VisualEvidence], **overrides) -> AgentResult:
    values = dict(agent_name="spatial_agent", answer="small-vehicle", evidence_items=items)
    values.update(overrides)
    return AgentResult(**values)


# ── 去重与合并 / dedup and merge ──────────────────────────────────────────


def test_merge_visual_evidence_dedups_overlapping_boxes() -> None:
    first = [_box("small-vehicle", [100, 100, 200, 200])]
    second = [_box("small-vehicle", [110, 110, 190, 190]), _box("small-vehicle", [400, 400, 500, 500])]
    merged = merge_visual_evidence(first, second)
    assert len(merged) == 2


def test_merge_keeps_adjacent_instances_distinct() -> None:
    first = [_box("small-vehicle", [100, 100, 200, 200])]
    second = [_box("small-vehicle", [220, 100, 320, 200])]
    assert len(merge_visual_evidence(first, second)) == 2


def test_same_visual_observation_point_and_box() -> None:
    point = VisualEvidence(label="small-vehicle", point=[150, 150])
    box = _box("small-vehicle", [100, 100, 200, 200])
    assert same_visual_observation(point, box) is True
    assert same_visual_observation(box, _box("large-vehicle", [100, 100, 200, 200])) is False


def test_box_iou_and_intersection_over_smaller() -> None:
    first = [100, 100, 200, 200]
    second = [110, 110, 190, 190]
    assert box_iou(first, second) > 0.5
    assert box_intersection_over_smaller(first, second) > 0.5
    assert normalized_box_center_distance(first, second) < 0.4
    assert point_distance([0, 0], [3, 4]) == 5.0


# ── 标签兼容 / label compatibility ────────────────────────────────────────


def test_compatible_evidence_labels() -> None:
    assert compatible_evidence_labels("small-vehicle", "small-vehicle") is True
    assert compatible_evidence_labels("Small Vehicle", "small-vehicle") is True
    assert compatible_evidence_labels("vehicle", "small-vehicle") is True
    assert compatible_evidence_labels("small-vehicle", "large-vehicle") is False
    assert compatible_evidence_labels("car", "truck") is False


def test_matches_target_label() -> None:
    item = _box("Small Vehicle", [100, 100, 200, 200])
    assert matches_target_label(item, "small-vehicle") is True
    assert matches_target_label(item, "large-vehicle") is False
    assert matches_target_label(item, None) is True


def test_canonical_answer() -> None:
    assert canonical_answer("small-vehicle") == "small-vehicle"
    assert canonical_answer("trucks") == "truck"


# ── 角点锚定 / corner anchoring ───────────────────────────────────────────


def test_is_corner_anchored_box() -> None:
    assert is_corner_anchored_box(_box("x", [0, 0, 50, 50])) is True
    assert is_corner_anchored_box(_box("x", [950, 950, 999, 999])) is True
    assert is_corner_anchored_box(_box("x", [100, 100, 200, 200])) is False


# ── 严重度与占位符 / severity and placeholders ───────────────────────────


def test_maximum_repair_severity() -> None:
    assert maximum_repair_severity("none", "low") == "low"
    assert maximum_repair_severity("high", "low") == "high"
    assert maximum_repair_severity("none", "none") == "none"


def test_is_status_answer_placeholder() -> None:
    assert is_status_answer_placeholder("completed") is True
    assert is_status_answer_placeholder(" PARTIAL ") is True
    assert is_status_answer_placeholder("small-vehicle") is False


# ── 复核判定 / review decision ───────────────────────────────────────────


def test_needs_review_false_for_non_spatial_operations() -> None:
    result = _result([_box("small-vehicle", [100, 100, 200, 200])])
    assert needs_candidate_review(result, operation="box_gap", target_label=None) is False


def test_needs_review_grid_position_single_target() -> None:
    """A single non-corner-anchored candidate needs no enumeration review.
    单一非角点锚定候选不需要枚举复核。"""
    result = _result([_box("small-vehicle", [100, 100, 200, 200])])
    assert needs_candidate_review(
        result, operation="grid_position", target_label="small-vehicle"
    ) is False


def test_needs_review_extreme_sufficient_evidence_skips() -> None:
    items = [
        _box("small-vehicle", [100, 700, 200, 800]),
        _box("small-vehicle", [100, 0, 200, 40]),  # touches top band / 触顶带
    ]
    result = _result(items, status="completed", geometry={"extreme_direction": "top"})
    assert extreme_evidence_is_sufficient(
        result, direction="top", target_label="small-vehicle"
    ) is True
    assert needs_candidate_review(
        result, operation="extreme_category", target_label="small-vehicle"
    ) is False


def test_needs_review_extreme_insufficient_evidence() -> None:
    result = _result([_box("small-vehicle", [100, 100, 200, 200])])
    assert needs_candidate_review(
        result, operation="extreme_category", target_label="small-vehicle"
    ) is True


# ── position review evidence / 位置复核证据恢复 ───────────────────────────


def test_position_review_evidence_labels_top_level_boxes() -> None:
    review = AgentResult(
        agent_name="spatial_agent",
        answer="",
        boxes=[[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]],
    )
    evidence, labeled = position_review_evidence(
        review, is_grid=True, target_label="small-vehicle"
    )
    assert labeled == 2
    assert all(item.box is not None for item in evidence)
    assert all(item.label == "small-vehicle" for item in evidence)


def test_position_review_evidence_non_grid_passthrough() -> None:
    review = AgentResult(
        agent_name="spatial_agent", answer="", boxes=[[100.0, 100.0, 200.0, 200.0]]
    )
    evidence, labeled = position_review_evidence(
        review, is_grid=False, target_label="small-vehicle"
    )
    assert evidence == []
    assert labeled == 0


# ── 边界 / boundaries ─────────────────────────────────────────────────────


def test_evidence_merge_has_no_dataset_branch() -> None:
    source = (REPO_ROOT / "agents" / "spatial" / "evidence_merge.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source
    assert "question" not in source
