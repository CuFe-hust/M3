"""Unit tests for pure spatial evidence-review rules.
空间证据复核纯规则的单元测试。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.spatial.evidence_merge import (
    box_intersection_over_smaller,
    box_iou,
    compatible_evidence_labels,
    extreme_vehicle_evidence_is_sufficient,
    is_corner_anchored_box,
    matches_position_target,
    maximum_repair_severity,
    merge_visual_evidence,
    needs_candidate_review,
    normalized_box_center_distance,
    point_distance,
    position_review_evidence,
    same_box_observation,
    vehicle_label_kind,
)
from spacers_agent.agents.spatial.candidate_review import SpatialCandidateReviewResult
from models.qwen_transformers import _validate_response
from spacers_agent.schemas import AgentResult, ImageRef, UnifiedSample, VisualEvidence


def _grid_sample(question: str = "Where is the large vehicle located?") -> UnifiedSample:
    return UnifiedSample(
        sample_id="grid",
        dataset="VRSBench",
        split="validation",
        task="general_vqa",
        images=[ImageRef(image_id="image", path=Path("image.png"), role="image")],
        question=question,
        metadata={"question_type": "position"},
    )


def test_needs_candidate_review_replaces_corner_placeholder_once() -> None:
    corner = VisualEvidence(label="large-vehicle", box=[0, 0, 200, 200], confidence=0.9)
    first = AgentResult(
        agent_name="spatial_agent",
        answer="top-left",
        evidence_items=[corner],
        status="completed",
    )
    reviewed = first.model_copy(
        update={
            "evidence_items": [
                VisualEvidence(label="large-vehicle", box=[400, 400, 520, 520], confidence=0.9)
            ],
            "geometry": {"candidate_review_used": True},
        }
    )

    assert needs_candidate_review(_grid_sample(), first) is True
    assert needs_candidate_review(_grid_sample(), reviewed) is False


def test_edge_complete_extreme_vehicle_skips_review() -> None:
    sample = _grid_sample("What object class is the bottom-most vehicle?").model_copy(
        update={"metadata": {"question_type": "object category"}}
    )
    result = AgentResult(
        agent_name="spatial_agent",
        answer="small-vehicle",
        evidence_items=[
            VisualEvidence(label="large-vehicle", box=[10, 300, 100, 600]),
            VisualEvidence(label="small-vehicle", box=[680, 970, 740, 999]),
        ],
        status="completed",
    )

    assert extreme_vehicle_evidence_is_sufficient(sample.question, result) is True
    assert needs_candidate_review(sample, result) is False


def test_central_extreme_and_arrangement_keep_review() -> None:
    extreme = _grid_sample("What object class is the bottom-most vehicle?").model_copy(
        update={"metadata": {"question_type": "object category"}}
    )
    arrangement = _grid_sample("Are the vehicles arranged in a line?").model_copy(
        update={"metadata": {"question_type": "object arrangement"}}
    )
    result = AgentResult(
        agent_name="spatial_agent",
        answer="small-vehicle",
        evidence_items=[
            VisualEvidence(label="large-vehicle", box=[10, 300, 100, 600]),
            VisualEvidence(label="small-vehicle", box=[600, 700, 680, 800]),
        ],
        status="completed",
    )

    assert extreme_vehicle_evidence_is_sufficient(extreme.question, result) is False
    assert needs_candidate_review(extreme, result) is True
    assert needs_candidate_review(arrangement, result) is True


def test_position_target_and_corner_rules_are_explicit() -> None:
    large = VisualEvidence(label="large vehicle", box=[0, 0, 100, 300], confidence=0.8)
    small = VisualEvidence(label="small-vehicle", box=[100, 100, 200, 200], confidence=0.8)

    assert matches_position_target("Where is the large vehicle?", large) is True
    assert matches_position_target("Where is the large vehicle?", small) is False
    assert is_corner_anchored_box(large) is True
    assert is_corner_anchored_box(small) is False


def test_position_review_recovers_labeled_top_level_boxes() -> None:
    review = AgentResult(
        agent_name="spatial_agent",
        answer="bottom-middle",
        boxes=[[420, 670, 520, 770]],
        evidence_items=[],
        status="completed",
    )

    evidence, recovered = position_review_evidence(
        "Where is the large vehicle?",
        "grid_position",
        review,
    )

    assert recovered == 1
    assert evidence[0].label == "large-vehicle"
    assert evidence[0].box == [420, 670, 520, 770]


def test_merge_visual_evidence_prefers_box_and_removes_duplicates() -> None:
    point = VisualEvidence(label="small-vehicle", point=[150, 150], confidence=0.95)
    box = VisualEvidence(label="small vehicle", box=[100, 100, 200, 200], confidence=0.8)
    distinct = VisualEvidence(label="small-vehicle", box=[400, 400, 500, 500], confidence=0.7)

    merged = merge_visual_evidence([point], [box, distinct])

    assert merged == [box, distinct]


def test_merge_visual_evidence_deduplicates_positional_vehicle_roles() -> None:
    first = [
        VisualEvidence(label="bottom-most vehicle", box=[592, 812, 638, 888], confidence=0.95),
        VisualEvidence(label="top-most vehicle", box=[632, 292, 678, 368], confidence=0.95),
    ]
    review = [
        VisualEvidence(label="small-vehicle", box=[596, 812, 642, 888], confidence=0.95),
        VisualEvidence(label="small-vehicle", box=[626, 296, 672, 362], confidence=0.95),
        VisualEvidence(label="small-vehicle", box=[696, 972, 742, 999], confidence=0.8),
        VisualEvidence(label="large-vehicle", box=[16, 396, 86, 592], confidence=0.95),
    ]

    merged = merge_visual_evidence(first, review)

    assert merged == review
    assert compatible_evidence_labels("bottom-most vehicle", "small-vehicle") is True
    assert compatible_evidence_labels("bottom-most vehicle", "large-vehicle") is True
    assert compatible_evidence_labels("small-vehicle", "large-vehicle") is False


def test_shifted_small_boxes_use_overlap_and_center_guard() -> None:
    first = [632, 292, 678, 368]
    shifted = [626, 296, 672, 362]
    adjacent = [675, 296, 721, 362]

    assert box_iou(first, shifted) < 0.7
    assert box_intersection_over_smaller(first, shifted) >= 0.8
    assert normalized_box_center_distance(first, shifted) <= 0.25
    assert same_box_observation(first, shifted) is True
    assert same_box_observation(first, adjacent) is False


@pytest.mark.parametrize(
    ("first_label", "first_box", "review_box"),
    [
        ("Reference: Small vehicles on road", [580, 280, 650, 350], [620, 295, 665, 360]),
        ("bottom-most vehicle", [592, 812, 638, 888], [615, 816, 660, 890]),
        ("isolated_vehicle", [826, 286, 906, 326], [853, 288, 916, 322]),
    ],
)
def test_report_shifted_vehicle_boxes_merge_to_tighter_review(
    first_label: str,
    first_box: list[int],
    review_box: list[int],
) -> None:
    first = VisualEvidence(label=first_label, box=first_box, confidence=0.95)
    review = VisualEvidence(label="small-vehicle", box=review_box, confidence=0.0)

    merged = merge_visual_evidence([first], [review])

    assert same_box_observation(first_box, review_box) is True
    assert merged == [review]


def test_relaxed_shift_guard_keeps_adjacent_vehicles_distinct() -> None:
    first = VisualEvidence(label="small-vehicle", box=[100, 100, 150, 170], confidence=0.9)
    adjacent = VisualEvidence(label="small-vehicle", box=[135, 100, 185, 170], confidence=0.9)

    assert box_intersection_over_smaller(first.box, adjacent.box) < 0.45
    assert same_box_observation(first.box, adjacent.box) is False
    assert merge_visual_evidence([first], [adjacent]) == [first, adjacent]


def test_generic_vehicle_roles_match_explicit_vehicle_classes() -> None:
    assert vehicle_label_kind("isolated_vehicle") == "vehicle"
    assert vehicle_label_kind("target vehicle") == "vehicle"
    assert vehicle_label_kind("reference-vehicle") == "vehicle"
    assert compatible_evidence_labels("isolated_vehicle", "small-vehicle") is True


def test_geometry_helpers_preserve_threshold_inputs() -> None:
    assert box_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert point_distance([0, 0], [3, 4]) == 5.0
    assert maximum_repair_severity("low", "high") == "high"


def test_compact_review_clamps_only_one_step_coordinate_drift() -> None:
    result = SpatialCandidateReviewResult.model_validate(
        {"boxes": [["small-vehicle", -1, 966, 738, 1000]], "complete": True}
    )

    assert result.boxes == [("small-vehicle", 0, 966, 738, 999)]

    with pytest.raises(ValueError):
        SpatialCandidateReviewResult.model_validate(
            {"boxes": [["small-vehicle", -2, 966, 738, 1001]], "complete": True}
        )


def test_compact_review_recovers_qwen_missing_inner_brackets_locally() -> None:
    malformed = (
        '{"boxes":[["large-vehicle",11,400,88,595],'
        '"small-vehicle",620,295,665,360],'
        '"small-vehicle",610,815,655,885],"complete":true}'
    )

    result = _validate_response(malformed, SpatialCandidateReviewResult)

    assert len(result.boxes) == 3
    assert result.complete is True
    assert result._local_recoveries == ["compact_candidate_sequence_recovered_locally"]
