from __future__ import annotations

import pytest

from spacers_agent.schemas import ExpertResult, VisualEvidence
from spacers_agent.vqa_geometry import (
    apply_vrsbench_geometry,
    execution_task_for_vrsbench,
    normalize_vrsbench_answer,
    vrsbench_answer_vocabulary,
    vrsbench_count_target,
    vrsbench_question_subtype,
)


def _result(*items: VisualEvidence, answer: str = "model answer") -> ExpertResult:
    return ExpertResult(
        expert="spatial_expert",
        answer=answer,
        evidence_items=list(items),
    )


def test_top_and_bottom_category_use_box_center_y_and_closed_vehicle_classes() -> None:
    top = VisualEvidence(label="car", box=[400, 100, 500, 180], confidence=0.9)
    bottom = VisualEvidence(label="truck", box=[400, 700, 520, 850], confidence=0.8)

    top_result = apply_vrsbench_geometry(
        "What object class is the top-most vehicle?",
        "object category",
        _result(bottom, top),
    )
    bottom_result = apply_vrsbench_geometry(
        "What object class is the bottom-most vehicle?",
        "object category",
        _result(bottom, top),
    )

    assert top_result.answer == "small-vehicle"
    assert bottom_result.answer == "large-vehicle"
    assert top_result.geometry["rule"] == "top_most_box_center_y"
    assert top_result.boxes == [[400, 700, 520, 850], [400, 100, 500, 180]]


def test_extreme_category_does_not_claim_geometry_from_one_candidate() -> None:
    only = VisualEvidence(label="bus", box=[0, 388, 107, 600], confidence=0.8)

    result = apply_vrsbench_geometry(
        "What object class is the top-most vehicle?",
        "object category",
        _result(only, answer="bus"),
    )

    assert result.answer == "large-vehicle"
    assert result.geometry["rule"] == "insufficient_extreme_candidates"
    assert result.geometry["candidate_count"] == 1
    assert result.geometry["evidence_complete"] is False


def test_position_uses_top_left_raster_coordinates_and_three_by_three_grid() -> None:
    target = VisualEvidence(label="large-vehicle", box=[50, 380, 200, 570], confidence=0.9)

    result = apply_vrsbench_geometry(
        "What is the position of the large vehicle in the image?",
        "object position",
        _result(target),
    )

    assert result.answer == "middle-left"
    assert result.geometry["coordinate_frame"] == "normalized_0_999_top_left"
    assert result.geometry["rule"] == "three_by_three_box_center"


@pytest.mark.parametrize("question_type", ["object direction", "object position"])
def test_direction_without_north_metadata_is_not_overridden(question_type: str) -> None:
    result = apply_vrsbench_geometry(
        "What is the orientation of the road in the image?",
        question_type,
        _result(answer="diagonal"),
    )

    assert result.answer == "diagonal"
    assert result.geometry["north_metadata_available"] is False
    assert result.geometry["answer_source"] == "qwen_visual_answer"


def test_unknown_official_question_type_falls_back_to_general_vqa() -> None:
    assert execution_task_for_vrsbench("unregistered type", "What kind of area is shown?") == "general_vqa"


def test_answer_normalization_uses_declared_vocabularies_only() -> None:
    assert normalize_vrsbench_answer("object existence", "Yes, a vehicle is visible.") == "yes"
    assert normalize_vrsbench_answer("object category", "car") == "small-vehicle"
    assert normalize_vrsbench_answer("object category", "The object is a bus.") == "large-vehicle"
    assert normalize_vrsbench_answer("object position", "It is in the top right.") == "top-right"
    assert normalize_vrsbench_answer("object color", "The vehicles are white.") == "white"


def test_vrsbench_count_target_has_fixed_vehicle_aliases() -> None:
    small = vrsbench_count_target("How many small vehicles are visible?")
    large = vrsbench_count_target("How many large vehicles are visible?")
    all_vehicles = vrsbench_count_target("How many vehicles are visible?")

    assert small.canonical_label == "small-vehicle" and "motorcycle" in small.aliases
    assert large.canonical_label == "large-vehicle" and "trailer" in large.aliases
    assert all_vehicles.canonical_label == "vehicle" and "car" in all_vehicles.aliases


def test_expert_result_normalizes_corner_pair_boxes_and_single_points() -> None:
    boxed = ExpertResult.model_validate(
        {
            "expert": "spatial_expert",
            "answer": "middle-left",
            "boxes": [[0, 427], [100, 599]],
            "evidence_items": [{"label": "large vehicle", "box": [0, 427], "confidence": 0.9}],
        }
    )
    pointed = ExpertResult.model_validate(
        {
            "expert": "spatial_expert",
            "answer": "yes",
            "boxes": [[220, 100]],
            "evidence_items": [{"label": "vehicle", "box": [220, 100], "confidence": 0.9}],
        }
    )

    assert boxed.boxes == [[0, 427, 100, 599]]
    assert boxed.evidence_items[0].box == [0, 427, 100, 599]
    assert "top_level_corner_pairs_combined_as_boxes" in boxed.geometry["input_normalizations"]
    assert pointed.boxes == []
    assert pointed.evidence_items[0].point == [220, 100]


def test_expert_result_repairs_degenerate_boxes_and_resolves_point_conflicts() -> None:
    result = ExpertResult.model_validate(
        {
            "expert": "spatial_expert",
            "answer": "yes",
            "boxes": [[999, 249], [0, 249]],
            "evidence_items": [
                {
                    "label": "small-vehicle",
                    "box": [127, 407, 127, 407],
                    "point": [127, 407],
                    "confidence": 0.95,
                }
            ],
        }
    )

    assert result.evidence_items[0].box is None
    assert result.evidence_items[0].point == [127, 407]
    assert result.boxes == []
    normalizations = result.geometry["input_normalizations"]
    assert "box_corners_reordered" in normalizations
    assert "degenerate_top_level_box_dropped" in normalizations
    assert "degenerate_evidence_box_dropped_in_favor_of_point" in normalizations
    assert result.geometry["repair_severity"] == "high"
    assert result.geometry["evidence_quality"] == ["trusted_point"]


def test_degenerate_line_without_point_is_retained_only_as_repaired_point() -> None:
    result = ExpertResult.model_validate(
        {
            "expert": "spatial_expert",
            "answer": "north-south",
            "boxes": [[100, 250, 300, 250]],
            "evidence_items": [
                {"label": "road", "box": [100, 250, 300, 250], "confidence": 0.8}
            ],
        }
    )

    assert result.boxes == []
    assert result.evidence_items[0].box is None
    assert result.evidence_items[0].point == [200, 250]
    assert result.geometry["evidence_quality"] == ["repaired_point"]


def test_semantic_subtype_overrides_coarse_position_type_without_reference_access() -> None:
    subtype = vrsbench_question_subtype(
        "What is the orientation of the road in the image?",
        "object position",
    )

    assert subtype == "orientation"
    assert vrsbench_answer_vocabulary(subtype, "What is the orientation of the road in the image?") == [
        "north-south",
        "east-west",
    ]
    result = apply_vrsbench_geometry(
        "What is the orientation of the road in the image?",
        "object position",
        _result(answer="The road runs from the top-right to the bottom-left."),
    )
    assert result.answer == "The road runs from the top-right to the bottom-left."
    assert result.geometry["semantic_subtype"] == "orientation"
    assert result.geometry["rule"] == "cardinal_direction_requires_dataset_north_up_assumption"


def test_top_most_existence_is_distinct_from_extreme_category() -> None:
    subtype = vrsbench_question_subtype(
        "Is there a vehicle located at the top-most position in the image?",
        "object existence",
    )

    assert subtype == "extreme_existence"
    assert vrsbench_answer_vocabulary(
        subtype,
        "Is there a vehicle located at the top-most position in the image?",
    ) == ["yes", "no"]


def test_arrangement_normalizes_descriptive_row_language_to_closed_vocabulary() -> None:
    result = apply_vrsbench_geometry(
        "What is the arrangement of the large vehicles?",
        "object direction",
        _result(
            VisualEvidence(label="truck", box=[100, 100, 200, 400], confidence=0.9),
            VisualEvidence(label="truck", box=[250, 100, 350, 400], confidence=0.9),
            answer="They are parallel and parked side by side.",
        ),
    )

    assert result.answer == "in rows"
    assert result.geometry["semantic_subtype"] == "arrangement"
    assert result.geometry["candidate_count"] == 2
    assert result.status == "completed"


@pytest.mark.parametrize(
    ("question", "question_type", "expected_subtype"),
    [
        ("What type of area is visible in the image?", "object category", "general"),
        ("What object class do these structures belong to?", "object category", "category"),
        ("Where is the lighter colored vehicle positioned?", "object position", "grid_position"),
        ("Where is the top-most harbor located?", "object position", "general"),
        ("Is the harbor closer to the top or bottom?", "object position", "general"),
    ],
)
def test_semantic_subtypes_do_not_inherit_coarse_official_assumptions(
    question: str,
    question_type: str,
    expected_subtype: str,
) -> None:
    """Keep coarse dataset labels from forcing unrelated specialized semantics.
    防止粗粒度数据集标签强制产生无关的专用语义。
    """

    assert vrsbench_question_subtype(question, question_type) == expected_subtype


def test_closed_vocabulary_requires_explicit_question_evidence() -> None:
    """Use vehicle labels only for an explicit vehicle-class question.
    仅对明确的车辆类别问题使用车辆封闭词表。
    """

    assert vrsbench_answer_vocabulary("category", "What kind of area is shown?") == []
    assert vrsbench_answer_vocabulary("category", "What object class are the ships?") == []
    assert vrsbench_answer_vocabulary("category", "What type of vehicles are visible?") == [
        "small-vehicle",
        "large-vehicle",
    ]


def test_general_answers_do_not_require_localized_boxes() -> None:
    """Do not downgrade a supported global answer only because it has no box.
    不因全局答案缺少目标框而将其降级为部分结果。
    """

    result = apply_vrsbench_geometry(
        "What kind of area is shown in the image?",
        "object category",
        _result(answer="residential"),
    )

    assert result.answer == "residential"
    assert result.status == "completed"
    assert result.geometry["semantic_subtype"] == "general"


def test_status_tokens_are_not_persisted_as_vqa_answers() -> None:
    """Separate incomplete workflow status from the semantic answer value.
    将未完成工作流状态与语义答案值分离。
    """

    result = apply_vrsbench_geometry(
        "What kind of area is shown in the image?",
        "object category",
        ExpertResult(expert="general_vqa_expert", answer="Partial.", status="partial"),
    )

    assert result.answer == ""
    assert result.status == "partial"
    assert result.geometry["status_answer_placeholder_removed"] is True
