from __future__ import annotations

import pytest

from spacers_agent.schemas import ExpertResult, VisualEvidence
from spacers_agent.vqa_geometry import apply_vrsbench_geometry, execution_task_for_vrsbench


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


def test_unknown_official_question_type_fails_visibly() -> None:
    with pytest.raises(ValueError, match="Unsupported VRSBench VQA type"):
        execution_task_for_vrsbench("unregistered type")
