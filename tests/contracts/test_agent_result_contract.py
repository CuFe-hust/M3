"""Contract tests for AgentResult and VisualEvidence.

AgentResult / VisualEvidence 契约测试：框/点严格验证、corner pair / flat box
防御性归一化、repair severity、AgentName。不定义 CountingResult。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.schema import AgentName, AgentResult, VisualEvidence


# ── VisualEvidence / 视觉证据 ───────────────────────────────────────────────


def test_evidence_requires_exactly_one_of_box_or_point() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        VisualEvidence(label="x")
    with pytest.raises(ValidationError, match="exactly one"):
        VisualEvidence(label="x", box=[1, 2, 3, 4], point=[5, 6])


def test_evidence_box_must_be_4_values_in_range_with_ordered_corners() -> None:
    with pytest.raises(ValidationError, match="0..999"):
        VisualEvidence(label="x", box=[1, 2, 3, 1000])
    with pytest.raises(ValidationError, match="x1<x2"):
        VisualEvidence(label="x", box=[3, 2, 1, 4])
    ok = VisualEvidence(label="x", box=[1, 2, 3, 4])
    assert ok.box == [1, 2, 3, 4]


def test_evidence_point_must_be_2_values_in_range() -> None:
    with pytest.raises(ValidationError, match="0..999"):
        VisualEvidence(label="x", point=[1, 1000])
    ok = VisualEvidence(label="x", point=[10, 20])
    assert ok.point == [10, 20]


def test_evidence_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        VisualEvidence.model_validate({"label": "x", "box": [1, 2, 3, 4], "score": 0.9})


# ── AgentName / Agent 名 ────────────────────────────────────────────────────


def test_all_agent_names_are_accepted() -> None:
    for name in ("counting_agent", "change_agent", "grounding_agent",
                 "general_vqa_agent", "caption_agent"):
        result = AgentResult(agent_name=name, answer="ok")
        assert result.agent_name == name


def test_unknown_agent_name_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentResult(agent_name="mystery_agent", answer="ok")


# ── AgentResult 归一化 / normalization ──────────────────────────────────────


def test_flat_top_level_box_is_wrapped() -> None:
    result = AgentResult(agent_name="general_vqa_agent", answer="a", boxes=[[10, 20, 30, 40]])
    assert result.boxes == [[10, 20, 30, 40]]


def test_corner_pairs_combined_into_boxes() -> None:
    result = AgentResult(
        agent_name="general_vqa_agent",
        answer="a",
        boxes=[[10, 20], [30, 40], [50, 60], [70, 80]],
    )
    assert result.boxes == [[10, 20, 30, 40], [50, 60, 70, 80]]
    assert "top_level_corner_pairs_combined_as_boxes" in result.geometry["input_normalizations"]
    assert result.geometry["repair_severity"] == "low"


def test_reversed_corners_are_reordered() -> None:
    result = AgentResult(agent_name="general_vqa_agent", answer="a", boxes=[[30, 40, 10, 20]])
    assert result.boxes == [[10, 20, 30, 40]]
    assert "box_corners_reordered" in result.geometry["input_normalizations"]


def test_degenerate_top_level_box_is_dropped() -> None:
    result = AgentResult(agent_name="general_vqa_agent", answer="a", boxes=[[10, 10, 10, 10]])
    assert result.boxes == []
    assert "degenerate_top_level_box_dropped" in result.geometry["input_normalizations"]
    assert result.geometry["repair_severity"] == "high"


def test_evidence_corner_pair_combined_as_boxes() -> None:
    result = AgentResult(
        agent_name="general_vqa_agent",
        answer="a",
        evidence_items=[
            {"label": "target", "box": [10, 20], "point": [30, 40]},
        ],
    )
    item = result.evidence_items[0]
    assert item.box == [10, 20, 30, 40]
    assert item.point is None
    assert "evidence_box_and_point_combined_as_corners" in result.geometry["input_normalizations"]
    assert result.boxes == [[10, 20, 30, 40]]


def test_degenerate_evidence_box_becomes_point() -> None:
    result = AgentResult(
        agent_name="general_vqa_agent",
        answer="a",
        evidence_items=[
            {"label": "target", "box": [10, 20, 10, 20]},
        ],
    )
    item = result.evidence_items[0]
    assert item.box is None
    assert item.point == [10, 20]
    assert "degenerate_evidence_box_reclassified_as_point" in result.geometry["input_normalizations"]
    assert result.geometry["repair_severity"] == "high"


def test_two_value_evidence_box_becomes_point() -> None:
    result = AgentResult(
        agent_name="grounding_agent",
        answer="a",
        evidence_items=[{"label": "target", "box": [10, 20]}],
    )
    item = result.evidence_items[0]
    assert item.box is None
    assert item.point == [10, 20]
    assert "two_value_evidence_box_reclassified_as_point" in result.geometry["input_normalizations"]


def test_evidence_boxes_retained_in_canonical_list() -> None:
    result = AgentResult(
        agent_name="grounding_agent",
        answer="a",
        evidence_items=[
            {"label": "a", "box": [1, 2, 3, 4]},
            {"label": "b", "box": [5, 6, 7, 8]},
        ],
    )
    assert result.boxes == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert result.geometry["evidence_quality"] == ["trusted_box", "trusted_box"]
    assert result.geometry["repair_severity"] == "none"


def test_no_normalizations_when_clean() -> None:
    result = AgentResult(agent_name="general_vqa_agent", answer="yes")
    assert result.geometry == {}
    assert result.boxes == []
