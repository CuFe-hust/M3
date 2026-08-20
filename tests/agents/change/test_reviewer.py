"""Contract tests for the conservative change claim reviewer.

变化声明保守规则复核契约测试：证据缺失、结论冲突、证据框越界、外观词告警、
关闭复核行为、partial 状态与 geometry 记录；纯规则、不调用模型。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.change.reviewer import review_outcome, review_result
from agents.change.schema import ChangeProposal
from agents.change.settings import ChangeReviewSettings
from agents.schema import AgentResult, VisualEvidence


def _proposal(
    proposal_id: str = "p1",
    box: list[int] | None = None,
    score: float = 0.9,
    area_ratio: float = 0.05,
) -> ChangeProposal:
    return ChangeProposal(
        proposal_id=proposal_id,
        box=box or [100, 100, 300, 300],
        pixel_box=box or [100, 100, 300, 300],
        score=score,
        area_ratio=area_ratio,
    )


def _result(answer: str = "A building was removed.", **overrides) -> AgentResult:
    values = dict(agent_name="change_agent", answer=answer)
    values.update(overrides)
    return AgentResult(**values)


def test_claim_without_proposal_evidence_is_warned() -> None:
    result = _result(answer="The tree disappeared.")
    reviewed, warnings = review_result(result, [], ChangeReviewSettings())
    assert "CHANGE_CLAIM_WITHOUT_PROPOSAL_EVIDENCE" in warnings
    assert reviewed.status == "partial"


def test_claim_without_evidence_when_disabled_passes_through() -> None:
    result = _result(answer="The tree disappeared.")
    reviewed, warnings = review_result(
        result, [], ChangeReviewSettings(enabled=False)
    )
    assert warnings == []
    assert reviewed is result


def test_no_change_with_high_score_proposals_is_conflict() -> None:
    result = _result(answer="No visible change.")
    reviewed, warnings = review_result(
        result, [_proposal("p1"), _proposal("p2", score=0.8)], ChangeReviewSettings()
    )
    assert "CHANGE_RESULT_CONFLICT" in warnings
    assert reviewed.status == "completed"


def test_no_change_with_single_proposal_is_not_conflict() -> None:
    result = _result(answer="No visible change.")
    _, warnings = review_result(
        result, [_proposal("p1", score=0.9, area_ratio=0.001)], ChangeReviewSettings()
    )
    assert "CHANGE_RESULT_CONFLICT" in warnings


def test_landcover_only_semantic_candidate_is_suppressed_from_adjudication() -> None:
    proposal = _proposal("landcover", score=0.9).model_copy(
        update={
            "component_scores": {"semantic": 0.9},
            "semantic_consensus": {
                "structural_support": 0.0,
                "landcover_support": 0.9,
                "transient_support": 0.0,
            },
            "semantic_transitions": [
                {"evidence_type": "landcover_candidate"}
            ],
        }
    )
    outcome = review_outcome(
        _result(answer="No significant semantic change detected."),
        [proposal],
        ChangeReviewSettings(),
        task="change_caption",
    )
    assert outcome.route == "accept"
    assert "NEGATIVE_LANDCOVER_ONLY_SUPPRESSED" in outcome.route_reasons


def test_structural_semantic_candidate_still_routes_to_adjudication() -> None:
    proposal = _proposal("building", score=0.5).model_copy(
        update={
            "component_scores": {"semantic": 0.5},
            "semantic_consensus": {
                "structural_support": 0.8,
                "landcover_support": 0.0,
                "transient_support": 0.0,
            },
            "semantic_transitions": [
                {"evidence_type": "structural_candidate"}
            ],
        }
    )
    outcome = review_outcome(
        _result(answer="No significant semantic change detected."),
        [proposal],
        ChangeReviewSettings(),
        task="change_caption",
    )
    assert outcome.route == "adjudicate_negative"
    assert "NEGATIVE_STRONG_PROPOSAL" in outcome.route_reasons


def test_canonical_no_change_phrase_uses_current_proposal_score_scale() -> None:
    result = _result(answer="No significant semantic change detected.")
    _, warnings = review_result(
        result,
        [
            _proposal("p1", score=0.23, area_ratio=0.006),
            _proposal("p2", score=0.20, area_ratio=0.005),
        ],
        ChangeReviewSettings(),
    )
    assert "CHANGE_RESULT_CONFLICT" not in warnings
    assert "CHANGE_CLAIM_WITHOUT_PROPOSAL_EVIDENCE" not in warnings


def test_evidence_box_outside_proposals_is_warned() -> None:
    result = _result(
        evidence_items=[
            VisualEvidence(label="building", box=[500, 500, 600, 600])
        ]
    )
    _, warnings = review_result(
        result, [_proposal("p1", box=[100, 100, 300, 300])], ChangeReviewSettings()
    )
    assert "EVIDENCE_OUTSIDE_PROPOSALS" in warnings


def test_evidence_box_overlapping_proposal_is_accepted() -> None:
    result = _result(
        evidence_items=[
            VisualEvidence(label="building", box=[150, 150, 250, 250])
        ]
    )
    _, warnings = review_result(
        result, [_proposal("p1", box=[100, 100, 300, 300])], ChangeReviewSettings()
    )
    assert "EVIDENCE_OUTSIDE_PROPOSALS" not in warnings


def test_appearance_wording_triggers_warning() -> None:
    result = _result(answer="The road looks brighter.")
    _, warnings = review_result(
        result, [_proposal("p1")], ChangeReviewSettings()
    )
    assert "APPEARANCE_CHANGE_NOT_SEMANTIC_EVIDENCE" in warnings


def test_semantic_change_with_proposals_is_clean() -> None:
    result = _result(
        answer="A building was removed.",
        evidence_items=[
            VisualEvidence(label="building", box=[150, 150, 250, 250])
        ],
    )
    reviewed, warnings = review_result(
        result, [_proposal("p1", box=[100, 100, 300, 300])], ChangeReviewSettings()
    )
    assert warnings == []
    assert reviewed.status == "completed"


def test_reviewer_does_not_treat_v2_component_hints_as_semantic_truth() -> None:
    proposal = _proposal().model_copy(
        update={
            "source": "fused_change_v2",
            "component_scores": {
                "low_level": 0.4,
                "feature": 0.9,
                "semantic": 0.8,
                "fused": 0.75,
            },
        }
    )
    result = _result(
        answer="A truck appeared.",
        evidence_items=[VisualEvidence(label="truck", box=[150, 150, 250, 250])],
    )

    reviewed, warnings = review_result(result, [proposal], ChangeReviewSettings())

    assert warnings == []
    assert reviewed.status == "completed"


def test_geometry_records_review_and_proposal_ids() -> None:
    result = _result()
    reviewed, _ = review_result(
        result, [_proposal("p1"), _proposal("p2")], ChangeReviewSettings()
    )
    assert reviewed.geometry["change_review"]["proposal_ids"] == ["p1", "p2"]
    assert reviewed.geometry["evidence_path_types"] == ["raw_crop", "harmonized_crop"]


def test_geometry_records_raw_full_when_no_proposals() -> None:
    reviewed, _ = review_result(_result(), [], ChangeReviewSettings())
    assert reviewed.geometry["change_review"]["warnings"] == [
        "CHANGE_CLAIM_WITHOUT_PROPOSAL_EVIDENCE"
    ]
    assert reviewed.geometry["evidence_path_types"] == ["raw_full"]


def test_unknown_proposal_reference_is_warned() -> None:
    result = _result(geometry={"proposal_ids": ["missing"]})
    _, warnings = review_result(
        result, [_proposal("p1")], ChangeReviewSettings()
    )
    assert "EVIDENCE_REFERENCES_UNKNOWN_PROPOSAL" in warnings


def test_invalid_overlap_evidence_reference_is_warned() -> None:
    result = _result(
        evidence_items=[
            VisualEvidence(
                label="candidate",
                image_id="p1:non_overlap_crop",
                box=[100, 100, 200, 200],
            )
        ]
    )
    _, warnings = review_result(
        result, [_proposal("p1")], ChangeReviewSettings()
    )
    assert "EVIDENCE_REFERENCES_INVALID_OVERLAP" in warnings


def test_one_sided_temporal_evidence_is_warned() -> None:
    result = _result(
        answer="No significant semantic change detected.",
        evidence_items=[
            VisualEvidence(
                label="No visible change.",
                image_id="raw_full_t1",
                box=[100, 100, 200, 200],
            )
        ],
    )
    _, warnings = review_result(
        result,
        [_proposal("p1", box=[100, 100, 300, 300], area_ratio=0.001)],
        ChangeReviewSettings(),
    )
    assert "EVIDENCE_MISSING_TEMPORAL_PAIR" in warnings


def test_reviewer_never_calls_models_or_dataset_logic() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "change" / "reviewer.py").read_text(
        encoding="utf-8"
    )
    for token in ("qwen", "deepseek", "complete_json", "vrsbench", "dataset"):
        assert token not in source.casefold(), token


def test_invalid_proposal_score_rejected() -> None:
    with pytest.raises(ValueError):
        _proposal(score=1.5)
