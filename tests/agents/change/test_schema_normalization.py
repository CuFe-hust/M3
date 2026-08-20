"""Regression fixtures for adjudication response normalization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.change.schema import (
    CANONICAL_NO_CHANGE,
    ChangeAdjudicationResult,
    ChangeCandidateReview,
    ChangeGlobalReview,
)


def _candidate(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "proposal_id": "change_000",
        "verdict": "appearance_only",
        "t1_state": "same",
        "t2_state": "same",
        "reason": "appearance differs only",
    }
    value.update(updates)
    return value


def _global(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "verdict": "no_persistent_change",
        "t1_state": "same",
        "t2_state": "same",
        "reason": "no durable change",
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    "verdict,category",
    [
        ("appearance_only", "vegetation_extent"),
        ("registration_artifact", "building_structure"),
        ("insufficient_visual_evidence", "water_geometry"),
    ],
)
def test_nonpersistent_candidate_category_is_cleared(verdict: str, category: str) -> None:
    result = ChangeCandidateReview.model_validate(
        _candidate(verdict=verdict, change_category=category)
    )
    assert result.change_category is None
    assert "ADJUDICATION_NONPERSISTENT_CATEGORY_CLEARED" in result.normalization_reasons


def test_transient_candidate_category_is_downgraded() -> None:
    result = ChangeCandidateReview.model_validate(
        _candidate(
            verdict="persistent_change",
            change_category="transient",
            persistent_geometry_changed=True,
        )
    )
    assert result.verdict == "transient"
    assert result.change_category is None
    assert result.persistent_geometry_changed is False
    assert "ADJUDICATION_TRANSIENT_CATEGORY_DOWNGRADED" in result.normalization_reasons


def test_nonpersistent_pseudo_category_on_persistent_candidate_is_downgraded() -> None:
    result = ChangeCandidateReview.model_validate(
        _candidate(verdict="persistent_change", change_category="appearance_only")
    )
    assert result.verdict == "appearance_only"
    assert result.change_category is None


def test_transient_global_category_is_downgraded_to_no_change() -> None:
    result = ChangeGlobalReview.model_validate(
        _global(verdict="persistent_change", change_category="transient")
    )
    assert result.verdict == "no_persistent_change"
    assert result.change_category is None
    assert result.persistent_geometry_changed is False


def test_invalid_schema_metadata_is_ignored_at_adjudication_boundary() -> None:
    result = ChangeAdjudicationResult.model_validate(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "agent_name": "change_agent",
            "global_review": _global(),
            "candidate_reviews": [_candidate()],
            "answer": CANONICAL_NO_CHANGE,
        }
    )
    assert result.agent_name == "change_agent"


def test_canonical_adjudication_negative_clears_model_evidence() -> None:
    result = ChangeAdjudicationResult.model_validate(
        {
            "agent_name": "change_agent",
            "global_review": _global(),
            "candidate_reviews": [_candidate()],
            "answer": CANONICAL_NO_CHANGE,
            "boxes": [[1, 2, 3, 4]],
            "evidence": ["change_000:reference_t1_crop"],
            "evidence_items": [{"image_id": "raw_full_t1", "description": "same"}],
        }
    )
    assert result.boxes == []
    assert result.evidence == []
    assert result.evidence_items == []
    assert result.geometry["change_input_normalizations"] == [
        "canonical_no_change_cleared_model_evidence"
    ]


def test_unknown_persistent_category_is_not_silently_invented() -> None:
    with pytest.raises(ValidationError):
        ChangeCandidateReview.model_validate(
            _candidate(verdict="persistent_change", change_category="not_a_real_category")
        )

from agents.change.schema import (
    BuildingRescueCandidateReview,
    BuildingRescueReview,
)


def test_building_rescue_review_rejects_duplicate_ids() -> None:
    review = BuildingRescueCandidateReview(
        candidate_id="c1", verdict="reject", reason="shadow"
    )
    with pytest.raises(ValueError, match="duplicate"):
        BuildingRescueReview(reviews=(review, review))


def test_building_rescue_review_requires_final_answer_only_for_confirmations() -> None:
    with pytest.raises(ValueError, match="final_answer"):
        BuildingRescueReview(
            reviews=(
                BuildingRescueCandidateReview(
                    candidate_id="c1", verdict="reject", reason="not a building"
                ),
            ),
            final_answer="a building was added",
        )
    valid = BuildingRescueReview(
        reviews=(
            BuildingRescueCandidateReview(
                candidate_id="c1", verdict="confirmed_added_building", reason="roof"
            ),
        ),
        final_answer=None,
    )
    assert valid.reviews[0].verdict == "confirmed_added_building"

