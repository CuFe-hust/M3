"""Conservative rule review for model change claims.

模型变化结论的保守规则复核。纯规则、不调用模型；添加告警但不反转模型
语义结论。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agents.change.schema import ChangeProposal
from agents.change.settings import ChangeReviewSettings
from agents.schema import AgentResult

ReviewRoute = Literal["accept", "adjudicate_negative", "adjudicate_positive"]


@dataclass(frozen=True)
class ChangeReviewOutcome:
    result: AgentResult
    warnings: tuple[str, ...]
    route: ReviewRoute
    route_reasons: tuple[str, ...]


def is_canonical_no_change(answer: str) -> bool:
    return answer.strip() == "No significant semantic change detected."


def is_unresolved_change_answer(answer: str) -> bool:
    return answer.strip() == "Unable to confirm a persistent semantic change from the available evidence."


def is_positive_change_answer(answer: str) -> bool:
    return bool(answer.strip()) and not is_canonical_no_change(answer) and not is_unresolved_change_answer(answer)


def is_no_change_answer(answer: str) -> bool:
    """Recognize completed no-change wording without treating uncertainty as negative."""
    if is_canonical_no_change(answer):
        return True
    normalized = answer.casefold()
    return any(token in normalized for token in (
        "no visible change", "no change", "no significant semantic change",
        "no significant change", "未见变化", "没有变化", "无显著语义变化", "无明显变化",
    ))


def meaningful_proposals(
    proposals: list[ChangeProposal], settings: ChangeReviewSettings
) -> list[ChangeProposal]:
    return [item for item in proposals if item.score >= settings.no_change_conflict_min_score]


def has_no_change_conflict(
    result: AgentResult, proposals: list[ChangeProposal], settings: ChangeReviewSettings
) -> bool:
    if not settings.enabled or not is_no_change_answer(result.answer):
        return False
    return _negative_conflict_reasons(proposals, settings) != []


def _is_edge(proposal: ChangeProposal, margin_ratio: float) -> bool:
    margin = round(999 * margin_ratio)
    x1, y1, x2, y2 = proposal.box
    return x1 <= margin or y1 <= margin or x2 >= 999 - margin or y2 >= 999 - margin


def _negative_conflict_reasons(
    proposals: list[ChangeProposal], settings: ChangeReviewSettings
) -> list[str]:
    reasons: list[str] = []
    for proposal in proposals:
        adjusted = proposal.score * (
            sum(proposal.reliability.values()) / len(proposal.reliability)
            if proposal.reliability else 1.0
        )
        active = sum(1 for value in proposal.component_scores.values() if value > 0.0)
        if adjusted >= settings.negative_strong_score:
            reasons.append("NEGATIVE_STRONG_PROPOSAL")
        if adjusted >= settings.negative_moderate_score and active >= settings.negative_min_reliable_components:
            reasons.append("NEGATIVE_CROSS_BRANCH_SUPPORT")
        if _is_edge(proposal, settings.negative_edge_margin_ratio) and adjusted >= settings.negative_edge_score:
            reasons.append("NEGATIVE_EDGE_RESCUE")
    if sum(item.area_ratio for item in meaningful_proposals(proposals, settings)) >= settings.negative_large_total_area_ratio:
        reasons.append("NEGATIVE_LARGE_COHERENT_SUPPORT")
    return list(dict.fromkeys(reasons))


def _positive_conflict_reasons(result: AgentResult, proposals: list[ChangeProposal], task: str) -> list[str]:
    if task != "change_caption" or not is_positive_change_answer(result.answer):
        return []
    answer = result.answer.casefold()
    reasons: list[str] = []
    persistent = ("building", "structure", "road", "cleared", "vegetation", "wooded", "land", "basin", "shoreline", "infrastructure")
    transient = ("vehicle", "truck", "car", "equipment")
    water_state = ("water-filled", "filled", "dry", "wet", "turbidity", "reflection")
    appearance = ("greener", "brown", "brightness", "brighter", "darker", "shadow", "color", "seasonal")
    if any(token in answer for token in transient) and not any(token in answer for token in persistent):
        reasons.append("POSITIVE_TRANSIENT_ONLY")
    if any(token in answer for token in water_state) and not any(token in answer for token in ("shoreline", "boundary", "basin", "constructed", "removed", "expanded", "contracted")):
        reasons.append("POSITIVE_WATER_STATE_ONLY")
    if any(token in answer for token in appearance) and not any(token in answer for token in ("cleared", "removed", "replaced", "extent")):
        reasons.append("POSITIVE_APPEARANCE_ONLY")
    local = any(token in answer for token in ("building", "structure", "road"))
    temporal = {_temporal_side(item.image_id) for item in result.evidence_items if item.image_id}
    if local and result.evidence_items and temporal != {"t1", "t2"}:
        reasons.append("POSITIVE_MISSING_TEMPORAL_PAIR")
    if local and proposals and result.evidence_items and all(
        item.box is None or not any(_iou(item.box, proposal.box) > 0.05 for proposal in proposals)
        for item in result.evidence_items
    ):
        reasons.append("POSITIVE_LOCAL_CLAIM_OUTSIDE_ATTENTION")
    return reasons


def review_outcome(result: AgentResult, proposals: list[ChangeProposal], settings: ChangeReviewSettings, *, task: str) -> ChangeReviewOutcome:
    reviewed, warnings = review_result(result, proposals, settings)
    negative = _negative_conflict_reasons(proposals, settings) if is_canonical_no_change(result.answer) else []
    positive = _positive_conflict_reasons(result, proposals, task)
    reasons = negative or positive
    route: ReviewRoute = "adjudicate_negative" if negative else "adjudicate_positive" if positive else "accept"
    return ChangeReviewOutcome(reviewed, tuple(warnings), route, tuple(reasons))


def review_result(
    result: AgentResult,
    proposals: list[ChangeProposal],
    settings: ChangeReviewSettings,
) -> tuple[AgentResult, list[str]]:
    """Add warnings without reversing the model's semantic conclusion.
    添加告警但不反转模型语义结论。"""
    if not settings.enabled:
        return result, []
    warnings: list[str] = []
    answer = result.answer.casefold()
    no_change = is_no_change_answer(result.answer)
    if (
        settings.require_proposal_evidence
        and not no_change
        and not proposals
        and not result.evidence_items
    ):
        warnings.append("CHANGE_CLAIM_WITHOUT_PROPOSAL_EVIDENCE")
    if has_no_change_conflict(result, proposals, settings):
        warnings.append("CHANGE_RESULT_CONFLICT")
    proposal_boxes = [item.box for item in proposals]
    for evidence in result.evidence_items:
        if (
            evidence.box is not None
            and proposal_boxes
            and not any(_iou(evidence.box, box) > 0.05 for box in proposal_boxes)
        ):
            warnings.append("EVIDENCE_OUTSIDE_PROPOSALS")
            break
    appearance_only = any(
        token in answer
        for token in ("brighter", "darker", "color", "blur", "变亮", "变暗", "颜色", "清晰度")
    )
    if appearance_only:
        warnings.append("APPEARANCE_CHANGE_NOT_SEMANTIC_EVIDENCE")
    geometry = dict(result.geometry)
    referenced_ids = geometry.get("proposal_ids")
    if isinstance(referenced_ids, list):
        known_ids = {item.proposal_id for item in proposals}
        unknown_ids = [
            str(item) for item in referenced_ids if str(item) not in known_ids
        ]
        if unknown_ids:
            warnings.append("EVIDENCE_REFERENCES_UNKNOWN_PROPOSAL")
    for evidence in result.evidence_items:
        image_id = evidence.image_id
        if not isinstance(image_id, str):
            continue
        proposal_id = image_id.split(":", 1)[0]
        if ":" in image_id and proposal_id not in {item.proposal_id for item in proposals}:
            warnings.append("EVIDENCE_REFERENCES_UNKNOWN_PROPOSAL")
        if any(token in image_id.casefold() for token in ("invalid", "non_overlap", "non-overlap")):
            warnings.append("EVIDENCE_REFERENCES_INVALID_OVERLAP")
    temporal_sides = {
        side
        for evidence in result.evidence_items
        if isinstance(evidence.image_id, str)
        if (side := _temporal_side(evidence.image_id)) is not None
    }
    if (
        settings.require_temporal_pair_evidence
        and temporal_sides
        and temporal_sides != {"t1", "t2"}
    ):
        warnings.append("EVIDENCE_MISSING_TEMPORAL_PAIR")
    warnings = list(dict.fromkeys(warnings))
    geometry.update(
        {
            "change_review": {
                "warnings": warnings,
                "proposal_ids": [item.proposal_id for item in proposals],
            },
            "evidence_path_types": (
                ["raw_crop", "harmonized_crop"] if proposals else ["raw_full"]
            ),
        }
    )
    return (
        result.model_copy(
            update={"geometry": geometry, "status": "partial" if warnings else result.status}
        ),
        warnings,
    )


def _iou(first: list[int] | list[float], second: list[int] | list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return float(intersection / union) if union > 0 else 0.0


def _temporal_side(image_id: str) -> str | None:
    normalized = image_id.casefold()
    t1_tokens = ("raw_full_t1", "reference_t1", "harmonized_t1", "_t1_crop")
    t2_tokens = (
        "raw_full_t2",
        "registered_t2",
        "harmonized_t2",
        "t2_registered",
        "t2_raw_fallback",
        "_t2_crop",
    )
    if any(token in normalized for token in t1_tokens):
        return "t1"
    if any(token in normalized for token in t2_tokens):
        return "t2"
    return None
