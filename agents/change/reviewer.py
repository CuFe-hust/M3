"""Conservative rule review for model change claims.

模型变化结论的保守规则复核。纯规则、不调用模型；添加告警但不反转模型
语义结论。
"""

from __future__ import annotations

from agents.change.schema import ChangeProposal
from agents.change.settings import ChangeReviewSettings
from agents.schema import AgentResult


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
    answer = result.answer.lower()
    no_change = any(
        token in answer
        for token in ("no visible change", "no change", "未见变化", "没有变化")
    )
    if (
        settings.require_proposal_evidence
        and not no_change
        and not proposals
        and not result.evidence_items
    ):
        warnings.append("CHANGE_CLAIM_WITHOUT_PROPOSAL_EVIDENCE")
    if no_change and sum(item.score >= 0.5 for item in proposals) >= 2:
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
