"""Dataset-neutral spatial geometry rules over labeled evidence.

数据集无关的带标签证据空间几何规则。只消费 SpatialQuerySpec 与 AgentResult
证据；仅在证据完整且规则可复现时确定性覆盖答案，否则保留视觉模型答案并
记录原因。本模块不读取问题文本、不做评测、不做任何数据集专用映射。
"""

from __future__ import annotations

import re
from typing import Any

from agents.schema import AgentResult, VisualEvidence
from agents.spatial.schema import SpatialQuerySpec


def apply_spatial_geometry(
    spec: SpatialQuerySpec,
    result: AgentResult,
) -> AgentResult:
    """Apply reproducible answer rules supported by labeled evidence geometry;
    the audit always records answer_source, rule, candidate_count, and whether
    the evidence was complete. 应用由带标签证据几何支持且可复现的答案规则；
    audit 始终记录 answer_source、rule、candidate_count 与证据是否完整。"""
    audit: dict[str, Any] = {
        "version": "evidence-geometry-v1",
        "coordinate_frame": "normalized_0_999_top_left",
        "operation": spec.operation,
        "answer_source": "qwen_visual_answer",
        "rule": "no_deterministic_override",
        "evidence_complete": False,
    }
    for key in (
        "input_normalizations",
        "evidence_quality",
        "repair_severity",
        "candidate_review_used",
        "candidate_review_added",
        "candidate_review_replaced",
        "candidate_review_labeled_boxes",
        "candidate_review_geometry",
        "candidate_review_error_type",
    ):
        if key in result.geometry:
            audit[key] = result.geometry[key]
    boxed = [item for item in result.evidence_items if item.box is not None]
    matched = _matching_evidence(spec, boxed)

    if spec.operation == "extreme_category":
        return _apply_extreme_category(spec, result, audit, matched)

    if spec.operation == "grid_position":
        return _apply_grid_position(spec, result, audit, matched)

    if spec.operation == "box_gap":
        return _apply_box_gap(result, audit, boxed)

    if spec.operation == "orientation_evidence":
        audit.update(
            {
                "rule": "cardinal_direction_requires_dataset_north_up_assumption",
                "north_metadata_available": False,
            }
        )
        return _finalize(result, audit)

    if spec.operation == "arrangement_evidence":
        audit.update(
            {
                "rule": "arrangement_requires_instance_set",
                "candidate_count": len(matched),
            }
        )
        return _finalize(result, audit)

    audit["rule"] = "unsupported_operation"
    return _finalize(result, audit)


def _apply_extreme_category(
    spec: SpatialQuerySpec,
    result: AgentResult,
    audit: dict[str, Any],
    matched: list[VisualEvidence],
) -> AgentResult:
    """Select the top-most/bottom-most matching box centre and override when
    the evidence is complete. 证据完整时选择最上/最下匹配框中心并覆盖。"""
    hint = (spec.target_hint or "").strip().casefold()
    extreme = "top" if hint in {"top", "topmost", "top-most"} else (
        "bottom" if hint in {"bottom", "bottommost", "bottom-most"} else None
    )
    audit["candidate_count"] = len(matched)
    if extreme is None:
        audit.update({"rule": "missing_extreme_hint", "evidence_complete": False})
        return _finalize(result, audit)
    if len(matched) < spec.min_candidates:
        audit.update(
            {
                "rule": "insufficient_extreme_candidates",
                "evidence_complete": False,
            }
        )
        return _finalize(result, audit)
    selected = (
        min(matched, key=_center_y)
        if extreme == "top"
        else max(matched, key=_center_y)
    )
    audit.update(
        {
            "answer_source": "deterministic_geometry",
            "rule": f"{extreme}_most_box_center_y",
            "selected_label": selected.label,
            "selected_box": selected.box,
            "selected_center_y": round(_center_y(selected), 3),
            "evidence_complete": True,
        }
    )
    return _finalize(result, audit, canonical_answer(selected.label))


def _apply_grid_position(
    spec: SpatialQuerySpec,
    result: AgentResult,
    audit: dict[str, Any],
    matched: list[VisualEvidence],
) -> AgentResult:
    """Place the target's box centre into a 3x3 grid when exactly one
    unambiguous candidate exists. 存在唯一无歧义候选时，将目标框中心放入
    3x3 网格。"""
    audit["candidate_count"] = len(matched)
    if len(matched) == 0:
        audit.update({"rule": "missing_position_target", "evidence_complete": False})
        return _finalize(result, audit)
    if len(matched) > 1:
        audit.update({"rule": "ambiguous_position_target", "evidence_complete": False})
        return _finalize(result, audit)
    target = matched[0]
    answer = _grid_position(target, spec.grid_boundaries)
    audit.update(
        {
            "answer_source": "deterministic_geometry",
            "rule": "three_by_three_box_center",
            "selected_label": target.label,
            "selected_box": target.box,
            "grid_boundaries": list(spec.grid_boundaries),
            "evidence_complete": True,
        }
    )
    return _finalize(result, audit, answer)


def _apply_box_gap(
    result: AgentResult,
    audit: dict[str, Any],
    boxed: list[VisualEvidence],
) -> AgentResult:
    """Record the nearest box gap without a threshold override.
    记录最近框间距，不做阈值覆盖。"""
    if len(boxed) >= 2:
        audit.update(
            {
                "rule": "box_gap_recorded_without_threshold_override",
                "nearest_box_gap": round(_box_gap(boxed[0], boxed[1]), 3),
            }
        )
    else:
        audit.update(
            {
                "rule": "insufficient_boxes_for_gap",
                "candidate_count": len(boxed),
            }
        )
    return _finalize(result, audit)


def _matching_evidence(
    spec: SpatialQuerySpec,
    boxed: list[VisualEvidence],
) -> list[VisualEvidence]:
    """Evidence whose normalized label equals the spec target label; without a
    target label, all boxed evidence matches. 归一化标签等于 spec 目标标签的
    证据；无目标标签时全部框证据匹配。"""
    if not spec.target_label:
        return list(boxed)
    wanted = _normalize_label(spec.target_label)
    return [item for item in boxed if _normalize_label(item.label) == wanted]


def _normalize_label(label: str) -> str:
    """Normalize a label for comparison without dataset-specific semantics.
    归一化标签以进行比较，不引入任何数据集特定语义。"""
    return re.sub(r"[\s_]+", "-", label.strip().casefold())


def canonical_answer(label: str) -> str:
    """Canonical answer for a selected label: the normalized label with any
    trailing plural 's' removed. 所选标签的规范答案：归一化标签并去除末尾
    复数 's'。"""
    normalized = _normalize_label(label)
    if normalized.endswith("s") and len(normalized) > 1:
        return normalized[:-1]
    return normalized


def _center_y(item: VisualEvidence) -> float:
    box = item.box
    if box is None:
        raise ValueError("box evidence required")
    return (box[1] + box[3]) / 2


def _grid_position(item: VisualEvidence, boundaries: tuple[int, int]) -> str:
    box = item.box
    if box is None:
        raise ValueError("box evidence required")
    lower, upper = boundaries
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    horizontal = "left" if center_x < lower else ("middle" if center_x < upper else "right")
    vertical = "top" if center_y < lower else ("middle" if center_y < upper else "bottom")
    return f"{vertical}-{horizontal}"


def _box_gap(first: VisualEvidence, second: VisualEvidence) -> float:
    first_box, second_box = first.box, second.box
    if first_box is None or second_box is None:
        raise ValueError("box evidence required")
    dx = max(first_box[0] - second_box[2], second_box[0] - first_box[2], 0)
    dy = max(first_box[1] - second_box[3], second_box[1] - first_box[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def _finalize(
    result: AgentResult,
    audit: dict[str, Any],
    answer: str | None = None,
) -> AgentResult:
    """Apply the audit and optional deterministic answer; incomplete evidence
    keeps the visual answer and downgrades status to partial.
    应用 audit 与可选确定性答案；证据不完整时保留视觉答案并将状态降级为
    partial。"""
    if answer is not None:
        final_answer = answer
    else:
        final_answer = result.answer
        audit["rule"] = audit.get("rule", "no_deterministic_override")
    audit["raw_answer"] = result.answer
    audit["final_answer"] = final_answer
    if audit.get("answer_source") == "deterministic_geometry" and audit.get("evidence_complete") is True:
        status = "completed"
    elif audit.get("evidence_complete") is False:
        status = "partial"
    else:
        status = result.status
    audit["workflow_status"] = status
    if final_answer != result.answer or audit != result.geometry:
        return result.model_copy(
            update={"answer": final_answer, "geometry": audit, "status": status}
        )
    return result
