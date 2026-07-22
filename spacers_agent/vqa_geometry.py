"""Deterministic VRSBench VQA routing profiles and evidence geometry.
VRSBench VQA 的确定性路由配置与证据几何计算。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from spacers_agent.schemas import CountTargetSpec, ExpertResult, VisualEvidence


ExecutionTask = Literal["counting", "spatial_relation", "general_vqa"]

VRSBENCH_TYPE_ROUTES: dict[str, ExecutionTask] = {
    "object quantity": "counting",
    "object existence": "spatial_relation",
    "object position": "spatial_relation",
    "object category": "spatial_relation",
    "object direction": "spatial_relation",
    "object color": "general_vqa",
}

_SMALL_VEHICLE_ALIASES = (
    "small vehicle",
    "small-vehicle",
    "car",
    "automobile",
    "passenger car",
    "motorcycle",
)
_LARGE_VEHICLE_ALIASES = (
    "large vehicle",
    "large-vehicle",
    "truck",
    "bus",
    "trailer",
    "semi-truck",
)


def execution_task_for_vrsbench(question_type: str) -> ExecutionTask:
    """Map an official VRSBench question type to a declared execution task.
    将 VRSBench 官方问题类型映射到已声明的执行任务。
    """

    normalized = " ".join(question_type.casefold().split())
    try:
        return VRSBENCH_TYPE_ROUTES[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported VRSBench VQA type: {question_type!r}") from error


def vrsbench_count_target(question: str) -> CountTargetSpec:
    """Return the audited VRSBench vehicle ontology without an LLM parse call.
    不调用大模型，返回经审计的 VRSBench 车辆计数类别体系。
    """

    lowered = " ".join(question.casefold().replace("-", " ").split())
    if "small vehicle" in lowered:
        return CountTargetSpec(
            canonical_label="small-vehicle",
            aliases=list(_SMALL_VEHICLE_ALIASES),
            inclusion_rule="Count each visible car, passenger car, or motorcycle as one small vehicle.",
            exclusion_rule="Exclude trucks, buses, trailers, and non-vehicle objects.",
        )
    if "large vehicle" in lowered:
        return CountTargetSpec(
            canonical_label="large-vehicle",
            aliases=list(_LARGE_VEHICLE_ALIASES),
            inclusion_rule="Count each visible truck, bus, trailer, or semi-truck as one large vehicle.",
            exclusion_rule="Exclude cars, motorcycles, and non-vehicle objects.",
        )
    return CountTargetSpec(
        canonical_label="vehicle",
        aliases=["vehicle", *_SMALL_VEHICLE_ALIASES, *_LARGE_VEHICLE_ALIASES],
        inclusion_rule="Count each visible small or large vehicle as one vehicle.",
        exclusion_rule="Exclude non-vehicle objects and do not count one vehicle more than once.",
    )


def apply_vrsbench_geometry(
    question: str,
    question_type: str,
    result: ExpertResult,
) -> ExpertResult:
    """Apply only reproducible answer rules supported by labeled evidence geometry.
    仅应用由带标签证据几何支持且可复现的答案规则。
    """

    audit: dict[str, Any] = {
        "version": "vrsbench-evidence-geometry-v2",
        "coordinate_frame": "normalized_0_999_top_left",
        "question_type": question_type,
        "answer_source": "qwen_visual_answer",
        "rule": "no_deterministic_override",
    }
    for key in (
        "input_normalizations",
        "candidate_review_used",
        "candidate_review_added",
        "candidate_review_error",
    ):
        if key in result.geometry:
            audit[key] = result.geometry[key]
    boxed = [item for item in result.evidence_items if item.box is not None]
    lowered = question.casefold()

    extreme = "top" if re.search(r"\btop[ -]?most\b", lowered) else (
        "bottom" if re.search(r"\bbottom[ -]?most\b", lowered) else None
    )
    vehicle_boxes = [item for item in boxed if vrsbench_vehicle_class(item.label) in {"small-vehicle", "large-vehicle"}]
    if question_type.casefold() == "object category" and extreme and len(vehicle_boxes) >= 2:
        selected = min(vehicle_boxes, key=_center_y) if extreme == "top" else max(vehicle_boxes, key=_center_y)
        canonical = vrsbench_vehicle_class(selected.label)
        audit.update(
            {
                "answer_source": "deterministic_geometry",
                "rule": f"{extreme}_most_box_center_y",
                "selected_label": selected.label,
                "selected_box": selected.box,
                "selected_center_y": _center_y(selected),
                "candidate_count": len(vehicle_boxes),
                "evidence_complete": True,
            }
        )
        return _finalize_vrsbench_answer(result, question_type, audit, canonical)

    if question_type.casefold() == "object category" and extreme:
        audit.update(
            {
                "rule": "insufficient_extreme_candidates",
                "candidate_count": len(vehicle_boxes),
                "evidence_complete": False,
            }
        )
        return _finalize_vrsbench_answer(result, question_type, audit, result.answer)

    if question_type.casefold() == "object position" and boxed and _asks_grid_position(lowered):
        target = _select_position_target(lowered, boxed)
        if target is not None:
            answer = _grid_position(target)
            audit.update(
                {
                    "answer_source": "deterministic_geometry",
                    "rule": "three_by_three_box_center",
                    "selected_label": target.label,
                    "selected_box": target.box,
                    "grid_boundaries": [333, 666],
                }
            )
            return _finalize_vrsbench_answer(result, question_type, audit, answer)

    if question_type.casefold() == "object direction" or "orientation" in lowered:
        audit.update(
            {
                "rule": "cardinal_direction_requires_dataset_north_up_assumption",
                "north_metadata_available": False,
            }
        )
    elif "near" in lowered and len(boxed) >= 2:
        first, second = boxed[0], boxed[1]
        audit.update(
            {
                "rule": "box_gap_recorded_without_threshold_override",
                "nearest_box_gap": round(_box_gap(first, second), 3),
            }
        )
    return _finalize_vrsbench_answer(result, question_type, audit, result.answer)


def _center_y(item: VisualEvidence) -> float:
    box = item.box
    if box is None:
        raise ValueError("box evidence required")
    return (box[1] + box[3]) / 2


def _grid_position(item: VisualEvidence) -> str:
    box = item.box
    if box is None:
        raise ValueError("box evidence required")
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    horizontal = "left" if center_x < 333 else ("middle" if center_x < 666 else "right")
    vertical = "top" if center_y < 333 else ("middle" if center_y < 666 else "bottom")
    return f"{vertical}-{horizontal}"


def _asks_grid_position(question: str) -> bool:
    return "what is the position of" in question or question.startswith("where is ")


def _select_position_target(question: str, boxed: list[VisualEvidence]) -> VisualEvidence | None:
    desired = "large-vehicle" if "large vehicle" in question else (
        "small-vehicle" if "small vehicle" in question else None
    )
    candidates = [item for item in boxed if desired is None or vrsbench_vehicle_class(item.label) == desired]
    return candidates[0] if len(candidates) == 1 else None


def normalize_vrsbench_answer(question_type: str, answer: str) -> str:
    """Normalize declared VRSBench answer vocabularies without using references.
    不读取参考答案，仅规范化已声明的 VRSBench 答案词表。
    """

    normalized_type = " ".join(question_type.casefold().split())
    lowered = " ".join(answer.casefold().replace("_", " ").split())
    if normalized_type == "object existence":
        if re.match(r"^(yes|yeah|true)\b", lowered):
            return "yes"
        if re.match(r"^(no|false)\b", lowered):
            return "no"
    if normalized_type == "object category":
        category = vrsbench_vehicle_class(answer)
        if category in {"small-vehicle", "large-vehicle"}:
            return category
    if normalized_type == "object position":
        compact = re.sub(r"[^a-z]+", "-", lowered).strip("-")
        for vertical in ("top", "middle", "bottom"):
            for horizontal in ("left", "middle", "right"):
                candidate = f"{vertical}-{horizontal}"
                if candidate in compact:
                    return candidate
    if normalized_type == "object color":
        colors = [
            color
            for color in ("black", "blue", "brown", "gray", "green", "orange", "red", "white", "yellow")
            if re.search(rf"\b{color}\b", lowered)
        ]
        if len(colors) == 1:
            return colors[0]
    if normalized_type == "object direction":
        if re.search(r"\bnorth[ -]?south\b", lowered):
            return "north-south"
        if re.search(r"\beast[ -]?west\b", lowered):
            return "east-west"
    return answer.strip()


def vrsbench_vehicle_class(label: str) -> str:
    """Map common vehicle names to the closed VRSBench vehicle classes.
    将常见车辆名称映射到封闭的 VRSBench 车辆类别。
    """

    normalized = re.sub(r"[_\s]+", "-", label.strip().casefold())
    if normalized in {value.replace(" ", "-") for value in _SMALL_VEHICLE_ALIASES}:
        return "small-vehicle"
    if normalized in {value.replace(" ", "-") for value in _LARGE_VEHICLE_ALIASES}:
        return "large-vehicle"
    for alias in sorted(_SMALL_VEHICLE_ALIASES, key=len, reverse=True):
        pattern = re.escape(alias).replace(r"\ ", r"[\s-]+")
        if re.search(rf"\b{pattern}\b", label.casefold()):
            return "small-vehicle"
    for alias in sorted(_LARGE_VEHICLE_ALIASES, key=len, reverse=True):
        pattern = re.escape(alias).replace(r"\ ", r"[\s-]+")
        if re.search(rf"\b{pattern}\b", label.casefold()):
            return "large-vehicle"
    return normalized


def _finalize_vrsbench_answer(
    result: ExpertResult,
    question_type: str,
    audit: dict[str, Any],
    answer: str,
) -> ExpertResult:
    raw_answer = result.answer
    normalized = normalize_vrsbench_answer(question_type, answer)
    audit.update(
        {
            "raw_answer": raw_answer,
            "normalized_answer": normalized,
            "answer_normalization_version": "vrsbench-answer-v1",
        }
    )
    return result.model_copy(update={"answer": normalized, "geometry": audit})


def _box_gap(first: VisualEvidence, second: VisualEvidence) -> float:
    first_box, second_box = first.box, second.box
    if first_box is None or second_box is None:
        raise ValueError("box evidence required")
    dx = max(first_box[0] - second_box[2], second_box[0] - first_box[2], 0)
    dy = max(first_box[1] - second_box[3], second_box[1] - first_box[3], 0)
    return (dx * dx + dy * dy) ** 0.5
