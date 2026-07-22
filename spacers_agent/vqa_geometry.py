"""Deterministic VRSBench VQA routing profiles and evidence geometry.
VRSBench VQA 的确定性路由配置与证据几何计算。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from spacers_agent.schemas import ExpertResult, VisualEvidence


ExecutionTask = Literal["counting", "spatial_relation", "general_vqa"]

VRSBENCH_TYPE_ROUTES: dict[str, ExecutionTask] = {
    "object quantity": "counting",
    "object existence": "spatial_relation",
    "object position": "spatial_relation",
    "object category": "spatial_relation",
    "object direction": "spatial_relation",
    "object color": "general_vqa",
}


def execution_task_for_vrsbench(question_type: str) -> ExecutionTask:
    """Map an official VRSBench question type to a declared execution task.
    将 VRSBench 官方问题类型映射到已声明的执行任务。
    """

    normalized = " ".join(question_type.casefold().split())
    try:
        return VRSBENCH_TYPE_ROUTES[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported VRSBench VQA type: {question_type!r}") from error


def apply_vrsbench_geometry(
    question: str,
    question_type: str,
    result: ExpertResult,
) -> ExpertResult:
    """Apply only reproducible answer rules supported by labeled evidence geometry.
    仅应用由带标签证据几何支持且可复现的答案规则。
    """

    audit: dict[str, Any] = {
        "version": "vrsbench-evidence-geometry-v1",
        "coordinate_frame": "normalized_0_999_top_left",
        "question_type": question_type,
        "answer_source": "qwen_visual_answer",
        "rule": "no_deterministic_override",
    }
    boxed = [item for item in result.evidence_items if item.box is not None]
    lowered = question.casefold()

    extreme = "top" if re.search(r"\btop[ -]?most\b", lowered) else (
        "bottom" if re.search(r"\bbottom[ -]?most\b", lowered) else None
    )
    vehicle_boxes = [item for item in boxed if _vrsbench_vehicle_class(item.label) in {"small-vehicle", "large-vehicle"}]
    if question_type.casefold() == "object category" and extreme and vehicle_boxes:
        selected = min(vehicle_boxes, key=_center_y) if extreme == "top" else max(vehicle_boxes, key=_center_y)
        canonical = _vrsbench_vehicle_class(selected.label)
        audit.update(
            {
                "answer_source": "deterministic_geometry",
                "rule": f"{extreme}_most_box_center_y",
                "selected_label": selected.label,
                "selected_box": selected.box,
                "selected_center_y": _center_y(selected),
                "candidate_count": len(vehicle_boxes),
            }
        )
        return result.model_copy(update={"answer": canonical, "geometry": audit})

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
            return result.model_copy(update={"answer": answer, "geometry": audit})

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
    return result.model_copy(update={"geometry": audit})


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
    candidates = [item for item in boxed if desired is None or _vrsbench_vehicle_class(item.label) == desired]
    return candidates[0] if len(candidates) == 1 else None


def _vrsbench_vehicle_class(label: str) -> str:
    normalized = re.sub(r"[_\s]+", "-", label.strip().casefold())
    if normalized in {"car", "motorcycle", "small-vehicle", "smallvehicle"}:
        return "small-vehicle"
    if normalized in {"truck", "bus", "large-vehicle", "largevehicle", "trailer", "semi-truck"}:
        return "large-vehicle"
    return normalized


def _box_gap(first: VisualEvidence, second: VisualEvidence) -> float:
    first_box, second_box = first.box, second.box
    if first_box is None or second_box is None:
        raise ValueError("box evidence required")
    dx = max(first_box[0] - second_box[2], second_box[0] - first_box[2], 0)
    dy = max(first_box[1] - second_box[3], second_box[1] - first_box[3], 0)
    return (dx * dx + dy * dy) ** 0.5
