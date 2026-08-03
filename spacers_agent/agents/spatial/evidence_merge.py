"""Pure spatial candidate-review and evidence-merging rules.
空间候选复核与证据合并的纯规则。
"""

from __future__ import annotations

import re

from spacers_agent.schemas import ExpertResult, UnifiedSample, VisualEvidence
from spacers_agent.vqa_geometry import (
    vrsbench_question_subtype,
    vrsbench_vehicle_class,
)


def needs_candidate_review(sample: UnifiedSample, result: ExpertResult) -> bool:
    """Return whether VRSBench spatial evidence still needs one review pass.
    返回 VRSBench 空间证据是否仍需一次复核。
    """

    if sample.dataset != "VRSBench":
        return False
    subtype = vrsbench_question_subtype(
        sample.question,
        str(sample.metadata.get("question_type", "")),
    )
    if subtype not in {"extreme_category", "grid_position", "arrangement"}:
        return False
    boxed = [item for item in result.evidence_items if item.box is not None]
    vehicles = [
        item
        for item in boxed
        if vrsbench_vehicle_class(item.label) in {"small-vehicle", "large-vehicle"}
    ]
    if subtype == "grid_position":
        targets = [item for item in boxed if matches_position_target(sample.question, item)]
        if result.geometry.get("candidate_review_used"):
            return len(targets) != 1
        return len(targets) != 1 or is_corner_anchored_box(targets[0])
    if not result.geometry.get("candidate_review_used"):
        if subtype == "extreme_category" and extreme_vehicle_evidence_is_sufficient(
            sample.question, result
        ):
            return False
        return True
    return len(vehicles) < 2


def extreme_vehicle_evidence_is_sufficient(
    question: str,
    result: ExpertResult,
    *,
    edge_margin: int = 40,
) -> bool:
    """Conservatively prove that a first-pass extreme vehicle is image-edge complete.
    保守证明首轮极值车辆证据已覆盖到相应图像边缘。
    """

    if result.status != "completed" or str(result.geometry.get("repair_severity", "none")) == "high":
        return False
    direction = _extreme_direction(question)
    if direction is None:
        return False
    vehicles = [
        item
        for item in result.evidence_items
        if item.box is not None
        and vrsbench_vehicle_class(item.label) in {"small-vehicle", "large-vehicle"}
    ]
    if len(vehicles) < 2:
        return False
    axis = 0 if direction in {"left", "right"} else 1
    centers = [((item.box[axis] + item.box[axis + 2]) / 2, item) for item in vehicles]
    extreme_center, extreme_item = (
        min(centers, key=lambda value: value[0])
        if direction in {"left", "top"}
        else max(centers, key=lambda value: value[0])
    )
    touches_extreme_band = (
        extreme_center <= edge_margin
        if direction in {"left", "top"}
        else extreme_center >= 999 - edge_margin
    )
    return touches_extreme_band and (
        vrsbench_vehicle_class(result.answer)
        == vrsbench_vehicle_class(extreme_item.label)
    )


def _extreme_direction(question: str) -> str | None:
    lowered = question.casefold()
    for direction in ("top", "bottom", "left", "right"):
        if re.search(rf"\b{direction}[\s-]*most\b", lowered):
            return direction
    return None


def matches_position_target(question: str, item: VisualEvidence) -> bool:
    """Match evidence to the vehicle class named by a position question.
    将证据与位置问题指定的车辆类别进行匹配。
    """

    desired = position_target_label(question)
    return desired is None or vrsbench_vehicle_class(item.label) == desired


def position_target_label(question: str) -> str | None:
    """Return the explicit vehicle class named by a position question.
    返回位置问题中明确指定的车辆类别。
    """

    lowered = question.casefold()
    if "large vehicle" in lowered:
        return "large-vehicle"
    if "small vehicle" in lowered:
        return "small-vehicle"
    return None


def position_review_evidence(
    question: str,
    subtype: str,
    review: ExpertResult,
) -> tuple[list[VisualEvidence], int]:
    """Recover labeled grid-position evidence from top-level review boxes.
    从复核结果的顶层框恢复带标签的九宫格位置证据。
    """

    evidence = list(review.evidence_items)
    target_label = position_target_label(question)
    if subtype != "grid_position" or any(item.box is not None for item in evidence):
        return evidence, 0
    review_label = target_label or "position-target"
    labeled = [
        VisualEvidence(
            label=review_label,
            box=[int(round(value)) for value in box],
            confidence=0.0,
        )
        for box in review.boxes
    ]
    return evidence + labeled, len(labeled)


def is_status_answer_placeholder(answer: str) -> bool:
    """Return whether an answer contains only a workflow status token.
    返回答案是否仅包含工作流状态词。
    """

    token = re.sub(r"[^a-z]+", "", answer.casefold())
    return token in {"completed", "failed", "partial"}


def is_corner_anchored_box(item: VisualEvidence, *, tolerance: int = 5) -> bool:
    """Flag boxes anchored to two image borders as answer-region placeholders.
    将同时贴住两条图像边界的框标记为答案区域占位框。
    """

    if item.box is None:
        return False
    left, top, right, bottom = item.box
    touches_horizontal_border = left <= tolerance or right >= 999 - tolerance
    touches_vertical_border = top <= tolerance or bottom >= 999 - tolerance
    return touches_horizontal_border and touches_vertical_border


def merge_visual_evidence(
    first: list[VisualEvidence],
    second: list[VisualEvidence],
) -> list[VisualEvidence]:
    """Merge evidence passes while suppressing strongly overlapping duplicates.
    合并两轮证据并去除高度重叠的重复项。
    """

    merged = list(first)
    for candidate in second:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if same_visual_observation(candidate, existing)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
        elif prefer_candidate_evidence(candidate, merged[duplicate_index]):
            merged[duplicate_index] = candidate
    return merged


def point_distance(first: list[int], second: list[int]) -> float:
    """Return Euclidean distance between normalized evidence points.
    返回归一化证据点之间的欧氏距离。
    """

    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def maximum_repair_severity(first: str, second: str) -> str:
    """Retain the highest evidence-repair severity across independent passes.
    在独立复核轮次之间保留最高的证据修复严重度。
    """

    rank = {"none": 0, "low": 1, "high": 2}
    return max((first, second), key=lambda value: rank.get(value, 2))


def box_iou(first: list[int], second: list[int]) -> float:
    """Return intersection over union for normalized axis-aligned boxes.
    返回归一化轴对齐框的交并比。
    """

    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def box_intersection_over_smaller(first: list[int], second: list[int]) -> float:
    """Return intersection area divided by the smaller box area.
    返回交集面积与较小框面积之比。
    """

    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area else 0.0


def normalized_box_center_distance(first: list[int], second: list[int]) -> float:
    """Return center distance normalized by the smaller box diagonal.
    返回以较小框对角线归一化的中心距离。
    """

    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    center_distance = (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5
    first_diagonal = ((first[2] - first[0]) ** 2 + (first[3] - first[1]) ** 2) ** 0.5
    second_diagonal = ((second[2] - second[0]) ** 2 + (second[3] - second[1]) ** 2) ** 0.5
    smaller_diagonal = min(first_diagonal, second_diagonal)
    return center_distance / smaller_diagonal if smaller_diagonal else float("inf")


def vehicle_label_kind(label: str) -> str | None:
    """Return a canonical class or a generic positional vehicle role.
    返回标准车辆类别或泛化的位置车辆角色。
    """

    canonical = vrsbench_vehicle_class(label)
    if canonical in {"small-vehicle", "large-vehicle"}:
        return canonical
    normalized_role = re.sub(r"[_-]+", " ", label.casefold())
    if re.search(r"\bvehicles?\b", normalized_role):
        return "vehicle"
    return None


def compatible_evidence_labels(first: str, second: str) -> bool:
    """Allow positional vehicle roles to match an explicit vehicle class.
    允许位置车辆角色与明确的车辆类别匹配。
    """

    first_vehicle = vehicle_label_kind(first)
    second_vehicle = vehicle_label_kind(second)
    if first_vehicle is not None and second_vehicle is not None:
        return (
            first_vehicle == second_vehicle
            or first_vehicle == "vehicle"
            or second_vehicle == "vehicle"
        )
    return vrsbench_vehicle_class(first) == vrsbench_vehicle_class(second)


def same_box_observation(first: list[int], second: list[int]) -> bool:
    """Match shifted small-object boxes without merging adjacent instances.
    匹配发生偏移的小目标框，同时避免合并相邻实例。
    """

    if box_iou(first, second) >= 0.7:
        return True
    return (
        box_intersection_over_smaller(first, second) >= 0.45
        and normalized_box_center_distance(first, second) <= 0.40
    )


def same_visual_observation(first: VisualEvidence, second: VisualEvidence) -> bool:
    """Return whether two evidence items describe the same observation.
    返回两条证据是否描述同一观测。
    """

    if not compatible_evidence_labels(first.label, second.label):
        return False
    if first.box is not None and second.box is not None:
        return same_box_observation(first.box, second.box)
    if first.point is not None and second.point is not None:
        return point_distance(first.point, second.point) <= 12
    box_item, point_item = (first, second) if first.box is not None else (second, first)
    if box_item.box is None or point_item.point is None:
        return False
    x, y = point_item.point
    return box_item.box[0] <= x <= box_item.box[2] and box_item.box[1] <= y <= box_item.box[3]


def prefer_candidate_evidence(candidate: VisualEvidence, existing: VisualEvidence) -> bool:
    """Prefer a box, then the higher-confidence duplicate observation.
    对重复观测优先保留框，其次保留置信度更高者。
    """

    if candidate.box is not None and existing.box is None:
        return True
    if candidate.box is None and existing.box is not None:
        return False
    candidate_vehicle = vehicle_label_kind(candidate.label)
    existing_vehicle = vehicle_label_kind(existing.label)
    if candidate_vehicle in {"small-vehicle", "large-vehicle"} and existing_vehicle == "vehicle":
        return True
    if candidate_vehicle == "vehicle" and existing_vehicle in {"small-vehicle", "large-vehicle"}:
        return False
    if candidate.box is not None and existing.box is not None:
        candidate_area = (candidate.box[2] - candidate.box[0]) * (candidate.box[3] - candidate.box[1])
        existing_area = (existing.box[2] - existing.box[0]) * (existing.box[3] - existing.box[1])
        if candidate_area <= existing_area * 0.85:
            return True
        if existing_area <= candidate_area * 0.85:
            return False
    return candidate.confidence > existing.confidence
