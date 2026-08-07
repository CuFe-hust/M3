"""Pure spatial candidate-review and evidence-merging rules.

空间候选复核与证据合并的纯规则。全部函数数据集无关：不读取问题文本、
不检查数据来源、不做数据集专用类别映射；目标标签由调用方从结构化
spec 提供。
"""

from __future__ import annotations

import re

from agents.schema import AgentResult, VisualEvidence


def needs_candidate_review(
    result: AgentResult,
    *,
    operation: str | None,
    target_label: str | None,
) -> bool:
    """Return whether spatial evidence still needs one review pass, based only
    on the structured operation and the current evidence. 仅基于结构化操作与
    当前证据判断空间证据是否仍需一次复核。"""
    if operation not in {"extreme_category", "grid_position", "arrangement"}:
        return False
    boxed = [item for item in result.evidence_items if item.box is not None]
    if operation == "grid_position":
        targets = [item for item in boxed if matches_target_label(item, target_label)]
        if result.geometry.get("candidate_review_used"):
            return len(targets) != 1
        if not targets:
            return True
        return len(targets) != 1 or is_corner_anchored_box(targets[0])
    if not result.geometry.get("candidate_review_used"):
        if operation == "extreme_category" and extreme_evidence_is_sufficient(
            result,
            direction=str(result.geometry.get("extreme_direction") or ""),
            target_label=target_label,
        ):
            return False
        return True
    vehicles = [item for item in boxed if canonical_label_kind(item.label) is not None]
    return len(vehicles) < 2


def extreme_evidence_is_sufficient(
    result: AgentResult,
    *,
    direction: str,
    target_label: str | None,
    edge_margin: int = 40,
) -> bool:
    """Conservatively prove that a first-pass extreme evidence set is
    image-edge complete. 保守证明首轮极值证据集合已覆盖到相应图像边缘。"""
    if result.status != "completed" or str(result.geometry.get("repair_severity", "none")) == "high":
        return False
    direction_key = _extreme_direction(direction)
    if direction_key is None:
        return False
    vehicles = [
        item
        for item in result.evidence_items
        if item.box is not None and matches_target_label(item, target_label)
    ]
    if len(vehicles) < 2:
        return False
    axis = 0 if direction_key in {"left", "right"} else 1
    centers = [((item.box[axis] + item.box[axis + 2]) / 2, item) for item in vehicles]
    extreme_center, extreme_item = (
        min(centers, key=lambda value: value[0])
        if direction_key in {"left", "top"}
        else max(centers, key=lambda value: value[0])
    )
    touches_extreme_band = (
        extreme_center <= edge_margin
        if direction_key in {"left", "top"}
        else extreme_center >= 999 - edge_margin
    )
    return touches_extreme_band and canonical_answer(result.answer) == canonical_answer(
        extreme_item.label
    )


def _extreme_direction(direction: str) -> str | None:
    lowered = direction.strip().casefold()
    for candidate in ("top", "bottom", "left", "right"):
        if candidate in lowered or f"{candidate}most" in lowered.replace("-", ""):
            return candidate
    return None


def canonical_answer(label: str) -> str:
    """Canonical normalized answer label. / 规范化答案标签。"""
    normalized = re.sub(r"[\s_]+", "-", label.strip().casefold())
    if normalized.endswith("s") and len(normalized) > 1:
        return normalized[:-1]
    return normalized


def matches_target_label(item: VisualEvidence, target_label: str | None) -> bool:
    """Match evidence to the canonical target label; without a target label all
    evidence matches. 将证据与规范目标标签匹配；无目标标签时全部证据匹配。"""
    if not target_label:
        return True
    return canonical_answer(item.label) == canonical_answer(target_label)


def position_review_evidence(
    review: AgentResult,
    *,
    is_grid: bool,
    target_label: str | None,
) -> tuple[list[VisualEvidence], int]:
    """Recover labeled grid-position evidence from top-level review boxes.
    从复核结果的顶层框恢复带标签的九宫格位置证据。"""
    evidence = list(review.evidence_items)
    if not is_grid or any(item.box is not None for item in evidence):
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
    返回答案是否仅包含工作流状态词。"""
    token = re.sub(r"[^a-z]+", "", answer.casefold())
    return token in {"completed", "failed", "partial"}


def is_corner_anchored_box(item: VisualEvidence, *, tolerance: int = 5) -> bool:
    """Flag boxes anchored to two image borders as answer-region placeholders.
    将同时贴住两条图像边界的框标记为答案区域占位框。"""
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
    合并两轮证据并去除高度重叠的重复项。"""
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
    返回归一化证据点之间的欧氏距离。"""
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def maximum_repair_severity(first: str, second: str) -> str:
    """Retain the highest evidence-repair severity across independent passes.
    在独立复核轮次之间保留最高的证据修复严重度。"""
    rank = {"none": 0, "low": 1, "high": 2}
    return max((first, second), key=lambda value: rank.get(value, 2))


def box_iou(first: list[int], second: list[int]) -> float:
    """Return intersection over union for normalized axis-aligned boxes.
    返回归一化轴对齐框的交并比。"""
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def box_intersection_over_smaller(first: list[int], second: list[int]) -> float:
    """Return intersection area divided by the smaller box area.
    返回交集面积与较小框面积之比。"""
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area else 0.0


def normalized_box_center_distance(first: list[int], second: list[int]) -> float:
    """Return center distance normalized by the smaller box diagonal.
    返回以较小框对角线归一化的中心距离。"""
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


def canonical_label_kind(label: str) -> str | None:
    """Return a canonical class or a generic positional vehicle role, without
    any dataset-specific alias table. 返回标准类别或泛化的位置车辆角色，不含
    任何数据集专用别名表。"""
    canonical = canonical_answer(label)
    if canonical in {"small-vehicle", "large-vehicle"}:
        return canonical
    normalized_role = re.sub(r"[_-]+", " ", label.casefold())
    if re.search(r"\bvehicles?\b", normalized_role):
        return "vehicle"
    return None


def compatible_evidence_labels(first: str, second: str) -> bool:
    """Allow positional vehicle roles to match an explicit vehicle class.
    允许位置车辆角色与明确的车辆类别匹配。"""
    first_vehicle = canonical_label_kind(first)
    second_vehicle = canonical_label_kind(second)
    if first_vehicle is not None and second_vehicle is not None:
        return (
            first_vehicle == second_vehicle
            or first_vehicle == "vehicle"
            or second_vehicle == "vehicle"
        )
    return canonical_answer(first) == canonical_answer(second)


def same_box_observation(first: list[int], second: list[int]) -> bool:
    """Match shifted small-object boxes without merging adjacent instances.
    匹配发生偏移的小目标框，同时避免合并相邻实例。"""
    if box_iou(first, second) >= 0.7:
        return True
    return (
        box_intersection_over_smaller(first, second) >= 0.45
        and normalized_box_center_distance(first, second) <= 0.40
    )


def same_visual_observation(first: VisualEvidence, second: VisualEvidence) -> bool:
    """Return whether two evidence items describe the same observation.
    返回两条证据是否描述同一观测。"""
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
    对重复观测优先保留框，其次保留置信度更高者。"""
    if candidate.box is not None and existing.box is None:
        return True
    if candidate.box is None and existing.box is not None:
        return False
    candidate_vehicle = canonical_label_kind(candidate.label)
    existing_vehicle = canonical_label_kind(existing.label)
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
