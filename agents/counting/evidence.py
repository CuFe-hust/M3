"""Counting evidence — pure functions for box→point, dedup, count parsing.

计数证据 — box→点转换、去重、数量解析的纯函数。全部纯确定性；不做任何
数据集分支（标签比较使用通用归一化），不调用 Qwen/YOLO。
"""

from __future__ import annotations

import re

from agents.counting.schema import GlobalPointObservation
from agents.schema import VisualEvidence


def _normalize_label(label: str) -> str:
    """Normalize a label for comparison without dataset-specific semantics.
    归一化标签以进行比较，不引入任何数据集特定语义。"""
    return re.sub(r"[\s_]+", "-", label.strip().casefold())


# ── count answer parsing / 数量答案解析 ──────────────────────────────────


def parse_count_answer(value: str) -> int:
    """Parse a non-negative integer from a count answer.
    从计数答案解析非负整数。"""
    normalized = value.strip()
    if re.fullmatch(r"\d+", normalized) is None:
        raise ValueError(f"Count proposal is not a non-negative integer: {value!r}")
    return int(normalized)


def recover_count_proposal_header(raw_response: str) -> int | None:
    """Recover integer answer from malformed JSON.
    从畸形 JSON 恢复整数答案。"""
    match = re.search(r'"answer"\s*:\s*"(\d+)"', raw_response)
    return int(match.group(1)) if match is not None else None


# ── box evidence / box 证据 ──────────────────────────────────────────────


def box_evidence(
    boxes: list[list[float]],
    target: str,
    image_id: str,
) -> list[VisualEvidence]:
    """Normalize legacy proposal boxes. / 规范化旧版提议框。"""
    evidence: list[VisualEvidence] = []
    for raw_box in boxes:
        if len(raw_box) != 4 or any(not isinstance(v, (int, float)) for v in raw_box):
            continue
        x1, y1, x2, y2 = [max(0, min(999, round(v))) for v in raw_box]
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        evidence.append(VisualEvidence(label=target, box=box, image_id=image_id))
    return evidence


# ── border fragment detection / 边界残片检测 ─────────────────────────────


def is_tiny_border_fragment(box: list[int]) -> bool:
    """Identify border-clipped fragments. / 识别边界截断残片。"""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return (
        (box[0] == 0 and cx < 25)
        or (box[1] == 0 and cy < 25)
        or (box[2] == 999 and cx > 974)
        or (box[3] == 999 and cy > 974)
    )


# ── count evidence merge / 计数证据合并 ──────────────────────────────────


def _box_iou(first: list[int], second: list[int]) -> float:
    iw = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    ih = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    inter = iw * ih
    a1 = (first[2] - first[0]) * (first[3] - first[1])
    a2 = (second[2] - second[0]) * (second[3] - second[1])
    union = a1 + a2 - inter
    return inter / union if union else 0.0


def _point_distance(first: list[int], second: list[int]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _same_count_observation(
    first: VisualEvidence,
    second: VisualEvidence,
) -> bool:
    """Return whether two observations describe the same instance, using
    generic label normalization — no dataset-specific class mapping.
    返回两条观测是否描述同一实例；使用通用标签归一化——无数据集特定映射。"""
    if _normalize_label(first.label) != _normalize_label(second.label):
        return False
    if first.box is not None and second.box is not None:
        return _box_iou(first.box, second.box) >= 0.9
    if first.point is not None and second.point is not None:
        return _point_distance(first.point, second.point) <= 12
    box_item, point_item = (first, second) if first.box is not None else (second, first)
    if box_item.box is None or point_item.point is None:
        return False
    bc = [
        round((box_item.box[0] + box_item.box[2]) / 2),
        round((box_item.box[1] + box_item.box[3]) / 2),
    ]
    return _point_distance(bc, point_item.point) <= 12


def merge_count_evidence(items: list[VisualEvidence]) -> list[VisualEvidence]:
    """Deduplicate count evidence (strict — adjacent vehicles remain distinct).
    去重计数证据（严格 — 相邻目标保持独立）。"""
    merged: list[VisualEvidence] = []
    for candidate in items:
        dup_idx = next(
            (
                i
                for i, existing in enumerate(merged)
                if _same_count_observation(candidate, existing)
            ),
            None,
        )
        if dup_idx is None:
            merged.append(candidate)
        elif candidate.box is not None and merged[dup_idx].box is None:
            merged[dup_idx] = candidate
    return merged


def accepted_count_evidence(
    evidence: list[VisualEvidence],
    target: str,
    image_id: str,
) -> tuple[list[VisualEvidence], list[list[int]], int]:
    """Deduplicate, drop border fragments, emit accepted centres.
    去重、丢弃边界残片、输出接受中心。"""
    merged = merge_count_evidence(evidence)
    raw_points: list[VisualEvidence] = []
    boxes: list[list[int]] = []
    dropped = len(evidence) - len(merged)
    for item in merged:
        if item.box is not None:
            if is_tiny_border_fragment(item.box):
                dropped += 1
                continue
            boxes.append(list(item.box))
            point = [
                round((item.box[0] + item.box[2]) / 2),
                round((item.box[1] + item.box[3]) / 2),
            ]
        elif item.point is not None:
            point = list(item.point)
        else:
            dropped += 1
            continue
        raw_points.append(
            VisualEvidence(label=target, point=point, image_id=image_id)
        )
    points = merge_count_evidence(raw_points)
    dropped += len(raw_points) - len(points)
    return points, boxes, max(0, dropped)


def global_count_point(
    sample_id: str,
    target: str,
    evidence: VisualEvidence,
    index: int,
    width: int,
    height: int,
) -> GlobalPointObservation:
    """Convert normalized centre to whole-image provenance.
    将归一化中心转为整图来源记录。"""
    if evidence.point is None:
        raise ValueError("Accepted count evidence requires a point")
    x, y = evidence.point
    local_id = f"p{index:03d}"
    return GlobalPointObservation(
        global_id=f"{sample_id}:whole_image_overview:{local_id}",
        target=target,
        source_tile_id="whole_image_overview",
        local_id=local_id,
        local_x_norm=x,
        local_y_norm=y,
        local_radius_norm=0,
        global_x_px=round(x * (width - 1) / 999),
        global_y_px=round(y * (height - 1) / 999),
        global_x_norm=x,
        global_y_norm=y,
        radius_px=0.0,
        # Public visual evidence intentionally carries no confidence. This
        # compatibility path records the confidence as unknown.
        # 公共视觉证据有意不携带置信度；此兼容路径将置信度记录为未知。
        confidence=0.0,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=True,
        short_evidence="whole-image localized instance centre",
    )
