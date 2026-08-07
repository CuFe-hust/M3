"""Deterministic, explainable difference-map proposals.

确定且可解释的差异图候选。RGB/edge/structure 加权差异图 + 连通域过滤；
候选数量、面积、score 与 box 坐标全部稳定可复现。不调用模型、不修改源
数组。
"""

from __future__ import annotations

import cv2
import numpy as np

from agents.change.schema import ChangeProposal
from agents.change.settings import ChangeProposalSettings


def propose_changes(
    t1: np.ndarray,
    t2: np.ndarray,
    settings: ChangeProposalSettings,
) -> tuple[np.ndarray, list[ChangeProposal]]:
    """Return normalized scores and connected-component proposals.
    返回归一化得分与连通域候选。"""

    first, second = t1.astype(np.float32), t2.astype(np.float32)
    rgb = np.mean(np.abs(first - second), axis=2) / 255.0
    gray1 = cv2.cvtColor(t1, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(t2, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edge = np.abs(
        cv2.Laplacian(gray1, cv2.CV_32F) - cv2.Laplacian(gray2, cv2.CV_32F)
    )
    structure = np.abs(
        cv2.GaussianBlur(gray1, (11, 11), 0) - cv2.GaussianBlur(gray2, (11, 11), 0)
    )
    weights = settings.rgb_weight + settings.edge_weight + settings.structure_weight
    score = (
        settings.rgb_weight * rgb
        + settings.edge_weight * np.clip(edge, 0, 1)
        + settings.structure_weight * structure
    ) / weights
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    if float(score.max()) <= 1e-8:
        return score, []
    threshold = float(np.quantile(score, settings.threshold_quantile))
    if threshold <= 1e-8:
        return score, []
    binary = (score >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = score.shape
    image_area = float(width * height)
    candidates: list[tuple[float, int, int, int, int, float]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        area_ratio = area / image_area
        if not settings.min_component_area_ratio <= area_ratio <= settings.max_component_area_ratio:
            continue
        component_score = float(np.mean(score[y : y + h, x : x + w]))
        candidates.append((component_score, x, y, w, h, area_ratio))
    candidates.sort(key=lambda item: (item[0], item[5]), reverse=True)
    proposals: list[ChangeProposal] = []
    for index, (component_score, x, y, w, h, area_ratio) in enumerate(
        candidates[: settings.max_proposals]
    ):
        pixel_box = [x, y, x + w, y + h]
        box = [
            round(x * 999 / width),
            round(y * 999 / height),
            round((x + w) * 999 / width),
            round((y + h) * 999 / height),
        ]
        box[2], box[3] = (
            min(999, max(box[0] + 1, box[2])),
            min(999, max(box[1] + 1, box[3])),
        )
        proposals.append(
            ChangeProposal(
                proposal_id=f"change_{index:03d}",
                box=box,
                pixel_box=pixel_box,
                score=component_score,
                area_ratio=area_ratio,
            )
        )
    return score, proposals


def render_overlay(image: np.ndarray, proposals: list[ChangeProposal]) -> np.ndarray:
    """Render proposal boxes without altering the source array.
    在不修改源数组的前提下绘制候选框。"""

    overlay = image.copy()
    for proposal in proposals:
        x1, y1, x2, y2 = proposal.pixel_box
        cv2.rectangle(
            overlay,
            (x1, y1),
            (max(x1, x2 - 1), max(y1, y2 - 1)),
            (255, 40, 40),
            2,
        )
        cv2.putText(
            overlay,
            proposal.proposal_id,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 40, 40),
            1,
        )
    return overlay
