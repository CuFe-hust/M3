"""Pure geometry helpers for canonical materialized visual views.

规范物化视觉视图的纯几何 helper。本模块不依赖 PIL；最终 Agent 使用的裁切框
已经由 VisualTaskPlanner 以源图像素坐标确定，这里只保留预览尺寸与局部框变换。
"""

from __future__ import annotations

from agents.general_vqa.evidence.schema import RoiEvidenceRecord

MAX_MODEL_SIDE = 1080


def compute_preview_size(
    size: tuple[int, int],
    *,
    max_side: int = MAX_MODEL_SIDE,
) -> tuple[int, int]:
    """Shrink to max_side when needed, never upscale a small image.
    需要时缩到 max_side，绝不放大小图。"""

    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {size!r}")
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def local_to_global(
    local_xyxy: tuple[float, float, float, float],
    view: RoiEvidenceRecord,
) -> tuple[float, float, float, float]:
    """Translate crop-local pixels by the materialized crop origin.
    将裁切局部像素坐标按物化裁切原点转换到整图像素坐标。"""

    x1, y1, x2, y2 = local_xyxy
    left, top = view.expanded_xyxy[0], view.expanded_xyxy[1]
    return x1 + left, y1 + top, x2 + left, y2 + top


def global_to_local(
    global_xyxy: tuple[float, float, float, float],
    view: RoiEvidenceRecord,
) -> tuple[float, float, float, float]:
    """Translate whole-image pixels into the materialized crop frame.
    将整图像素坐标转换到物化裁切坐标系。"""

    x1, y1, x2, y2 = global_xyxy
    left, top = view.expanded_xyxy[0], view.expanded_xyxy[1]
    return x1 - left, y1 - top, x2 - left, y2 - top
