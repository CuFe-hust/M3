"""Pure geometry helpers for canonical materialized visual views.

规范物化视觉视图的纯几何 helper。本模块不依赖 PIL/NumPy/torch/模型实现；
最终 Agent 使用的裁切框已经由 VisualTaskPlanner 以源图像素坐标确定，这里
保留预览尺寸、局部框变换与 1024×1024 tile 的确定性纯几何。
"""

from __future__ import annotations

import math

from agents.general_vqa.evidence.schema import EvidenceTileRecord, RoiEvidenceRecord

MODEL_INPUT_SIZE = 1024

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


def partition_axis(
    length: int,
    tile_size: int = 1024,
) -> tuple[tuple[int, int], ...]:
    """Greedy non-overlapping row-major partition of one axis into half-open
    integer intervals. Every pixel belongs to exactly one interval; a length
    divisible by tile_size produces no zero-size tail.
    一条轴的贪心无重叠 row-major 切分，产生半开整数区间。每个像素恰好属于
    一个区间；长度可被 tile_size 整除时不会产生零尺寸尾部。"""

    if not isinstance(length, int) or length <= 0:
        raise ValueError(f"axis length must be a positive integer, got {length!r}")
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError(f"tile_size must be a positive integer, got {tile_size!r}")
    intervals: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + tile_size, length)
        intervals.append((start, end))
        start = end
    return tuple(intervals)


def partition_roi(
    record: RoiEvidenceRecord,
    tile_size: int = 1024,
) -> tuple[EvidenceTileRecord, ...]:
    """Deterministic row-major Cartesian partition of a materialized ROI crop
    into model tiles. source_tile_xyxy stays in the crop-local frame; tile ids
    are stable ``<roi_id>-r<row>-c<column>`` identities, never completion
    order or hash values. 对已物化 ROI 裁切做确定性 row-major Cartesian 切分，
    生成 model tiles。source_tile_xyxy 保持裁切局部坐标系；tile id 是稳定的
    ``<roi_id>-r<row>-c<column>`` 身份，绝不来自完成顺序或 hash。"""

    width, height = record.crop_size
    xs = partition_axis(width, tile_size)
    ys = partition_axis(height, tile_size)
    tiles: list[EvidenceTileRecord] = []
    for row, (y0, y1) in enumerate(ys):
        for column, (x0, x1) in enumerate(xs):
            source_size = (x1 - x0, y1 - y0)
            tiles.append(
                EvidenceTileRecord(
                    tile_id=f"{record.roi_id}-r{row}-c{column}",
                    roi_id=record.roi_id,
                    row=row,
                    column=column,
                    source_tile_xyxy=(x0, y0, x1, y1),
                    source_tile_size=source_size,
                    scale_x=MODEL_INPUT_SIZE / source_size[0],
                    scale_y=MODEL_INPUT_SIZE / source_size[1],
                    resize_applied=source_size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                )
            )
    return tuple(tiles)


def model_xyxy_to_roi_xyxy(
    box: tuple[float, float, float, float],
    tile_record: EvidenceTileRecord,
) -> tuple[float, float, float, float]:
    """Inverse-map a model-tile box into the ROI-local crop frame by scale
    division and tile offset, clamping to the source tile extent. Degenerate
    or non-finite model boxes fail closed — stretched model coordinates are
    never treated as ROI coordinates. 将 model-tile 框按 scale 除法与 tile
    偏移逆映射到 ROI 局部裁切坐标系，并裁剪到源 tile 范围。退化或非有限
    model 框严格失败——拉伸后的 model 坐标绝不直接当作 ROI 坐标。"""

    if len(box) != 4 or any(not isinstance(value, (int, float)) for value in box):
        raise ValueError("model box must contain four numbers")
    x1, y1, x2, y2 = box
    try:
        finite = all(math.isfinite(value) for value in (x1, y1, x2, y2))
    except TypeError:
        finite = False
    if not finite:
        raise ValueError("model box must be finite")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("model box must be non-degenerate")
    source_width, source_height = tile_record.source_tile_size
    tile_x0, tile_y0 = tile_record.source_tile_xyxy[0], tile_record.source_tile_xyxy[1]
    roi_box = (
        max(0.0, min(float(source_width), x1 / tile_record.scale_x)) + tile_x0,
        max(0.0, min(float(source_height), y1 / tile_record.scale_y)) + tile_y0,
        max(0.0, min(float(source_width), x2 / tile_record.scale_x)) + tile_x0,
        max(0.0, min(float(source_height), y2 / tile_record.scale_y)) + tile_y0,
    )
    if roi_box[0] >= roi_box[2] or roi_box[1] >= roi_box[3]:
        raise ValueError("inverse-mapped box degenerated outside the source tile")
    return roi_box
