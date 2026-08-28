"""Pure geometry helpers for canonical materialized visual views.

规范物化视觉视图的纯几何 helper。本模块不依赖 PIL/NumPy/torch/模型实现；
最终 Agent 使用的裁切框已经由 VisualTaskPlanner 以源图像素坐标确定，这里
保留预览尺寸、局部框变换与 1024×1024 tile 的确定性纯几何，以及有界执行
所需的 NEAREST 采样查找表（26 规范 Gate 1/3）。
"""

from __future__ import annotations

import math

from agents.general_vqa.evidence.schema import (
    EvidenceTileRecord,
    RoiEvidenceRecord,
    SegFormerPreprocessRecord,
)

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


def tile_global_xyxy(
    record: RoiEvidenceRecord,
    tile_record: EvidenceTileRecord,
) -> tuple[int, int, int, int]:
    """Translate one tile's crop-local source box into the whole-image pixel
    frame by adding the materialized crop origin. Pure deterministic
    geometry: the worker reads exactly this box from the region source just
    before execution, so no full ROI crop and no eager tile list ever exist.
    把 tile 的裁切局部源框按物化裁切原点平移到整图像素坐标系。纯确定性几何：
    worker 仅在执行前从 region source 读取该框，因此全程不存在完整 ROI 裁切
    或提前物化的 tile 列表。"""

    origin_x, origin_y = record.expanded_xyxy[0], record.expanded_xyxy[1]
    x0, y0, x1, y1 = tile_record.source_tile_xyxy
    return origin_x + x0, origin_y + y0, origin_x + x1, origin_y + y1


def nearest_lookup(source_size: int, target_size: int) -> tuple[int, ...]:
    """Deterministic PIL-NEAREST source-index lookup for one axis of a pure
    scale, replicating Pillow's affine nearest path (ImagingScaleAffine):
    the source coordinate starts at ``0.5 * source_size / target_size`` and
    accumulates the double-precision scale, and each output pixel takes the
    truncated coordinate. This pure replication is parity-proven against
    Pillow's resize for exhaustive small sizes and the production sizes
    (1024 -> padded extent, crop -> preview). It costs O(target_size) memory
    even for multi-hundred-megapixel sources.
    单轴纯缩放的确定性 PIL-NEAREST 源索引查找，复刻 Pillow 仿射 nearest
    路径（ImagingScaleAffine）：源坐标从 ``0.5 * source_size / target_size``
    起按双精度 scale 累加，每个输出像素取截断坐标。该纯复刻已对穷举小尺寸
    与生产尺寸（1024 -> padded、裁切 -> preview）与 Pillow resize 逐点
    parity 验证。即使源为数亿像素，内存也只有 O(target_size)。"""

    if not isinstance(source_size, int) or source_size <= 0:
        raise ValueError(f"source_size must be a positive integer, got {source_size!r}")
    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError(f"target_size must be a positive integer, got {target_size!r}")
    scale = source_size / target_size
    position = 0.5 * scale
    lookup: list[int] = []
    for _ in range(target_size):
        # COORD() in Geometry.c: truncation toward zero for non-negative.
        # Geometry.c 的 COORD()：非负时向零截断。
        lookup.append(int(position) if position >= 0.0 else -1)
        position += scale
    return tuple(lookup)


def segformer_preview_lookups(
    preprocess: SegFormerPreprocessRecord,
    preview_size: tuple[int, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compose the two NEAREST steps of the legacy restore into one direct
    preview->model lookup pair: preview pixel p first maps to the restored
    crop index via the crop->preview scale, then to the 1024 model-mask
    index via the model->padded scale. Because both steps are exact PIL
    NEAREST replications and the crop is an exact pixel restriction, the
    composed lookup samples exactly the class id the legacy pipeline would
    have placed in the preview — padding is never reachable because every
    preview column stays strictly below the padded width.
    把旧恢复路径的两步 NEAREST 合成一对直接 preview->model 查找：preview
    像素 p 先经 crop->preview 比例映射到恢复裁切索引，再经 model->padded
    比例映射到 1024 model-mask 索引。两步都是精确的 PIL NEAREST 复刻且
    crop 是精确像素限制，因此合成查找采样的 class id 与旧管线放入 preview
    的完全一致——preview 每一列都严格小于 padded 宽度，padding 永远不可达。"""

    width, height = preprocess.source_size
    preview_width, preview_height = preview_size
    crop_x = nearest_lookup(width, preview_width)
    crop_y = nearest_lookup(height, preview_height)
    model_x = nearest_lookup(MODEL_INPUT_SIZE, preprocess.padded_size[0])
    model_y = nearest_lookup(MODEL_INPUT_SIZE, preprocess.padded_size[1])
    return (
        tuple(model_x[index] for index in crop_x),
        tuple(model_y[index] for index in crop_y),
    )


def segformer_model_extent(
    preprocess: SegFormerPreprocessRecord,
) -> tuple[int, int]:
    """The largest model-mask column and row reachable from any source-ROI
    pixel under the NEAREST restore: the legacy restored WxH grid reads
    exactly the model-mask prefix rectangle [0..mx] x [0..my], because the
    NEAREST mapping is non-decreasing, starts at 0 and advances by at most
    one per pixel. The old full-resolution hit decision ("any restored pixel
    of the leaf class") is therefore computable from this prefix rectangle
    with O(1024*1024) memory instead of a WxH mask.
    在 NEAREST 恢复下从源 ROI 任一像素可达的最大 model-mask 列与行：旧
    WxH 恢复网格恰好读取 model-mask 前缀矩形 [0..mx] x [0..my]，因为
    NEAREST 映射非递减、从 0 起且每像素最多前进 1。因此旧的整分辨率命中
    判定（“任一恢复像素属于叶子类别”）只需 O(1024*1024) 内存计算，无需
    WxH mask。"""

    width, height = preprocess.source_size
    model_x = nearest_lookup(MODEL_INPUT_SIZE, preprocess.padded_size[0])
    model_y = nearest_lookup(MODEL_INPUT_SIZE, preprocess.padded_size[1])
    return model_x[width - 1], model_y[height - 1]
