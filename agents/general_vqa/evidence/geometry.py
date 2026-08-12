"""Pure geometry for the frozen VQA ROI contract (14B).

14B 冻结 VQA ROI 契约的纯几何层。本模块不依赖 PIL，只做数值映射；EXIF/RGB
与裁切复用 models/images.py 的 crop_image_region，不复制实现。核心不变式：
core/expanded/crop 与 crop_image_region 对同一 (box, halo) 的输出逐位一致；
ROI 计划整体非法或空计划时回退唯一整图，绝不截断前三个 ROI、绝不重试
Planner；多 ROI 分别映射，绝不合并外接矩形、绝不绑定目标类别。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

from agents.general_vqa.evidence.schema import RoiEvidenceRecord
from agents.schema import FirstQwenVisualPlan, RoiRegion

# Frozen halo ratio: each side of a mapped ROI expands by 10% of its own
# width/height. 冻结 halo 比例：映射后 ROI 每边按自身宽/高扩张 10%。
HALO_RATIO = 0.10

# Frozen preview/final-Qwen longest-side cap: images are only ever shrunk to
# this side, never upscaled. 冻结预览/最终 Qwen 最长边上限：图片只缩放到该
# 边长，绝不放大。
MAX_MODEL_SIDE = 1080

# Frozen coordinate frame for VQA ROIs. / VQA ROI 的冻结坐标制式。
COORDINATE_FRAME = Literal["normalized_0_1_top_left"]


def compute_preview_size(
    size: tuple[int, int],
    *,
    max_side: int = MAX_MODEL_SIDE,
) -> tuple[int, int]:
    """Return the preview target size: shrink to max_side only when the
    longest side exceeds it; never upscale a small image. Pure and
    deterministic (round-half-to-even). 返回预览目标尺寸：仅当最长边超过
    max_side 时等比缩到 max_side；绝不放大小图。纯函数且确定性
    （round-half-to-even）。"""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {size!r}")
    longest = max(width, height)
    if longest <= max_side:
        return (width, height)
    scale = max_side / longest
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def map_roi(
    region: RoiRegion,
    source_size: tuple[int, int],
    *,
    halo_ratio: float = HALO_RATIO,
) -> RoiEvidenceRecord:
    """Map one normalized [0,1] ROI to the frozen pixel records: core is the
    floor/ceil mapped box, expanded is core plus the per-side halo fraction
    clamped to the image; crop_size is the expanded extent. The math mirrors
    crop_image_region exactly so the rendering layer can verify zero drift.
    将一个归一化 [0,1] ROI 映射为冻结像素记录：core 是 floor/ceil 映射框，
    expanded 是 core 加每边 halo 比例并 clamp 到图像；crop_size 是 expanded
    范围。数学与 crop_image_region 完全一致，使渲染层可以验证零漂移。"""
    width, height = source_size
    if width <= 0 or height <= 0:
        raise ValueError(f"source size must be positive, got {source_size!r}")
    x0, y0, x1, y1 = region.xyxy
    # The [0,1] frame upper bound is 1.0, so mapping is x * width; keep the
    # same divide-then-multiply shape as crop_image_region for bit parity.
    # [0,1] 制式上界为 1.0，因此映射为 x * width；为与 crop_image_region
    # 逐位一致保留相同的先除后乘形式。
    x0_px = x0 / 1.0 * width
    y0_px = y0 / 1.0 * height
    x1_px = x1 / 1.0 * width
    y1_px = y1 / 1.0 * height
    core = (
        math.floor(x0_px),
        math.floor(y0_px),
        math.ceil(x1_px),
        math.ceil(y1_px),
    )
    halo_x = (x1_px - x0_px) * halo_ratio
    halo_y = (y1_px - y0_px) * halo_ratio
    expanded = (
        max(0, min(width, math.floor(x0_px - halo_x))),
        max(0, min(height, math.floor(y0_px - halo_y))),
        max(0, min(width, math.ceil(x1_px + halo_x))),
        max(0, min(height, math.ceil(y1_px + halo_y))),
    )
    return RoiEvidenceRecord(
        roi_id=region.roi_id,
        image_id=region.image_id,
        source_size=(width, height),
        core_xyxy=core,
        expanded_xyxy=expanded,
        crop_size=(expanded[2] - expanded[0], expanded[3] - expanded[1]),
    )


def full_image_roi(
    image_id: str,
    source_size: tuple[int, int],
) -> RoiEvidenceRecord:
    """The unique full-image ROI: the "no reliable spatial constraint" plan
    maps to this single ROI, never to a truncated or guessed region.
    唯一整图 ROI：“无可靠空间约束”的计划映射到该唯一 ROI，绝不截断或猜测。"""
    width, height = source_size
    return RoiEvidenceRecord(
        roi_id="full",
        image_id=image_id,
        source_size=(width, height),
        core_xyxy=(0, 0, width, height),
        expanded_xyxy=(0, 0, width, height),
        crop_size=(width, height),
    )


def roi_records_from_plan(
    plan: FirstQwenVisualPlan,
    sizes: Mapping[str, tuple[int, int]],
    *,
    halo_ratio: float = HALO_RATIO,
) -> list[RoiEvidenceRecord]:
    """Map every plan ROI in plan order (cross-platform stable). An unknown
    image_id fails with a stable error instead of guessing a size; more than
    three ROIs are impossible here because the plan schema rejects them —
    geometry never truncates. 按计划顺序映射每个计划 ROI（跨平台稳定）。
    未知 image_id 以稳定错误失败而非猜测尺寸；超过三个 ROI 因计划 schema
    拒绝而在此不可能出现——几何层绝不截断。"""
    records: list[RoiEvidenceRecord] = []
    for region in plan.roi_plan.rois:
        if region.image_id not in sizes:
            raise ValueError(
                f"plan references unknown image_id {region.image_id!r}"
            )
        records.append(map_roi(region, sizes[region.image_id], halo_ratio=halo_ratio))
    return records


def resolve_roi_records(
    plan: FirstQwenVisualPlan,
    sizes: Mapping[str, tuple[int, int]],
    *,
    fallback_image_id: str,
    halo_ratio: float = HALO_RATIO,
) -> list[RoiEvidenceRecord]:
    """Frozen ROI resolution: a plan with no spatial constraint falls back to
    the unique full-image ROI of the fallback image; a valid non-empty plan
    maps directly. An invalid raw plan is rejected by schema validation
    before this layer and falls back in the executor without retry.
    冻结 ROI 解析：无空间约束的计划回退为 fallback 图的唯一整图 ROI；合法
    非空计划直接映射。非法原始计划在到达本层前被 schema 校验拒绝，并由执行
    器直接回退，绝不重试。"""
    if not plan.roi_plan.rois:
        if fallback_image_id not in sizes:
            raise ValueError(
                f"fallback image_id {fallback_image_id!r} has no size"
            )
        return [full_image_roi(fallback_image_id, sizes[fallback_image_id])]
    return roi_records_from_plan(plan, sizes, halo_ratio=halo_ratio)


def local_to_global(
    local_xyxy: tuple[float, float, float, float],
    roi: RoiEvidenceRecord,
) -> tuple[float, float, float, float]:
    """Explicit crop-local to whole-image pixel transform (offset by the
    expanded crop origin). Detections from the runtime adapter arrive in the
    crop-local frame after the explicit letterbox inverse; this transform is
    the second, explicit hop. 显式 crop 局部到整图像素变换（以 expanded crop
    原点为偏移）。运行时适配器在显式 letterbox 逆变换后返回 crop 局部坐标；
    本变换是第二个显式跳跃。"""
    x1, y1, x2, y2 = local_xyxy
    left, top = roi.expanded_xyxy[0], roi.expanded_xyxy[1]
    return (x1 + left, y1 + top, x2 + left, y2 + top)


def global_to_local(
    global_xyxy: tuple[float, float, float, float],
    roi: RoiEvidenceRecord,
) -> tuple[float, float, float, float]:
    """Inverse of local_to_global. / local_to_global 的逆变换。"""
    x1, y1, x2, y2 = global_xyxy
    left, top = roi.expanded_xyxy[0], roi.expanded_xyxy[1]
    return (x1 - left, y1 - top, x2 - left, y2 - top)
