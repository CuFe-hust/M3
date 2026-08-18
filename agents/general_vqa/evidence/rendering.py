"""VQA evidence rendering: ROI crops, previews, and mask overlays.

VQA 证据渲染：ROI 裁切、预览与掩膜 overlay。任务无关的 EXIF/RGB 与裁切
复用 models/images.py 的 crop_image_region，不复制实现。证据以文本记录提供，
clean ROI 不画检测框；SegFormer 证据以每个 ROI 的独立半透明 overlay 提供，
掩膜不转框、不计数。调色表为纯内存确定性哈希色（同叶子类别跨 ROI/样本
稳定）；具体图片格式/质量等持久化参数尚未批准，本模块不写任何文件、不
自行选择 JPEG/PNG 参数。
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps

from agents.general_vqa.evidence.geometry import (
    MAX_MODEL_SIDE,
    compute_preview_size,
)
from agents.general_vqa.evidence.schema import RoiEvidenceRecord
from models.images import (
    crop_image_box,
    image_to_data_url,
    image_sha256,
    materialize_quantized_roi as _materialize_quantized_roi,
    read_normalized_image,
    QuantizedRoi,
)

# Internal overlay transparency; the frozen palette/persistence parameters are
# not yet approved, so these defaults are in-memory seams only and never
# persisted. 内部 overlay 透明度；冻结调色表/持久化参数尚未批准，因此这些
# 默认值只是内存 seam，绝不持久化。
_OVERLAY_ALPHA = 0.40


def normalized_image_size(path: Path) -> tuple[int, int]:
    """Return the EXIF/RGB-normalized source size for planner geometry.
    返回规划器几何所需的 EXIF/RGB 规范化源尺寸。"""
    return read_normalized_image(path).size


def materialize_quantized_roi(
    source_size: tuple[int, int],
    roi_xyxy: tuple[int, int, int, int],
    *,
    roi_quantum: int = 1024,
) -> QuantizedRoi:
    """Expose the shared quantized-ROI primitive through the agents seam.
    通过 agents seam 暴露共享的量化 ROI 原语。"""
    return _materialize_quantized_roi(
        source_size,
        roi_xyxy,
        roi_quantum=roi_quantum,
    )


def render_roi_crop(
    image: Image.Image,
    record: RoiEvidenceRecord,
) -> Image.Image:
    """Render the exact source-pixel box recorded by the materialized view.
    按已物化视图记录的精确源像素框渲染图像。"""

    crop = crop_image_box(image, record.expanded_xyxy)
    if crop.size != record.crop_size:
        raise ValueError(
            f"ROI crop drift: geometry predicts {record.crop_size!r} but "
            f"pixel crop rendered {crop.size!r}"
        )
    return crop


def make_preview(
    image: Image.Image,
    *,
    max_side: int = MAX_MODEL_SIDE,
) -> Image.Image:
    """EXIF/RGB-normalize an in-memory image (same semantics as
    read_normalized_image) and shrink it only when the longest side exceeds
    max_side; never upscale. 对内存图像做 EXIF/RGB 归一化（与
    read_normalized_image 语义一致），仅当最长边超过 max_side 时缩小；
    绝不放大。"""
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    target = compute_preview_size(normalized.size, max_side=max_side)
    if target == normalized.size:
        return normalized
    # In-memory deterministic resize; persistence format/quality parameters
    # are not approved and are deliberately never chosen here.
    # 内存内确定性缩放；持久化格式/质量参数尚未批准，此处刻意不选择。
    return normalized.resize(target, resample=Image.Resampling.LANCZOS)


def preview_from_path(
    path: Path,
    *,
    max_side: int = MAX_MODEL_SIDE,
) -> tuple[str, str]:
    """Read one image file (EXIF/RGB-normalized), build the safe shrink-only
    preview, and return (data_url, sha256): deterministic in-memory PNG
    transport bytes plus the honest digest of exactly what the model receives.
    The PNG choice is a transport seam only — persistence format/quality
    parameters stay unchosen and nothing is ever written to disk here.
    读取一个图像文件（EXIF/RGB 归一化），构建安全只缩预览，返回
    （data_url, sha256）：确定性内存 PNG 传输字节及模型实际收到内容的真实
    摘要。PNG 选择只是传输 seam——持久化格式/质量参数保持未选择，此处绝不
    写盘。"""
    image = read_normalized_image(path)
    preview = make_preview(image, max_side=max_side)
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    data = buffer.getvalue()
    return image_to_data_url(data, "image/png"), image_sha256(data)


def stable_palette_color(leaf_category: str) -> tuple[int, int, int]:
    """Deterministic per-leaf RGB color: the same leaf category maps to the
    same color across ROIs and samples, with no mutable global palette state.
    每叶子类别确定性 RGB 颜色：同一叶子类别跨 ROI/样本映射到同一颜色，
    不依赖任何可变全局调色表状态。"""
    digest = hashlib.sha256(leaf_category.encode("utf-8")).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))


def overlay_mask(
    source: Image.Image,
    mask: Image.Image,
    *,
    color: tuple[int, int, int],
    alpha: float = _OVERLAY_ALPHA,
) -> Image.Image:
    """Return a NEW composite of source with one independent semi-transparent
    mask overlay; neither input is modified, masks never blend across ROIs,
    and no box or count is ever derived here (mask evidence stays a mask).
    返回 source 与一个独立半透明掩膜 overlay 合成的新图像；两个输入都不被
    修改，掩膜跨 ROI 绝不融合，此处绝不导出框或计数（掩膜证据保持为掩膜）。"""
    if mask.size != source.size:
        raise ValueError(
            f"mask size {mask.size!r} must match source {source.size!r}"
        )
    base = source.convert("RGBA")
    overlay = Image.new("RGBA", source.size, (*color, 0))
    # Presence masks are boolean (0/1), so any nonzero pixel is fully present;
    # scale the presence by alpha instead of treating the value as 0..255
    # gray, which would make boolean-true pixels invisible.
    # presence 掩膜是布尔（0/1）的，因此任何非零像素都是“存在”；直接用 alpha
    # 缩放存在性，而不是把值当作 0..255 灰度，否则布尔 true 像素会不可见。
    opacity = mask.convert("L").point(
        lambda value: round((255.0 if value else 0.0) * alpha)
    )
    overlay.putalpha(opacity)
    return Image.alpha_composite(base, overlay).convert("RGB")
