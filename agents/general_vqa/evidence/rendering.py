"""VQA evidence rendering: ROI crops, previews, and the three-branch protocol.

VQA 证据渲染：ROI 裁切、预览与三分支最终图像协议。任务无关的 EXIF/RGB 与
裁切复用 models/images.py 的 crop_image_region，不复制实现。

Frozen protocol (14.12) / 冻结协议（14.12）：

- 每个 ROI 按证据分支稳定输出：仅 YOLO -> 标注 ROI；仅 SegFormer -> 纯色
  mask；两者都有 -> YOLO-on-pure-mask + clean ROI；均无 -> clean ROI；
- YOLO 高对比标注：黑色外描边 5px、亮品红内描边 3px（均为 <=1080 输出尺寸），
  标签为黑底、品红边框、白色叶子文字，confidence 绝不写入；
- SegFormer 调色表按调用方稳定叶子顺序确定性生成，颜色满足冻结距离约束
  （与品红 >= 128、与黑色背景 >= 96、彼此 >= 48），sha256(leaf|attempt)
  重采样，有限尝试预算耗尽时稳定失败；
- 所有 final-Qwen 图像最长边超过 1080 才缩小，小图绝不放大；掩膜类图像用
  NEAREST 保持纯色；
- 具体图片格式/质量等持久化参数尚未批准，本模块不写任何文件、不自行选择
  JPEG/PNG 参数。
"""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from agents.general_vqa.evidence.geometry import (
    MAX_MODEL_SIDE,
    MODEL_INPUT_SIZE,
    compute_preview_size,
)
from agents.general_vqa.evidence.schema import (
    EvidenceTileRecord,
    RoiEvidenceRecord,
    SegFormerPreprocessRecord,
)
from models.images import (
    crop_image_box,
    image_to_data_url,
    image_sha256,
    materialize_quantized_roi as _materialize_quantized_roi,
    read_normalized_image,
    QuantizedRoi,
)

# Frozen annotation palette (14.12.2): stroke widths are specified at <=1080
# output, so annotation always draws on the shrunk preview.
# 冻结标注调色（14.12.2）：描边宽度按 <=1080 输出指定，因此标注始终画在
# 已缩小的预览上。
_YOLO_MAGENTA = (255, 0, 255)
_MASK_BACKGROUND = (0, 0, 0)
_YOLO_OUTER_STROKE = (0, 0, 0)
_YOLO_INNER_STROKE = (255, 0, 255)
_YOLO_OUTER_WIDTH = 5
_YOLO_INNER_WIDTH = 3

# Deterministic SegFormer mask palette budget and distance constraints. The
# version is public: it pins the palette identity inside evidence_identity so
# any palette change alters the final-Qwen request hash (14.13).
# 确定性 SegFormer mask 调色表预算与距离约束。版本为公开常量：它通过
# evidence_identity 固定调色表身份，使任何调色表变化都会改变最终 Qwen 请求
# hash（14.13）。
PALETTE_VERSION = "v1"
_PALETTE_MAX_ATTEMPTS = 256
_PALETTE_MIN_DIST_MAGENTA = 128.0
_PALETTE_MIN_DIST_BACKGROUND = 96.0
_PALETTE_MIN_DIST_PAIRWISE = 48.0
_LABEL_FONT_SIZE = 18
_LABEL_PLATE_PADDING = 3


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


def prepare_model_tile(
    roi_image: Image.Image,
    tile_record: EvidenceTileRecord,
) -> Image.Image:
    """Materialize one strict 1024×1024 RGB model tile from the ROI-local
    crop: exact crop of source_tile_xyxy, full tiles pass through untouched,
    remainders stretch with LANCZOS. A new image is always returned; the ROI
    source is never modified. 从 ROI 局部裁切物化一个严格 1024×1024 RGB model
    tile：精确裁切 source_tile_xyxy，完整 tile 原样通过，余块用 LANCZOS 拉伸。
    始终返回新图像；绝不修改 ROI 源。"""

    x0, y0, x1, y1 = tile_record.source_tile_xyxy
    if x1 > roi_image.width or y1 > roi_image.height:
        raise ValueError(
            f"tile box {tile_record.source_tile_xyxy!r} exceeds ROI image size {roi_image.size!r}"
        )
    tile = crop_image_box(roi_image, (x0, y0, x1, y1)).convert("RGB")
    if tile.size != tile_record.source_tile_size:
        raise ValueError(
            f"tile crop drift: geometry predicts {tile_record.source_tile_size!r} "
            f"but pixel crop rendered {tile.size!r}"
        )
    if tile.size == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        return tile
    return tile.resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        resample=Image.Resampling.LANCZOS,
    )


def restore_class_id_mask(
    model_mask: Image.Image,
    tile_record: EvidenceTileRecord,
) -> Image.Image:
    """Restore one 1024×1024 integer class-id model mask to the source tile
    size with NEAREST interpolation — never bilinear/LANCZOS, which could
    fabricate class ids. The input must be a strict 1024 square integer grid
    ("L" or "I" mode); continuous probability images are rejected here so the
    class-id NEAREST helper never sees them. 将一个 1024×1024 整数 class-id
    model mask 用 NEAREST 插值恢复到源 tile 尺寸——绝不用 bilinear/LANCZOS，
    以免杜撰出不存在的 class id。输入必须是严格 1024 方形整数网格（"L" 或
    "I" 模式）；此处拒绝连续概率图，使 class-id NEAREST helper 绝不收到它们。"""

    if model_mask.mode not in ("L", "I"):
        raise ValueError(
            f"model mask mode {model_mask.mode!r} must be an integer class-id grid"
        )
    if model_mask.size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError(
            "model mask must be a strict 1024x1024 model tile"
        )
    if tile_record.source_tile_size == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        return model_mask.copy()
    return model_mask.resize(
        tile_record.source_tile_size,
        resample=Image.Resampling.NEAREST,
    )


def prepare_segformer_roi(
    roi_image: Image.Image,
    *,
    roi_id: str,
    source_size: tuple[int, int],
) -> tuple[SegFormerPreprocessRecord, Image.Image]:
    """Materialize the single strict 1024×1024 RGB model input of the fresh
    SegFormer pad protocol (26 §3.1-3.3): pad the whole ROI on the right and
    bottom with constant black to the minimal 1024 multiples, then resize the
    padded canvas to 1024 square with LANCZOS. The ROI origin stays (0, 0)
    with no coordinate shift, so restore/crop and the YOLO ROI-local boxes
    stay aligned. A new image is always returned; the ROI source is never
    modified. 物化新鲜 SegFormer pad 协议（26 §3.1-3.3）的单一严格 1024×1024
    RGB 模型输入：整张 ROI 在右侧与底部以固定黑色 padding 到 1024 最小倍数，
    再把 padded canvas 用 LANCZOS 缩放到 1024 方形。ROI 原点保持 (0, 0)、无
    坐标平移，因此恢复/裁切与 YOLO ROI-local 框保持对齐。始终返回新图像；
    绝不修改 ROI 源。"""

    width, height = source_size
    if width <= 0 or height <= 0:
        raise ValueError(f"source_size must be positive, got {source_size!r}")
    if roi_image.size != source_size:
        raise ValueError(
            f"ROI image size {roi_image.size!r} does not match source_size "
            f"{source_size!r}"
        )
    padded_width = ((width + 1023) // 1024) * 1024
    padded_height = ((height + 1023) // 1024) * 1024
    canvas = Image.new("RGB", (padded_width, padded_height), (0, 0, 0))
    canvas.paste(roi_image.convert("RGB"), (0, 0))
    model_input = canvas.resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        resample=Image.Resampling.LANCZOS,
    )
    preprocess = SegFormerPreprocessRecord(
        roi_id=roi_id,
        source_size=(width, height),
        padded_size=(padded_width, padded_height),
        padding_right=padded_width - width,
        padding_bottom=padded_height - height,
        model_input_size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        scale_x=MODEL_INPUT_SIZE / padded_width,
        scale_y=MODEL_INPUT_SIZE / padded_height,
        padding_mode="constant-black-right-bottom",
        rgb_interpolation="lanczos",
        mask_inverse_interpolation="nearest",
    )
    return preprocess, model_input


def restore_segformer_class_id_mask(
    model_mask: Image.Image,
    preprocess: SegFormerPreprocessRecord,
) -> Image.Image:
    """Restore one strict 1024×1024 integer class-id model mask to the source
    ROI size: NEAREST back to the padded canvas, then crop [0:W, 0:H] so the
    padding region can never appear in the final mask. The input must be a
    strict 1024 square integer grid ("L" or "I" mode); bilinear/LANCZOS are
    never used on class ids, which could fabricate class values. The returned
    grid is exactly preprocess.source_size, aligned to the ROI-local frame.
    将一个严格 1024×1024 整数 class-id model mask 恢复到源 ROI 尺寸：NEAREST
    缩回 padded canvas，再裁切 [0:W, 0:H] 使 padding 区域绝不出现在最终 mask。
    输入必须是严格 1024 方形整数网格（"L" 或 "I" 模式）；class id 绝不使用
    bilinear/LANCZOS，以免杜撰类别值。返回网格尺寸恰为 preprocess.source_size，
    与 ROI 局部坐标系对齐。"""

    if model_mask.mode not in ("L", "I"):
        raise ValueError(
            f"model mask mode {model_mask.mode!r} must be an integer class-id grid"
        )
    if model_mask.size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("model mask must be a strict 1024x1024 model input")
    minimum, _ = model_mask.getextrema()
    if minimum < 0:
        raise ValueError("model mask class ids must be non-negative")
    width, height = preprocess.source_size
    padded_width, padded_height = preprocess.padded_size
    if (padded_width, padded_height) == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        restored = model_mask.copy()
    else:
        restored = model_mask.resize(
            (padded_width, padded_height),
            resample=Image.Resampling.NEAREST,
        )
    cropped = restored.crop((0, 0, width, height))
    if cropped.size != (width, height):
        raise ValueError(
            f"restored mask size {cropped.size!r} must equal source_size "
            f"{(width, height)!r}"
        )
    return cropped


def stitch_class_id_masks(
    restored_tiles: Sequence[tuple[EvidenceTileRecord, Image.Image]],
    roi_size: tuple[int, int],
) -> Image.Image:
    """Stitch restored per-tile class-id masks back into one ROI-local canvas.
    Partitions are overlap-free by construction, so every ROI pixel must be
    written exactly once: a hole, duplicate write, size drift, or out-of-bounds
    placement fails closed instead of being guessed. The returned canvas is an
    integer class-id grid ("I" mode) starting from class 0. 将恢复后的逐 tile
    class-id mask 拼接回一个 ROI 局部 canvas。partition 构造上无重叠，因此每
    个 ROI 像素必须恰好写入一次：hole、重复写入、尺寸漂移或越界放置都严格
    失败，绝不猜测修复。返回的 canvas 是以 class 0 为底色的整数 class-id
    网格（"I" 模式）。"""

    width, height = roi_size
    if width <= 0 or height <= 0:
        raise ValueError(f"roi_size must be positive, got {roi_size!r}")
    canvas = Image.new("I", roi_size, 0)
    coverage = bytearray(width * height)
    for tile_record, mask in restored_tiles:
        x0, y0, x1, y1 = tile_record.source_tile_xyxy
        if x1 > width or y1 > height:
            raise ValueError(
                f"tile box {tile_record.source_tile_xyxy!r} exceeds ROI canvas {roi_size!r}"
            )
        if mask.size != tile_record.source_tile_size:
            raise ValueError(
                f"restored mask size {mask.size!r} must match tile record {tile_record.source_tile_size!r}"
            )
        if mask.mode not in ("L", "I"):
            raise ValueError(f"restored mask mode {mask.mode!r} must be an integer class-id grid")
        for y in range(y0, y1):
            start = y * width + x0
            region = coverage[start : start + (x1 - x0)]
            if any(region):
                raise ValueError(
                    f"overlapping tile placement for {tile_record.tile_id!r}"
                )
        canvas.paste(mask, (x0, y0))
        for y in range(y0, y1):
            start = y * width + x0
            coverage[start : start + (x1 - x0)] = b"\x01" * (x1 - x0)
    if not all(coverage):
        missing = sum(1 for value in coverage if value == 0)
        raise ValueError(f"tile coverage leaves {missing} ROI pixels unwritten")
    return canvas


def make_preview(
    image: Image.Image,
    *,
    max_side: int = MAX_MODEL_SIDE,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """EXIF/RGB-normalize an in-memory image (same semantics as
    read_normalized_image) and shrink it only when the longest side exceeds
    max_side; never upscale. Mask-class images pass NEAREST so flat palette
    colors stay exact. 对内存图像做 EXIF/RGB 归一化（与 read_normalized_image
    语义一致），仅当最长边超过 max_side 时缩小；绝不放大。掩膜类图像用
    NEAREST 保持平坦调色色精确。"""
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    target = compute_preview_size(normalized.size, max_side=max_side)
    if target == normalized.size:
        return normalized
    # In-memory deterministic resize; persistence format/quality parameters
    # are not approved and are deliberately never chosen here.
    # 内存内确定性缩放；持久化格式/质量参数尚未批准，此处刻意不选择。
    return normalized.resize(target, resample=resample)


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


def _color_distance(
    a: tuple[int, int, int], b: tuple[int, int, int]
) -> float:
    """RGB Euclidean distance. RGB 欧氏距离。"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def segformer_palette(leaves: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """Deterministic SegFormer mask palette over an ordered leaf list: colors
    satisfy the frozen distance constraints against the YOLO magenta and the
    black background and pairwise among themselves, resampled via
    sha256(leaf|attempt) with a bounded attempt budget; exhaustion fails
    stably. Same ordered leaves -> same colors, no RNG or process state.
    按有序叶子列表确定性生成 SegFormer mask 调色表：颜色满足与 YOLO 品红及
    黑色背景、以及彼此间的冻结距离约束，用 sha256(leaf|attempt) 有限预算
    重采样；预算耗尽稳定失败。相同有序叶子 -> 相同颜色，无 RNG 或进程状态。"""
    palette: dict[str, tuple[int, int, int]] = {}
    for leaf in leaves:
        palette[leaf] = _resample_palette_color(leaf, tuple(palette.values()))
    return palette


def _resample_palette_color(
    leaf: str,
    previous: tuple[tuple[int, int, int], ...],
) -> tuple[int, int, int]:
    for attempt in range(_PALETTE_MAX_ATTEMPTS):
        digest = hashlib.sha256(f"{leaf}|{attempt}".encode("utf-8")).digest()
        color = (int(digest[0]), int(digest[1]), int(digest[2]))
        if _color_distance(color, _YOLO_MAGENTA) < _PALETTE_MIN_DIST_MAGENTA:
            continue
        if _color_distance(color, _MASK_BACKGROUND) < _PALETTE_MIN_DIST_BACKGROUND:
            continue
        if any(
            _color_distance(color, other) < _PALETTE_MIN_DIST_PAIRWISE
            for other in previous
        ):
            continue
        return color
    raise ValueError(f"segformer palette exhausted for leaf {leaf!r}")


def render_pure_mask(
    size: tuple[int, int],
    leaf_masks: Sequence[tuple[str, Image.Image]],
    palette: Mapping[str, tuple[int, int, int]],
) -> Image.Image:
    """Compose per-leaf presence masks into one flat pure-color mask on the
    black background; later leaves overwrite earlier ones on overlap, so the
    caller's stable leaf order decides deterministic precedence. No box or
    count is ever derived here (mask evidence stays a mask).
    将逐叶子 presence 掩膜合成到黑色背景上的纯色 mask；重叠处靠后叶子覆盖
    靠前叶子，由调用方稳定叶子顺序决定确定性优先级。此处绝不派生框或计数
    （掩膜证据保持为掩膜）。"""
    canvas = Image.new("RGBA", size, (*_MASK_BACKGROUND, 255))
    for leaf, mask in leaf_masks:
        if mask.size != size:
            raise ValueError(
                f"mask size {mask.size!r} must match canvas {size!r}"
            )
        layer = Image.new("RGBA", size, (*palette[leaf], 255))
        layer.putalpha(mask.convert("L"))
        canvas = Image.alpha_composite(canvas, layer)
    return canvas.convert("RGB")


def render_yolo_annotation(
    image: Image.Image,
    boxes: Sequence[tuple[str, tuple[float, float, float, float]]],
    *,
    source_size: tuple[int, int] | None = None,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Annotate one image with the frozen high-contrast YOLO boxes: shrink
    only to <=1080 first (never upscale), then a black 5px outer stroke, a
    magenta 3px inner stroke, and a black label plate with a magenta border
    and white leaf text — confidence never appears. Boxes are given in the
    input image's pixel frame and scale with the preview. When the caller
    already shrunk the image (e.g. a NEAREST pre-shrunk pure mask) it must
    pass ``source_size`` — the pixel frame the boxes live in — so the boxes
    scale onto the preview instead of being drawn at scale 1.0 on the smaller
    canvas. A new image is always returned; the source is never modified.
    用冻结高对比 YOLO 框标注一张图像：先只缩到 <=1080（绝不放大），再画黑色
    5px 外描边、品红 3px 内描边，以及黑底、品红边框、白色叶子文字的标签底板
    ——confidence 绝不出现。框以输入图像像素帧给出，随预览等比缩放。调用方
    若已预先缩小图像（如 NEAREST 预缩的纯色 mask），必须传 ``source_size``
    ——框所在的像素帧——使框按预览缩放，而不是以 scale 1.0 画在更小的画布上。
    始终返回新图像；绝不修改源。"""
    preview = make_preview(image, resample=resample)
    if source_size is not None:
        src_width, src_height = source_size
        if src_width <= 0 or src_height <= 0:
            raise ValueError("source_size must be positive")
        scale_x = preview.width / src_width
        scale_y = preview.height / src_height
    elif preview.size == image.size:
        scale_x = scale_y = 1.0
    else:
        scale_x = preview.width / image.width
        scale_y = preview.height / image.height
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default(size=_LABEL_FONT_SIZE)
    for leaf, (x0, y0, x1, y1) in boxes:
        scaled = (
            x0 * scale_x,
            y0 * scale_y,
            x1 * scale_x,
            y1 * scale_y,
        )
        draw.rectangle(scaled, outline=_YOLO_OUTER_STROKE, width=_YOLO_OUTER_WIDTH)
        draw.rectangle(scaled, outline=_YOLO_INNER_STROKE, width=_YOLO_INNER_WIDTH)
        _draw_label_plate(draw, font, leaf, scaled)
    return preview


def _draw_label_plate(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    leaf: str,
    box: tuple[float, float, float, float],
) -> None:
    """Draw the leaf label plate just above the box (inside it when the box
    touches the top edge); the plate is black with a magenta border and white
    text. 在框上方绘制叶子标签底板（框贴顶时画在框内）；底板为黑色、品红
    边框、白色文字。"""
    left, top, right, _ = box
    image_width = draw.im.size[0]
    image_height = draw.im.size[1]
    text_width, text_height = draw.textbbox((0, 0), leaf, font=font)[2:4]
    plate_width = text_width + 2 * _LABEL_PLATE_PADDING
    plate_height = text_height + 2 * _LABEL_PLATE_PADDING
    # Clamp the plate inside the image. A degenerate/near-edge YOLO box must
    # not produce an inverted rectangle (x1 < x0 or y1 < y0).
    # 把底板限制在图像内。退化/贴边 YOLO 框绝不能产生反转矩形（x1 < x0 或 y1 < y0）。
    max_left = max(0, image_width - plate_width - 1)
    plate_left = min(max(0, int(left)), max_left)
    plate_top = int(top)
    if plate_top - plate_height - 2 >= 0:
        plate_top = plate_top - plate_height - 2
    max_top = max(0, image_height - plate_height - 1)
    plate_top = min(max(0, plate_top), max_top)
    plate_right = min(image_width - 1, plate_left + plate_width)
    plate_bottom = min(image_height - 1, plate_top + plate_height)
    if plate_right < plate_left:
        plate_right = plate_left
    if plate_bottom < plate_top:
        plate_bottom = plate_top
    draw.rectangle(
        (plate_left, plate_top, plate_right, plate_bottom),
        fill=_MASK_BACKGROUND,
        outline=_YOLO_INNER_STROKE,
        width=1,
    )
    draw.text(
        (plate_left + _LABEL_PLATE_PADDING, plate_top + _LABEL_PLATE_PADDING),
        leaf,
        font=font,
        fill=(255, 255, 255),
    )
