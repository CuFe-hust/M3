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

import array
import hashlib
import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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


def normalize_model_tile(
    tile: Image.Image,
    tile_record: EvidenceTileRecord,
) -> Image.Image:
    """Normalize one already-cropped source tile into a strict 1024×1024 RGB
    model tile: full tiles pass through untouched, remainders stretch with
    LANCZOS; size drift fails closed. Shared by the crop-based tile seam and
    the region-source bounded reader. 把一张已裁切的源 tile 规范化为严格
    1024×1024 RGB model tile：完整 tile 原样通过，余块 LANCZOS 拉伸；尺寸
    漂移严格失败。由基于裁切的 tile seam 与 region-source 有界读取共用。"""

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


def prepare_model_tile(
    roi_image: Image.Image,
    tile_record: EvidenceTileRecord,
) -> Image.Image:
    """Materialize one strict 1024×1024 RGB model tile from the ROI-local
    crop: exact crop of source_tile_xyxy, then normalize_model_tile. A new
    image is always returned; the ROI source is never modified.
    从 ROI 局部裁切物化一个严格 1024×1024 RGB model tile：精确裁切
    source_tile_xyxy，再经 normalize_model_tile。始终返回新图像；绝不修改
    ROI 源。"""

    x0, y0, x1, y1 = tile_record.source_tile_xyxy
    if x1 > roi_image.width or y1 > roi_image.height:
        raise ValueError(
            f"tile box {tile_record.source_tile_xyxy!r} exceeds ROI image size {roi_image.size!r}"
        )
    tile = crop_image_box(roi_image, (x0, y0, x1, y1)).convert("RGB")
    return normalize_model_tile(tile, tile_record)


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


def class_id_grid_from_any(class_id_map: Any) -> Image.Image:
    """Convert any two-dimensional integer class-id structure into an
    in-memory "I" grid without importing NumPy or torch: either a
    shape-bearing array or a list of equal-width rows. Non-integer values
    fail closed. 在不导入 NumPy 或 torch 的情况下，把任意二维整数 class-id
    结构转换为内存 "I" 网格：支持带 shape 的数组或等宽行 list。非整数值严格
    失败。"""

    shape = getattr(class_id_map, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError("class_id_map must be two-dimensional")
        height, width = int(shape[0]), int(shape[1])
        flattened = class_id_map.reshape(-1)
        values = (
            flattened.tolist() if hasattr(flattened, "tolist") else list(flattened)
        )
    elif isinstance(class_id_map, (list, tuple)) and class_id_map:
        height = len(class_id_map)
        width = len(class_id_map[0])
        if any(
            not isinstance(row, (list, tuple)) or len(row) != width
            for row in class_id_map
        ):
            raise ValueError("class_id_map rows must have equal width")
        values = [value for row in class_id_map for value in row]
    else:
        raise ValueError("class_id_map must be a two-dimensional integer grid")
    if width <= 0 or height <= 0:
        raise ValueError("class_id_map dimensions must be positive")
    for value in values:
        integer = int(value)
        if integer != value:
            raise ValueError("class_id_map contains a non-integer class ID")
    data = array.array("i", (int(value) for value in values))
    return Image.frombytes("I", (width, height), data.tobytes())


def leaf_boolean_grid(grid: Image.Image, class_ids: frozenset[int]) -> Image.Image:
    """Boolean presence grid of one leaf over an integer class-id grid:
    pixels whose class id belongs to the leaf become 255. Unrequested classes
    stay 0 (background) and never reach the segments/legend. PIL's point()
    with a callable is a no-op on I-mode images, so the grid is scanned via
    raw bytes instead. 一张整数 class-id grid 上某叶子的布尔存在网格：类别 id
    属于该叶子的像素为 255。未请求类别保持 0（背景），绝不进入
    segments/legend。PIL 的 point() 在 I 模式下对 callable 是空操作，因此改为
    扫描原始字节。"""

    values = array.array("i", grid.tobytes())
    present = bytes(255 if class_id in class_ids else 0 for class_id in values)
    return Image.frombytes("L", grid.size, present)


def sample_class_id_grid(
    model_mask: Image.Image,
    x_lookup: tuple[int, ...],
    y_lookup: tuple[int, ...],
) -> Image.Image:
    """Directly sample an integer class-id preview grid from a strict
    1024x1024 model mask through the composed NEAREST lookups: every preview
    pixel takes exactly the class id the legacy restore-then-shrink pipeline
    would have produced, with O(Vw*Vh) memory instead of a WxH/WpxHp grid.
    Indices come from pure geometry (26 Gate 3) and never touch the padding
    region. 通过合成 NEAREST 查找从严格 1024×1024 model mask 直接采样整数
    class-id preview grid：每个 preview 像素恰好取旧“恢复后缩小”管线会产生的
    class id，内存 O(Vw*Vh) 而非 WxH/WpxHp 网格。索引来自纯几何（26 Gate 3），
    绝不触碰 padding 区域。"""

    if model_mask.mode != "I":
        raise ValueError(
            f"model mask mode {model_mask.mode!r} must be an integer class-id grid"
        )
    if model_mask.size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("model mask must be a strict 1024x1024 model input")
    minimum, _ = model_mask.getextrema()
    if minimum < 0:
        raise ValueError("model mask class ids must be non-negative")
    values = array.array("i", model_mask.tobytes())
    stride = MODEL_INPUT_SIZE
    preview_width = len(x_lookup)
    preview_height = len(y_lookup)
    if preview_width <= 0 or preview_height <= 0:
        raise ValueError("preview lookups must be non-empty")
    flat = array.array("i")
    for y in y_lookup:
        base = y * stride
        for x in x_lookup:
            flat.append(values[base + x])
    return Image.frombytes("I", (preview_width, preview_height), flat.tobytes())


def class_ids_in_prefix_rect(
    model_mask: Image.Image,
    extent: tuple[int, int],
) -> frozenset[int]:
    """The set of class ids present in the model-mask prefix rectangle
    [0..mx] x [0..my] — the exact source of the legacy full-resolution
    restored grid (26 Gate 3). Bounded by 1024x1024 pixels regardless of the
    source ROI size. 存在于 model-mask 前缀矩形 [0..mx] x [0..my] 内的类别 id
    集合——旧整分辨率恢复网格的精确来源（26 Gate 3）。无论源 ROI 多大，都
    以 1024×1024 像素为界。"""

    if model_mask.mode != "I":
        raise ValueError(
            f"model mask mode {model_mask.mode!r} must be an integer class-id grid"
        )
    if model_mask.size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("model mask must be a strict 1024x1024 model input")
    max_x, max_y = extent
    if max_x < 0 or max_y < 0 or max_x >= MODEL_INPUT_SIZE or max_y >= MODEL_INPUT_SIZE:
        raise ValueError(f"model extent {extent!r} must stay within the 1024 model grid")
    if max_x == 0 and max_y == 0:
        # Keep the scan O(1) for the pathological 1x1 ROI. 病态 1x1 ROI 下保持 O(1)。
        values = array.array("i", model_mask.tobytes())
        return frozenset({values[0]})
    cropped = model_mask.crop((0, 0, max_x + 1, max_y + 1))
    return frozenset(array.array("i", cropped.tobytes()))


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
