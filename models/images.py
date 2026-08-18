"""Model-input image utilities: EXIF, RGB, MIME, data URLs, hashing.

模型输入图像工具：EXIF 方向、RGB 转换、MIME 判断、data URL、图片哈希。
计数切片与 tile 几何不属于本模块（见 agents/counting/geometry.py）。
"""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError

_MIME_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
    "BMP": "image/bmp",
}


class UnsupportedImageFormatError(ValueError):
    """Raised when an image file cannot be identified as a supported format.
    无法将图片文件识别为受支持格式时抛出。"""


def read_normalized_image(path: Path) -> Image.Image:
    """Read one image, apply EXIF orientation, and convert it to RGB.
    读取一张图像，应用 EXIF 方向并转换为 RGB。"""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


CoordinateFrame = Literal[
    "normalized_0_1_top_left",
    "normalized_0_999_top_left",
]


@dataclass(frozen=True)
class QuantizedRoi:
    """Deterministic audit geometry for one quantized ROI.
    一个量化 ROI 的确定性审计几何。"""

    requested_roi_xyxy_0_999: tuple[int, int, int, int]
    requested_pixel_xyxy: tuple[int, int, int, int]
    roi_quantum: int
    quantized_side: int
    ideal_square_xyxy: tuple[int, int, int, int]
    crop_xyxy: tuple[int, int, int, int]
    crop_size: tuple[int, int]
    was_clipped: bool


def materialize_quantized_roi(
    source_size: tuple[int, int],
    roi_xyxy: Sequence[int],
    *,
    roi_quantum: int = 1024,
) -> QuantizedRoi:
    """Map a strict 0..999 box to a quantized, center-preserving ROI.
    将严格 0..999 矩形映射为中心保持、按量化单位扩展的 ROI。

    The model rectangle is mapped outward to source pixels, the longest side
    is rounded up to a positive multiple of roi_quantum, and the ideal square
    is clipped directly against the source. It is intentionally not shifted
    or resized after clipping. 模型矩形先向外映射到源图像素，最长边向上量化
    为 roi_quantum 的正整数倍，再与源图直接求交；截断后绝不平移或二次缩放。
    """
    width, height = source_size
    if width <= 0 or height <= 0:
        raise ValueError("source_size must be positive")
    if roi_quantum <= 0:
        raise ValueError("roi_quantum must be positive")
    values = tuple(roi_xyxy)
    if len(values) != 4 or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise ValueError("roi_xyxy must contain four strict integers")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 999 and 0 <= y0 < y1 <= 999):
        raise ValueError("roi_xyxy must be a non-degenerate box within [0, 999]")

    left = (x0 * width) // 999
    top = (y0 * height) // 999
    right = (x1 * width + 998) // 999
    bottom = (y1 * height + 998) // 999
    requested_pixel = (left, top, right, bottom)
    requested_width = right - left
    requested_height = bottom - top
    longest_side = max(requested_width, requested_height)
    quantized_side = max(
        roi_quantum,
        ((longest_side + roi_quantum - 1) // roi_quantum) * roi_quantum,
    )

    # Integer floor division exactly implements floor for the possibly
    # negative ideal origin, avoiding float tie drift. 使用整数向下除法精确
    # 实现可能为负的理想原点 floor，避免浮点边界漂移。
    ideal_left = (left + right - quantized_side) // 2
    ideal_top = (top + bottom - quantized_side) // 2
    ideal = (
        ideal_left,
        ideal_top,
        ideal_left + quantized_side,
        ideal_top + quantized_side,
    )
    crop = (
        max(0, ideal[0]),
        max(0, ideal[1]),
        min(width, ideal[2]),
        min(height, ideal[3]),
    )
    if crop[0] >= crop[2] or crop[1] >= crop[3]:
        raise ValueError("quantized ROI does not intersect source image")
    return QuantizedRoi(
        requested_roi_xyxy_0_999=values,
        requested_pixel_xyxy=requested_pixel,
        roi_quantum=roi_quantum,
        quantized_side=quantized_side,
        ideal_square_xyxy=ideal,
        crop_xyxy=crop,
        crop_size=(crop[2] - crop[0], crop[3] - crop[1]),
        was_clipped=ideal != crop,
    )


def crop_image_box(
    image: Path | Image.Image,
    box: Sequence[int],
) -> Image.Image:
    """Crop an exact integer half-open box from a normalized RGB image.
    从规范化 RGB 图像按整数半开区间精确裁切，并返回独立图像。"""
    values = tuple(box)
    if len(values) != 4 or not all(isinstance(value, int) for value in values):
        raise ValueError("box must contain four integer coordinates")
    x0, y0, x1, y1 = values
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("box must be a positive half-open rectangle")
    if isinstance(image, Path):
        normalized = read_normalized_image(image)
    elif isinstance(image, Image.Image):
        normalized = ImageOps.exif_transpose(image).convert("RGB")
    else:
        raise TypeError(
            f"image must be a Path or PIL.Image.Image, got {type(image).__name__}"
        )
    width, height = normalized.size
    if x1 > width or y1 > height:
        raise ValueError("box must be inside the image")
    return normalized.crop((x0, y0, x1, y1))

# Max normalized coordinate per frame; the image edge maps to this value.
# 每种坐标制式允许的最大归一化坐标，图片边缘对应此值。
_FRAME_UPPER_BOUND: dict[str, float] = {
    "normalized_0_1_top_left": 1.0,
    "normalized_0_999_top_left": 999.0,
}


def crop_image_region(
    image: Path | Image.Image,
    box: Sequence[float],
    *,
    coordinate_frame: CoordinateFrame,
    halo_ratio: float = 0.0,
) -> Image.Image:
    """Crop a rectangular region of interest from an image in memory.

    Consumes xyxy boxes produced directly by a vision model. Both the
    14A 0..999 frame and the 14B frozen [0,1] frame are accepted; the box
    is mapped to pixels, expanded by ``halo_ratio`` (a fraction of the
    mapped ROI width/height on each side), rounded outward to Pillow's
    half-open pixel boundary, clamped to the image, and returned as a
    standalone RGB crop. The input image object is never modified.
    在内存中裁切矩形 ROI：直接消费模型输出的 xyxy 坐标，兼容 14A 的
    0..999 制式与 14B 冻结的 [0,1] 制式。坐标映射为像素、按 halo_ratio
    （映射后 ROI 宽高的比例）向四边扩张、边界向外取整到 Pillow 半开像素
    边界并裁剪到原图范围，返回独立 RGB 裁切图；不修改输入图片对象。
    """
    if coordinate_frame not in _FRAME_UPPER_BOUND:
        raise ValueError(f"unsupported coordinate_frame: {coordinate_frame!r}")

    box = tuple(box)
    if len(box) != 4:
        raise ValueError(f"box must contain exactly 4 coordinates, got {len(box)}")
    x0, y0, x1, y1 = box
    try:
        finite = all(math.isfinite(value) for value in (x0, y0, x1, y1))
    except TypeError:
        finite = False
    if not finite:
        raise ValueError("box coordinates must be finite numeric values")

    upper = _FRAME_UPPER_BOUND[coordinate_frame]
    for name, value in (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1)):
        if not 0.0 <= value <= upper:
            raise ValueError(
                f"box coordinate {name} out of range [0, {upper}] for "
                f"{coordinate_frame}: {value!r}"
            )
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "box must be a non-degenerate rectangle with x1 > x0 and y1 > y0"
        )

    if not math.isfinite(halo_ratio) or halo_ratio < 0.0:
        raise ValueError(f"halo_ratio must be finite and >= 0, got {halo_ratio!r}")

    if isinstance(image, Path):
        loaded = read_normalized_image(image)
    elif isinstance(image, Image.Image):
        loaded = ImageOps.exif_transpose(image).convert("RGB")
    else:
        raise TypeError(
            f"image must be a Path or PIL.Image.Image, got {type(image).__name__}"
        )

    width, height = loaded.size
    # Divide first, then multiply: for equivalent boxes in the two frames
    # (e.g. 0.25 in [0,1] vs 249.75 in [0,999]) this yields bit-identical
    # pixel values, so both frames crop the same region.
    # 先除后乘：两种制式下等价的 box（如 [0,1] 的 0.25 与 [0,999] 的 249.75）
    # 会得到完全一致的像素值，从而裁切出相同区域。
    x0_px = x0 / upper * width
    y0_px = y0 / upper * height
    x1_px = x1 / upper * width
    y1_px = y1 / upper * height

    halo_x = (x1_px - x0_px) * halo_ratio
    halo_y = (y1_px - y0_px) * halo_ratio

    # Round outward to Pillow's half-open pixel boundary, then clamp the
    # expanded region to the original image extent.
    # 向外取整到 Pillow 半开像素边界，再把扩张区域裁剪到原图范围。
    left = max(0, min(width, math.floor(x0_px - halo_x)))
    right = max(0, min(width, math.ceil(x1_px + halo_x)))
    top = max(0, min(height, math.floor(y0_px - halo_y)))
    bottom = max(0, min(height, math.ceil(y1_px + halo_y)))

    return loaded.crop((left, top, right, bottom))


def detect_image_mime(path: Path) -> str:
    """Detect the MIME type from the real file content via Pillow; unknown or
    corrupt files fail explicitly and are never masked as JPEG.
    通过 Pillow 按真实文件内容检测 MIME 类型；未知或损坏文件显式失败，
    绝不伪装成 JPEG。"""
    try:
        with Image.open(path) as image:
            format_name = image.format
    except (UnidentifiedImageError, OSError) as error:
        raise UnsupportedImageFormatError(
            f"cannot identify image format: {type(error).__name__}"
        ) from error
    mime = _MIME_BY_FORMAT.get(format_name)
    if mime is None:
        raise UnsupportedImageFormatError(
            f"unsupported image format: {format_name or 'unknown'}"
        )
    return mime


def image_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Encode image bytes as one OpenAI-compatible data URL.
    将图像字节编码为一条 OpenAI 兼容的数据 URL。"""
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")
    if not mime.startswith("image/"):
        raise ValueError("mime must identify an image type")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw image bytes.
    返回原始图像字节的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(data).hexdigest()
