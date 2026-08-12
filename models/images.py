"""Model-input image utilities: EXIF, RGB, MIME, data URLs, hashing.

模型输入图像工具：EXIF 方向、RGB 转换、MIME 判断、data URL、图片哈希。
计数切片与 tile 几何不属于本模块（见 agents/counting/geometry.py）。
"""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Sequence
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
