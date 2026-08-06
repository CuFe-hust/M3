"""Model-input image utilities: EXIF, RGB, MIME, data URLs, hashing.

模型输入图像工具：EXIF 方向、RGB 转换、MIME 判断、data URL、图片哈希。
计数切片与 tile 几何不属于本模块（见 agents/counting/geometry.py）。
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

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
