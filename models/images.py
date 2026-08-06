"""Model-input image utilities: EXIF, RGB, MIME, data URLs, hashing.

模型输入图像工具：EXIF 方向、RGB 转换、MIME 判断、data URL、图片哈希。
计数切片与 tile 几何不属于本模块（见 agents/counting/geometry.py）。
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from PIL import Image, ImageOps

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def read_normalized_image(path: Path) -> Image.Image:
    """Read one image, apply EXIF orientation, and convert it to RGB.
    读取一张图像，应用 EXIF 方向并转换为 RGB。"""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def guess_image_mime(path: Path) -> str:
    """Guess the MIME type from the file suffix; defaults to image/jpeg.
    根据文件后缀猜测 MIME 类型；未知时默认 image/jpeg。"""
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")


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
