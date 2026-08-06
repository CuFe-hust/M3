"""Pure seam-crop construction and visual markers. / 纯 seam 裁剪构造和视觉标记。"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageDraw

from spacers_agent.imaging import pixel_to_norm
from spacers_agent.schemas import GlobalPointObservation, PixelRect


@dataclass(frozen=True)
class SeamCrop:
    """Marked local seam evidence without altering the source image. / 不修改源图的标记局部 seam 证据。"""

    rect: PixelRect
    image: Image.Image
    first_pixel: tuple[int, int]
    second_pixel: tuple[int, int]
    first_norm: tuple[int, int]
    second_norm: tuple[int, int]
    touches_source_border: bool
    sha256: str


def build_seam_crop(image: Image.Image, first: GlobalPointObservation, second: GlobalPointObservation, *, margin_px: int, max_side: int) -> SeamCrop:
    """Build and mark a bounded seam crop in a copied image. / 在副本图像中构建并标记有界 seam 裁剪。"""

    if margin_px < 0 or max_side <= 0:
        raise ValueError("invalid seam crop limits")
    left, top = max(0, min(first.global_x_px, second.global_x_px) - margin_px), max(0, min(first.global_y_px, second.global_y_px) - margin_px)
    right, bottom = min(image.width, max(first.global_x_px, second.global_x_px) + margin_px + 1), min(image.height, max(first.global_y_px, second.global_y_px) + margin_px + 1)
    rect = PixelRect(left=left, top=top, right=right, bottom=bottom)
    marked = image.crop((left, top, right, bottom)).copy()
    if max(marked.size) > max_side:
        ratio = max_side / max(marked.size)
        marked = marked.resize((round(marked.width * ratio), round(marked.height * ratio)), Image.Resampling.LANCZOS)
    scale_x, scale_y = marked.width / rect.width, marked.height / rect.height
    a, b = (round((first.global_x_px-left)*scale_x), round((first.global_y_px-top)*scale_y)), (round((second.global_x_px-left)*scale_x), round((second.global_y_px-top)*scale_y))
    draw = ImageDraw.Draw(marked)
    for label, point, color in (("A", a, "red"), ("B", b, "blue")):
        draw.ellipse((point[0]-7, point[1]-7, point[0]+7, point[1]+7), outline=color, width=2)
        draw.text((point[0]+8, point[1]-8), label, fill=color)
    with io.BytesIO() as buffer:
        marked.save(buffer, format="PNG")
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    return SeamCrop(rect, marked, a, b, (pixel_to_norm(a[0], marked.width), pixel_to_norm(a[1], marked.height)), (pixel_to_norm(b[0], marked.width), pixel_to_norm(b[1], marked.height)), left == 0 or top == 0 or right == image.width or bottom == image.height, digest)
