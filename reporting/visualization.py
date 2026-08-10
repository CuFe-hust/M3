"""Auditable counting overlays rendered from persisted point geometry.

从持久化点几何渲染可审计计数标注图。输入为源图像与 CountingResult；绝不
导入 CountingAgent/YOLO backend，绝不调用模型。图像尺寸必须与
CountingResult 声明的源尺寸一致，否则稳定失败（绝不静默缩放）。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from agents.counting.schema import CountingResult

# Accepted / rejected point colors (green / red).
# 接受点/拒绝点颜色（绿/红）。
_ACCEPTED_COLOR = (34, 197, 94)
_REJECTED_COLOR = (220, 38, 38)


def render_counting_overlay(
    image: Image.Image,
    *,
    result: CountingResult,
    output_path: Path,
) -> Path:
    """Render accepted (green) and rejected (red) points onto an RGB copy of
    the source image. The image size must match the CountingResult source
    dimensions, otherwise the render fails stably instead of silently
    rescaling. 在源图像 RGB 副本上渲染接受点（绿）与拒绝点（红）。图像尺寸
    必须与 CountingResult 声明的源尺寸一致，否则稳定失败而非静默缩放。"""

    canvas = image.convert("RGB").copy()
    if canvas.size != (result.source_width, result.source_height):
        raise ValueError(
            "counting overlay image size does not match CountingResult dimensions"
        )
    draw = ImageDraw.Draw(canvas)
    for point in result.global_points:
        color = _ACCEPTED_COLOR if point.accepted else _REJECTED_COLOR
        radius = max(3, round(point.radius_px))
        draw.ellipse(
            (
                point.global_x_px - radius,
                point.global_y_px - radius,
                point.global_x_px + radius,
                point.global_y_px + radius,
            ),
            outline=color,
            width=2,
        )
        marker = point.local_id if point.accepted else f"{point.local_id}!"
        draw.text(
            (point.global_x_px + radius + 1, point.global_y_px + radius + 1),
            marker,
            fill=color,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
