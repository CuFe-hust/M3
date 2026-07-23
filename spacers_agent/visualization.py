"""Auditable counting overlays rendered from persisted geometry and point records.
从持久化几何和点记录渲染可审计计数标注图。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from spacers_agent.schemas import CountingResult, TileSpec


def render_counting_overlay(
    image: Image.Image,
    *,
    result: CountingResult,
    tiles: Sequence[TileSpec],
    output_path: Path,
) -> None:
    """Render core grids plus accepted and rejected points into an explicit output image.
    将 core 网格、接受点和拒绝点渲染到显式指定的输出图像。
    """

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for tile in tiles:
        core = tile.owner_core_global
        draw.rectangle((core.left, core.top, core.right - 1, core.bottom - 1), outline=(60, 130, 246), width=1)
        draw.text((core.left + 2, core.top + 2), tile.tile_id, fill=(60, 130, 246))
    for point in result.global_points:
        color = (34, 197, 94) if point.accepted else (220, 38, 38)
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
        draw.text((point.global_x_px + radius + 1, point.global_y_px + radius + 1), marker, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
