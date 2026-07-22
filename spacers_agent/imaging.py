"""Pure image normalization, owner-core tiling, and coordinate conversions.
纯图像规范化、owner core 切片与坐标换算。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from spacers_agent.schemas import GlobalPointObservation, LocalPointObservation, PixelRect, TileSpec


def read_normalized_image(path: Path) -> Image.Image:
    """Read one image, apply EXIF orientation, and convert it to RGB.
    读取一张图像，应用 EXIF 方向并转换为 RGB。
    """

    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def resize_dimensions_without_upscaling(width: int, height: int, model_max_side: int) -> tuple[int, int]:
    """Return aspect-preserving model dimensions without enlarging an image.
    返回保持纵横比且不放大图像的模型输入尺寸。
    """

    if width <= 0 or height <= 0 or model_max_side <= 0:
        raise ValueError("image dimensions and model_max_side must be positive")
    if max(width, height) <= model_max_side:
        return width, height
    scale = model_max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def build_core_halo_tiles(
    width: int,
    height: int,
    *,
    core_size: int,
    halo_size: int,
    model_max_side: int,
) -> list[TileSpec]:
    """Build row-major non-overlapping owner cores with clipped halo crops.
    构建按行优先排列、互不重叠且 halo 被裁剪的 owner core。
    """

    if width <= 0 or height <= 0 or core_size <= 0 or halo_size < 0:
        raise ValueError("invalid source, core, or halo dimensions")
    tiles: list[TileSpec] = []
    row = 0
    for top in range(0, height, core_size):
        bottom = min(top + core_size, height)
        col = 0
        for left in range(0, width, core_size):
            right = min(left + core_size, width)
            core = PixelRect(left=left, top=top, right=right, bottom=bottom)
            crop = PixelRect(
                left=max(0, left - halo_size),
                top=max(0, top - halo_size),
                right=min(width, right + halo_size),
                bottom=min(height, bottom + halo_size),
            )
            local_core = PixelRect(
                left=core.left - crop.left,
                top=core.top - crop.top,
                right=core.right - crop.left,
                bottom=core.bottom - crop.top,
            )
            model_width, model_height = resize_dimensions_without_upscaling(crop.width, crop.height, model_max_side)
            tiles.append(
                TileSpec(
                    tile_id=f"r{row:03d}_c{col:03d}",
                    row=row,
                    col=col,
                    crop_global=crop,
                    owner_core_global=core,
                    owner_core_local=local_core,
                    source_width=width,
                    source_height=height,
                    model_input_width=model_width,
                    model_input_height=model_height,
                )
            )
            col += 1
        row += 1
    return tiles


def should_tile_image(
    width: int,
    height: int,
    *,
    model_max_side: int,
    max_pixels_without_tiling: int,
) -> bool:
    """Return whether an image exceeds the configured single-tile limits.
    返回图像是否超过配置的单切片限制。
    """

    if width <= 0 or height <= 0 or model_max_side <= 0 or max_pixels_without_tiling <= 0:
        raise ValueError("image and tiling limits must be positive")
    return max(width, height) > model_max_side or width * height > max_pixels_without_tiling


def split_tile_owner_core(tile: TileSpec, *, halo_size: int, model_max_side: int) -> list[TileSpec]:
    """Split one owner core into four child cores while retaining halo context.
    将一个 owner core 分为四个子 core，同时保留 halo 上下文。
    """

    if halo_size < 0 or model_max_side <= 0:
        raise ValueError("halo_size and model_max_side must be valid")
    core = tile.owner_core_global
    if core.width < 2 or core.height < 2:
        raise ValueError("owner core is too small to split")
    middle_x = core.left + core.width // 2
    middle_y = core.top + core.height // 2
    cores = (
        PixelRect(left=core.left, top=core.top, right=middle_x, bottom=middle_y),
        PixelRect(left=middle_x, top=core.top, right=core.right, bottom=middle_y),
        PixelRect(left=core.left, top=middle_y, right=middle_x, bottom=core.bottom),
        PixelRect(left=middle_x, top=middle_y, right=core.right, bottom=core.bottom),
    )
    children: list[TileSpec] = []
    for index, child_core in enumerate(cores):
        crop = PixelRect(
            left=max(0, child_core.left - halo_size),
            top=max(0, child_core.top - halo_size),
            right=min(tile.source_width, child_core.right + halo_size),
            bottom=min(tile.source_height, child_core.bottom + halo_size),
        )
        model_width, model_height = resize_dimensions_without_upscaling(
            crop.width, crop.height, model_max_side
        )
        children.append(
            TileSpec(
                tile_id=f"{tile.tile_id}_d{tile.recursive_depth + 1}_{index}",
                row=tile.row * 2 + index // 2,
                col=tile.col * 2 + index % 2,
                crop_global=crop,
                owner_core_global=child_core,
                owner_core_local=PixelRect(
                    left=child_core.left - crop.left,
                    top=child_core.top - crop.top,
                    right=child_core.right - crop.left,
                    bottom=child_core.bottom - crop.top,
                ),
                source_width=tile.source_width,
                source_height=tile.source_height,
                model_input_width=model_width,
                model_input_height=model_height,
                recursive_depth=tile.recursive_depth + 1,
                parent_tile_id=tile.tile_id,
            )
        )
    return children


def crop_for_tile(image: Image.Image, tile: TileSpec) -> Image.Image:
    """Crop normalized source pixels and resize only when model limits require it.
    裁剪规范化源像素，并仅在模型限制需要时缩放。
    """

    crop = image.crop((tile.crop_global.left, tile.crop_global.top, tile.crop_global.right, tile.crop_global.bottom))
    if crop.size == (tile.model_input_width, tile.model_input_height):
        return crop
    return crop.resize((tile.model_input_width, tile.model_input_height), Image.Resampling.LANCZOS)


def norm_to_pixel(value: int, length: int, coord_max: int = 999) -> int:
    """Convert one normalized coordinate to an inclusive pixel index.
    将一个归一化坐标转换为包含端点的像素索引。
    """

    if length <= 0 or coord_max <= 0:
        raise ValueError("length and coord_max must be positive")
    if not 0 <= value <= coord_max:
        raise ValueError("normalized coordinate is outside range")
    return round(value / coord_max * (length - 1))


def pixel_to_norm(pixel: int, length: int, coord_max: int = 999) -> int:
    """Convert one valid pixel index to the shared normalized coordinate range.
    将一个合法像素索引转换为共享归一化坐标范围。
    """

    if length <= 0 or coord_max <= 0:
        raise ValueError("length and coord_max must be positive")
    if not 0 <= pixel < length:
        raise ValueError("pixel is outside image range")
    if length == 1:
        return 0
    return round(pixel / (length - 1) * coord_max)


def local_norm_to_crop_pixel(value: int, crop_length: int, model_input_length: int) -> int:
    """Map a model-local normalized coordinate back to source crop pixels.
    将模型局部归一化坐标映射回源 crop 像素。
    """

    input_pixel = norm_to_pixel(value, model_input_length)
    if crop_length == 1 or model_input_length == 1:
        return 0
    return round(input_pixel / (model_input_length - 1) * (crop_length - 1))


def crop_pixel_to_global_pixel(crop_pixel: int, crop_start: int, source_length: int) -> tuple[int, bool]:
    """Translate crop pixels to global pixels and report defensive clamping.
    将 crop 像素平移为全局像素并报告防御性截断。
    """

    if source_length <= 0:
        raise ValueError("source_length must be positive")
    raw = crop_start + crop_pixel
    clamped = min(max(raw, 0), source_length - 1)
    return clamped, clamped != raw


def global_pixel_to_global_norm(pixel: int, source_length: int) -> int:
    """Convert a global pixel to a global normalized coordinate.
    将全局像素转换为全局归一化坐标。
    """

    return pixel_to_norm(pixel, source_length)


def contains_point(rect: PixelRect, x: int, y: int) -> bool:
    """Return whether a point belongs to a half-open rectangle.
    返回一个点是否属于半开矩形。
    """

    return rect.left <= x < rect.right and rect.top <= y < rect.bottom


def ownership_tolerance_px(source_width: int, source_height: int) -> int:
    """Return the fixed small tolerance used only for boundary review candidates.
    返回仅用于边界复核候选的固定小容差。
    """

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    return max(2, round(0.003 * min(source_width, source_height)))


def within_owner_tolerance(rect: PixelRect, x: int, y: int, tolerance: int) -> bool:
    """Return whether an out-of-core point is close enough for boundary review.
    返回 core 外点是否足够接近以进入边界复核。
    """

    if tolerance < 0:
        raise ValueError("tolerance must not be negative")
    return rect.left - tolerance <= x < rect.right + tolerance and rect.top - tolerance <= y < rect.bottom + tolerance


def is_near_core_boundary(rect: PixelRect, x: int, y: int, band_px: int) -> bool:
    """Return whether an accepted point is near an internal owner-core boundary.
    返回接受点是否接近内部 owner core 边界。
    """

    if not contains_point(rect, x, y):
        return False
    if band_px < 0:
        raise ValueError("band_px must not be negative")
    return min(x - rect.left, rect.right - 1 - x, y - rect.top, rect.bottom - 1 - y) < band_px


def owner_core_prompt_norm(tile: TileSpec) -> list[int]:
    """Express owner core bounds in the tile crop's normalized coordinate system.
    在切片 crop 的归一化坐标系中表示 owner core 边界。
    """

    local = tile.owner_core_local
    return [
        pixel_to_norm(local.left, tile.crop_global.width),
        pixel_to_norm(local.top, tile.crop_global.height),
        pixel_to_norm(local.right - 1, tile.crop_global.width),
        pixel_to_norm(local.bottom - 1, tile.crop_global.height),
    ]


def convert_local_point_to_global(
    point: LocalPointObservation,
    tile: TileSpec,
    *,
    sample_id: str,
    target: str,
    boundary_band_px: int,
    tolerance_px: int | None = None,
) -> GlobalPointObservation:
    """Convert one tile-local point and enforce strict owner-core acceptance.
    转换一条切片局部点并强制严格 owner core 接受规则。
    """

    crop_x = local_norm_to_crop_pixel(point.x, tile.crop_global.width, tile.model_input_width)
    crop_y = local_norm_to_crop_pixel(point.y, tile.crop_global.height, tile.model_input_height)
    global_x, x_was_clamped = crop_pixel_to_global_pixel(crop_x, tile.crop_global.left, tile.source_width)
    global_y, y_was_clamped = crop_pixel_to_global_pixel(crop_y, tile.crop_global.top, tile.source_height)
    core = tile.owner_core_global
    ownership_valid = contains_point(core, global_x, global_y)
    effective_tolerance = tolerance_px if tolerance_px is not None else ownership_tolerance_px(
        tile.source_width, tile.source_height
    )
    boundary_candidate = not ownership_valid and within_owner_tolerance(core, global_x, global_y, effective_tolerance)
    near_boundary = is_near_core_boundary(core, global_x, global_y, boundary_band_px) or boundary_candidate
    radius_px = point.radius / 999 * min(tile.crop_global.width, tile.crop_global.height)
    return GlobalPointObservation(
        global_id=f"{sample_id}:{tile.tile_id}:{point.local_id}",
        target=target,
        source_tile_id=tile.tile_id,
        local_id=point.local_id,
        local_x_norm=point.x,
        local_y_norm=point.y,
        local_radius_norm=point.radius,
        global_x_px=global_x,
        global_y_px=global_y,
        global_x_norm=global_pixel_to_global_norm(global_x, tile.source_width),
        global_y_norm=global_pixel_to_global_norm(global_y, tile.source_height),
        global_x_was_clamped=x_was_clamped,
        global_y_was_clamped=y_was_clamped,
        radius_px=radius_px,
        confidence=point.confidence,
        ownership_valid=ownership_valid,
        near_core_boundary=near_boundary,
        accepted=ownership_valid,
        rejection_reason=None if ownership_valid else "POINT_OUTSIDE_CORE",
        short_evidence=point.short_evidence,
    )
