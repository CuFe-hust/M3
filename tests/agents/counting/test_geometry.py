"""Contract tests for counting geometry.

计数几何契约测试：坐标端点、1px 图片、边界点、重叠 halo、owner-core 切片、
严格接受规则与坐标换算的纯确定性。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from agents.counting.geometry import (
    build_core_halo_tiles,
    contains_point,
    convert_local_point_to_global,
    crop_for_tile,
    crop_pixel_to_global_pixel,
    global_pixel_to_global_norm,
    is_near_core_boundary,
    local_norm_to_crop_pixel,
    norm_to_pixel,
    ownership_tolerance_px,
    owner_core_prompt_norm,
    pixel_to_norm,
    resize_dimensions_without_upscaling,
    should_tile_image,
    split_tile_owner_core,
    within_owner_tolerance,
)
from agents.counting.schema import LocalPointObservation, PixelRect, TileSpec

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── 缩放与切片 / resize and tiling ─────────────────────────────────────────


def test_resize_without_upscaling() -> None:
    assert resize_dimensions_without_upscaling(100, 50, 1280) == (100, 50)
    assert resize_dimensions_without_upscaling(2560, 1280, 1280) == (1280, 640)
    with pytest.raises(ValueError):
        resize_dimensions_without_upscaling(0, 10, 1280)


def test_should_tile_image_limits() -> None:
    assert should_tile_image(100, 100, model_max_side=1280, max_pixels_without_tiling=1_600_000) is False
    assert should_tile_image(2000, 100, model_max_side=1280, max_pixels_without_tiling=1_600_000) is True
    assert should_tile_image(1600, 1600, model_max_side=1280, max_pixels_without_tiling=1_600_000) is True
    with pytest.raises(ValueError):
        should_tile_image(0, 10, model_max_side=1280, max_pixels_without_tiling=100)


def _tile(
    *,
    crop: PixelRect,
    core: PixelRect,
    width: int = 1000,
    height: int = 1000,
    model_width: int = 1024,
    model_height: int = 1024,
) -> TileSpec:
    return TileSpec(
        tile_id="t0",
        row=0,
        col=0,
        crop_global=crop,
        owner_core_global=core,
        owner_core_local=PixelRect(
            left=core.left - crop.left,
            top=core.top - crop.top,
            right=core.right - crop.left,
            bottom=core.bottom - crop.top,
        ),
        source_width=width,
        source_height=height,
        model_input_width=model_width,
        model_input_height=model_height,
    )


def test_build_core_halo_tiles_non_overlapping_cores_overlapping_halo() -> None:
    """Owner cores never overlap; halo crops do. owner core 互不重叠；
    halo crop 允许重叠。"""
    tiles = build_core_halo_tiles(2000, 2000, core_size=896, halo_size=128, model_max_side=1280)
    assert [t.tile_id for t in tiles] == [
        "r000_c000", "r000_c001", "r000_c002",
        "r001_c000", "r001_c001", "r001_c002",
        "r002_c000", "r002_c001", "r002_c002",
    ]
    cores = [t.owner_core_global for t in tiles]
    for i, first in enumerate(cores):
        for second in cores[i + 1:]:
            overlap = (
                first.left < second.right
                and second.left < first.right
                and first.top < second.bottom
                and second.top < first.bottom
            )
            assert not overlap, "owner cores must not overlap"
    # The centre tile carries a halo on every side. / 中心切片四侧带 halo。
    interior = tiles[4]  # r001_c001
    assert interior.crop_global.left == interior.owner_core_global.left - 128
    assert interior.crop_global.top == interior.owner_core_global.top - 128
    assert interior.crop_global.right == interior.owner_core_global.right + 128
    assert interior.crop_global.bottom == interior.owner_core_global.bottom + 128
    # Corner tiles clip the halo at the source border. / 角部切片在源边界裁剪 halo。
    corner = tiles[0]
    assert corner.crop_global.left == 0
    assert corner.crop_global.top == 0


def test_build_core_halo_tiles_1px_image() -> None:
    """A 1x1 source still yields a valid single tile.
    1x1 源仍产生一条合法单切片。"""
    tiles = build_core_halo_tiles(1, 1, core_size=896, halo_size=128, model_max_side=1280)
    assert len(tiles) == 1
    assert tiles[0].crop_global == PixelRect(left=0, top=0, right=1, bottom=1)
    assert tiles[0].owner_core_global == PixelRect(left=0, top=0, right=1, bottom=1)


def test_split_tile_owner_core_quarters() -> None:
    tile = _tile(
        crop=PixelRect(left=0, top=0, right=100, bottom=100),
        core=PixelRect(left=10, top=10, right=90, bottom=90),
        width=100,
        height=100,
    )
    children = split_tile_owner_core(tile, halo_size=10, model_max_side=1280)
    assert len(children) == 4
    cores = [child.owner_core_global for child in children]
    assert cores == [
        PixelRect(left=10, top=10, right=50, bottom=50),
        PixelRect(left=50, top=10, right=90, bottom=50),
        PixelRect(left=10, top=50, right=50, bottom=90),
        PixelRect(left=50, top=50, right=90, bottom=90),
    ]
    assert all(child.recursive_depth == 1 for child in children)
    assert all(child.parent_tile_id == "t0" for child in children)
    # Halo retained for interior children. / 内部子切片保留 halo。
    assert children[0].crop_global.left == 0
    assert children[3].crop_global.right == 100


def test_split_tile_owner_core_rejects_tiny_core() -> None:
    tile = _tile(
        crop=PixelRect(left=0, top=0, right=3, bottom=3),
        core=PixelRect(left=1, top=1, right=2, bottom=2),
        width=3,
        height=3,
    )
    with pytest.raises(ValueError, match="too small"):
        split_tile_owner_core(tile, halo_size=0, model_max_side=1280)


def test_crop_for_tile_accepts_opened_image() -> None:
    image = Image.new("RGB", (100, 100), (1, 2, 3))
    tile = _tile(
        crop=PixelRect(left=10, top=10, right=60, bottom=60),
        core=PixelRect(left=20, top=20, right=50, bottom=50),
        width=100,
        height=100,
        model_width=50,
        model_height=50,
    )
    crop = crop_for_tile(image, tile)
    assert crop.size == (50, 50)


# ── 坐标换算端点 / coordinate endpoints ────────────────────────────────────


def test_norm_to_pixel_endpoints() -> None:
    assert norm_to_pixel(0, 100) == 0
    assert norm_to_pixel(999, 100) == 99
    assert norm_to_pixel(500, 100) == 50
    with pytest.raises(ValueError, match="outside range"):
        norm_to_pixel(1000, 100)


def test_pixel_to_norm_endpoints() -> None:
    assert pixel_to_norm(0, 100) == 0
    assert pixel_to_norm(99, 100) == 999
    assert pixel_to_norm(0, 1) == 0  # 1px image / 1px 图片
    with pytest.raises(ValueError, match="outside image range"):
        pixel_to_norm(100, 100)


def test_local_norm_to_crop_pixel_1px_crop() -> None:
    assert local_norm_to_crop_pixel(500, 1, 1024) == 0
    assert local_norm_to_crop_pixel(500, 100, 1) == 0


def test_crop_pixel_to_global_pixel_clamping() -> None:
    value, clamped = crop_pixel_to_global_pixel(10, 990, 1000)
    assert value == 999 and clamped is True
    value2, clamped2 = crop_pixel_to_global_pixel(5, 10, 1000)
    assert value2 == 15 and clamped2 is False


def test_global_pixel_to_global_norm() -> None:
    assert global_pixel_to_global_norm(0, 100) == 0
    assert global_pixel_to_global_norm(99, 100) == 999


# ── 边界点与 owner 规则 / boundary points and ownership ───────────────────


def test_contains_point_half_open_semantics() -> None:
    rect = PixelRect(left=10, top=10, right=20, bottom=20)
    assert contains_point(rect, 10, 10) is True
    assert contains_point(rect, 19, 19) is True
    assert contains_point(rect, 20, 10) is False  # right edge excluded / 右缘排除
    assert contains_point(rect, 10, 20) is False  # bottom edge excluded / 下缘排除


def test_ownership_tolerance_and_within() -> None:
    assert ownership_tolerance_px(1000, 1000) == 3
    assert ownership_tolerance_px(100, 100) == 2
    rect = PixelRect(left=10, top=10, right=20, bottom=20)
    assert within_owner_tolerance(rect, 8, 10, 2) is True
    assert within_owner_tolerance(rect, 7, 10, 2) is False


def test_is_near_core_boundary() -> None:
    rect = PixelRect(left=10, top=10, right=20, bottom=20)
    assert is_near_core_boundary(rect, 10, 15, 2) is True
    assert is_near_core_boundary(rect, 15, 15, 2) is False
    assert is_near_core_boundary(rect, 5, 15, 2) is False  # outside / 外部
    with pytest.raises(ValueError, match="band_px"):
        is_near_core_boundary(rect, 15, 15, -1)


def test_owner_core_prompt_norm() -> None:
    tile = _tile(
        crop=PixelRect(left=0, top=0, right=100, bottom=100),
        core=PixelRect(left=10, top=10, right=90, bottom=90),
    )
    bounds = owner_core_prompt_norm(tile)
    assert len(bounds) == 4
    assert bounds[0] == pixel_to_norm(10, 100)
    assert bounds[2] == pixel_to_norm(89, 100)


def test_convert_local_point_accepts_inside_core() -> None:
    tile = _tile(
        crop=PixelRect(left=0, top=0, right=100, bottom=100),
        core=PixelRect(left=10, top=10, right=90, bottom=90),
        width=100,
        height=100,
        model_width=100,
        model_height=100,
    )
    point = LocalPointObservation(
        local_id="p1", x=300, y=300, confidence=0.9, short_evidence="e"
    )
    global_point = convert_local_point_to_global(
        point, tile, sample_id="s1", target="car", boundary_band_px=5
    )
    assert global_point.accepted is True
    assert global_point.ownership_valid is True
    assert global_point.rejection_reason is None
    assert global_point.global_id == "s1:t0:p1"


def test_convert_local_point_rejects_outside_core() -> None:
    """Strict owner-core acceptance: a point outside the core is rejected.
    严格 owner core 接受规则：core 外点被拒绝。"""
    tile = _tile(
        crop=PixelRect(left=0, top=0, right=100, bottom=100),
        core=PixelRect(left=10, top=10, right=90, bottom=90),
        width=100,
        height=100,
        model_width=100,
        model_height=100,
    )
    point = LocalPointObservation(
        local_id="p1", x=0, y=0, confidence=0.9, short_evidence="e"
    )
    global_point = convert_local_point_to_global(
        point, tile, sample_id="s1", target="car", boundary_band_px=5
    )
    assert global_point.accepted is False
    assert global_point.rejection_reason == "POINT_OUTSIDE_CORE"


def test_convert_local_point_outside_but_within_tolerance_is_boundary_candidate() -> None:
    """Out-of-core points within the small tolerance become boundary-review
    candidates. core 外但在小容差内的点成为边界复核候选。"""
    tile = _tile(
        crop=PixelRect(left=0, top=0, right=100, bottom=100),
        core=PixelRect(left=10, top=10, right=90, bottom=90),
        width=100,
        height=100,
        model_width=100,
        model_height=100,
    )
    # x=91 maps to crop pixel 9 — one pixel left of the core (tolerance 2).
    # x=91 映射到 crop 像素 9——core 左侧 1px（容差 2）。
    point = LocalPointObservation(
        local_id="p1", x=91, y=91, confidence=0.9, short_evidence="e"
    )
    global_point = convert_local_point_to_global(
        point, tile, sample_id="s1", target="car", boundary_band_px=5
    )
    assert global_point.accepted is False
    assert global_point.near_core_boundary is True


def test_convert_local_point_clamps_halo_overflow() -> None:
    tile = _tile(
        crop=PixelRect(left=0, top=0, right=100, bottom=100),
        core=PixelRect(left=10, top=10, right=90, bottom=90),
        width=100,
        height=100,
        model_width=100,
        model_height=100,
    )
    point = LocalPointObservation(
        local_id="p1", x=0, y=0, confidence=0.9, short_evidence="e"
    )
    global_point = convert_local_point_to_global(
        point, tile, sample_id="s1", target="car", boundary_band_px=0
    )
    assert 0 <= global_point.global_x_px < 100
    assert 0 <= global_point.global_y_px < 100


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_geometry_never_reads_files() -> None:
    """The module must never open files; crop_for_tile receives an opened
    image. 模块绝不打开文件；crop_for_tile 接收已打开的图片。"""
    source = (REPO_ROOT / "agents" / "counting" / "geometry.py").read_text(encoding="utf-8")
    for token in ("Image.open", "open(", ".read_bytes", "imread"):
        assert token not in source, token


def test_geometry_has_no_dataset_branch() -> None:
    source = (REPO_ROOT / "agents" / "counting" / "geometry.py").read_text(encoding="utf-8")
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source
