from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from spacers_agent.imaging import (
    build_core_halo_tiles,
    contains_point,
    convert_local_point_to_global,
    crop_for_tile,
    global_pixel_to_global_norm,
    is_near_core_boundary,
    local_norm_to_crop_pixel,
    norm_to_pixel,
    owner_core_prompt_norm,
    pixel_to_norm,
    read_normalized_image,
    resize_dimensions_without_upscaling,
    within_owner_tolerance,
)
from spacers_agent.schemas import (
    CountingResult,
    GlobalPointObservation,
    LocalPointObservation,
    PixelRect,
    TileCountResponse,
    stable_sample_id,
)


@pytest.mark.parametrize("length", [1, 2, 10, 896, 1152, 10000])
@pytest.mark.parametrize("value", [0, 1, 500, 998, 999])
def test_normalized_coordinate_endpoints_and_round_trip(length: int, value: int) -> None:
    pixel = norm_to_pixel(value, length)

    assert norm_to_pixel(0, length) == 0
    assert norm_to_pixel(999, length) == length - 1
    assert abs(norm_to_pixel(pixel_to_norm(pixel, length), length) - pixel) <= 1


def test_core_tiles_cover_each_pixel_once_and_clip_halo() -> None:
    width, height = 1933, 1777
    tiles = build_core_halo_tiles(width, height, core_size=896, halo_size=128, model_max_side=1280)

    assert [tile.tile_id for tile in tiles] == ["r000_c000", "r000_c001", "r000_c002", "r001_c000", "r001_c001", "r001_c002"]
    for tile in tiles:
        assert tile.crop_global.left >= 0 and tile.crop_global.top >= 0
        assert tile.crop_global.right <= width and tile.crop_global.bottom <= height
        assert tile.crop_global.left <= tile.owner_core_global.left
        assert tile.crop_global.right >= tile.owner_core_global.right
    for x in range(0, width, 37):
        for y in range(0, height, 41):
            assert sum(contains_point(tile.owner_core_global, x, y) for tile in tiles) == 1


def test_randomized_core_coverage_is_complete_and_non_overlapping() -> None:
    dimensions = [(1, 1), (17, 931), (1025, 1027), (2003, 1999), (4097, 2731)]
    for width, height in dimensions:
        tiles = build_core_halo_tiles(width, height, core_size=224, halo_size=32, model_max_side=640)
        for x in range(width):
            for y in range(0, height, max(1, height // 19)):
                assert sum(contains_point(tile.owner_core_global, x, y) for tile in tiles) == 1


def test_small_image_uses_one_tile_without_upscaling() -> None:
    tile = build_core_halo_tiles(80, 60, core_size=896, halo_size=128, model_max_side=1280)[0]

    assert tile.crop_global == tile.owner_core_global
    assert (tile.model_input_width, tile.model_input_height) == (80, 60)
    assert resize_dimensions_without_upscaling(4000, 2000, 1280) == (1280, 640)


def test_scaled_coordinate_mapping_and_owner_boundary_rules() -> None:
    tile = build_core_halo_tiles(3000, 1000, core_size=896, halo_size=128, model_max_side=640)[1]
    core = tile.owner_core_global
    crop_x = local_norm_to_crop_pixel(999, tile.crop_global.width, tile.model_input_width)

    assert crop_x == tile.crop_global.width - 1
    assert global_pixel_to_global_norm(2999, 3000) == 999
    assert within_owner_tolerance(core, core.left - 2, core.top, 2)
    assert not within_owner_tolerance(core, core.left - 3, core.top, 2)
    assert is_near_core_boundary(core, core.left, core.top, 32)
    assert not is_near_core_boundary(core, core.left + 100, core.top + 100, 32)
    assert all(0 <= value <= 999 for value in owner_core_prompt_norm(tile))


def test_local_point_conversion_tracks_global_coordinates_and_rejection() -> None:
    tile = build_core_halo_tiles(1000, 1000, core_size=500, halo_size=100, model_max_side=1280)[1]
    accepted = LocalPointObservation(local_id="p1", x=500, y=500, confidence=0.8, radius=20, short_evidence="roof")
    rejected = LocalPointObservation(local_id="p2", x=0, y=500, confidence=0.8, radius=20, short_evidence="halo roof")

    accepted_global = convert_local_point_to_global(accepted, tile, sample_id="sample", target="building", boundary_band_px=32)
    rejected_global = convert_local_point_to_global(rejected, tile, sample_id="sample", target="building", boundary_band_px=32)

    assert accepted_global.accepted
    assert accepted_global.global_id == "sample:r000_c001:p1"
    assert not rejected_global.accepted
    assert rejected_global.rejection_reason == "POINT_OUTSIDE_CORE"


def test_tile_crop_and_exif_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "rotated.jpg"
    source = Image.new("RGB", (20, 10), "red")
    exif = Image.Exif()
    exif[274] = 6
    source.save(image_path, exif=exif)

    normalized = read_normalized_image(image_path)
    tile = build_core_halo_tiles(*normalized.size, core_size=8, halo_size=2, model_max_side=8)[0]

    assert normalized.size == (10, 20)
    assert crop_for_tile(normalized, tile).size == (tile.model_input_width, tile.model_input_height)


def test_schema_invariants_and_stable_sample_id() -> None:
    with pytest.raises(ValidationError, match="invalid half-open"):
        PixelRect(left=1, top=1, right=1, bottom=2)
    with pytest.raises(ValidationError, match="reported_count"):
        TileCountResponse(target="building", tile_id="r000_c000", points=[], reported_count=1)
    with pytest.raises(ValidationError, match="duplicate local_id"):
        point = LocalPointObservation(local_id="p1", x=0, y=0, confidence=1.0, short_evidence="roof")
        TileCountResponse(target="building", tile_id="r000_c000", points=[point, point], reported_count=2)

    global_point = GlobalPointObservation(
        global_id="s:r:p", target="building", source_tile_id="r", local_id="p", local_x_norm=0, local_y_norm=0,
        local_radius_norm=0, global_x_px=0, global_y_px=0, global_x_norm=0, global_y_norm=0, radius_px=0.0,
        confidence=1.0, ownership_valid=True, near_core_boundary=False, accepted=True, short_evidence="roof"
    )
    with pytest.raises(ValidationError, match="final_count"):
        CountingResult(
            sample_id="s", target="building", question="count", source_width=1, source_height=1, tile_count=1,
            global_points=[global_point], final_count=0, status="completed"
        )
    assert stable_sample_id(None, Path("images/a.png"), "count buildings", 2) == stable_sample_id(
        None, Path("images/a.png"), "count buildings", 2
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": -1},
        {"y": 1000},
        {"confidence": 1.1},
        {"radius": -1},
    ],
)
def test_local_point_schema_rejects_out_of_range_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "local_id": "p1",
        "x": 0,
        "y": 0,
        "confidence": 0.5,
        "radius": 0,
        "short_evidence": "roof",
    }
    values.update(kwargs)

    with pytest.raises(ValidationError):
        LocalPointObservation(**values)
