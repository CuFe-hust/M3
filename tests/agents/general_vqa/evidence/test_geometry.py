"""Tests for preview sizing, pixel-frame transforms, and 1024×1024 tiles.

预览尺寸、像素坐标系变换与 1024×1024 tile 纯几何测试。ROI 几何本身由 v2
planner 物化，本文件不再测试已删除的归一化 ROI 规划桥接。
"""

from __future__ import annotations

import pytest

from agents.general_vqa.evidence.geometry import (
    MAX_MODEL_SIDE,
    MODEL_INPUT_SIZE,
    compute_preview_size,
    global_to_local,
    local_to_global,
    model_xyxy_to_roi_xyxy,
    partition_axis,
    partition_roi,
)
from agents.general_vqa.evidence.schema import EvidenceTileRecord, RoiEvidenceRecord


def _record() -> RoiEvidenceRecord:
    return RoiEvidenceRecord(
        roi_id="quantized_roi-0",
        image_id="img1",
        source_size=(2048, 1536),
        core_xyxy=(1024, 512, 2048, 1536),
        expanded_xyxy=(1024, 512, 2048, 1536),
        crop_size=(1024, 1024),
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((4000, 2000), (1080, 540)),
        ((2000, 4000), (540, 1080)),
        ((1080, 1080), (1080, 1080)),
        ((100, 50), (100, 50)),
        ((4000, 3000), (1080, 810)),
    ],
)
def test_preview_size_is_shrink_only(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert compute_preview_size(size) == expected


def test_preview_size_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_preview_size((0, 100))
    with pytest.raises(ValueError, match="positive"):
        compute_preview_size((100, -1))
    assert MAX_MODEL_SIDE == 1080


def test_pixel_frame_transforms_round_trip_through_materialized_view() -> None:
    view = _record()
    local = (10.0, 20.0, 60.0, 70.0)
    global_box = local_to_global(local, view)
    assert global_box == (1034.0, 532.0, 1084.0, 582.0)
    assert global_to_local(global_box, view) == local


# ── 1024×1024 tile partition (14.7.2) / tile 纯几何 ──────────────────────


def _roi(size: tuple[int, int], roi_id: str = "roi-0") -> RoiEvidenceRecord:
    width, height = size
    return RoiEvidenceRecord(
        roi_id=roi_id,
        image_id="img1",
        source_size=size,
        core_xyxy=(0, 0, width, height),
        expanded_xyxy=(0, 0, width, height),
        crop_size=size,
    )


def _coverage_counts(
    size: tuple[int, int],
    tiles: tuple[EvidenceTileRecord, ...],
) -> bytearray:
    """Per-pixel overlap count over the ROI-local frame.
    计算 ROI 局部坐标系下逐像素覆盖次数。"""
    width, height = size
    counts = bytearray(width * height)
    for tile in tiles:
        x0, y0, x1, y1 = tile.source_tile_xyxy
        for y in range(y0, y1):
            row = y * width
            for x in range(x0, x1):
                counts[row + x] += 1
    return counts


_TILE_SIZES = [
    (1, 1),
    (600, 400),
    (1024, 1024),
    (1024, 1536),
    (1536, 1024),
    (2000, 2000),
    (2048, 1536),
    (2048, 2048),
]


@pytest.mark.parametrize("size", _TILE_SIZES)
def test_partition_covers_every_roi_pixel_exactly_once(size: tuple[int, int]) -> None:
    tiles = partition_roi(_roi(size))
    assert tiles
    counts = _coverage_counts(size, tiles)
    assert all(count == 1 for count in counts)
    assert sum(counts) == size[0] * size[1]
    assert tiles[0].source_tile_xyxy[:2] == (0, 0)
    assert tiles[-1].source_tile_xyxy[2:] == size
    for tile in tiles:
        assert tile.model_input_size == (1024, 1024)
        assert tile.scale_x == pytest.approx(1024 / tile.source_tile_size[0])
        assert tile.scale_y == pytest.approx(1024 / tile.source_tile_size[1])


def test_partition_is_stable_row_major_with_exact_tile_ids() -> None:
    tiles = partition_roi(_roi((2000, 2000), roi_id="roi-0"))
    assert [tile.tile_id for tile in tiles] == [
        "roi-0-r0-c0",
        "roi-0-r0-c1",
        "roi-0-r1-c0",
        "roi-0-r1-c1",
    ]
    assert [tile.row for tile in tiles] == [0, 0, 1, 1]
    assert [tile.column for tile in tiles] == [0, 1, 0, 1]
    assert [tile.source_tile_size for tile in tiles] == [
        (1024, 1024),
        (976, 1024),
        (1024, 976),
        (976, 976),
    ]
    assert [tile.resize_applied for tile in tiles] == [False, True, True, True]
    assert tiles[0].scale_x == 1.0 and tiles[0].scale_y == 1.0


def test_partition_matches_plan_examples() -> None:
    assert [t.source_tile_size for t in partition_roi(_roi((2000, 2000)))] == [
        (1024, 1024),
        (976, 1024),
        (1024, 976),
        (976, 976),
    ]
    assert [t.source_tile_size for t in partition_roi(_roi((2048, 1536)))] == [
        (1024, 1024),
        (1024, 1024),
        (1024, 512),
        (1024, 512),
    ]
    assert [t.source_tile_size for t in partition_roi(_roi((1024, 1536)))] == [
        (1024, 1024),
        (1024, 512),
    ]


def test_partition_axis_uses_half_open_intervals_without_zero_tail() -> None:
    assert partition_axis(1) == ((0, 1),)
    assert partition_axis(1024) == ((0, 1024),)
    assert partition_axis(1025) == ((0, 1024), (1024, 1025))
    assert partition_axis(2000) == ((0, 1024), (1024, 2000))
    assert partition_axis(2048) == ((0, 1024), (1024, 2048))
    with pytest.raises(ValueError, match="positive"):
        partition_axis(0)
    with pytest.raises(ValueError, match="positive"):
        partition_axis(-5)
    with pytest.raises(ValueError, match="positive"):
        partition_axis(100, tile_size=0)


def test_model_xyxy_identity_on_full_tile() -> None:
    tiles = partition_roi(_roi((1024, 1024)))
    assert tiles[0].resize_applied is False
    box = model_xyxy_to_roi_xyxy((100.0, 200.0, 300.0, 400.0), tiles[0])
    assert box == (100.0, 200.0, 300.0, 400.0)


def test_model_xyxy_inverse_maps_remainder_tile_with_scale_and_offset() -> None:
    tiles = partition_roi(_roi((2000, 1024)))
    remainder = tiles[1]
    assert remainder.source_tile_xyxy == (1024, 0, 2000, 1024)
    assert remainder.source_tile_size == (976, 1024)
    # The full model tile maps back to exactly the source partition box.
    # 完整 model tile 恰好映射回源 partition 框。
    full = model_xyxy_to_roi_xyxy((0.0, 0.0, 1024.0, 1024.0), remainder)
    assert full == (1024.0, 0.0, 2000.0, 1024.0)
    mid = model_xyxy_to_roi_xyxy((488.0, 0.0, 536.0, 1024.0), remainder)
    assert mid[0] == pytest.approx(488 * 976 / 1024 + 1024)
    assert mid[1] == 0.0
    assert mid[2] == pytest.approx(536 * 976 / 1024 + 1024)
    assert mid[3] == 1024.0


def test_model_xyxy_clamps_out_of_tile_boxes_to_source_extent() -> None:
    tiles = partition_roi(_roi((2000, 1024)))
    remainder = tiles[1]
    clamped = model_xyxy_to_roi_xyxy((-10.0, -20.0, 2048.0, 2100.0), remainder)
    assert clamped == (1024.0, 0.0, 2000.0, 1024.0)


def test_model_xyxy_rejects_degenerate_or_non_finite_boxes() -> None:
    tiles = partition_roi(_roi((1024, 1024)))
    with pytest.raises(ValueError, match="non-degenerate"):
        model_xyxy_to_roi_xyxy((10.0, 10.0, 10.0, 20.0), tiles[0])
    with pytest.raises(ValueError, match="finite"):
        model_xyxy_to_roi_xyxy((float("inf"), 0.0, 10.0, 20.0), tiles[0])
    with pytest.raises(ValueError, match="four numbers"):
        model_xyxy_to_roi_xyxy(("a", 0.0, 10.0, 20.0), tiles[0])
    assert MODEL_INPUT_SIZE == 1024


def test_full_image_pixel_frame_has_zero_origin() -> None:
    view = RoiEvidenceRecord(
        roi_id="full",
        image_id="img1",
        source_size=(100, 80),
        core_xyxy=(0, 0, 100, 80),
        expanded_xyxy=(0, 0, 100, 80),
        crop_size=(100, 80),
    )
    assert local_to_global((0.0, 0.0, 10.0, 10.0), view) == (0.0, 0.0, 10.0, 10.0)
