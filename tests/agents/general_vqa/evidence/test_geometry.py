"""Tests for preview sizing and pixel-frame transforms.

预览尺寸与像素坐标系变换测试。ROI 几何本身由 v2 planner 物化，本文件不再
测试已删除的归一化 ROI 规划桥接。
"""

from __future__ import annotations

import pytest

from agents.general_vqa.evidence.geometry import (
    MAX_MODEL_SIDE,
    compute_preview_size,
    global_to_local,
    local_to_global,
)
from agents.general_vqa.evidence.schema import RoiEvidenceRecord


def _record() -> RoiEvidenceRecord:
    return RoiEvidenceRecord(
        roi_id="fixed_roi-0",
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
