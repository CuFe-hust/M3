"""Contract tests for the frozen VQA ROI geometry layer.

冻结 VQA ROI 几何层契约测试：preview 尺寸（横/竖/方/奇数/1px/超大图）、
ROI 像素映射与边缘 halo、整图 fallback、多 ROI 稳定顺序、未知 image_id、
local-global 显式变换与跨平台稳定性。
"""

from __future__ import annotations

import pytest

from agents.general_vqa.evidence.geometry import (
    MAX_MODEL_SIDE,
    compute_preview_size,
    full_image_roi,
    global_to_local,
    local_to_global,
    map_roi,
    resolve_roi_records,
    roi_records_from_plan,
)
from agents.schema import FirstQwenVisualPlan, ObjectEvidenceRequest, RoiPlan, RoiRegion

ROI_PLAN_SIZE = (1000, 800)


def _region(roi_id: str = "roi-1", xyxy=(0.25, 0.25, 0.75, 0.75)) -> RoiRegion:
    return RoiRegion(roi_id=roi_id, image_id="img1", xyxy=xyxy)


def _plan(rois: list[RoiRegion]) -> FirstQwenVisualPlan:
    return FirstQwenVisualPlan.model_validate(
        {
            "version": "first-qwen-plan-v1",
            "execution_family": "object_evidence_vqa",
            "confidence": 0.9,
            "roi_plan": RoiPlan(rois=rois),
            "evidence_request": ObjectEvidenceRequest(
                composite_categories=["vehicle"]
            ),
        }
    )


_SIZES = {"img1": ROI_PLAN_SIZE, "img2": (2000, 1000)}


# ── 预览尺寸 / preview sizing ─────────────────────────────────────────────


def test_preview_landscape_shrinks_to_1080() -> None:
    assert compute_preview_size((4000, 2000)) == (1080, 540)
    assert compute_preview_size((4000, 2000), max_side=1080) == (1080, 540)


def test_preview_portrait_shrinks_to_1080() -> None:
    assert compute_preview_size((2000, 4000)) == (540, 1080)


def test_preview_square_at_cap_is_unchanged() -> None:
    assert compute_preview_size((1080, 1080)) == (1080, 1080)
    assert compute_preview_size((1081, 1081)) == (1080, 1080)


def test_preview_never_upscales_small_images() -> None:
    assert compute_preview_size((100, 50)) == (100, 50)
    assert compute_preview_size((1, 1)) == (1, 1)
    assert compute_preview_size((1079, 600)) == (1079, 600)


def test_preview_odd_sizes_stay_deterministic() -> None:
    assert compute_preview_size((999, 777)) == (999, 777)
    # 4000 -> 1080 keeps aspect ratio within round-half-to-even; same input
    # always yields the same output. 4000 -> 1080 在 round-half-to-even 下
    # 保持宽高比；相同输入永远得到相同输出。
    assert compute_preview_size((4000, 3000)) == compute_preview_size((4000, 3000))


def test_preview_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_preview_size((0, 100))
    with pytest.raises(ValueError, match="positive"):
        compute_preview_size((100, -1))


def test_preview_cap_constant_is_frozen() -> None:
    assert MAX_MODEL_SIDE == 1080


# ── ROI 像素映射 / ROI pixel mapping ──────────────────────────────────────


def test_map_roi_basic_mapping_with_halo() -> None:
    record = map_roi(_region(), ROI_PLAN_SIZE)
    assert record.roi_id == "roi-1"
    assert record.image_id == "img1"
    assert record.source_size == (1000, 800)
    # mapped core: (250, 200, 750, 600); halo 10% -> (200, 160, 800, 640)
    # 映射 core：(250, 200, 750, 600)；halo 10% -> (200, 160, 800, 640)。
    assert record.core_xyxy == (250, 200, 750, 600)
    assert record.expanded_xyxy == (200, 160, 800, 640)
    assert record.crop_size == (600, 480)


def test_map_roi_floor_ceil_outward() -> None:
    # x0_px = 0.2 -> floor 200, x1_px = 0.4 -> ceil 400.
    # x0_px = 0.2 -> floor 200，x1_px = 0.4 -> ceil 400。
    record = map_roi(_region("f", (0.2, 0.2, 0.4, 0.4)), (1000, 1000))
    assert record.core_xyxy == (200, 200, 400, 400)
    # fractional mapped coords round outward / 小数映射坐标向外取整
    record = map_roi(_region("f", (0.200001, 0.200001, 0.400001, 0.400001)), (1000, 1000))
    assert record.core_xyxy == (200, 200, 401, 401)


def test_map_roi_edge_halo_is_clamped() -> None:
    # ROI touching the top-left corner: halo expands beyond the image and is
    # clamped, never producing negative or out-of-bounds pixels.
    # 贴左上角的 ROI：halo 扩张超出图像后被 clamp，绝不产生负值或越界像素。
    record = map_roi(_region("c", (0.0, 0.0, 0.1, 0.1)), ROI_PLAN_SIZE)
    assert record.core_xyxy == (0, 0, 100, 80)
    assert record.expanded_xyxy == (0, 0, 110, 88)
    assert record.crop_size == (110, 88)
    # Bottom-right edge likewise clamps to the source extent.
    # 右下边缘同样 clamp 到源图范围。
    record = map_roi(_region("c", (0.9, 0.9, 1.0, 1.0)), ROI_PLAN_SIZE)
    assert record.expanded_xyxy == (890, 712, 1000, 800)
    assert record.crop_size == (110, 88)


def test_map_roi_one_pixel_image() -> None:
    record = map_roi(_region("tiny", (0.0, 0.0, 1.0, 1.0)), (1, 1))
    assert record.core_xyxy == (0, 0, 1, 1)
    assert record.expanded_xyxy == (0, 0, 1, 1)
    assert record.crop_size == (1, 1)


def test_map_roi_rejects_unknown_frame_and_bad_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        map_roi(_region(), (0, 100))


# ── 整图 fallback / full-image fallback ───────────────────────────────────


def test_full_image_roi_is_unique_and_exact() -> None:
    record = full_image_roi("img1", ROI_PLAN_SIZE)
    assert record.roi_id == "full"
    assert record.core_xyxy == (0, 0, 1000, 800)
    assert record.expanded_xyxy == (0, 0, 1000, 800)
    assert record.crop_size == (1000, 800)


def test_resolve_empty_plan_falls_back_to_full_image() -> None:
    records = resolve_roi_records(
        _plan([]), _SIZES, fallback_image_id="img1"
    )
    assert len(records) == 1
    assert records[0].roi_id == "full"
    assert records[0].crop_size == ROI_PLAN_SIZE


def test_resolve_empty_plan_unknown_fallback_fails() -> None:
    with pytest.raises(ValueError, match="fallback image_id"):
        resolve_roi_records(_plan([]), _SIZES, fallback_image_id="ghost")


# ── 多 ROI 计划 / multi-ROI plans ─────────────────────────────────────────


def test_plan_maps_rois_in_stable_order() -> None:
    plan = _plan([_region("roi-1"), _region("roi-2", (0.1, 0.1, 0.2, 0.2))])
    records = roi_records_from_plan(plan, _SIZES)
    assert [record.roi_id for record in records] == ["roi-1", "roi-2"]
    assert records[0].crop_size == (600, 480)
    # (0.1..0.2) x (0.1..0.2) on 1000x800: 100px core + 10% -> 120px wide,
    # 80px core + 10% -> 96px tall. (0.1..0.2) x (0.1..0.2) 在 1000x800 上：
    # 100px 宽 core + 10% -> 120px，80px 高 core + 10% -> 96px。
    assert records[1].crop_size == (120, 96)


def test_plan_maps_rois_from_different_images() -> None:
    plan = _plan(
        [
            _region("roi-1", (0.0, 0.0, 0.5, 0.5)),
            RoiRegion(roi_id="roi-2", image_id="img2", xyxy=(0.0, 0.0, 0.25, 0.25)),
        ]
    )
    records = roi_records_from_plan(plan, _SIZES)
    assert records[0].source_size == (1000, 800)
    assert records[1].source_size == (2000, 1000)
    # 500x250 core + 10% halo per side -> 550x275 / 500x250 core + 每边 10%
    # halo -> 550x275。
    assert records[1].crop_size == (550, 275)


def test_plan_unknown_image_id_fails_stable() -> None:
    plan = _plan([RoiRegion(roi_id="x", image_id="ghost", xyxy=(0.0, 0.0, 0.5, 0.5))])
    with pytest.raises(ValueError, match="unknown image_id 'ghost'"):
        roi_records_from_plan(plan, _SIZES)


def test_plan_never_truncates_and_maps_every_roi() -> None:
    plan = _plan(
        [
            _region("roi-1", (0.0, 0.0, 0.3, 0.3)),
            _region("roi-2", (0.3, 0.3, 0.6, 0.6)),
            _region("roi-3", (0.6, 0.6, 1.0, 1.0)),
        ]
    )
    records = roi_records_from_plan(plan, _SIZES)
    assert [record.roi_id for record in records] == ["roi-1", "roi-2", "roi-3"]
    assert all(record.crop_size[0] > 0 and record.crop_size[1] > 0 for record in records)


# ── local-global 变换 / local-global transforms ───────────────────────────


def test_local_global_roundtrip() -> None:
    record = map_roi(_region(), ROI_PLAN_SIZE)
    local = (10.0, 20.0, 60.0, 70.0)
    global_box = local_to_global(local, record)
    # expanded origin is (200, 160) / expanded 原点为 (200, 160)
    assert global_box == (210.0, 180.0, 260.0, 230.0)
    assert global_to_local(global_box, record) == local


def test_local_global_uses_expanded_origin() -> None:
    record = full_image_roi("img1", (1000, 800))
    assert local_to_global((0.0, 0.0, 100.0, 100.0), record) == (0.0, 0.0, 100.0, 100.0)
