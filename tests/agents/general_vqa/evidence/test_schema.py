"""Contract tests for the v5 visual plan and strict VQA evidence schemas.

v5 视觉计划与严格 VQA 证据 schema 契约测试：计划与物化视图保持独立，证据
几何使用源像素坐标，mask 不转框且所有持久化字段 JSON-safe。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.general_vqa.evidence.schema import (
    EvidencePreprocessing,
    EvidenceState,
    EvidenceTileRecord,
    LayerStateRecord,
    ModelCallAudit,
    RoiEvidenceRecord,
    SegFormerEvidenceRecord,
    SegFormerPreprocessRecord,
    VqaEvidenceBundle,
    YoloDetectionRecord,
)
from agents.schema import (
    MaterializedVisualView,
    VisualTaskPlan,
)


def _plan(**overrides) -> VisualTaskPlan:
    data = {
        "version": "visual-task-plan-v5",
        "task": "general_vqa",
        "needs_visual_assistance": True,
        "object_categories": ["small-vehicle"],
        "count_target": None,
        "region_request": {
            "explicit": True,
            "image_index": 0,
            "roi_xyxy": (400, 400, 600, 600),
        },
        "reason_codes": ["test"],
    }
    data.update(overrides)
    return VisualTaskPlan.model_validate(data)


def test_visual_task_plan_validates_assistance_linkage() -> None:
    plan = _plan()
    assert plan.version == "visual-task-plan-v5"
    assert plan.task == "general_vqa"
    assert plan.object_categories == ["small-vehicle"]
    with pytest.raises(ValidationError, match="requires object_categories"):
        _plan(needs_visual_assistance=True, object_categories=[])
    with pytest.raises(ValidationError, match="require visual assistance"):
        _plan(needs_visual_assistance=False, object_categories=["small-vehicle"])


def test_visual_task_plan_rejects_unknown_or_path_like_fields() -> None:
    with pytest.raises(ValidationError):
        _plan(task="not-a-task")
    with pytest.raises(ValidationError, match="path-like"):
        _plan(object_categories=["/models/vehicle"])
    with pytest.raises(ValidationError, match="valid integer"):
        _plan(region_request={"explicit": True, "image_index": 0, "roi_xyxy": (1.0, 2, 3, 4)})
    with pytest.raises(ValidationError, match="within"):
        _plan(region_request={"explicit": True, "image_index": 0, "roi_xyxy": (-1, 2, 3, 4)})
    with pytest.raises(ValidationError, match="non-degenerate"):
        _plan(region_request={"explicit": True, "image_index": 0, "roi_xyxy": (4, 2, 3, 4)})
    with pytest.raises(ValidationError):
        _plan(version="visual-task-plan-v4")


def test_visual_task_plan_v5_schema_has_count_target_linkage_and_no_confidence() -> None:
    schema = VisualTaskPlan.model_json_schema()
    assert "confidence" not in schema["properties"]
    assert "confidence" not in schema.get("required", [])
    with pytest.raises(ValidationError):
        _plan(confidence=0.9)
    counting = _plan(
        task="counting",
        count_target="small-vehicle",
        object_categories=["small-vehicle"],
    )
    assert counting.count_target == "small-vehicle"
    with pytest.raises(ValidationError, match="requires count_target"):
        _plan(task="counting", count_target=None)
    with pytest.raises(ValidationError, match="requires count_target"):
        _plan(task="fine_grained_counting", count_target=None)
    with pytest.raises(ValidationError, match="non-counting"):
        _plan(count_target="small-vehicle")


@pytest.mark.parametrize("target", ["", " vehicle", "vehicle ", "../vehicle", "3", "ship\n"])
def test_visual_task_plan_v5_rejects_unsafe_count_target(target: str) -> None:
    with pytest.raises(ValidationError):
        _plan(task="counting", count_target=target)


def test_visual_task_plan_v5_allows_at_most_eight_leaf_categories() -> None:
    categories = [f"leaf-{index}" for index in range(8)]
    assert _plan(object_categories=categories).object_categories == categories
    with pytest.raises(ValidationError):
        _plan(object_categories=[*categories, "leaf-8"])
    with pytest.raises(ValidationError, match="duplicates"):
        _plan(object_categories=["small-vehicle", "small-vehicle"])


def test_materialized_view_is_exact_source_pixel_geometry() -> None:
    view = MaterializedVisualView(
        image_id="img1",
        view_mode="quantized_roi",
        source_size=(2048, 1536),
        crop_xyxy=(1024, 640, 2048, 1536),
        crop_size=(1024, 896),
        requested_roi_xyxy_0_999=(500, 500, 999, 999),
        requested_pixel_xyxy=(1025, 768, 2048, 1536),
        roi_quantum=1024,
        quantized_side=1024,
        ideal_square_xyxy=(1024, 640, 2048, 1664),
        was_clipped=True,
    )
    dumped = view.model_dump(mode="json")
    assert dumped["crop_xyxy"] == [1024, 640, 2048, 1536]
    assert dumped["crop_size"] == [1024, 896]
    assert dumped["was_clipped"] is True
    with pytest.raises(ValidationError):
        MaterializedVisualView(
            image_id="img1",
            view_mode="fixed_roi",
            source_size=(2048, 1536),
            crop_xyxy=(1024, 512, 2048, 1536),
            crop_size=(1024, 1024),
        )
    with pytest.raises(ValidationError, match="full_image"):
        MaterializedVisualView(
            image_id="img1",
            view_mode="full_image",
            source_size=(100, 80),
            crop_xyxy=(1, 0, 100, 80),
            crop_size=(99, 80),
        )


# ── VQA 证据 schema / VQA evidence schema ────────────────────────────────


def test_evidence_states_are_closed() -> None:
    assert set(EvidenceState.__args__) == {
        "hit",
        "missing",
        "unsupported",
        "unavailable",
        "error",
        "not_run",
    }


def test_detection_record_has_no_confidence_and_keeps_both_frames() -> None:
    record = YoloDetectionRecord(
        leaf_category="small_vehicle",
        roi_id="roi-1",
        local_xyxy=(10.0, 20.0, 60.0, 70.0),
        local_roi_size=(100, 100),
        global_xyxy=(210.0, 320.0, 260.0, 370.0),
        global_image_size=(1000, 800),
    )
    dumped = record.model_dump()
    assert "confidence" not in dumped
    assert dumped["local_xyxy"] == (10.0, 20.0, 60.0, 70.0)
    assert dumped["global_xyxy"] == (210.0, 320.0, 260.0, 370.0)
    record.model_dump_json()


def test_detection_record_rejects_confidence_as_extra() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        YoloDetectionRecord(
            leaf_category="small_vehicle",
            roi_id="roi-1",
            local_xyxy=(0.0, 0.0, 10.0, 10.0),
            local_roi_size=(100, 100),
            global_xyxy=(0.0, 0.0, 10.0, 10.0),
            global_image_size=(1000, 800),
            confidence=0.95,
        )


def test_detection_record_validates_geometry_and_size_refs() -> None:
    with pytest.raises(ValidationError, match="non-degenerate"):
        YoloDetectionRecord(
            leaf_category="x", roi_id="r",
            local_xyxy=(0.0, 0.0, 0.0, 10.0),
            local_roi_size=(100, 100),
            global_xyxy=(0.0, 0.0, 10.0, 10.0),
            global_image_size=(1000, 800),
        )
    with pytest.raises(ValidationError, match="exceeds"):
        YoloDetectionRecord(
            leaf_category="x", roi_id="r",
            local_xyxy=(0.0, 0.0, 101.0, 10.0),
            local_roi_size=(100, 100),
            global_xyxy=(0.0, 0.0, 10.0, 10.0),
            global_image_size=(1000, 800),
        )
    with pytest.raises(ValidationError, match="finite"):
        YoloDetectionRecord(
            leaf_category="x", roi_id="r",
            local_xyxy=(0.0, 0.0, 10.0, float("nan")),
            local_roi_size=(100, 100),
            global_xyxy=(0.0, 0.0, 10.0, 10.0),
            global_image_size=(1000, 800),
        )


def test_segformer_record_has_no_box_or_count() -> None:
    record = SegFormerEvidenceRecord(leaf_category="small_vehicle", roi_id="roi-1")
    dumped = record.model_dump()
    assert "box" not in dumped
    assert "count" not in dumped
    assert "confidence" not in dumped
    record.model_dump_json()
    with pytest.raises(ValidationError, match="Extra inputs"):
        SegFormerEvidenceRecord(
            leaf_category="small_vehicle",
            roi_id="roi-1",
            instance_count=3,
        )


def test_roi_evidence_record_validates_nested_geometry() -> None:
    record = RoiEvidenceRecord(
        roi_id="roi-1",
        image_id="img1",
        source_size=(1000, 800),
        core_xyxy=(100, 100, 200, 200),
        expanded_xyxy=(90, 90, 210, 210),
        crop_size=(120, 120),
    )
    assert record.crop_size == (120, 120)
    with pytest.raises(ValidationError, match="exceeds"):
        RoiEvidenceRecord(
            roi_id="r", image_id="i",
            source_size=(1000, 800),
            core_xyxy=(100, 100, 200, 200),
            expanded_xyxy=(90, 90, 1100, 210),
            crop_size=(1010, 120),
        )
    with pytest.raises(ValidationError, match="inside expanded"):
        RoiEvidenceRecord(
            roi_id="r", image_id="i",
            source_size=(1000, 800),
            core_xyxy=(50, 100, 200, 200),
            expanded_xyxy=(90, 90, 210, 210),
            crop_size=(120, 120),
        )
    with pytest.raises(ValidationError, match="crop_size width"):
        RoiEvidenceRecord(
            roi_id="r", image_id="i",
            source_size=(1000, 800),
            core_xyxy=(100, 100, 200, 200),
            expanded_xyxy=(90, 90, 210, 210),
            crop_size=(200, 120),
        )


def test_layer_state_record_validates_not_run_placement() -> None:
    record = LayerStateRecord(leaf_category="x", layer="segformer", state="not_run")
    assert record.state == "not_run"
    with pytest.raises(ValidationError, match="not_run"):
        LayerStateRecord(leaf_category="x", layer="yolo", state="not_run")


def test_model_call_audit_requires_error_code_on_failure() -> None:
    ok = ModelCallAudit(
        layer="yolo",
        roi_id="roi-1",
        input_size=(1024, 1024),
        logical_model_id="det-a",
        weights_sha256="a" * 64,
    )
    assert ok.status == "succeeded"
    with pytest.raises(ValidationError, match="error_code"):
        ModelCallAudit(
            layer="yolo",
            roi_id="roi-1",
            input_size=(1024, 1024),
            logical_model_id="det-a",
            status="failed",
        )
    with pytest.raises(ValidationError, match="must not carry"):
        ModelCallAudit(
            layer="yolo",
            roi_id="roi-1",
            input_size=(1024, 1024),
            logical_model_id="det-a",
            error_code="boom",
        )


def _bundle(**overrides) -> VqaEvidenceBundle:
    data = {
        "catalog_version": "test-catalog-v1",
        "rois": [
            RoiEvidenceRecord(
                roi_id="roi-1",
                image_id="img1",
                source_size=(1000, 800),
                core_xyxy=(100, 100, 200, 200),
                expanded_xyxy=(90, 90, 210, 210),
                crop_size=(120, 120),
            )
        ],
        "detections": [
            YoloDetectionRecord(
                leaf_category="small_vehicle",
                roi_id="roi-1",
                local_xyxy=(10.0, 20.0, 60.0, 70.0),
                local_roi_size=(120, 120),
                global_xyxy=(210.0, 320.0, 260.0, 370.0),
                global_image_size=(1000, 800),
            )
        ],
        "segments": [
            SegFormerEvidenceRecord(leaf_category="large_vehicle", roi_id="roi-1")
        ],
        "missing_leaves": [],
        "leaf_states": {
            "small_vehicle": "hit",
            "large_vehicle": "hit",
        },
    }
    data.update(overrides)
    return VqaEvidenceBundle.model_validate(data)


def test_bundle_validates_cross_references() -> None:
    bundle = _bundle()
    assert bundle.workflow == "object_evidence_vqa"
    assert bundle.leaf_states["small_vehicle"] == "hit"
    bundle.model_dump_json()  # fully JSON-safe / 完全 JSON 安全

    with pytest.raises(ValidationError, match="unknown roi_id"):
        _bundle(
            detections=[
                YoloDetectionRecord(
                    leaf_category="x",
                    roi_id="ghost",
                    local_xyxy=(0.0, 0.0, 10.0, 10.0),
                    local_roi_size=(120, 120),
                    global_xyxy=(0.0, 0.0, 10.0, 10.0),
                    global_image_size=(1000, 800),
                )
            ]
        )


def test_bundle_rejects_hit_leaf_in_missing_leaves() -> None:
    with pytest.raises(ValidationError, match="must not appear in missing_leaves"):
        _bundle(
            missing_leaves=["small_vehicle"],
            leaf_states={"small_vehicle": "hit"},
        )


def test_bundle_requires_leaf_states_for_missing_leaves() -> None:
    with pytest.raises(ValidationError, match="absent from leaf_states"):
        _bundle(
            missing_leaves=["large_vehicle"],
            leaf_states={"small_vehicle": "hit"},
        )


def test_bundle_has_no_valid_empty_state() -> None:
    """A successful-but-empty filter is missing, never a valid_empty state.
    成功但筛选为空属于 missing，绝不存在 valid_empty 状态。"""
    assert "valid_empty" not in EvidenceState.__args__


def test_bundle_json_is_free_of_unsafe_payloads() -> None:
    text = _bundle().model_dump_json()
    for token in ("data:image", "base64", "sk-", "secret", "C:\\", "/Users"):
        assert token not in text


# ── Tile schema (14.7.1) / tile 契约 ─────────────────────────────────────


def _tile(**overrides) -> EvidenceTileRecord:
    data = {
        "tile_id": "roi-1-r0-c1",
        "roi_id": "roi-1",
        "row": 0,
        "column": 1,
        "source_tile_xyxy": (1024, 0, 2000, 1024),
        "source_tile_size": (976, 1024),
        "scale_x": 1024 / 976,
        "scale_y": 1.0,
        "resize_applied": True,
    }
    data.update(overrides)
    return EvidenceTileRecord(**data)


def test_tile_record_holds_remainder_geometry_and_is_json_safe() -> None:
    tile = _tile()
    assert tile.model_input_size == (1024, 1024)
    assert tile.scale_x == pytest.approx(1024 / 976)
    assert tile.resize_applied is True
    payload = tile.model_dump_json()
    assert "tile_id" in payload
    with pytest.raises(ValidationError, match="Extra inputs"):
        _tile(confidence=0.5)


def test_tile_record_full_tile_keeps_scale_one_and_no_resize() -> None:
    tile = _tile(
        tile_id="roi-1-r0-c0",
        column=0,
        source_tile_xyxy=(0, 0, 1024, 1024),
        source_tile_size=(1024, 1024),
        scale_x=1.0,
        scale_y=1.0,
        resize_applied=False,
    )
    assert tile.model_input_size == (1024, 1024)
    assert tile.resize_applied is False


def test_tile_record_rejects_inconsistent_geometry() -> None:
    with pytest.raises(ValidationError, match="non-degenerate"):
        _tile(source_tile_xyxy=(10, 0, 10, 20))
    with pytest.raises(ValidationError, match="must match"):
        _tile(source_tile_xyxy=(1024, 0, 2001, 1024), source_tile_size=(976, 1024))
    with pytest.raises(ValidationError, match="scale_x must equal"):
        _tile(scale_x=1.5)
    with pytest.raises(ValidationError, match="resize_applied must be true"):
        _tile(resize_applied=False)
    with pytest.raises(ValidationError, match="resize_applied must be true"):
        _tile(
            source_tile_xyxy=(0, 0, 1024, 1024),
            source_tile_size=(1024, 1024),
            scale_x=1.0,
            scale_y=1.0,
            resize_applied=True,
        )
    with pytest.raises(ValidationError, match="keep scale 1"):
        _tile(
            tile_id="roi-1-r0-c0",
            column=0,
            source_tile_xyxy=(0, 0, 1024, 1024),
            source_tile_size=(1024, 1024),
            scale_x=1.0,
            scale_y=1.00000000001,
            resize_applied=False,
        )
    with pytest.raises(ValidationError):
        _tile(row=-1)
    with pytest.raises(ValidationError):
        _tile(tile_id="../roi-1-r0-c1")


def test_model_call_audit_accepts_safe_tile_id() -> None:
    ok = ModelCallAudit(
        layer="segformer",
        roi_id="roi-1",
        tile_id="roi-1-r0-c0",
        input_size=(1024, 1024),
        logical_model_id="seg-a",
    )
    assert ok.tile_id == "roi-1-r0-c0"
    with pytest.raises(ValidationError):
        ModelCallAudit(
            layer="segformer",
            roi_id="roi-1",
            tile_id="../roi-1-r0-c0",
            input_size=(1024, 1024),
            logical_model_id="seg-a",
        )


def test_bundle_accepts_legacy_artifacts_without_tile_fields() -> None:
    """Old artifacts may lack preprocessing_version/tiles; fresh executors
    must fill them. 旧 artifact 可以缺少 preprocessing_version/tiles；fresh
    executor 必须填满。"""
    legacy = _bundle()
    assert legacy.preprocessing_version is None
    assert legacy.tiles == []
    filled = _bundle(
        preprocessing_version="greedy-1024-stretch-v1",
        tiles=[_tile(), _tile(tile_id="roi-1-r1-c0", row=1, column=0)],
    )
    assert filled.preprocessing_version == "greedy-1024-stretch-v1"
    assert [tile.tile_id for tile in filled.tiles] == [
        "roi-1-r0-c1",
        "roi-1-r1-c0",
    ]
    filled.model_dump_json()  # fully JSON-safe / 完全 JSON 安全


# ── SegFormer pad preprocessing (26 §3.1/§5.2) / SegFormer pad 预处理 ─────


def _pad_record(**overrides) -> SegFormerPreprocessRecord:
    data = {
        "roi_id": "roi-1",
        "source_size": (2000, 1536),
        "padded_size": (2048, 2048),
        "padding_right": 48,
        "padding_bottom": 512,
        "scale_x": 1024 / 2048,
        "scale_y": 1024 / 2048,
    }
    data.update(overrides)
    return SegFormerPreprocessRecord(**data)


@pytest.mark.parametrize(
    "source_size,padded_size,padding_right,padding_bottom",
    [
        ((1024, 1024), (1024, 1024), 0, 0),
        ((1024, 2048), (1024, 2048), 0, 0),
        ((976, 1024), (1024, 1024), 48, 0),
        ((1025, 1025), (2048, 2048), 1023, 1023),
        ((2000, 1536), (2048, 2048), 48, 512),
        ((1, 1), (1024, 1024), 1023, 1023),
    ],
)
def test_pad_record_enforces_minimal_ceiling_padding(
    source_size: tuple[int, int],
    padded_size: tuple[int, int],
    padding_right: int,
    padding_bottom: int,
) -> None:
    record = SegFormerPreprocessRecord(
        roi_id="roi-1",
        source_size=source_size,
        padded_size=padded_size,
        padding_right=padding_right,
        padding_bottom=padding_bottom,
        scale_x=1024 / padded_size[0],
        scale_y=1024 / padded_size[1],
    )
    assert record.model_input_size == (1024, 1024)
    assert record.padding_mode == "constant-black-right-bottom"
    assert record.rgb_interpolation == "lanczos"
    assert record.mask_inverse_interpolation == "nearest"
    record.model_dump_json()  # fully JSON-safe / 完全 JSON 安全


def test_pad_record_rejects_over_padding_and_inconsistent_geometry() -> None:
    """The validator recomputes the minimal ceiling padding: over-padding or
    any hidden offset fails closed. validator 重新计算最小上取整 padding：过度
    padding 或任何隐式偏移都严格失败。"""
    with pytest.raises(ValidationError, match="minimal 1024 multiples"):
        SegFormerPreprocessRecord(
            roi_id="r",
            source_size=(1000, 800),
            padded_size=(2048, 1024),
            padding_right=1048,
            padding_bottom=224,
            scale_x=1024 / 2048,
            scale_y=1024 / 1024,
        )
    with pytest.raises(ValidationError, match="minimal 1024 multiples"):
        _pad_record(padded_size=(3072, 2048), padding_right=1072, padding_bottom=512)
    with pytest.raises(ValidationError, match="padding_right must equal"):
        _pad_record(padding_right=47)
    with pytest.raises(ValidationError, match="padding_bottom must equal"):
        _pad_record(padding_bottom=511)
    with pytest.raises(ValidationError, match="scale_x must equal"):
        _pad_record(scale_x=1.0)
    with pytest.raises(ValidationError, match="scale_y must equal"):
        _pad_record(scale_y=0.25)
    with pytest.raises(ValidationError, match="strict 1024x1024"):
        _pad_record(model_input_size=(512, 512))
    with pytest.raises(ValidationError, match="positive"):
        _pad_record(source_size=(0, 1536))
    with pytest.raises(ValidationError, match="positive"):
        _pad_record(source_size=(2000, -1))
    with pytest.raises(ValidationError):
        _pad_record(padding_right=-1)
    with pytest.raises(ValidationError, match="Extra inputs"):
        _pad_record(confidence=0.9)


def test_pad_record_is_json_safe_and_path_free() -> None:
    text = _pad_record().model_dump_json()
    for token in ("data:image", "base64", "sk-", "C:\\", "/Users", "/models"):
        assert token not in text


def test_evidence_preprocessing_defaults_to_fresh_v2_identity() -> None:
    preprocessing = EvidencePreprocessing()
    assert preprocessing.version == "yolo-v1-segformer-pad-v1"
    assert preprocessing.yolo_version == "greedy-1024-stretch-v1"
    assert preprocessing.segformer_version == "pad-multiple-1024-resize-square-v1"
    assert preprocessing.segformer_padding_mode == "constant-black-right-bottom"
    assert preprocessing.segformer_rgb_interpolation == "lanczos"
    assert preprocessing.segformer_mask_inverse_interpolation == "nearest"
    assert preprocessing.max_tile_concurrency == 4


def _v1_preprocessing() -> EvidencePreprocessing:
    return EvidencePreprocessing(
        version="greedy-1024-stretch-v1",
        yolo_version=None,
        segformer_version=None,
        segformer_padding_mode=None,
        segformer_rgb_interpolation=None,
        segformer_mask_inverse_interpolation=None,
    )


def test_evidence_preprocessing_v1_never_carries_v2_fields() -> None:
    assert _v1_preprocessing().version == "greedy-1024-stretch-v1"
    with pytest.raises(ValidationError, match="must not carry v2-only"):
        EvidencePreprocessing(version="greedy-1024-stretch-v1")
    with pytest.raises(ValidationError, match="requires every v2 field"):
        EvidencePreprocessing(segformer_version=None)
    with pytest.raises(ValidationError, match="requires every v2 field"):
        EvidencePreprocessing(
            version="yolo-v1-segformer-pad-v1",
            yolo_version=None,
        )


def test_bundle_carries_and_validates_segformer_preprocess_records() -> None:
    filled = _bundle(segformer_preprocess=[_pad_record()])
    assert [record.roi_id for record in filled.segformer_preprocess] == ["roi-1"]
    filled.model_dump_json()
    # Old artifacts without the field parse with an empty list.
    # 旧 artifact 缺该字段时解析为空列表。
    assert _bundle().segformer_preprocess == []
    with pytest.raises(ValidationError, match="unknown roi_id"):
        _bundle(segformer_preprocess=[_pad_record(roi_id="ghost")])
