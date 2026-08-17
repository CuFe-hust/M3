"""Contract tests for the v3 visual plan and strict VQA evidence schemas.

v3 视觉计划与严格 VQA 证据 schema 契约测试：计划与物化视图保持独立，证据
几何使用源像素坐标，mask 不转框且所有持久化字段 JSON-safe。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.general_vqa.evidence.schema import (
    EvidenceState,
    LayerStateRecord,
    ModelCallAudit,
    RoiEvidenceRecord,
    SegFormerEvidenceRecord,
    VqaEvidenceBundle,
    YoloDetectionRecord,
)
from agents.schema import (
    MaterializedVisualView,
    VisualTaskPlan,
)


def _plan(**overrides) -> VisualTaskPlan:
    data = {
        "version": "visual-task-plan-v4",
        "task": "general_vqa",
        "needs_visual_assistance": True,
        "object_categories": ["small-vehicle"],
        "count_target": None,
        "region_request": {
            "explicit": True,
            "image_index": 0,
            "focus_xy_norm": (0.5, 0.5),
        },
        "reason_codes": ["test"],
    }
    data.update(overrides)
    return VisualTaskPlan.model_validate(data)


def test_visual_task_plan_validates_assistance_linkage() -> None:
    plan = _plan()
    assert plan.version == "visual-task-plan-v4"
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
    with pytest.raises(ValidationError, match="finite"):
        _plan(region_request={"explicit": True, "image_index": 0, "focus_xy_norm": (float("nan"), 0.5)})
    with pytest.raises(ValidationError):
        _plan(version="visual-task-plan-v2")


def test_visual_task_plan_v4_schema_has_count_target_linkage_and_no_confidence() -> None:
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
    with pytest.raises(ValidationError, match="non-counting"):
        _plan(count_target="small-vehicle")


@pytest.mark.parametrize("target", ["", " vehicle", "vehicle ", "../vehicle", "3", "ship\n"])
def test_visual_task_plan_v4_rejects_unsafe_count_target(target: str) -> None:
    with pytest.raises(ValidationError):
        _plan(task="counting", count_target=target)


def test_visual_task_plan_v4_allows_at_most_eight_leaf_categories() -> None:
    categories = [f"leaf-{index}" for index in range(8)]
    assert _plan(object_categories=categories).object_categories == categories
    with pytest.raises(ValidationError):
        _plan(object_categories=[*categories, "leaf-8"])


def test_materialized_view_is_exact_source_pixel_geometry() -> None:
    view = MaterializedVisualView(
        image_id="img1",
        view_mode="fixed_roi",
        source_size=(2048, 1536),
        crop_xyxy=(1024, 512, 2048, 1536),
        crop_size=(1024, 1024),
    )
    assert view.model_dump(mode="json")["crop_xyxy"] == [1024, 512, 2048, 1536]
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
