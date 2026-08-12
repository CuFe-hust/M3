"""Contract tests for the shared first-Qwen plan schema and the strict VQA
evidence schema.

共享第一次 Qwen 计划 schema 与严格 VQA 证据 schema 的契约测试：strict
fields、版本/类别/ROI 约束、required 联动、missing 语义、confidence 隔离、
mask 不转框与 JSON 安全。
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
    FirstQwenVisualPlan,
    ObjectEvidenceRequest,
    RoiPlan,
    RoiRegion,
)


def _region(roi_id: str = "roi-1", xyxy=(0.0, 0.0, 0.5, 0.5)) -> RoiRegion:
    return RoiRegion(roi_id=roi_id, image_id="img1", xyxy=xyxy)


def _plan(**overrides) -> FirstQwenVisualPlan:
    data = {
        "version": "first-qwen-plan-v1",
        "execution_family": "object_evidence_vqa",
        "confidence": 0.9,
        "roi_plan": RoiPlan(rois=[_region()]),
        "evidence_request": ObjectEvidenceRequest(
            composite_categories=["vehicle"]
        ),
    }
    data.update(overrides)
    return FirstQwenVisualPlan.model_validate(data)


# ── 共享计划 schema / shared plan schema ────────────────────────────────


def test_plan_accepts_valid_object_evidence() -> None:
    plan = _plan()
    assert plan.version == "first-qwen-plan-v1"
    assert plan.execution_family == "object_evidence_vqa"
    assert plan.roi_plan.rois[0].xyxy == (0.0, 0.0, 0.5, 0.5)
    assert plan.evidence_request is not None
    assert plan.evidence_request.composite_categories == ["vehicle"]


def test_plan_accepts_direct_vqa_without_evidence() -> None:
    plan = _plan(
        execution_family="direct_vqa",
        evidence_request=None,
        roi_plan=RoiPlan(rois=[]),
    )
    assert plan.execution_family == "direct_vqa"
    assert plan.evidence_request is None
    # Empty ROI plan means no spatial constraint; geometry maps to full image.
    # 空 ROI 计划表示无空间约束；几何层映射为整图。
    assert plan.roi_plan.rois == []


def test_plan_accepts_full_image_roi() -> None:
    plan = _plan(roi_plan=RoiPlan(rois=[_region("full", (0.0, 0.0, 1.0, 1.0))]))
    assert plan.roi_plan.rois[0].xyxy == (0.0, 0.0, 1.0, 1.0)


def test_plan_rejects_wrong_version() -> None:
    with pytest.raises(ValidationError, match="version"):
        _plan(version="second-qwen-plan-v2")


def test_plan_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _plan(backend="ultralytics", checkpoint="/models/det.pt", answer="42")


def test_plan_rejects_family_evidence_mismatch() -> None:
    with pytest.raises(ValidationError, match="requires an evidence_request"):
        _plan(execution_family="object_evidence_vqa", evidence_request=None)
    with pytest.raises(ValidationError, match="must not carry"):
        _plan(
            execution_family="direct_vqa",
            evidence_request=ObjectEvidenceRequest(composite_categories=["vehicle"]),
        )


def test_plan_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        _plan(confidence=1.5)
    with pytest.raises(ValidationError):
        _plan(confidence=-0.1)


def test_plan_rejects_more_than_three_composite_categories() -> None:
    with pytest.raises(ValidationError):
        ObjectEvidenceRequest(
            composite_categories=["a", "b", "c", "d"]
        )


def test_plan_rejects_empty_evidence_request() -> None:
    with pytest.raises(ValidationError):
        ObjectEvidenceRequest(composite_categories=[])


def test_plan_rejects_path_like_category() -> None:
    with pytest.raises(ValidationError, match="path-like"):
        ObjectEvidenceRequest(composite_categories=["/home/user/vehicle"])


def test_plan_accepts_more_than_three_rois() -> None:
    """The count is not capped at the schema: 14B §6.2 lets the planner
    collapse an over-limit plan to the unique full-image ROI instead of
    rejecting it. 数量不在 schema 封顶：14B §6.2 让规划器把超限计划折叠为
    唯一整图 ROI 而非拒绝。"""
    rois = [_region(f"roi-{index}") for index in range(1, 4)]
    rois.append(_region("roi-4"))
    assert len(RoiPlan(rois=rois).rois) == 4


def test_plan_accepts_degenerate_and_out_of_range_rois() -> None:
    """Geometric validity is decided by the planner (14B §6.2 full-image
    fallback), so the schema parses finite out-of-range or degenerate boxes.
    几何合法性由规划器决定（14B §6.2 整图回退），因此 schema 接受有限的越界
    或退化框。"""
    assert _region("d", (0.2, 0.2, 0.2, 0.6)).xyxy == (0.2, 0.2, 0.2, 0.6)
    assert _region("d", (0.2, 0.2, 0.6, 0.2)).xyxy == (0.2, 0.2, 0.6, 0.2)
    assert _region("o", (0.0, 0.0, 1.1, 1.0)).xyxy == (0.0, 0.0, 1.1, 1.0)
    assert _region("o", (-0.1, 0.0, 0.5, 0.5)).xyxy == (-0.1, 0.0, 0.5, 0.5)


def test_plan_rejects_non_finite_roi_coordinates() -> None:
    with pytest.raises(ValidationError, match="finite"):
        _region("n", (0.0, 0.0, float("nan"), 0.5))
    with pytest.raises(ValidationError, match="finite"):
        _region("n", (0.0, 0.0, float("inf"), 0.5))


def test_plan_rejects_bad_roi_id_and_empty_image_id() -> None:
    with pytest.raises(ValidationError):
        _region("bad id")
    with pytest.raises(ValidationError):
        RoiRegion(roi_id="roi-1", image_id="", xyxy=(0.0, 0.0, 0.5, 0.5))


def test_plan_never_carries_backend_or_answer_fields() -> None:
    plan = _plan()
    dumped = plan.model_dump()
    for forbidden in ("backend", "checkpoint", "device", "answer", "box_id", "mask"):
        assert forbidden not in dumped
    plan.model_dump_json()  # JSON-safe / JSON 安全


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
