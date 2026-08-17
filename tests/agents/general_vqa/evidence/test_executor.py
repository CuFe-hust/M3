"""Contract tests for the v2 VQA evidence executor.

v2 VQA 证据执行器契约测试：执行器只消费 VisualTaskPlan 与已物化源像素视图，
按视图运行检测，保留稳定状态和审计字段，不回建旧 ROI 计划。
"""

from __future__ import annotations

import dataclasses
import inspect
import json

import pytest
from PIL import Image

from agents.evidence_catalog import EvidenceCatalog
from agents.general_vqa.evidence.executor import EvidencePolicy, ObjectEvidenceExecutor
from agents.schema import MaterializedVisualView, VisualTaskPlan
from models.base import ModelCacheIdentity, ObjectDetectionOutput

_CATALOG_DATA = {
    "catalog_version": "test-catalog-v1",
    "aliases": {},
    "parents": {
        "vehicle": ["small-vehicle", "large-vehicle"],
        "building": ["building-outline"],
    },
    "leaves": {
        "small-vehicle": {"yolo_labels": ["small_vehicle"], "yolo_enabled": True},
        "large-vehicle": {"yolo_labels": ["large_vehicle"], "yolo_enabled": True},
        "building-outline": {"yolo_labels": ["building"], "yolo_enabled": True},
    },
    "task_capabilities": {
        task: ["small-vehicle", "large-vehicle", "building-outline"]
        for task in ("counting", "fine_grained_counting", "general_vqa", "grounding")
    },
}


class _FakeYolo:
    """Return deterministic detections and record the exact crop sizes.
    返回确定性检测，并记录每次收到的精确裁切尺寸。"""

    def __init__(self, labels: tuple[str, ...], *, error: Exception | None = None) -> None:
        self.labels = labels
        self.error = error
        self.calls: list[Image.Image] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="yolo-test-v1",
            generation={"weights_sha256": "a" * 64},
            client_version="test",
        )

    def detect(
        self,
        image: Image.Image,
        *,
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
        max_detections: int,
    ) -> list[ObjectDetectionOutput]:
        self.calls.append(image)
        if self.error is not None:
            raise self.error
        return [
            ObjectDetectionOutput(
                label=label,
                confidence=0.9,
                xyxy=(10.0, 10.0, 40.0, 40.0),
                polygon=None,
                input_width=image.width,
                input_height=image.height,
                logical_model_id="yolo-test-v1",
                weights_sha256="a" * 64,
                provider_audit={},
            )
            for label in self.labels
        ]


def _image(size: tuple[int, int] = (1000, 800)) -> Image.Image:
    return Image.new("RGB", size, (7, 8, 9))


def _plan(
    categories: tuple[str, ...] = (
        "small-vehicle", "large-vehicle", "building-outline"
    ),
) -> VisualTaskPlan:
    return VisualTaskPlan(
        version="visual-task-plan-v4",
        task="general_vqa",
        needs_visual_assistance=True,
        object_categories=list(categories),
        reason_codes=["test"],
    )


def _view(
    *,
    image_id: str = "img1",
    source_size: tuple[int, int] = (1000, 800),
    mode: str = "full_image",
    box: tuple[int, int, int, int] | None = None,
) -> MaterializedVisualView:
    crop = box or (0, 0, *source_size)
    return MaterializedVisualView(
        image_id=image_id,
        view_mode=mode,  # type: ignore[arg-type]
        source_size=source_size,
        crop_xyxy=crop,
        crop_size=(crop[2] - crop[0], crop[3] - crop[1]),
    )


def _policy(**overrides) -> EvidencePolicy:
    values = {
        "confidence_threshold": 0.5,
        "nms_iou_threshold": 0.5,
        "max_detections": 5,
    }
    values.update(overrides)
    return EvidencePolicy(**values)


def _executor(
    *,
    yolo: _FakeYolo | None = None,
    policy: EvidencePolicy | None = None,
) -> ObjectEvidenceExecutor:
    return ObjectEvidenceExecutor(
        catalog=EvidenceCatalog(_CATALOG_DATA),
        policy=policy or _policy(),
        yolo_client=yolo,
        yolo_device="cpu",
        yolo_image_size=640,
        segformer_client=None,
    )


def _execute(
    executor: ObjectEvidenceExecutor,
    *,
    plan: VisualTaskPlan | None = None,
    images: dict[str, Image.Image] | None = None,
    views: tuple[MaterializedVisualView, ...] | None = None,
):
    actual_images = images or {"img1": _image()}
    actual_views = views or (_view(),)
    return executor.execute(
        plan or _plan(),
        actual_images,
        fallback_image_id="img1",
        materialized_views=actual_views,
    )


def test_policy_is_inject_only_and_rejects_invalid_values() -> None:
    for field in dataclasses.fields(EvidencePolicy):
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING
    with pytest.raises(ValueError):
        _policy(confidence_threshold=1.1)
    with pytest.raises(ValueError):
        _policy(max_detections=0)


def test_executor_signature_consumes_only_v2_plan_and_views() -> None:
    signature = inspect.signature(ObjectEvidenceExecutor.execute)
    assert set(signature.parameters) == {
        "self",
        "plan",
        "images",
        "fallback_image_id",
        "materialized_views",
    }


def test_yolo_runs_once_per_materialized_view_and_keeps_requested_leaves() -> None:
    yolo = _FakeYolo(("small_vehicle", "large_vehicle", "building"))
    execution = _execute(_executor(yolo=yolo))
    assert len(yolo.calls) == 1
    assert yolo.calls[0].size == (1000, 800)
    assert set(execution.bundle.leaf_states) == {
        "small-vehicle",
        "large-vehicle",
        "building-outline",
    }
    assert all(state == "hit" for state in execution.bundle.leaf_states.values())
    assert execution.bundle.rois[0].expanded_xyxy == (0, 0, 1000, 800)


def test_executor_consumes_exact_fixed_roi_pixels() -> None:
    yolo = _FakeYolo(("small_vehicle", "large_vehicle"))
    view = _view(
        source_size=(2048, 1536),
        mode="fixed_roi",
        box=(1024, 512, 2048, 1536),
    )
    execution = _execute(
        _executor(yolo=yolo),
        plan=_plan(("small-vehicle", "large-vehicle")),
        images={"img1": _image((2048, 1536))},
        views=(view,),
    )
    assert yolo.calls[0].size == (1024, 1024)
    assert execution.bundle.rois[0].core_xyxy == (1024, 512, 2048, 1536)


def test_executor_requires_assistance_and_materialized_views() -> None:
    direct = VisualTaskPlan(
        version="visual-task-plan-v4",
        task="general_vqa",
    )
    with pytest.raises(ValueError, match="visual assistance"):
        _execute(_executor(), plan=direct)
    with pytest.raises(ValueError, match="materialized views"):
        _executor(yolo=_FakeYolo(("small_vehicle",))).execute(
            _plan(("small-vehicle", "large-vehicle")),
            {"img1": _image()},
            fallback_image_id="img1",
            materialized_views=(),
        )


def test_unknown_materialized_image_fails_before_model_call() -> None:
    yolo = _FakeYolo(("small_vehicle",))
    ghost = _view(image_id="ghost")
    with pytest.raises(ValueError, match="source size"):
        _execute(_executor(yolo=yolo), views=(ghost,))
    assert yolo.calls == []


def test_detector_error_is_stable_and_does_not_leak_exception_text() -> None:
    yolo = _FakeYolo(("small_vehicle",), error=RuntimeError("/private/model.pt secret"))
    execution = _execute(
        _executor(yolo=yolo),
        plan=_plan(("small-vehicle", "large-vehicle")),
    )
    dumped = execution.bundle.model_dump_json()
    assert "private/model.pt" not in dumped
    assert "secret" not in dumped
    assert execution.bundle.leaf_states["small-vehicle"] == "unsupported"
    assert any(
        state.leaf_category == "small-vehicle"
        and state.layer == "yolo"
        and state.state == "error"
        for state in execution.layer_states
    )


def test_bundle_is_json_safe_and_does_not_persist_confidence() -> None:
    execution = _execute(
        _executor(yolo=_FakeYolo(("small_vehicle",))),
        plan=_plan(("small-vehicle", "large-vehicle")),
    )
    payload = json.loads(execution.bundle.model_dump_json())
    assert "confidence" not in json.dumps(payload)
    assert "base64" not in json.dumps(payload)
    assert execution.masks == {}
