"""Contract tests for the v2 grounding evidence seam.

v2 定位证据 seam 契约测试：YOLO 与最终 Qwen 只消费已物化源像素视图，输出
稳定的整图 0..999 框，错误与审计字段不泄漏原始异常。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.evidence_catalog import EvidenceCatalog
from agents.grounding.evidence import (
    GroundingEvidenceError,
    GroundingEvidenceExecutor,
    GroundingEvidencePolicy,
    GroundingQwenResponse,
)
from agents.schema import MaterializedVisualView, VisualTaskPlan
from agents.visual_base import PromptBinding
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import ModelCacheIdentity, ObjectDetectionOutput

_CATALOG_DATA = {
    "catalog_version": "test-catalog-v1",
    "composites": {"building": ["building_outline"]},
    "leaves": {
        "building_outline": {
            "yolo_labels": ["building"],
            "yolo_enabled": True,
        }
    },
}


class _FakeYolo:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[Image.Image] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="yolo-test-v1",
            generation={"weights_sha256": "a" * 64},
            client_version="test",
        )

    def detect(self, image: Image.Image, **kwargs) -> list[ObjectDetectionOutput]:
        self.calls.append(image)
        if self.error is not None:
            raise self.error
        return [
            ObjectDetectionOutput(
                label="building",
                confidence=0.9,
                xyxy=(10.0, 10.0, 40.0, 40.0),
                polygon=None,
                input_width=image.width,
                input_height=image.height,
                logical_model_id="yolo-test-v1",
                weights_sha256="a" * 64,
                provider_audit={},
            )
        ]


class _FakeQwen:
    def __init__(self, response: dict[str, Any], *, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(model="qwen-test-v1", generation={"temperature": 0.0}, client_version="test")

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append({"messages": messages, "request_meta": request_meta})
        if self.error is not None:
            raise self.error
        return response_model.model_validate(self.response)


def _sample(tmp_path: Path, size: tuple[int, int] = (100, 80)) -> UnifiedSample:
    Image.new("RGB", size, (1, 2, 3)).save(tmp_path / "img.png", format="PNG")
    return UnifiedSample(
        sample_id="s1",
        dataset="demo",
        split="test",
        task="grounding",
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="Locate the building.",
        ground_truth=GroundTruth(answers=["building"]),
    )


def _plan() -> VisualTaskPlan:
    return VisualTaskPlan(
        version="visual-task-plan-v3",
        task="grounding",
        needs_visual_assistance=True,
        object_categories=["building"],
        reason_codes=["test"],
    )


def _view(
    *,
    mode: str = "full_image",
    box: tuple[int, int, int, int] = (0, 0, 100, 80),
    source_size: tuple[int, int] = (100, 80),
) -> MaterializedVisualView:
    return MaterializedVisualView(
        image_id="img1",
        view_mode=mode,  # type: ignore[arg-type]
        source_size=source_size,
        crop_xyxy=box,
        crop_size=(box[2] - box[0], box[3] - box[1]),
    )


def _executor(qwen, *, yolo=None, policy=None) -> GroundingEvidenceExecutor:
    return GroundingEvidenceExecutor(
        catalog=EvidenceCatalog(_CATALOG_DATA),
        qwen_client=qwen,
        prompt=PromptBinding(text="Locate the requested object.", version="test-v1"),
        policy=policy or GroundingEvidencePolicy(
            confidence_threshold=0.5,
            nms_iou_threshold=0.5,
            max_detections=5,
        ),
        yolo_client=yolo,
        yolo_device="cpu",
        yolo_image_size=640 if yolo is not None else None,
    )


def _run(
    executor,
    tmp_path: Path,
    *,
    view: MaterializedVisualView | None = None,
    source_size: tuple[int, int] = (100, 80),
):
    return asyncio.run(
        executor.run(
            _plan(),
            _sample(tmp_path, source_size),
            {"img1": Image.open(tmp_path / "img.png")},
            fallback_image_id="img1",
            artifact_dir=tmp_path / "artifacts",
            budget=None,
            materialized_views=(view or _view(),),
        )
    )


def test_policy_defaults_explicitly_disable_uncalibrated_detector() -> None:
    policy = GroundingEvidencePolicy()
    assert policy.yolo_enabled is False
    with pytest.raises(ValueError):
        GroundingEvidencePolicy(confidence_threshold=1.1)


def test_v2_executor_calls_each_model_once_and_returns_whole_image_box(tmp_path: Path) -> None:
    yolo = _FakeYolo()
    qwen = _FakeQwen({"selected_box_ids": ["full-box-1"], "fallback_boxes": []})
    result = _run(_executor(qwen, yolo=yolo), tmp_path)

    assert len(yolo.calls) == 1
    assert len(qwen.calls) == 1
    assert result.whole_image_boxes[0].label == "building_outline"
    assert result.whole_image_boxes[0].box == (100, 125, 400, 500)
    payload = result.bundle.model_dump_json()
    assert "confidence" not in payload
    assert "base64" not in payload


def test_v2_executor_uses_exact_fixed_roi_pixels(tmp_path: Path) -> None:
    yolo = _FakeYolo()
    qwen = _FakeQwen({"selected_box_ids": ["fixed_roi-0-box-1"], "fallback_boxes": []})
    view = _view(
        mode="fixed_roi",
        box=(1024, 512, 2048, 1536),
        source_size=(2048, 1536),
    )
    result = _run(
        _executor(qwen, yolo=yolo),
        tmp_path,
        view=view,
        source_size=(2048, 1536),
    )
    assert yolo.calls[0].size == (1024, 1024)
    assert result.bundle.rois[0].core_xyxy == (1024, 512, 2048, 1536)


def test_uncalibrated_detector_allows_authorized_missing_leaf_fallback(tmp_path: Path) -> None:
    qwen = _FakeQwen(
        {
            "selected_box_ids": [],
            "fallback_boxes": [
                {"leaf_category": "building_outline", "roi_id": "full", "xyxy": [100, 100, 500, 500]}
            ],
        }
    )
    result = _run(
        _executor(qwen, yolo=None, policy=GroundingEvidencePolicy()),
        tmp_path,
    )
    assert result.bundle.leaf_states["building_outline"] == "unavailable"
    assert result.whole_image_boxes[0].box == (100, 100, 500, 500)


def test_invalid_materialized_view_fails_before_model_calls(tmp_path: Path) -> None:
    yolo = _FakeYolo()
    qwen = _FakeQwen({"selected_box_ids": [], "fallback_boxes": []})
    bad = _view()
    with pytest.raises(GroundingEvidenceError, match="PLAN_INVALID"):
        asyncio.run(
            _executor(qwen, yolo=yolo).run(
                _plan(),
                _sample(tmp_path),
                {"other": Image.open(tmp_path / "img.png")},
                fallback_image_id="img1",
                artifact_dir=tmp_path / "artifacts",
                budget=None,
                materialized_views=(bad,),
            )
        )
    assert yolo.calls == []
    assert qwen.calls == []


def test_qwen_error_is_stable_and_does_not_leak_text(tmp_path: Path) -> None:
    yolo = _FakeYolo()
    qwen = _FakeQwen(
        {"selected_box_ids": ["full-box-1"], "fallback_boxes": []},
        error=RuntimeError("/private/model.pt sk-secret"),
    )
    with pytest.raises(GroundingEvidenceError, match="CLIENT_ERROR") as info:
        _run(_executor(qwen, yolo=yolo), tmp_path)
    assert "/private/model.pt" not in str(info.value)
    assert "sk-secret" not in str(info.value)


def test_grounding_evidence_has_no_vqa_evidence_import() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "grounding" / "evidence.py").read_text(
        encoding="utf-8"
    )
    assert "agents.general_vqa.evidence" not in source
    assert "map_grounding_roi" not in source
