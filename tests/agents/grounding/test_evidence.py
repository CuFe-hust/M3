"""Fake-client contract tests for the grounding evidence seam (C6, 14A2).

C6 Grounding 证据 seam 的 fake-client 契约测试（只遵守 14C）：单层
YOLO→最终 Qwen、YOLO-hit 叶子只能选 box_id / 缺失叶子可自由框、权限强制与
稳定丢弃计数、ROI-local [0,1]→整图 0..999 的确定性后处理、未校准能力关闭、
confidence 绝不进入 bundle、稳定错误码与逐 ROI 独立性。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from agents.evidence_catalog import EvidenceCatalog
from agents.grounding.evidence import (
    GroundingEvidenceBundle,
    GroundingEvidenceError,
    GroundingEvidenceExecutor,
    GroundingEvidencePolicy,
    GroundingQwenResponse,
    map_grounding_roi,
)
from agents.schema import (
    FirstQwenVisualPlan,
    ObjectEvidenceRequest,
    RoiPlan,
    RoiRegion,
)
from agents.visual_base import PromptBinding
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import (
    ModelCacheIdentity,
    ObjectDetectionOutput,
)
from models.images import crop_image_region

_CATALOG_DATA = {
    "catalog_version": "test-catalog-v1",
    "composites": {
        "vehicle": ["small_vehicle", "large_vehicle"],
        "building": ["building_outline"],
    },
    "leaves": {
        "small_vehicle": {
            "yolo_labels": ["small_vehicle"],
            "yolo_enabled": True,
        },
        "large_vehicle": {
            "yolo_labels": ["large_vehicle"],
            "yolo_enabled": True,
        },
        "building_outline": {
            "yolo_labels": ["building"],
            "yolo_enabled": True,
        },
    },
}

_YOLO_ID = "yolo-test-v1"
_YOLO_DIGEST = "a" * 64
_QWEN_ID = "qwen-test-v1"
_QWEN_DIGEST = "c" * 64

# Ordered leaf expansion of vehicle + building / vehicle 与 building 的有序叶子展开。
_ALL_LEAVES = ("small_vehicle", "large_vehicle", "building_outline")


class _FakeYolo:
    """Records every call, returns per-call per-label detections, and can
    raise typed errors on selected call indices (1-based) — same contract as
    the VQA evidence tests. 记录每次调用，按调用序返回逐标签检测，并可在指定
    调用序号（1 起）抛类型化错误——与 VQA 证据测试同契约。"""

    def __init__(
        self,
        responses: dict[str, list[tuple[float, tuple[float, float, float, float]]]]
        | list[dict[str, list[tuple[float, tuple[float, float, float, float]]]]],
        *,
        fail_calls: set[int] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = [responses] if isinstance(responses, dict) else responses
        self.fail_calls = set(fail_calls or ())
        self.error = error
        self.calls: list[tuple[Image.Image, dict[str, object]]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model=_YOLO_ID,
            generation={"weights_sha256": _YOLO_DIGEST},
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
        self.calls.append(
            (
                image,
                {
                    "confidence": confidence,
                    "iou": iou,
                    "image_size": image_size,
                    "device": device,
                    "max_detections": max_detections,
                },
            )
        )
        if self.error is not None and (
            not self.fail_calls or len(self.calls) in self.fail_calls
        ):
            raise self.error
        index = min(len(self.calls), len(self.responses)) - 1
        outputs: list[ObjectDetectionOutput] = []
        for label, detections in self.responses[index].items():
            for score, xyxy in detections:
                outputs.append(
                    ObjectDetectionOutput(
                        label=label,
                        confidence=score,
                        xyxy=xyxy,
                        polygon=None,
                        input_width=image.width,
                        input_height=image.height,
                        logical_model_id=_YOLO_ID,
                        weights_sha256=_YOLO_DIGEST,
                        provider_audit={},
                    )
                )
        return outputs


class _FakeQwen:
    """Records every call; validates the configured response against the
    request's response_model so schema-invalid mode raises the same
    ValidationError the live client would. 记录每次调用；对配置响应按请求的
    response_model 校验，使 schema-invalid 模式抛出与 live client 相同的
    ValidationError。"""

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        *,
        error: Exception | None = None,
        schema_invalid: bool = False,
        no_identity: bool = False,
    ) -> None:
        self.response = response or {"selected_box_ids": [], "fallback_boxes": []}
        self.error = error
        self.schema_invalid = schema_invalid
        self.no_identity = no_identity
        self.calls: list[dict[str, Any]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity | None:
        if self.no_identity:
            return None
        return ModelCacheIdentity(
            model=_QWEN_ID,
            generation={"weights_sha256": _QWEN_DIGEST},
            client_version="test",
        )

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[Any],
        request_meta: Any,
        max_tokens: int | None = None,
    ) -> Any:
        self.calls.append({"messages": messages, "request_meta": request_meta})
        if self.schema_invalid:
            return response_model.model_validate({"selected_box_ids": 123})
        if self.error is not None:
            raise self.error
        return response_model.model_validate(self.response)


class _FakeBudget:
    def __init__(self, *, exhausted: bool = False) -> None:
        self.qwen_calls = 0
        self.exhausted = exhausted

    def reserve_qwen(self) -> None:
        if self.exhausted:
            raise RuntimeError("qwen budget exhausted")
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


# ── helpers / 辅助 ────────────────────────────────────────────────────────


def _image(size: tuple[int, int] = (1000, 800), fill: int = 7) -> Image.Image:
    return Image.new("RGB", size, (fill, fill + 1, fill + 2))


def _roi(roi_id: str, xyxy: tuple[float, float, float, float]) -> RoiRegion:
    return RoiRegion(roi_id=roi_id, image_id="img1", xyxy=xyxy)


def _plan(
    categories: tuple[str, ...] = ("vehicle", "building"),
    *,
    rois: list[RoiRegion] | None = None,
) -> FirstQwenVisualPlan:
    return FirstQwenVisualPlan(
        version="first-qwen-plan-v1",
        execution_family="object_evidence_vqa",
        confidence=0.95,
        roi_plan=RoiPlan(rois=rois or []),
        evidence_request=ObjectEvidenceRequest(composite_categories=list(categories)),
        reason_codes=[],
    )


def _policy(**overrides) -> GroundingEvidencePolicy:
    kwargs = {
        "confidence_threshold": 0.5,
        "nms_iou_threshold": 0.5,
        "max_detections": 5,
    }
    kwargs.update(overrides)
    return GroundingEvidencePolicy(**kwargs)


def _executor(
    catalog: EvidenceCatalog,
    *,
    qwen: _FakeQwen | None = None,
    yolo: _FakeYolo | None = None,
    policy: GroundingEvidencePolicy | None = None,
    **overrides,
) -> GroundingEvidenceExecutor:
    kwargs = {
        "catalog": catalog,
        "qwen_client": qwen or _FakeQwen(),
        "prompt": PromptBinding(text="final grounding prompt", version="grounding_v1"),
        "policy": policy or _policy(),
        "yolo_client": yolo,
        "yolo_device": "cpu",
        "yolo_image_size": 640,
    }
    kwargs.update(overrides)
    return GroundingEvidenceExecutor(**kwargs)


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="grounding",
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="Locate vehicles and buildings.",
        ground_truth=GroundTruth(answers=["vehicle"], boxes=[[300, 300, 400, 400]]),
    )


def _run(
    executor: GroundingEvidenceExecutor,
    artifact_dir: Path,
    *,
    rois: list[RoiRegion] | None = None,
    categories: tuple[str, ...] = ("vehicle", "building"),
    images: dict[str, Image.Image] | None = None,
    budget: _FakeBudget | None = None,
    sample: UnifiedSample | None = None,
):
    return asyncio.run(
        executor.run(
            _plan(categories, rois=rois),
            sample or _sample(),
            images or {"img1": _image()},
            fallback_image_id="img1",
            artifact_dir=artifact_dir,
            budget=budget,
        )
    )


def _bundle_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(value)
        for item in value.values():
            keys |= _bundle_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys |= _bundle_keys(item)
    return keys


# ── policy 与构造 / policy and construction ───────────────────────────────


def test_policy_none_defaults_mean_capability_disabled() -> None:
    """The unfrozen parameters are inject-only: every field defaults to None
    meaning "not calibrated"; the YOLO phase is active only when a threshold
    is provided. 未冻结参数仅可注入：每个字段默认 None 表示“未校准”；仅当提供
    阈值时 YOLO 阶段才激活。"""
    policy = GroundingEvidencePolicy()
    assert policy.confidence_threshold is None
    assert policy.nms_iou_threshold is None
    assert policy.max_detections is None
    assert policy.yolo_enabled is False
    enabled = GroundingEvidencePolicy(confidence_threshold=0.5)
    assert enabled.yolo_enabled is True


def test_policy_rejects_invalid_values() -> None:
    for bad in (
        {"confidence_threshold": 1.5},
        {"confidence_threshold": -0.1},
        {"confidence_threshold": float("nan")},
        {"nms_iou_threshold": 1.1},
        {"nms_iou_threshold": float("inf")},
        {"max_detections": 0},
    ):
        with pytest.raises(ValueError):
            _policy(**bad)


def test_executor_rejects_invalid_call_parameters() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    with pytest.raises(ValueError):
        _executor(catalog, halo_ratio=-0.1)
    with pytest.raises(ValueError):
        _executor(catalog, max_side=0)
    with pytest.raises(ValueError):
        _executor(catalog, yolo_image_size=0)
    # The image size is required when the YOLO phase is enabled.
    # YOLO 阶段启用时必须提供图像尺寸。
    with pytest.raises(ValueError, match="yolo_image_size"):
        _executor(catalog, yolo_image_size=None)


# ── 命中路径 / happy path ─────────────────────────────────────────────────


def test_hit_leaves_selected_via_box_ids_and_converted(tmp_path: Path) -> None:
    """All leaves hit at YOLO; the final Qwen selects existing box_ids only and
    the deterministic postprocess converts ROI-local [0,1] to whole-image
    normalized_0_999_top_left. 所有叶子在 YOLO 命中；最终 Qwen 只能选择已有
    box_id，确定性后处理将 ROI-local [0,1] 转为整图 0..999。"""
    catalog = EvidenceCatalog(_CATALOG_DATA)
    qwen = _FakeQwen(
        {
            "selected_box_ids": ["roi-1-box-1", "roi-1-box-2", "roi-1-box-3"],
            "fallback_boxes": [],
        }
    )
    budget = _FakeBudget()
    yolo = _FakeYolo(
        {
            "small_vehicle": [(0.9, (100, 80, 200, 160))],
            "large_vehicle": [(0.7, (500, 300, 600, 400))],
            "building": [(0.8, (300, 200, 400, 300))],
        }
    )
    result = _run(
        _executor(catalog, qwen=qwen, yolo=yolo),
        tmp_path,
        rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))],
        budget=budget,
    )
    assert len(yolo.calls) == 1
    assert len(qwen.calls) == 1
    assert budget.qwen_calls == 1
    bundle = result.bundle
    assert isinstance(bundle, GroundingEvidenceBundle)
    assert bundle.missing_leaves == []
    assert bundle.leaf_states == {
        "small_vehicle": "hit",
        "large_vehicle": "hit",
        "building_outline": "hit",
    }
    assert bundle.selected_box_ids == ["roi-1-box-1", "roi-1-box-2", "roi-1-box-3"]
    assert bundle.fallback_boxes == []
    # box_ids are assigned in the deterministic dedup order: confidence
    # descending (stable), then ROI/leaf order. 0.9 > 0.8 > 0.7 so the
    # candidate order is small_vehicle -> building_outline -> large_vehicle.
    # box_id 按确定性去重顺序分配：置信度降序（稳定），再 ROI/叶子顺序。
    # 0.9 > 0.8 > 0.7，因此候选顺序为 small_vehicle -> building_outline
    # -> large_vehicle。
    assert [candidate.box_id for candidate in bundle.candidates] == [
        "roi-1-box-1",
        "roi-1-box-2",
        "roi-1-box-3",
    ]
    assert [candidate.leaf_category for candidate in bundle.candidates] == [
        "small_vehicle",
        "building_outline",
        "large_vehicle",
    ]
    # ROI (0.25,0.25,0.75,0.75) on 1000x800 maps to expanded (200,160,800,640).
    # 1000x800 上 ROI (0.25,0.25,0.75,0.75) 映射到 expanded (200,160,800,640)。
    assert bundle.rois[0].core_xyxy == (250, 200, 750, 600)
    assert bundle.rois[0].expanded_xyxy == (200, 160, 800, 640)
    assert bundle.rois[0].crop_size == (600, 480)
    # Conversion is whole-image pixels -> normalized_0_999_top_left.
    # 转换是整图像素 -> normalized_0_999_top_left。
    assert [(box.label, box.box) for box in result.whole_image_boxes] == [
        ("small_vehicle", (300, 300, 400, 400)),
        ("building_outline", (500, 450, 599, 574)),
        ("large_vehicle", (699, 574, 799, 699)),
    ]
    # Audits: one YOLO call plus the single final Qwen call.
    # 审计：一次 YOLO 调用加唯一一次最终 Qwen 调用。
    assert [(a.layer, a.status) for a in bundle.call_audit] == [
        ("yolo", "succeeded"),
        ("final_qwen", "succeeded"),
    ]
    assert bundle.call_audit[-1].logical_model_id == _QWEN_ID
    # The request hash covers the messages and the final prompt version.
    # request hash 覆盖消息与最终 prompt 版本。
    assert qwen.calls[0]["request_meta"].prompt_version == "grounding_v1"


def test_missing_leaf_free_box_authority(tmp_path: Path) -> None:
    """A leaf without any retained candidate stays missing and the final Qwen
    may box it directly; the free box is converted to the same whole-image
    frame. 无保留候选的叶子保持缺失，最终 Qwen 可直接框选；自由框同样转为
    整图坐标系。"""
    catalog = EvidenceCatalog(_CATALOG_DATA)
    qwen = _FakeQwen(
        {
            "selected_box_ids": ["roi-1-box-1"],
            "fallback_boxes": [
                {
                    "leaf_category": "large_vehicle",
                    "roi_id": "roi-1",
                    "bbox": [0.5, 0.5, 0.6, 0.6],
                },
                {
                    "leaf_category": "building_outline",
                    "roi_id": "roi-1",
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                },
            ],
        }
    )
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    result = _run(
        _executor(catalog, qwen=qwen, yolo=yolo),
        tmp_path,
        rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))],
    )
    bundle = result.bundle
    assert bundle.leaf_states == {
        "small_vehicle": "hit",
        "large_vehicle": "missing",
        "building_outline": "missing",
    }
    assert bundle.missing_leaves == ["large_vehicle", "building_outline"]
    assert [box.model_dump() for box in bundle.fallback_boxes] == [
        {
            "leaf_category": "large_vehicle",
            "roi_id": "roi-1",
            "bbox": (0.5, 0.5, 0.6, 0.6),
        },
        {
            "leaf_category": "building_outline",
            "roi_id": "roi-1",
            "bbox": (0.1, 0.1, 0.2, 0.2),
        },
    ]
    # Selected candidates first, then fallback boxes in response order.
    # 已选候选在前，自由框按响应顺序在后。
    assert [(box.label, box.box) for box in result.whole_image_boxes] == [
        ("small_vehicle", (300, 300, 400, 400)),
        ("large_vehicle", (500, 500, 559, 559)),
        ("building_outline", (260, 260, 320, 320)),
    ]
    assert bundle.dropped == {}


def test_yolo_disabled_capability_off_all_leaves_fallback(tmp_path: Path) -> None:
    """Uncalibrated detector policy disables the YOLO phase entirely: the
    detector is never called, every leaf falls back to the final-Qwen visual
    path, and no YOLO audit entry is fabricated. 未校准检测策略整体关闭 YOLO
    阶段：检测器绝不调用，全部叶子走最终 Qwen 视觉兜底，且不伪造任何 YOLO
    审计记录。"""
    catalog = EvidenceCatalog(_CATALOG_DATA)
    qwen = _FakeQwen(
        {
            "selected_box_ids": [],
            "fallback_boxes": [
                {
                    "leaf_category": "small_vehicle",
                    "roi_id": "roi-1",
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                }
            ],
        }
    )
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    budget = _FakeBudget()
    result = _run(
        _executor(catalog, qwen=qwen, yolo=yolo, policy=GroundingEvidencePolicy()),
        tmp_path,
        rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))],
        budget=budget,
    )
    assert yolo.calls == []
    assert budget.qwen_calls == 1
    bundle = result.bundle
    assert bundle.leaf_states == {
        "small_vehicle": "unavailable",
        "large_vehicle": "unavailable",
        "building_outline": "unavailable",
    }
    assert bundle.missing_leaves == list(_ALL_LEAVES)
    assert bundle.candidates == []
    assert [(a.layer, a.status) for a in bundle.call_audit] == [
        ("final_qwen", "succeeded")
    ]
    assert [(box.label, box.box) for box in result.whole_image_boxes] == [
        ("small_vehicle", (260, 260, 320, 320))
    ]


def test_empty_plan_falls_back_to_unique_full_image_roi(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    qwen = _FakeQwen({"selected_box_ids": ["full-box-1"], "fallback_boxes": []})
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    result = _run(_executor(catalog, qwen=qwen, yolo=yolo), tmp_path)
    bundle = result.bundle
    assert len(bundle.rois) == 1
    assert bundle.rois[0].roi_id == "full"
    assert bundle.rois[0].crop_size == (1000, 800)
    assert yolo.calls[0][0].size == (1000, 800)
    assert bundle.candidates[0].box_id == "full-box-1"
    # Local and global frames coincide for the full-image ROI: the whole-image
    # box is the ROI-local box scaled by 999. 整图 ROI 下局部与全局坐标系一致：
    # 整图框即 ROI-local 框按 999 缩放。
    assert [(box.label, box.box) for box in result.whole_image_boxes] == [
        ("small_vehicle", (round(100 / 1000 * 999), round(80 / 800 * 999),
                           round(200 / 1000 * 999), round(160 / 800 * 999)))
    ]


def test_exactly_one_qwen_call_and_one_budget_reservation(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    qwen = _FakeQwen()
    budget = _FakeBudget()
    with pytest.raises(GroundingEvidenceError) as excinfo:
        _run(
            _executor(catalog, qwen=qwen, policy=GroundingEvidencePolicy()),
            tmp_path,
            budget=budget,
        )
    assert excinfo.value.code == "NO_VALID_BOXES"
    # One attempted call, one consumed budget entry, nothing more.
    # 一次尝试调用、一次消费的 budget，绝无更多。
    assert len(qwen.calls) == 1
    assert budget.qwen_calls == 1


# ── 权限强制 / authority enforcement ──────────────────────────────────────


def test_postprocess_drops_every_out_of_authority_item(tmp_path: Path) -> None:
    """Unknown box ids, duplicated box ids, free boxes for hit/unrequested
    leaves or unknown ROIs, and invalid coordinates are all dropped with
    stable counters. 未知 box_id、重复 box_id、hit/未请求叶子的自由框、未知
    ROI 与非法坐标全部以稳定计数丢弃。"""
    catalog = EvidenceCatalog(_CATALOG_DATA)
    qwen = _FakeQwen(
        {
            "selected_box_ids": ["roi-1-box-1", "roi-1-box-1", "ghost-box"],
            "fallback_boxes": [
                # Hit leaf: out of authority. / hit 叶子：越权。
                {
                    "leaf_category": "small_vehicle",
                    "roi_id": "roi-1",
                    "bbox": [0.3, 0.3, 0.4, 0.4],
                },
                # Leaf never requested. / 从未请求的叶子。
                {
                    "leaf_category": "airplane",
                    "roi_id": "roi-1",
                    "bbox": [0.3, 0.3, 0.4, 0.4],
                },
                # ROI never mapped. / 未映射的 ROI。
                {
                    "leaf_category": "large_vehicle",
                    "roi_id": "ghost",
                    "bbox": [0.3, 0.3, 0.4, 0.4],
                },
                # Degenerate box. / 退化框。
                {
                    "leaf_category": "large_vehicle",
                    "roi_id": "roi-1",
                    "bbox": [0.5, 0.5, 0.5, 0.6],
                },
            ],
        }
    )
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    result = _run(
        _executor(catalog, qwen=qwen, yolo=yolo),
        tmp_path,
        rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))],
    )
    bundle = result.bundle
    assert bundle.selected_box_ids == ["roi-1-box-1"]
    assert bundle.fallback_boxes == []
    assert bundle.dropped == {
        "box_id_duplicate": 1,
        "unknown_box_id": 1,
        "free_box_for_hit_leaf": 1,
        "free_box_unrequested_leaf": 1,
        "free_box_unknown_roi": 1,
        "free_box_invalid_coordinates": 1,
    }
    # small_vehicle stays the only whole-image box.
    # small_vehicle 仍是唯一的整图框。
    assert [(box.label, box.box) for box in result.whole_image_boxes] == [
        ("small_vehicle", (300, 300, 400, 400))
    ]


def test_no_valid_boxes_after_cleanup_is_explicit_failure(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    executor = _executor(catalog, qwen=_FakeQwen(), yolo=yolo)
    with pytest.raises(GroundingEvidenceError) as excinfo:
        _run(executor, tmp_path, rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))])
    assert excinfo.value.code == "NO_VALID_BOXES"


# ── 跨 ROI 去重与逐 ROI 独立 / dedup and per-ROI independence ─────────────


def test_cross_roi_dedup_higher_confidence_wins(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    qwen = _FakeQwen({"selected_box_ids": ["roi-2-box-1"], "fallback_boxes": []})
    yolo = _FakeYolo(
        [
            {"small_vehicle": [(0.6, (100, 80, 200, 160))]},
            {"small_vehicle": [(0.95, (100, 80, 200, 160))]},
        ]
    )
    result = _run(
        _executor(catalog, qwen=qwen, yolo=yolo),
        tmp_path,
        rois=[
            _roi("roi-1", (0.25, 0.25, 0.75, 0.75)),
            _roi("roi-2", (0.25, 0.25, 0.75, 0.75)),
        ],
    )
    # Identical global targets dedup in whole-image coordinates; the higher
    # internal confidence wins regardless of ROI order. 相同全局目标在
    # whole-image 坐标去重；内部置信度更高者胜出，与 ROI 顺序无关。
    assert len(result.bundle.candidates) == 1
    assert result.bundle.candidates[0].box_id == "roi-2-box-1"
    assert result.bundle.leaf_states["small_vehicle"] == "hit"


def test_yolo_error_is_stable_code_never_raw_exception(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        {},
        error=RuntimeError("boom: /Users/troy/secret.pth failed to load"),
    )
    qwen = _FakeQwen(
        {
            "selected_box_ids": [],
            "fallback_boxes": [
                {
                    "leaf_category": "small_vehicle",
                    "roi_id": "roi-1",
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                }
            ],
        }
    )
    result = _run(
        _executor(catalog, qwen=qwen, yolo=yolo),
        tmp_path,
        rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))],
    )
    bundle = result.bundle
    assert bundle.leaf_states["small_vehicle"] == "error"
    assert bundle.leaf_states["large_vehicle"] == "error"
    yolo_audit = [a for a in bundle.call_audit if a.layer == "yolo"]
    assert len(yolo_audit) == 1
    assert yolo_audit[0].status == "failed"
    assert yolo_audit[0].error_code == "RuntimeError"
    dumped = json.dumps(bundle.model_dump())
    assert "boom" not in dumped
    assert "/Users" not in dumped
    assert "secret.pth" not in dumped
    # The error leaves still feed the final-Qwen fallback.
    # 错误叶子仍进入最终 Qwen 回退。
    assert bundle.missing_leaves == list(_ALL_LEAVES)
    assert len(result.whole_image_boxes) == 1


def test_yolo_error_keeps_other_roi_evidence(tmp_path: Path) -> None:
    """Per-ROI independence: one failing ROI keeps every other ROI's evidence,
    and the final Qwen still runs exactly once over the available evidence.
    逐 ROI 独立：单 ROI 失败保留其他 ROI 的成功证据，最终 Qwen 仍只调用一次
    综合可用证据。"""
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        [
            {"small_vehicle": [(0.9, (100, 80, 200, 160))]},
            {"small_vehicle": [(0.9, (100, 80, 200, 160))]},
        ],
        fail_calls={1},
        error=RuntimeError("roi1 yolo failed"),
    )
    qwen = _FakeQwen({"selected_box_ids": ["roi-2-box-1"], "fallback_boxes": []})
    result = _run(
        _executor(catalog, qwen=qwen, yolo=yolo),
        tmp_path,
        rois=[
            _roi("roi-1", (0.25, 0.25, 0.75, 0.75)),
            _roi("roi-2", (0.25, 0.25, 0.75, 0.75)),
        ],
    )
    bundle = result.bundle
    # ROI1 error + ROI2 miss aggregates to error for the still-missing leaves;
    # small_vehicle hit in ROI2 keeps its successful evidence.
    # ROI1 error + ROI2 missing 聚合为 error；small_vehicle 在 ROI2 命中并保留
    # 成功证据。
    assert bundle.leaf_states == {
        "small_vehicle": "hit",
        "large_vehicle": "error",
        "building_outline": "error",
    }
    assert len(bundle.candidates) == 1
    assert bundle.candidates[0].box_id == "roi-2-box-1"
    assert len(bundle.call_audit) == 3
    assert bundle.call_audit[0].roi_id == "roi-1"
    assert bundle.call_audit[0].status == "failed"
    assert bundle.call_audit[1].roi_id == "roi-2"
    assert bundle.call_audit[1].status == "succeeded"
    assert bundle.call_audit[2].layer == "final_qwen"


# ── 错误与不可用 / error and unavailable ─────────────────────────────────


def test_schema_invalid_final_response_fails(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    budget = _FakeBudget()
    executor = _executor(
        catalog,
        qwen=_FakeQwen(schema_invalid=True),
        policy=GroundingEvidencePolicy(),
    )
    with pytest.raises(GroundingEvidenceError) as excinfo:
        _run(executor, tmp_path, budget=budget)
    assert excinfo.value.code == "SCHEMA_INVALID"
    assert budget.qwen_calls == 1


def test_client_unavailable_without_cache_identity(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    budget = _FakeBudget()
    executor = _executor(
        catalog,
        qwen=_FakeQwen(no_identity=True),
        policy=GroundingEvidencePolicy(),
    )
    with pytest.raises(GroundingEvidenceError) as excinfo:
        _run(executor, tmp_path, budget=budget)
    assert excinfo.value.code == "CLIENT_UNAVAILABLE"
    # Identity failures never consume budget. / 身份失败绝不消费 budget。
    assert budget.qwen_calls == 0


def test_budget_exhausted_fails_before_call(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    qwen = _FakeQwen()
    executor = _executor(
        catalog,
        qwen=qwen,
        policy=GroundingEvidencePolicy(),
    )
    with pytest.raises(GroundingEvidenceError) as excinfo:
        _run(executor, tmp_path, budget=_FakeBudget(exhausted=True))
    assert excinfo.value.code == "BUDGET_EXHAUSTED"
    assert qwen.calls == []


def test_client_error_maps_to_stable_code(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    budget = _FakeBudget()
    executor = _executor(
        catalog,
        qwen=_FakeQwen(error=RuntimeError("boom /etc/passwd")),
        policy=GroundingEvidencePolicy(),
    )
    with pytest.raises(GroundingEvidenceError) as excinfo:
        _run(executor, tmp_path, budget=budget)
    assert excinfo.value.code == "CLIENT_ERROR"
    assert budget.qwen_calls == 1


# ── bundle 与审计契约 / bundle and audit contracts ────────────────────────


def test_bundle_never_contains_confidence(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        {
            "small_vehicle": [(0.9, (100, 80, 200, 160))],
            "building": [(0.8, (300, 200, 400, 300))],
        }
    )
    result = _run(
        _executor(
            catalog,
            qwen=_FakeQwen({"selected_box_ids": ["roi-1-box-1", "roi-1-box-2"]}),
            yolo=yolo,
        ),
        tmp_path,
        rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))],
    )
    keys = _bundle_keys(result.bundle.model_dump())
    assert "confidence" not in keys
    # The internal scores decide dedup and top-k but never survive into any
    # persisted field. 内部分数决定去重与 top-k，但绝不进入任何持久化字段。
    assert "0.9" not in json.dumps(result.bundle.model_dump())


def test_bundle_is_json_safe_and_free_of_paths_secrets_base64(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    result = _run(
        _executor(
            catalog,
            qwen=_FakeQwen({"selected_box_ids": ["roi-1-box-1"]}),
            yolo=yolo,
        ),
        tmp_path,
        rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))],
    )
    dumped = json.dumps(result.bundle.model_dump())
    for forbidden in ("base64", "data:image", "/Users", "C:\\", "sk-", "secret", "password"):
        assert forbidden not in dumped


def test_run_is_stateless_and_deterministic_across_calls(tmp_path: Path) -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    executor = _executor(
        catalog,
        qwen=_FakeQwen({"selected_box_ids": ["roi-1-box-1"]}),
        yolo=yolo,
    )
    first = _run(
        executor, tmp_path, rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]
    )
    second = _run(
        executor, tmp_path, rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]
    )
    assert first.bundle.model_dump() == second.bundle.model_dump()
    assert first.whole_image_boxes == second.whole_image_boxes


# ── 几何与共享原语 / geometry and shared primitives ───────────────────────


def test_geometry_and_rendered_crop_zero_drift() -> None:
    """map_grounding_roi mirrors crop_image_region exactly: every mapped
    crop_size equals the actually rendered crop for the same box, including
    halo clamps at the image edges. map_grounding_roi 与 crop_image_region
    完全一致：同一 box 的每个映射 crop_size 都等于实际渲染裁切，包括图像边缘
    的 halo clamp。"""
    image = _image()
    for region in (
        _roi("center", (0.25, 0.25, 0.75, 0.75)),
        _roi("corner", (0.0, 0.0, 0.1, 0.1)),
        _roi("edge", (0.9, 0.2, 1.0, 0.4)),
        _roi("tiny", (0.49, 0.49, 0.51, 0.51)),
    ):
        record = map_grounding_roi(region, image.size)
        rendered = crop_image_region(
            image,
            region.xyxy,
            coordinate_frame="normalized_0_1_top_left",
            halo_ratio=0.10,
        )
        assert rendered.size == record.crop_size


def test_executor_has_no_segformer_and_no_vqa_evidence_import() -> None:
    """The grounding seam is single-layer by construction: no SegFormer
    protocol, no VQA evidence import, and no ground-truth access.
    Grounding seam 在构造上就是单层：无 SegFormer 协议、无 VQA evidence
    import、无 ground-truth 访问。"""
    source = (
        Path(__file__).resolve().parents[3]
        / "agents" / "grounding" / "evidence.py"
    ).read_text(encoding="utf-8")
    assert "from agents.general_vqa" not in source
    assert "import agents.general_vqa" not in source
    assert "DenseSemanticClient" not in source
    assert "ground_truth" not in source


def test_import_does_not_load_vqa_evidence_package() -> None:
    # The invariant is on the grounding import graph itself: importing
    # agents.grounding must not pull in the VQA evidence package. Measure the
    # delta so an earlier general-vqa test in the same session cannot mask it.
    # 不变量针对 grounding 自身的 import 图：导入 agents.grounding 不得拉入
    # VQA evidence 包。测量增量，使会话中更早的 general-vqa 测试无法掩盖该
    # 约束。
    before = set(sys.modules)
    import agents.grounding  # noqa: F401

    added = set(sys.modules) - before
    assert not any(
        module.startswith("agents.general_vqa.evidence") for module in added
    )


def test_grounding_qwen_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        GroundingQwenResponse.model_validate(
            {
                "selected_box_ids": [],
                "fallback_boxes": [],
                "hack": True,
            }
        )
