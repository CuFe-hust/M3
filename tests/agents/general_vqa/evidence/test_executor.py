"""Fake-client contract tests for the frozen VQA evidence executor (C5, 14A1).

C5 冻结 VQA 证据执行器的 fake-client 契约测试：六态状态机、按 ROI 不按类别的
调用次数、跨 ROI 全局去重、confidence 绝不进入 bundle、mask 纯内存、稳定
错误码与不填写任何生产默认值的注入策略。
"""

from __future__ import annotations

import dataclasses
import inspect
import json

import numpy as np
import pytest
from PIL import Image

from agents.evidence_catalog import EvidenceCatalog
from agents.general_vqa.evidence.executor import (
    EvidenceExecution,
    EvidencePolicy,
    ObjectEvidenceExecutor,
    RoiLeafOutcome,
)
from agents.schema import (
    FirstQwenVisualPlan,
    ObjectEvidenceRequest,
    RoiPlan,
    RoiRegion,
)
from models.base import (
    DenseSemanticOutput,
    ModelCacheIdentity,
    ObjectDetectionOutput,
)

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
            "segformer_labels": ["vehicle_small"],
            "segformer_enabled": True,
        },
        "large_vehicle": {
            "yolo_labels": ["large_vehicle"],
            "yolo_enabled": True,
        },
        "building_outline": {
            "yolo_labels": ["building"],
            "yolo_enabled": True,
            "segformer_labels": ["building"],
            "segformer_enabled": True,
        },
    },
}

_YOLO_ID = "yolo-test-v1"
_YOLO_DIGEST = "a" * 64
_SEG_ID = "segformer-test-v1"
_SEG_DIGEST = "b" * 64
_SEG_CLASS_NAMES = ("vehicle_small", "building", "background")

# Ordered leaf expansion of vehicle + building / vehicle 与 building 的有序叶子展开。
_ALL_LEAVES = ("small_vehicle", "large_vehicle", "building_outline")


class _FakeYolo:
    """Records every call, returns per-call per-label detections, and can
    raise typed errors on selected call indices (1-based).
    记录每次调用，按调用序返回逐标签检测，并可在指定调用序号（1 起）抛类型化
    错误。"""

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
        # A configured error raises on every call unless fail_calls restricts
        # the raising to selected call indices. 配置错误默认每次调用都抛，除非
        # fail_calls 限制只在指定调用序号抛出。
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


class _FakeSegFormer:
    """Records every call and returns a dense output whose presence pattern is
    driven by the per-call strategy ('hit'/'miss' per leaf label); class_names
    may be overridden to exercise the class-map mismatch seam.
    记录每次调用，按逐调用策略（每叶子标签 'hit'/'miss'）返回稠密输出；
    class_names 可覆盖以验证类别映射不匹配 seam。"""

    def __init__(
        self,
        strategies: list[dict[str, str]],
        *,
        class_names: tuple[str, ...] = _SEG_CLASS_NAMES,
        fail_calls: set[int] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.strategies = strategies
        self.class_names = class_names
        self.fail_calls = set(fail_calls or ())
        self.error = error
        self.calls: list[tuple[Image.Image, dict[str, object]]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model=_SEG_ID,
            generation={"weights_sha256": _SEG_DIGEST},
            client_version="test",
        )

    def infer(
        self,
        image: Image.Image,
        *,
        tile_size: int,
        tile_overlap: int,
        feature_stage: int,
    ) -> DenseSemanticOutput:
        self.calls.append(
            (
                image,
                {
                    "tile_size": tile_size,
                    "tile_overlap": tile_overlap,
                    "feature_stage": feature_stage,
                },
            )
        )
        if self.error is not None and (
            not self.fail_calls or len(self.calls) in self.fail_calls
        ):
            raise self.error
        height, width = image.height, image.width
        probabilities = np.zeros((len(self.class_names), height, width), dtype=np.float32)
        probabilities[-1] = 1.0
        index = min(len(self.calls), len(self.strategies)) - 1
        # Unknown names are skipped so a class-map mismatch surfaces from the
        # executor's own index lookup, not from this fake.
        # 未知名称跳过，使类别映射不匹配从执行器自身的查找中暴露，而非本 fake。
        for name, mode in self.strategies[index].items():
            if mode == "hit" and name in self.class_names:
                channel = self.class_names.index(name)
                probabilities[channel, 1:-1, 1:-1] = 0.9
                probabilities[-1, 1:-1, 1:-1] = 0.1
        return DenseSemanticOutput(
            probabilities=probabilities,
            features=np.zeros((2, height, width), dtype=np.float32),
            semantic_stride=(1.0, 1.0),
            feature_stride=(1.0, 1.0),
            original_size=(width, height),
            class_names=self.class_names,
            diagnostics={},
            weights_sha256=_SEG_DIGEST,
        )


# ── helpers / 辅助 ────────────────────────────────────────────────────────


def _image(size: tuple[int, int] = (1000, 800), fill: int = 7) -> Image.Image:
    return Image.new("RGB", size, (fill, fill + 1, fill + 2))


def _roi(roi_id: str, xyxy: tuple[float, float, float, float]) -> RoiRegion:
    return RoiRegion(roi_id=roi_id, image_id="img1", xyxy=xyxy)


def _plan(
    categories: tuple[str, ...] = ("vehicle", "building"),
    *,
    rois: list[RoiRegion] | None = None,
    family: str = "object_evidence_vqa",
) -> FirstQwenVisualPlan:
    return FirstQwenVisualPlan(
        version="first-qwen-plan-v1",
        execution_family=family,
        confidence=0.95,
        roi_plan=RoiPlan(rois=rois or []),
        evidence_request=(
            ObjectEvidenceRequest(composite_categories=list(categories))
            if family == "object_evidence_vqa"
            else None
        ),
        reason_codes=[],
    )


def _policy(**overrides) -> EvidencePolicy:
    kwargs = {
        "confidence_threshold": 0.5,
        "nms_iou_threshold": 0.5,
        "max_detections": 5,
    }
    kwargs.update(overrides)
    return EvidencePolicy(**kwargs)


def _executor(
    catalog: EvidenceCatalog,
    *,
    yolo: _FakeYolo | None = None,
    seg: _FakeSegFormer | None = None,
    policy: EvidencePolicy | None = None,
    **overrides,
) -> ObjectEvidenceExecutor:
    kwargs = {
        "catalog": catalog,
        "policy": policy or _policy(),
        "yolo_client": yolo,
        "yolo_device": "cpu",
        "yolo_image_size": 640,
        "segformer_client": seg,
        "segformer_tile_size": 512,
        "segformer_tile_overlap": 64,
        "segformer_feature_stage": 3,
    }
    kwargs.update(overrides)
    return ObjectEvidenceExecutor(**kwargs)


def _bundle_keys(value) -> set[str]:
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


def test_policy_requires_all_values_no_production_defaults() -> None:
    """The unfrozen parameters are inject-only: every field is required and
    no production default is filled anywhere. 未冻结参数仅可注入：每个字段必填，
    任何位置不填写生产默认值。"""
    for field in dataclasses.fields(EvidencePolicy):
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING


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
        _executor(catalog, yolo_image_size=0)
    with pytest.raises(ValueError):
        _executor(catalog, segformer_tile_size=0)
    with pytest.raises(ValueError):
        _executor(catalog, segformer_tile_overlap=512)
    with pytest.raises(ValueError):
        _executor(catalog, segformer_feature_stage=-1)


def test_executor_has_no_task_concept() -> None:
    """The executor never sees the sample or its task: its inputs are the
    plan, the in-memory images, and the fallback image id. 执行器绝不接触
    sample 或其 task：输入只有计划、内存图像与 fallback 图像 id。"""
    signature = inspect.signature(ObjectEvidenceExecutor.execute)
    assert set(signature.parameters) == {"self", "plan", "images", "fallback_image_id"}


# ── 命中路径 / happy path ─────────────────────────────────────────────────


def test_yolo_hit_all_leaves_single_roi() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        {
            "small_vehicle": [(0.9, (100, 80, 200, 160))],
            "large_vehicle": [(0.7, (500, 300, 600, 400))],
            "building": [(0.8, (300, 200, 400, 300))],
        }
    )
    seg = _FakeSegFormer([{"vehicle_small": "miss", "building": "miss"}])
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    assert len(yolo.calls) == 1
    assert seg.calls == []
    assert len(bundle.detections) == 3
    assert bundle.missing_leaves == []
    assert bundle.leaf_states == {
        "small_vehicle": "hit",
        "large_vehicle": "hit",
        "building_outline": "hit",
    }
    assert len(bundle.call_audit) == 1
    assert bundle.call_audit[0].layer == "yolo"
    assert bundle.call_audit[0].status == "succeeded"
    # A yolo hit leaf gets a not_run segformer record only when capable; the
    # leaf without a segformer mapping stays single-layer.
    # yolo 命中叶子仅在具备 segformer 能力时给出 not_run 记录；无 segformer
    # 映射的叶子保持单层记录。
    layers = {(record.leaf_category, record.layer, record.state) for record in execution.layer_states}
    assert layers == {
        ("small_vehicle", "yolo", "hit"),
        ("small_vehicle", "segformer", "not_run"),
        ("large_vehicle", "yolo", "hit"),
        ("building_outline", "yolo", "hit"),
        ("building_outline", "segformer", "not_run"),
    }


def test_missing_leaf_falls_to_segformer_and_hits() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({"building": [(0.8, (300, 200, 400, 300))]})
    seg = _FakeSegFormer([{"vehicle_small": "hit", "building": "miss"}])
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    assert len(bundle.segments) == 1
    assert bundle.segments[0].leaf_category == "small_vehicle"
    assert bundle.segments[0].roi_id == "roi-1"
    # large_vehicle has no approved segformer capability: unsupported at that
    # layer, still missing for the final-Qwen fallback.
    # large_vehicle 无批准 segformer 能力：该层 unsupported，仍属最终 Qwen
    # 回退的缺失集合。
    assert bundle.missing_leaves == ["large_vehicle"]
    assert bundle.leaf_states == {
        "small_vehicle": "hit",
        "large_vehicle": "unsupported",
        "building_outline": "hit",
    }
    # The mask travels in memory only. / mask 只在内存中传递。
    assert ("roi-1", "small_vehicle") in execution.masks
    assert "masks" not in _bundle_keys(bundle.model_dump())
    states = {(o.leaf_category, o.layer, o.state) for o in execution.outcomes}
    assert ("small_vehicle", "yolo", "missing") in states
    assert ("small_vehicle", "segformer", "hit") in states
    assert ("large_vehicle", "segformer", "unsupported") in states
    assert ("building_outline", "segformer", "not_run") in states


def test_success_but_empty_is_missing_not_valid_empty() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({})
    seg = _FakeSegFormer([{"vehicle_small": "miss", "building": "miss"}])
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    assert bundle.detections == []
    assert bundle.segments == []
    # A successful-but-empty call is missing, never a special empty state.
    # 成功但为空属于 missing，绝不存在特殊空状态。
    assert bundle.missing_leaves == list(_ALL_LEAVES)
    assert bundle.leaf_states["small_vehicle"] == "missing"
    assert bundle.leaf_states["large_vehicle"] == "unsupported"
    assert bundle.leaf_states["building_outline"] == "missing"
    missing_order = [o for o in execution.outcomes if o.state == "missing"]
    rois = {o.roi_id for o in missing_order}
    assert rois == {"roi-1"}


def test_yolo_hit_accumulates_across_rois_and_segformer_never_runs() -> None:
    """A hit leaf keeps accumulating evidence per ROI and is never re-run or
    overwritten at a deeper layer: with every leaf hit at yolo, segformer is
    never called at all. 命中叶子按 ROI 持续累积证据，绝不在更深层重跑或被
    覆盖：当所有叶子都在 yolo 命中时，segformer 完全不调用。"""
    catalog = EvidenceCatalog(_CATALOG_DATA)
    # The second ROI crop is 110x88, so its local boxes must fit that crop;
    # the executor rejects out-of-crop local frames with a strict error.
    # 第二个 ROI 裁切为 110x88，其局部框必须落在裁切内；执行器对越界局部框
    # 以严格错误拒绝。
    yolo = _FakeYolo(
        [
            {
                "small_vehicle": [(0.9, (100, 80, 200, 160))],
                "large_vehicle": [(0.7, (500, 300, 600, 400))],
                "building": [(0.8, (300, 200, 400, 300))],
            },
            {
                "small_vehicle": [(0.9, (5, 5, 20, 16))],
                "large_vehicle": [(0.7, (30, 10, 50, 30))],
                "building": [(0.8, (60, 40, 90, 70))],
            },
        ]
    )
    seg = _FakeSegFormer([{"vehicle_small": "miss", "building": "miss"}])
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(
            rois=[
                _roi("roi-1", (0.25, 0.25, 0.75, 0.75)),
                _roi("roi-2", (0.0, 0.0, 0.1, 0.1)),
            ]
        ),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    assert len(yolo.calls) == 2
    assert seg.calls == []
    assert len(execution.bundle.detections) == 6
    assert execution.bundle.missing_leaves == []
    not_run = {o.roi_id for o in execution.outcomes if o.state == "not_run"}
    assert not_run == {"roi-1", "roi-2"}


# ── 错误与不可用 / error and unavailable ─────────────────────────────────


def test_yolo_error_is_stable_code_never_raw_exception() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        {}, error=RuntimeError("boom: /Users/troy/secret.pth failed to load")
    )
    seg = _FakeSegFormer([{"vehicle_small": "miss", "building": "miss"}])
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    errors = [o for o in execution.outcomes if o.state == "error"]
    assert errors
    assert all(o.error_code == "RuntimeError" for o in errors)
    assert bundle.call_audit[0].status == "failed"
    assert bundle.call_audit[0].error_code == "RuntimeError"
    # The raw exception text and the machine path never reach the bundle.
    # 原始异常文本与机器路径绝不进入 bundle。
    dumped = json.dumps(bundle.model_dump())
    assert "boom" not in dumped
    assert "/Users" not in dumped
    assert "secret.pth" not in dumped
    # The error leaves are still missing for the final-Qwen fallback.
    # 错误叶子仍属最终 Qwen 回退的缺失集合。
    assert bundle.missing_leaves == list(_ALL_LEAVES)


def test_segformer_error_is_stable_code_never_raw_exception() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({})
    seg = _FakeSegFormer(
        [{"vehicle_small": "miss", "building": "miss"}],
        error=ValueError("bad tile /etc/passwd"),
    )
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    seg_audit = [a for a in bundle.call_audit if a.layer == "segformer"]
    assert len(seg_audit) == 1
    assert seg_audit[0].status == "failed"
    assert seg_audit[0].error_code == "ValueError"
    dumped = json.dumps(bundle.model_dump())
    assert "/etc" not in dumped
    assert "passwd" not in dumped
    # The error state is honest (error > missing for the leaf's deepest
    # layer) and the leaf still feeds the final-Qwen fallback.
    # 错误状态如实保留（叶子最深层 error > missing），该叶子仍进入最终 Qwen
    # 回退。
    assert bundle.leaf_states["small_vehicle"] == "error"
    assert "small_vehicle" in bundle.missing_leaves


def test_yolo_unavailable_without_client_produces_no_audit() -> None:
    """No client means no call: unavailable outcomes, and no audit entry is
    fabricated for a call that never happened. 无客户端即无调用：outcome 为
    unavailable，且绝不伪造一次未发生调用的审计记录。"""
    catalog = EvidenceCatalog(_CATALOG_DATA)
    seg = _FakeSegFormer([{"vehicle_small": "miss", "building": "miss"}])
    execution = _executor(catalog, seg=seg).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    assert len(bundle.call_audit) == 1
    assert bundle.call_audit[0].layer == "segformer"
    yolo_states = {(o.leaf_category, o.state) for o in execution.outcomes if o.layer == "yolo"}
    assert yolo_states == {(leaf, "unavailable") for leaf in _ALL_LEAVES}


def test_segformer_unavailable_without_client() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({})
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    assert len(bundle.call_audit) == 1
    assert bundle.call_audit[0].layer == "yolo"
    seg_states = {
        (o.leaf_category, o.state)
        for o in execution.outcomes
        if o.layer == "segformer"
    }
    # Capable leaves are unavailable; the leaf without a mapping is
    # unsupported, not unavailable — capability is catalog fact, not absence
    # of a client. 具备能力的叶子为 unavailable；无映射的叶子为 unsupported 而
    # 非 unavailable——能力是目录事实，不是客户端缺失。
    assert seg_states == {
        ("small_vehicle", "unavailable"),
        ("large_vehicle", "unsupported"),
        ("building_outline", "unavailable"),
    }


def test_segformer_class_map_mismatch_is_stable_error() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({})
    seg = _FakeSegFormer(
        [{"vehicle_small": "hit", "building": "hit"}],
        class_names=("background",),
    )
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    mismatches = [o for o in execution.outcomes if o.error_code is not None]
    assert mismatches
    assert all(
        o.error_code == "SEGFORMER_CLASS_MAP_MISMATCH"
        and o.state == "error"
        and o.layer == "segformer"
        for o in mismatches
    )
    # The call itself succeeded; only the approved mapping is wrong, and the
    # leaf stays missing for the fallback. 调用本身成功；只是已批准映射错误，
    # 叶子保持缺失进入回退。
    assert bundle.call_audit[0].status == "succeeded"
    assert bundle.missing_leaves == list(_ALL_LEAVES)


def test_error_severity_dominates_missing_across_rois() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        [
            {"large_vehicle": [(0.7, (500, 300, 600, 400))]},
            {"large_vehicle": [(0.7, (500, 300, 600, 400))]},
        ],
        fail_calls={1},
        error=RuntimeError("roi1 yolo failed"),
    )
    seg = _FakeSegFormer(
        [
            {"vehicle_small": "miss", "building": "miss"},
            {"vehicle_small": "miss", "building": "miss"},
        ]
    )
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(
            rois=[
                _roi("roi-1", (0.25, 0.25, 0.75, 0.75)),
                _roi("roi-2", (0.25, 0.25, 0.75, 0.75)),
            ]
        ),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    # ROI1 yolo error + ROI2 yolo miss aggregates to error for the yolo layer
    # of small_vehicle/building_outline (error > missing); large_vehicle hit
    # in ROI2 keeps its successful evidence — a hit never demotes to error.
    # ROI1 yolo error + ROI2 yolo missing 聚合为该叶子 yolo 层 error
    # （error > missing）；large_vehicle 在 ROI2 命中并保留成功证据——命中绝不
    # 降级为 error。
    layers = {
        (record.leaf_category, record.layer): record.state
        for record in execution.layer_states
    }
    assert layers[("small_vehicle", "yolo")] == "error"
    assert layers[("building_outline", "yolo")] == "error"
    assert layers[("large_vehicle", "yolo")] == "hit"
    assert len(execution.bundle.detections) == 1
    assert execution.bundle.detections[0].roi_id == "roi-2"


# ── 状态机不变式 / state machine invariants ───────────────────────────────


def test_segformer_hit_in_one_roi_is_not_run_in_later_roi() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({})
    seg = _FakeSegFormer(
        [
            {"vehicle_small": "hit", "building": "miss"},
            {"vehicle_small": "miss", "building": "hit"},
        ]
    )
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(
            rois=[
                _roi("roi-1", (0.25, 0.25, 0.75, 0.75)),
                _roi("roi-2", (0.25, 0.25, 0.75, 0.75)),
            ]
        ),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    assert len(seg.calls) == 2
    segments = {(s.leaf_category, s.roi_id) for s in bundle.segments}
    assert segments == {("small_vehicle", "roi-1"), ("building_outline", "roi-2")}
    # small_vehicle hit in ROI1: not_run in ROI2, never re-filtered, and its
    # mask exists only for ROI1. A leaf is not_run only AFTER it was hit — in
    # ROI1 building_outline was still missing, so it has no not_run record
    # there. small_vehicle 在 ROI1 命中：ROI2 为 not_run，绝不重筛，其 mask
    # 只存在于 ROI1。叶子只在命中之后才出现 not_run——ROI1 中 building_outline
    # 仍缺失，因此那里没有 not_run 记录。
    not_run = [o for o in execution.outcomes if o.state == "not_run"]
    assert set((o.roi_id, o.leaf_category) for o in not_run) == {
        ("roi-2", "small_vehicle"),
    }
    assert ("roi-1", "small_vehicle") in execution.masks
    assert ("roi-2", "small_vehicle") not in execution.masks
    assert bundle.leaf_states["small_vehicle"] == "hit"
    assert bundle.leaf_states["building_outline"] == "hit"
    assert bundle.leaf_states["large_vehicle"] == "unsupported"


def test_single_composite_single_leaf_expands_and_hits() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({"building": [(0.8, (300, 200, 400, 300))]})
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(("building",), rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    assert execution.bundle.leaf_states == {"building_outline": "hit"}
    assert execution.bundle.missing_leaves == []
    assert len(execution.bundle.detections) == 1
    assert len(execution.bundle.call_audit) == 1


def test_three_rois_produce_exactly_three_yolo_calls() -> None:
    """The schema allows up to three ROIs; the executor maps each to exactly
    one model call and assembles the bundle in roi order.
    Schema 允许最多三个 ROI；执行器把每个映射为恰好一次模型调用，并按 ROI
    顺序组装 bundle。"""
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        [
            {"small_vehicle": [(0.9, (100, 80, 200, 160))]},
            {"small_vehicle": [(0.9, (100, 80, 200, 160))]},
            {"small_vehicle": [(0.9, (100, 80, 200, 160))]},
        ]
    )
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(
            rois=[
                _roi("roi-1", (0.25, 0.25, 0.75, 0.75)),
                _roi("roi-2", (0.25, 0.25, 0.75, 0.75)),
                _roi("roi-3", (0.25, 0.25, 0.75, 0.75)),
            ]
        ),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    assert len(yolo.calls) == 3
    assert len(execution.bundle.call_audit) == 3
    # Identical global targets dedup to one detection; the winner is the
    # first of the equal-confidence ties, keeping roi order.
    # 相同全局目标去重为一条；等置信度 tie 保留首次出现，维持 roi 顺序。
    assert len(execution.bundle.detections) == 1
    assert execution.bundle.detections[0].roi_id == "roi-1"
    assert [record.roi_id for record in execution.bundle.rois] == [
        "roi-1",
        "roi-2",
        "roi-3",
    ]


def test_calls_scale_with_rois_not_with_leaves() -> None:
    """Three requested leaves, two ROIs: exactly two YOLO and two SegFormer
    calls — model inputs are ROI crops, never per-category images.
    三个请求叶子、两个 ROI：恰好两次 YOLO 与两次 SegFormer 调用——模型输入是
    ROI 裁切，绝不按类别重复喂图。"""
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({})
    seg = _FakeSegFormer(
        [
            {"vehicle_small": "miss", "building": "miss"},
            {"vehicle_small": "miss", "building": "miss"},
        ]
    )
    _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(
            rois=[
                _roi("roi-1", (0.25, 0.25, 0.75, 0.75)),
                _roi("roi-2", (0.0, 0.0, 0.1, 0.1)),
            ]
        ),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    assert len(yolo.calls) == 2
    assert len(seg.calls) == 2


def test_only_requested_labels_are_kept() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        {
            "small_vehicle": [(0.9, (100, 80, 200, 160))],
            "person": [(0.95, (50, 50, 150, 150))],
            "unrelated_class": [(0.7, (400, 400, 500, 500))],
        }
    )
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    dumped = json.dumps(execution.bundle.model_dump())
    assert "person" not in dumped
    assert "unrelated_class" not in dumped
    assert len(execution.bundle.detections) == 1
    assert execution.bundle.detections[0].leaf_category == "small_vehicle"


def test_max_detections_truncates_highest_confidence() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        {
            "small_vehicle": [
                (0.7, (700, 500, 750, 550)),
                (0.9, (100, 80, 200, 160)),
                (0.8, (400, 300, 500, 400)),
            ]
        }
    )
    execution = _executor(catalog, yolo=yolo, policy=_policy(max_detections=2)).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    assert len(execution.bundle.detections) == 2
    # The two highest internal scores survive; confidence never leaves the
    # executor. 内部分数最高的两条保留；confidence 绝不离开执行器。
    global_boxes = {d.global_xyxy for d in execution.bundle.detections}
    assert (300.0, 240.0, 400.0, 320.0) in global_boxes
    assert (600.0, 460.0, 700.0, 560.0) in global_boxes
    assert (900.0, 660.0, 950.0, 710.0) not in global_boxes


def test_cross_roi_dedup_higher_confidence_wins() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        [
            {"small_vehicle": [(0.6, (100, 80, 200, 160))]},
            {"small_vehicle": [(0.95, (100, 80, 200, 160))]},
        ]
    )
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(
            rois=[
                _roi("roi-1", (0.25, 0.25, 0.75, 0.75)),
                _roi("roi-2", (0.25, 0.25, 0.75, 0.75)),
            ]
        ),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    # Identical global targets across ROIs dedup in whole-image coordinates;
    # the higher internal confidence wins regardless of ROI order.
    # 跨 ROI 相同全局目标在 whole-image 坐标去重；内部置信度更高者胜出，与
    # ROI 顺序无关。
    assert len(execution.bundle.detections) == 1
    assert execution.bundle.detections[0].roi_id == "roi-2"


def test_cross_roi_disjoint_targets_are_both_kept() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        [
            {"small_vehicle": [(0.9, (100, 80, 200, 160))]},
            {"small_vehicle": [(0.8, (10, 8, 60, 44))]},
        ]
    )
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(
            rois=[
                _roi("roi-1", (0.25, 0.25, 0.75, 0.75)),
                _roi("roi-2", (0.0, 0.0, 0.1, 0.1)),
            ]
        ),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    assert len(execution.bundle.detections) == 2


def test_empty_plan_falls_back_to_unique_full_image_roi() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    bundle = execution.bundle
    assert len(bundle.rois) == 1
    assert bundle.rois[0].roi_id == "full"
    assert bundle.rois[0].crop_size == (1000, 800)
    # The whole image was the model input for the single call.
    # 整图是唯一一次调用的模型输入。
    assert yolo.calls[0][0].size == (1000, 800)
    # Local and global frames coincide for the full-image ROI.
    # 整图 ROI 下局部与全局坐标系一致。
    assert bundle.detections[0].global_xyxy == bundle.detections[0].local_xyxy


def test_execute_is_stateless_and_deterministic_across_calls() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    seg = _FakeSegFormer([{"vehicle_small": "miss", "building": "miss"}])
    executor = _executor(catalog, yolo=yolo, seg=seg)
    first = executor.execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    second = executor.execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    assert first.bundle.model_dump() == second.bundle.model_dump()
    assert first.outcomes == second.outcomes
    # A third plan on the same executor leaks no state from earlier passes.
    # 同一执行器上的第三个计划不泄漏前次执行的状态。
    third = executor.execute(
        _plan(("vehicle",)),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    assert {o.leaf_category for o in third.outcomes} == {"small_vehicle", "large_vehicle"}
    assert len(third.bundle.call_audit) == 1


# ── bundle 与审计契约 / bundle and audit contracts ────────────────────────


def test_bundle_never_contains_confidence() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        {
            "small_vehicle": [(0.9, (100, 80, 200, 160))],
            "building": [(0.8, (300, 200, 400, 300))],
        }
    )
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    keys = _bundle_keys(execution.bundle.model_dump())
    assert "confidence" not in keys


def test_bundle_is_json_safe_and_free_of_paths_secrets_base64() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo(
        {
            "small_vehicle": [(0.9, (100, 80, 200, 160))],
            "building": [(0.8, (300, 200, 400, 300))],
        }
    )
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    dumped = json.dumps(execution.bundle.model_dump())
    for forbidden in ("base64", "data:image", "/Users", "C:\\", "sk-", "secret", "password"):
        assert forbidden not in dumped


def test_masks_travel_in_memory_only() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({})
    seg = _FakeSegFormer([{"vehicle_small": "hit", "building": "miss"}])
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    mask = execution.masks[("roi-1", "small_vehicle")]
    # The presence mask is a boolean grid at the crop resolution.
    # 存在掩膜是 crop 分辨率的布尔网格。
    assert mask.shape == (480, 600)
    assert bool(mask.any())
    # No mask payload survives into the persisted bundle.
    # 持久化 bundle 中不残留任何掩膜载荷。
    dumped = json.dumps(execution.bundle.model_dump())
    assert "probabilities" not in dumped
    assert "mask" not in dumped


def test_detection_global_frame_transform() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    record = execution.bundle.detections[0]
    # ROI (0.25,0.25,0.75,0.75) on 1000x800 maps to expanded (200,160,800,640);
    # the local box shifts by exactly that origin.
    # 1000x800 上 ROI (0.25,0.25,0.75,0.75) 映射到 expanded (200,160,800,640)；
    # 局部框精确平移该原点。
    assert record.local_xyxy == (100.0, 80.0, 200.0, 160.0)
    assert record.local_roi_size == (600, 480)
    assert record.global_xyxy == (300.0, 240.0, 400.0, 320.0)
    assert record.global_image_size == (1000, 800)


def test_audit_identity_comes_from_model_outputs_for_yolo() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({"small_vehicle": [(0.9, (100, 80, 200, 160))]})
    execution = _executor(catalog, yolo=yolo).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    audit = execution.bundle.call_audit[0]
    assert audit.layer == "yolo"
    assert audit.logical_model_id == _YOLO_ID
    assert audit.weights_sha256 == _YOLO_DIGEST
    assert audit.input_size == (600, 480)


def test_audit_identity_comes_from_client_for_segformer() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({})
    seg = _FakeSegFormer([{"vehicle_small": "hit", "building": "miss"}])
    execution = _executor(catalog, yolo=yolo, seg=seg).execute(
        _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
        {"img1": _image()},
        fallback_image_id="img1",
    )
    seg_audit = [a for a in execution.bundle.call_audit if a.layer == "segformer"]
    assert len(seg_audit) == 1
    assert seg_audit[0].logical_model_id == _SEG_ID
    assert seg_audit[0].weights_sha256 == _SEG_DIGEST


# ── 输入守卫 / input guards ───────────────────────────────────────────────


def test_direct_vqa_plan_is_rejected() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    executor = _executor(catalog)
    with pytest.raises(ValueError, match="object_evidence"):
        executor.execute(
            _plan(family="direct_vqa"),
            {"img1": _image()},
            fallback_image_id="img1",
        )


def test_unknown_image_id_fails_before_any_model_call() -> None:
    catalog = EvidenceCatalog(_CATALOG_DATA)
    yolo = _FakeYolo({})
    executor = _executor(catalog, yolo=yolo)
    with pytest.raises(ValueError, match="unknown image_id"):
        executor.execute(
            _plan(rois=[_roi("roi-1", (0.25, 0.25, 0.75, 0.75))]),
            {"other": _image()},
            fallback_image_id="img1",
        )
    assert yolo.calls == []
