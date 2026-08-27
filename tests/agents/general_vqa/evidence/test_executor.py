"""Contract tests for the v2 VQA evidence executor.

v2 VQA 证据执行器契约测试：执行器只消费 VisualTaskPlan 与已物化源像素视图，
按冻结 1024×1024 tile 协议调度 YOLO、按 pad-multiple-1024-resize-square
协议调度 SegFormer（每个（ROI，binding）恰好一次调用），有界并发、稳定合并、
逐调用审计，聚合顺序 hit > error > unavailable > missing > unsupported。
本文件同时证明活动 job 峰值受 max_tile_concurrency 约束、乱序完成产生相同
bundle、单（ROI，binding）失败隔离、以及逻辑 client 一次 execution 只构造
一次（组合根负责）。

第 14.9/14.10/14.11 节与 26 §6：YOLO 调用次数按 tile、SegFormer 按
（ROI，binding），都不按类别增长；YOLO confidence 仅内部消费；SegFormer
掩膜逐 ROI 独立保留、恢复后严格裁回 ROI crop 尺寸；跨 ROI 去重在
whole-image 坐标。
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import threading
import time

import pytest
from PIL import Image

from agents.evidence_catalog import EvidenceCatalog
from agents.general_vqa.evidence.executor import (
    EvidencePolicy,
    ObjectEvidenceExecutor,
)
from agents.general_vqa.evidence.geometry import MODEL_INPUT_SIZE
from agents.general_vqa.evidence.rendering import materialize_quantized_roi
from agents.general_vqa.evidence.schema import EvidencePreprocessing
from agents.schema import MaterializedVisualView, VisualTaskPlan
from models.base import (
    ModelCacheIdentity,
    ObjectDetectionOutput,
    SemanticMaskOutput,
)

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

_SEG_CATALOG_DATA = {
    "catalog_version": "test-catalog-seg-v1",
    "aliases": {},
    "parents": {},
    "leaves": {
        "vehicle": {
            "yolo_labels": ["vehicle"],
            "yolo_enabled": True,
            "segformer_labels": ["vehicle"],
            "segformer_binding": "seg_001",
            "segformer_enabled": True,
        },
        "building": {
            "yolo_labels": ["building"],
            "yolo_enabled": True,
            "segformer_labels": ["building"],
            "segformer_binding": "seg_002",
            "segformer_enabled": True,
        },
        "water": {
            "yolo_labels": [],
            "yolo_enabled": False,
            "segformer_labels": ["water"],
            "segformer_binding": "seg_001",
            "segformer_enabled": True,
        },
    },
    "task_capabilities": {
        task: ["vehicle", "building", "water"]
        for task in ("counting", "fine_grained_counting", "general_vqa", "grounding")
    },
}


class _Latch:
    """Count-release latch: the first ``count`` arrivals block until all have
    arrived, proving real parallelism; waiters time out instead of hanging when
    the implementation regresses to serial execution. 计数释放闩锁：前
    ``count`` 个到达者阻塞直到全部到达，以证明真实并行；若实现退化为串行，
    等待者超时返回而不是挂死。"""

    def __init__(self, count: int) -> None:
        self._count = count
        self._arrived = 0
        self._condition = threading.Condition()

    def wait(self) -> None:
        with self._condition:
            self._arrived += 1
            if self._arrived >= self._count:
                self._condition.notify_all()
            else:
                self._condition.wait(timeout=10)


class _FakeYolo:
    """Deterministic fake detector with concurrency/error/delay hooks. Each
    call records its submission index; hooks keyed by that index address the
    stable tile plan order, never completion order. 带并发/错误/延迟钩子的
    确定性假检测器。每次调用记录其提交序号；按该序号索引的钩子对应稳定
    tile plan 顺序，绝不按完成顺序。"""

    def __init__(
        self,
        labels: tuple[str, ...],
        *,
        error: Exception | None = None,
        error_for: dict[int, Exception] | None = None,
        box_by_index: dict[int, tuple[float, float, float, float]] | None = None,
        delay_for: dict[int, float] | None = None,
        wrong_input_size: bool = False,
        latch_count: int | None = None,
    ) -> None:
        self.labels = labels
        self._error = error
        self._error_for = error_for or {}
        # When provided, only listed indexes return detections; otherwise every
        # call returns every label at the default box. 提供时仅列出的序号返回
        # 检测；否则每次调用以默认框返回全部标签。
        self._box_by_index = box_by_index
        self._delay_for = delay_for or {}
        self._wrong_input_size = wrong_input_size
        self._latch = _Latch(latch_count) if latch_count else None
        self.calls: list[Image.Image] = []
        self._lock = threading.Lock()
        self._call_count = 0
        self._active = 0
        self.peak = 0

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
        with self._lock:
            index = self._call_count
            self._call_count += 1
            self._active += 1
            self.peak = max(self.peak, self._active)
            self.calls.append(image)
        try:
            if self._latch is not None:
                self._latch.wait()
            delay = self._delay_for.get(index, 0.0)
            if delay:
                time.sleep(delay)
            if self._error is not None:
                raise self._error
            if index in self._error_for:
                raise self._error_for[index]
            if self._box_by_index is not None:
                box = self._box_by_index.get(index)
                if box is None:
                    return []
            else:
                box = (10.0, 10.0, 40.0, 40.0)
            size = (
                (image.width, image.height)
                if not self._wrong_input_size
                else (640, 640)
            )
            return [
                ObjectDetectionOutput(
                    label=label,
                    confidence=0.9,
                    xyxy=box,
                    polygon=None,
                    input_width=size[0],
                    input_height=size[1],
                    logical_model_id="yolo-test-v1",
                    weights_sha256="a" * 64,
                    provider_audit={},
                )
                for label in self.labels
            ]
        finally:
            with self._lock:
                self._active -= 1


def _grid_from_image(image: Image.Image) -> list[list[int]]:
    """Deterministic class-id grid derived from the tile pixels: red-dominant
    pixels become class 1, blue-dominant class 2, everything else 0. Purely
    python byte scanning; no numpy dependency in tests. 从 tile 像素导出确定性
    class-id grid：红色主导像素为 class 1，蓝色主导为 class 2，其余为 0。
    纯 Python 字节扫描；测试不依赖 numpy。"""
    width, height = image.size
    data = image.tobytes()
    grid: list[list[int]] = []
    for y in range(height):
        row = [0] * width
        offset = y * width * 3
        position = 0
        for pixel in range(offset, offset + width * 3, 3):
            red = data[pixel]
            green = data[pixel + 1]
            blue = data[pixel + 2]
            if red >= 100 and red - green >= 40 and red - blue >= 40:
                row[position] = 1
            elif blue >= 100 and blue - red >= 40 and blue - green >= 40:
                row[position] = 2
            position += 1
        grid.append(row)
    return grid


class _FakeSegmenter:
    """Deterministic fake semantic-mask client with the same concurrency/error
    hooks as the fake detector; ``grid_source="empty"`` skips pixel scanning
    for pure-scheduling tests. 与假检测器相同并发/错误钩子的确定性假语义 mask
    客户端；grid_source="empty" 为纯调度测试跳过像素扫描。"""

    def __init__(
        self,
        *,
        labels: dict[int, str],
        error_for: dict[int, Exception] | None = None,
        wrong_size: bool = False,
        grid_source: str = "image",
        latch_count: int | None = None,
    ) -> None:
        self.id_to_label = labels
        self._error_for = error_for or {}
        self._wrong_size = wrong_size
        self._grid_source = grid_source
        self._latch = _Latch(latch_count) if latch_count else None
        self.calls: list[Image.Image] = []
        self._lock = threading.Lock()
        self._call_count = 0
        self._active = 0
        self.peak = 0

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="seg-test-v1",
            generation={"weights_sha256": "b" * 64},
            client_version="test",
        )

    def segment(self, image: Image.Image) -> SemanticMaskOutput:
        with self._lock:
            index = self._call_count
            self._call_count += 1
            self._active += 1
            self.peak = max(self.peak, self._active)
            self.calls.append(image)
        try:
            if self._latch is not None:
                self._latch.wait()
            if index in self._error_for:
                raise self._error_for[index]
            if self._grid_source == "empty":
                grid: list[list[int]] = [
                    [0] * MODEL_INPUT_SIZE for _ in range(MODEL_INPUT_SIZE)
                ]
            else:
                grid = _grid_from_image(image)
            return SemanticMaskOutput(
                class_id_map=grid,
                id_to_label=self.id_to_label,
                original_size=(
                    (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
                    if not self._wrong_size
                    else (512, 512)
                ),
                weights_sha256="b" * 64,
                diagnostics={"logical_model_id": "seg-test-v1"},
            )
        finally:
            with self._lock:
                self._active -= 1


def _image(size: tuple[int, int] = (1000, 800)) -> Image.Image:
    return Image.new("RGB", size, (7, 8, 9))


def _region_image(size: tuple[int, int] = (2000, 1024)) -> Image.Image:
    """Gray base with a red box (class 1) and a blue box (class 2); masks are
    asserted at box centers, away from LANCZOS blend edges.
    灰底加红框（class 1）与蓝框（class 2）；mask 断言取框中心，避开 LANCZOS
    混合边缘。"""
    image = Image.new("RGB", size, (10, 12, 14))
    red = Image.new("RGB", (300, 200), (200, 30, 40))
    image.paste(red, (0, 0))
    blue = Image.new("RGB", (300, 300), (20, 50, 200))
    image.paste(blue, (400, 300))
    return image


def _plan(
    categories: tuple[str, ...] = (
        "small-vehicle", "large-vehicle", "building-outline"
    ),
) -> VisualTaskPlan:
    return VisualTaskPlan(
        version="visual-task-plan-v5",
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
    requested_roi: tuple[int, int, int, int] | None = None,
) -> MaterializedVisualView:
    if mode == "full_image":
        return MaterializedVisualView(
            image_id=image_id,
            view_mode="full_image",
            source_size=source_size,
            crop_xyxy=(0, 0, *source_size),
            crop_size=source_size,
        )
    # Quantized views are derived from the real quantizer so the audit
    # geometry is always self-consistent (the schema rejects fabricated
    # values). 量化视图由真实量化器推导，保证审计几何自洽（schema 拒绝伪造值）。
    quantized = materialize_quantized_roi(source_size, requested_roi)
    return MaterializedVisualView(
        image_id=image_id,
        view_mode="quantized_roi",
        source_size=source_size,
        crop_xyxy=quantized.crop_xyxy,
        crop_size=quantized.crop_size,
        requested_roi_xyxy_0_999=quantized.requested_roi_xyxy_0_999,
        requested_pixel_xyxy=quantized.requested_pixel_xyxy,
        roi_quantum=quantized.roi_quantum,
        quantized_side=quantized.quantized_side,
        ideal_square_xyxy=quantized.ideal_square_xyxy,
        was_clipped=quantized.was_clipped,
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
    segmenters: dict[str, _FakeSegmenter] | None = None,
    preprocessing: EvidencePreprocessing | None = None,
    catalog_data: dict | None = None,
) -> ObjectEvidenceExecutor:
    return ObjectEvidenceExecutor(
        catalog=EvidenceCatalog(catalog_data or _CATALOG_DATA),
        policy=policy or _policy(),
        yolo_client=yolo,
        yolo_device="cpu" if yolo is not None else None,
        yolo_image_size=MODEL_INPUT_SIZE if yolo is not None else None,
        segmenter_clients=segmenters or {},
        preprocessing=preprocessing or EvidencePreprocessing(),
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


def _audit(execution, *, layer: str) -> list[dict]:
    return [
        record.model_dump()
        for record in execution.bundle.call_audit
        if record.layer == layer
    ]


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


def test_yolo_runs_once_per_tile_and_keeps_requested_leaves() -> None:
    yolo = _FakeYolo(("small_vehicle", "large_vehicle", "building"))
    execution = _execute(_executor(yolo=yolo))
    # One 1000x800 ROI is a single stretched tile: one call at strict 1024.
    # 单个 1000x800 ROI 是一个拉伸 tile：1024 方形下仅一次调用。
    assert len(yolo.calls) == 1
    assert yolo.calls[0].size == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    assert set(execution.bundle.leaf_states) == {
        "small-vehicle",
        "large-vehicle",
        "building-outline",
    }
    assert all(state == "hit" for state in execution.bundle.leaf_states.values())
    assert execution.bundle.rois[0].expanded_xyxy == (0, 0, 1000, 800)
    assert [tile.tile_id for tile in execution.bundle.tiles] == ["full-r0-c0"]
    # Detections are inverse-mapped to the ROI pixel frame through the tile
    # scale (1024/1000, 1024/800). 检测经 tile 比例 (1024/1000, 1024/800)
    # 逆映射回 ROI 像素坐标系。
    box = execution.bundle.detections[0].global_xyxy
    assert box == pytest.approx(
        (10 * 1000 / 1024, 10 * 800 / 1024, 40 * 1000 / 1024, 40 * 800 / 1024)
    )
    yolo_audits = _audit(execution, layer="yolo")
    assert len(yolo_audits) == 1
    assert yolo_audits[0]["tile_id"] == "full-r0-c0"
    assert yolo_audits[0]["status"] == "succeeded"
    assert yolo_audits[0]["input_size"] == (1024, 1024)


def test_executor_consumes_exact_quantized_roi_pixels() -> None:
    yolo = _FakeYolo(("small_vehicle", "large_vehicle"))
    view = _view(
        source_size=(2048, 1536),
        mode="quantized_roi",
        requested_roi=(500, 500, 999, 999),
    )
    execution = _execute(
        _executor(yolo=yolo),
        plan=_plan(("small-vehicle", "large-vehicle")),
        images={"img1": _image((2048, 1536))},
        views=(view,),
    )
    # The quantized crop stays a single stretched 1024-square tile.
    # 量化裁切仍是一个拉伸的 1024 方形 tile。
    assert yolo.calls[0].size == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    assert execution.bundle.rois[0].core_xyxy == view.crop_xyxy
    assert [tile.tile_id for tile in execution.bundle.tiles] == [
        "quantized_roi-0-r0-c0"
    ]
    # Inverse map through the tile scale, then offset by the ROI box; the
    # expectation is derived from the quantizer's own geometry, never guessed.
    # 先经 tile 比例逆映射，再偏移 ROI 框；期望值由量化器自身几何推导，绝不猜测。
    x0, y0, x1, y1 = view.crop_xyxy
    crop_width, crop_height = x1 - x0, y1 - y0
    box = execution.bundle.detections[0].global_xyxy
    assert box == pytest.approx(
        (
            x0 + 10 * crop_width / 1024,
            y0 + 10 * crop_height / 1024,
            x0 + 40 * crop_width / 1024,
            y0 + 40 * crop_height / 1024,
        )
    )


def test_executor_requires_assistance_and_materialized_views() -> None:
    direct = VisualTaskPlan(
        version="visual-task-plan-v5",
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
    audit = _audit(execution, layer="yolo")[0]
    assert audit["status"] == "failed"
    assert audit["error_code"] == "RuntimeError"
    assert audit["logical_model_id"] == "yolo-test-v1"


def test_bundle_is_json_safe_and_does_not_persist_confidence() -> None:
    execution = _execute(
        _executor(yolo=_FakeYolo(("small_vehicle",))),
        plan=_plan(("small-vehicle", "large-vehicle")),
    )
    payload = json.loads(execution.bundle.model_dump_json())
    assert "confidence" not in json.dumps(payload)
    assert "base64" not in json.dumps(payload)
    assert execution.masks == {}


def test_constructor_rejects_non_1024_yolo_image_size() -> None:
    with pytest.raises(ValueError, match="1024"):
        ObjectEvidenceExecutor(
            catalog=EvidenceCatalog(_CATALOG_DATA),
            policy=_policy(),
            yolo_client=_FakeYolo(("small_vehicle",)),
            yolo_device="cpu",
            yolo_image_size=640,
            segmenter_clients={},
            preprocessing=EvidencePreprocessing(),
        )


# ── bounded concurrency (14.9) / 有界并发 ────────────────────────────────


def test_yolo_concurrency_peak_is_bounded_by_pool() -> None:
    # 2048x1536 -> 4 tiles; a latch of 4 proves all run in parallel.
    # 2048x1536 -> 4 个 tile；闩锁 4 证明全部并行执行。
    yolo = _FakeYolo(("small_vehicle",), latch_count=4)
    execution = _execute(
        _executor(yolo=yolo),
        plan=_plan(("small-vehicle",)),
        images={"img1": _image((2048, 1536))},
        views=(_view(source_size=(2048, 1536)),),
    )
    assert len(yolo.calls) == 4
    assert yolo.peak == 4
    assert execution.bundle.leaf_states["small-vehicle"] == "hit"
    assert [tile.tile_id for tile in execution.bundle.tiles] == [
        "full-r0-c0",
        "full-r0-c1",
        "full-r1-c0",
        "full-r1-c1",
    ]


def test_yolo_concurrency_respects_max_tile_concurrency() -> None:
    # 2000x2000 -> 4 tiles but only 2 workers: pairs, never 3 in flight.
    # 2000x2000 -> 4 个 tile 但只有 2 个 worker：成对执行，绝不超过 2 并发。
    yolo = _FakeYolo(("small_vehicle",), latch_count=2)
    preprocessing = EvidencePreprocessing(max_tile_concurrency=2)
    execution = _execute(
        _executor(yolo=yolo, preprocessing=preprocessing),
        plan=_plan(("small-vehicle",)),
        images={"img1": _image((2000, 2000))},
        views=(_view(source_size=(2000, 2000)),),
    )
    assert len(yolo.calls) == 4
    assert yolo.peak == 2
    assert execution.bundle.leaf_states["small-vehicle"] == "hit"


def test_out_of_order_yolo_completion_produces_identical_bundle() -> None:
    def run(delay: dict[int, float]):
        yolo = _FakeYolo(("small_vehicle",), delay_for=delay)
        execution = _execute(
            _executor(yolo=yolo),
            plan=_plan(("small-vehicle",)),
            images={"img1": _image((2000, 2000))},
            views=(_view(source_size=(2000, 2000)),),
        )
        return execution.bundle.model_dump_json()

    # Staggered delays force different completion orders across the two runs;
    # merging by stable tile index must make the bundles byte-identical.
    # 错峰延迟迫使两次运行以不同顺序完成；按稳定 tile index 合并必须使两个
    # bundle 字节级相同。
    slow_first = run({0: 0.3, 1: 0.2, 2: 0.1})
    slow_last = run({1: 0.1, 2: 0.2, 3: 0.3})
    assert slow_first == slow_last


# ── tile-level YOLO failure isolation and mapping (14.10) ────────────────


def test_single_tile_yolo_failure_is_isolated_and_hit_wins() -> None:
    yolo = _FakeYolo(
        ("small_vehicle",),
        box_by_index={0: (10.0, 10.0, 40.0, 40.0)},
        error_for={1: RuntimeError("secret leak /private/weights.pt")},
    )
    execution = _execute(
        _executor(yolo=yolo),
        plan=_plan(("small-vehicle",)),
        images={"img1": _image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    dumped = execution.bundle.model_dump_json()
    assert "secret" not in dumped
    # Evidence exists on the healthy tile, so the leaf is still a hit.
    # 健康 tile 上存在证据，因此叶子仍是 hit。
    assert execution.bundle.leaf_states["small-vehicle"] == "hit"
    assert len(execution.bundle.detections) == 1
    audits = _audit(execution, layer="yolo")
    assert [audit["tile_id"] for audit in audits] == ["full-r0-c0", "full-r0-c1"]
    failed = [audit for audit in audits if audit["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["tile_id"] == "full-r0-c1"
    assert failed[0]["error_code"] == "RuntimeError"


def test_remainder_tile_boxes_inverse_map_with_scale_and_offset() -> None:
    yolo = _FakeYolo(
        ("small_vehicle",),
        box_by_index={1: (488.0, 0.0, 536.0, 1024.0)},
    )
    execution = _execute(
        _executor(yolo=yolo),
        plan=_plan(("small-vehicle",)),
        images={"img1": _image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    assert execution.bundle.leaf_states["small-vehicle"] == "hit"
    assert len(execution.bundle.detections) == 1
    box = execution.bundle.detections[0].global_xyxy
    assert box == pytest.approx(
        (
            1024 + 488 * 976 / 1024,
            0.0,
            1024 + 536 * 976 / 1024,
            1024.0,
        )
    )


def test_degenerate_inverse_box_is_dropped_with_stable_outcome() -> None:
    yolo = _FakeYolo(
        ("small_vehicle",),
        box_by_index={0: (100.0, 100.0, 100.0, 200.0)},
    )
    execution = _execute(
        _executor(yolo=yolo),
        plan=_plan(("small-vehicle",)),
    )
    assert execution.bundle.detections == []
    assert execution.bundle.leaf_states["small-vehicle"] == "unsupported"
    assert any(
        outcome.leaf_category == "small-vehicle"
        and outcome.layer == "yolo"
        and outcome.state == "missing"
        and outcome.error_code == "degenerate_box"
        for outcome in execution.outcomes
    )


def test_unexpected_model_input_size_fails_closed() -> None:
    yolo = _FakeYolo(("small_vehicle",), wrong_input_size=True)
    execution = _execute(
        _executor(yolo=yolo),
        plan=_plan(("small-vehicle",)),
    )
    audit = _audit(execution, layer="yolo")[0]
    assert audit["status"] == "failed"
    assert audit["error_code"] == "unexpected_model_input_size"
    assert any(
        outcome.leaf_category == "small-vehicle"
        and outcome.layer == "yolo"
        and outcome.state == "error"
        and outcome.error_code == "unexpected_model_input_size"
        for outcome in execution.outcomes
    )


def test_non_yolo_leaves_make_zero_yolo_calls() -> None:
    yolo = _FakeYolo(("vehicle",))
    water = _FakeSegmenter(labels={2: "water"})
    execution = _execute(
        _executor(yolo=yolo, segmenters={"seg_001": water}, catalog_data=_SEG_CATALOG_DATA),
        plan=_plan(("water",)),
        images={"img1": _region_image((1000, 800))},
        views=(_view(source_size=(1000, 800)),),
    )
    assert yolo.calls == []
    assert execution.bundle.leaf_states["water"] == "hit"
    mask = execution.masks[("full", "water")]
    assert mask.size == (1000, 800)
    assert mask.getpixel((550, 450)) == 255
    assert mask.getpixel((150, 100)) == 0


# ── SegFormer aggregation (14.11 / 26 §6) ────────────────────────────────


def test_segformer_calls_once_per_roi_binding() -> None:
    """Fresh SegFormer runs once per (ROI, binding) on the whole ROI under
    the pad protocol — never per leaf and never per YOLO tile — and the
    restored class map is cropped back to the exact ROI crop size.
    新鲜 SegFormer 在 pad 协议下按（ROI，binding）对整张 ROI 各调用一次——
    绝不按 leaf、也绝不按 YOLO tile——恢复后的 class map 裁切回精确 ROI
    crop 尺寸。"""
    seg_001 = _FakeSegmenter(labels={1: "vehicle"})
    seg_002 = _FakeSegmenter(labels={2: "building"})
    execution = _execute(
        _executor(segmenters={"seg_001": seg_001, "seg_002": seg_002},
                  catalog_data=_SEG_CATALOG_DATA),
        plan=_plan(("vehicle", "building")),
        images={"img1": _region_image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    assert execution.bundle.leaf_states == {"vehicle": "hit", "building": "hit"}
    assert [s.model_dump() for s in execution.bundle.segments] == [
        {"leaf_category": "vehicle", "roi_id": "full"},
        {"leaf_category": "building", "roi_id": "full"},
    ]
    vehicle = execution.masks[("full", "vehicle")]
    building = execution.masks[("full", "building")]
    assert vehicle.size == (2000, 1024)
    assert building.size == (2000, 1024)
    assert vehicle.getpixel((150, 100)) == 255  # red box / 红框
    assert vehicle.getpixel((550, 450)) == 0    # blue box stays background / 蓝框保持背景
    assert vehicle.getpixel((1500, 500)) == 0   # gray remainder / 灰色区域
    assert building.getpixel((550, 450)) == 255
    assert building.getpixel((150, 100)) == 0
    # One strict 1024-square call per (ROI, binding): the whole ROI is padded
    # to 1024 multiples and resized once, never tiled.
    # 每个（ROI，binding）恰好一次严格 1024 方形调用：整张 ROI 一次性 padding
    # 到 1024 倍数并缩放，绝不切 tile。
    assert len(seg_001.calls) == 1
    assert len(seg_002.calls) == 1
    assert seg_001.calls[0].size == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    assert seg_002.calls[0].size == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    # The geometry record carries the pad geometry of the 2000x1024 ROI.
    # 几何记录携带 2000x1024 ROI 的 pad 几何。
    [preprocess] = execution.bundle.segformer_preprocess
    assert preprocess.roi_id == "full"
    assert preprocess.source_size == (2000, 1024)
    assert preprocess.padded_size == (2048, 1024)
    assert preprocess.padding_right == 48
    assert preprocess.padding_bottom == 0
    assert preprocess.model_input_size == (1024, 1024)
    assert preprocess.padding_mode == "constant-black-right-bottom"
    assert preprocess.rgb_interpolation == "lanczos"
    assert preprocess.mask_inverse_interpolation == "nearest"
    # Audits merge in roi order -> binding order with no fabricated tile id.
    # 审计按 roi order -> binding order 合并，且不伪造 tile id。
    seg_audits = _audit(execution, layer="segformer")
    assert [(a["roi_id"], a["logical_model_id"], a["tile_id"])
            for a in seg_audits] == [
        ("full", "seg-test-v1", None),
        ("full", "seg-test-v1", None),
    ]
    assert all(a["input_size"] == (1024, 1024) for a in seg_audits)


def test_one_restored_class_map_serves_all_requested_leaves_of_a_binding() -> None:
    """One (ROI, binding) call restores the class map once and serves every
    still-missing requested leaf of that binding — never one call per leaf.
    一次（ROI，binding）调用只恢复一次 class map，并为该 binding 下全部仍
    缺失的请求叶子生成 mask——绝不按 leaf 重复调用。"""
    catalog_data = {
        "catalog_version": "test-catalog-seg-same-binding-v1",
        "aliases": {},
        "parents": {},
        "leaves": {
            "vehicle": {
                "yolo_labels": [],
                "yolo_enabled": False,
                "segformer_labels": ["vehicle"],
                "segformer_binding": "seg_001",
                "segformer_enabled": True,
            },
            "water": {
                "yolo_labels": [],
                "yolo_enabled": False,
                "segformer_labels": ["water"],
                "segformer_binding": "seg_001",
                "segformer_enabled": True,
            },
        },
        "task_capabilities": {
            task: ["vehicle", "water"]
            for task in ("counting", "fine_grained_counting", "general_vqa", "grounding")
        },
    }
    seg = _FakeSegmenter(labels={1: "vehicle", 2: "water"})
    execution = _execute(
        _executor(segmenters={"seg_001": seg}, catalog_data=catalog_data),
        plan=_plan(("vehicle", "water")),
        images={"img1": _region_image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    assert len(seg.calls) == 1
    assert execution.bundle.leaf_states == {"vehicle": "hit", "water": "hit"}
    vehicle = execution.masks[("full", "vehicle")]
    water = execution.masks[("full", "water")]
    assert vehicle.getpixel((150, 100)) == 255
    assert vehicle.getpixel((550, 450)) == 0
    assert water.getpixel((550, 450)) == 255
    assert water.getpixel((150, 100)) == 0
    # One geometry record for the ROI, one audit for the single call.
    # ROI 一条几何记录，单次调用一条审计。
    assert len(execution.bundle.segformer_preprocess) == 1
    seg_audits = _audit(execution, layer="segformer")
    assert len(seg_audits) == 1
    assert seg_audits[0]["tile_id"] is None


def test_unrequested_classes_stay_background() -> None:
    # The model's map contains class 2 (blue), but the plan never requests it.
    # 模型 map 含 class 2（蓝色），但计划从未请求它。
    seg = _FakeSegmenter(labels={1: "vehicle", 2: "building"})
    execution = _execute(
        _executor(segmenters={"seg_001": seg}, catalog_data=_SEG_CATALOG_DATA),
        plan=_plan(("vehicle",)),
        images={"img1": _region_image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    mask = execution.masks[("full", "vehicle")]
    assert mask.getpixel((150, 100)) == 255
    assert mask.getpixel((550, 450)) == 0
    assert "building" not in execution.masks
    assert len(seg.calls) == 1


def test_class_map_mismatch_fails_closed() -> None:
    seg = _FakeSegmenter(labels={1: "person"})
    execution = _execute(
        _executor(segmenters={"seg_001": seg}, catalog_data=_SEG_CATALOG_DATA),
        plan=_plan(("vehicle",)),
        images={"img1": _region_image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    assert execution.bundle.leaf_states["vehicle"] == "error"
    assert execution.masks == {}
    assert any(
        outcome.leaf_category == "vehicle"
        and outcome.layer == "segformer"
        and outcome.state == "error"
        and outcome.error_code == "class_map_mismatch"
        for outcome in execution.outcomes
    )
    assert any(
        state.leaf_category == "vehicle"
        and state.layer == "segformer"
        and state.state == "error"
        for state in execution.layer_states
    )


def test_yolo_hit_skips_segformer_with_not_run_record() -> None:
    yolo = _FakeYolo(("vehicle", "building"))
    seg_001 = _FakeSegmenter(labels={1: "vehicle"})
    seg_002 = _FakeSegmenter(labels={2: "building"})
    execution = _execute(
        _executor(yolo=yolo, segmenters={"seg_001": seg_001, "seg_002": seg_002},
                  catalog_data=_SEG_CATALOG_DATA),
        plan=_plan(("vehicle", "building")),
        images={"img1": _region_image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    assert execution.bundle.leaf_states == {"vehicle": "hit", "building": "hit"}
    assert seg_001.calls == []
    assert seg_002.calls == []
    assert execution.masks == {}
    not_run = [
        state
        for state in execution.layer_states
        if state.layer == "segformer" and state.state == "not_run"
    ]
    assert {state.leaf_category for state in not_run} == {"vehicle", "building"}


def test_empty_segformer_mask_is_missing() -> None:
    seg = _FakeSegmenter(labels={1: "vehicle"})
    execution = _execute(
        _executor(segmenters={"seg_001": seg}, catalog_data=_SEG_CATALOG_DATA),
        plan=_plan(("vehicle",)),
        images={"img1": _image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    assert execution.bundle.leaf_states["vehicle"] == "missing"
    assert execution.bundle.missing_leaves == ["vehicle"]
    assert execution.bundle.segments == []
    assert execution.masks == {}
    assert any(
        outcome.leaf_category == "vehicle"
        and outcome.layer == "segformer"
        and outcome.state == "missing"
        for outcome in execution.outcomes
    )


def test_segformer_roi_failure_is_isolated_fail_closed() -> None:
    # Vehicle's (ROI, binding) call fails: incomplete evidence must not crash
    # the execution, must not fabricate a mask, and must not corrupt the other
    # binding's evidence. 车辆（ROI，binding）调用失败：不完整证据不得使执行
    # 崩溃、不得伪造 mask、不得污染另一个 binding 的证据。
    seg_001 = _FakeSegmenter(
        labels={1: "vehicle"},
        error_for={0: RuntimeError("secret seg leak")},
    )
    seg_002 = _FakeSegmenter(labels={2: "building"})
    execution = _execute(
        _executor(segmenters={"seg_001": seg_001, "seg_002": seg_002},
                  catalog_data=_SEG_CATALOG_DATA),
        plan=_plan(("vehicle", "building")),
        images={"img1": _region_image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    dumped = execution.bundle.model_dump_json()
    assert "secret" not in dumped
    assert execution.bundle.leaf_states == {"vehicle": "error", "building": "hit"}
    assert ("full", "vehicle") not in execution.masks
    assert execution.masks[("full", "building")].getpixel((550, 450)) == 255
    assert any(
        outcome.leaf_category == "vehicle"
        and outcome.layer == "segformer"
        and outcome.state == "error"
        and outcome.error_code == "RuntimeError"
        for outcome in execution.outcomes
    )
    seg_audits = _audit(execution, layer="segformer")
    failed = [a for a in seg_audits if a["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["roi_id"] == "full"
    assert failed[0]["tile_id"] is None
    assert failed[0]["error_code"] == "RuntimeError"


def test_segformer_audit_order_is_roi_then_binding() -> None:
    seg_001 = _FakeSegmenter(labels={1: "vehicle"}, grid_source="empty")
    seg_002 = _FakeSegmenter(labels={2: "building"}, grid_source="empty")
    views = (
        _view(source_size=(2048, 1536), mode="quantized_roi",
              requested_roi=(500, 500, 999, 999)),
        _view(source_size=(2048, 1536), mode="quantized_roi",
              requested_roi=(0, 0, 499, 499)),
    )
    execution = _execute(
        _executor(segmenters={"seg_001": seg_001, "seg_002": seg_002},
                  catalog_data=_SEG_CATALOG_DATA),
        plan=_plan(("vehicle", "building")),
        images={"img1": _image((2048, 1536))},
        views=views,
    )
    seg_audits = _audit(execution, layer="segformer")
    assert [(a["roi_id"], a["tile_id"]) for a in seg_audits] == [
        ("quantized_roi-0", None),
        ("quantized_roi-0", None),
        ("quantized_roi-1", None),
        ("quantized_roi-1", None),
    ]


def test_segformer_concurrency_peak_is_bounded() -> None:
    # Two ROIs make two (ROI, binding) groups; a latch of 2 proves they run
    # in parallel and never exceed the pool bound.
    # 两个 ROI 形成两个（ROI，binding）组；闩锁 2 证明并行执行且不超过池上限。
    seg = _FakeSegmenter(labels={1: "vehicle"}, grid_source="empty", latch_count=2)
    views = (
        _view(source_size=(2048, 1536), mode="quantized_roi",
              requested_roi=(500, 500, 999, 999)),
        _view(source_size=(2048, 1536), mode="quantized_roi",
              requested_roi=(0, 0, 499, 499)),
    )
    execution = _execute(
        _executor(segmenters={"seg_001": seg}, catalog_data=_SEG_CATALOG_DATA),
        plan=_plan(("vehicle",)),
        images={"img1": _image((2048, 1536))},
        views=views,
    )
    assert len(seg.calls) == 2
    assert seg.peak == 2
    assert execution.bundle.leaf_states["vehicle"] == "missing"


def test_legacy_v1_preprocessing_fails_closed_for_segformer() -> None:
    """Under the legacy v1 identity a fresh SegFormer call is never made:
    every still-missing segformer-capable leaf records the stable
    legacy_segformer_protocol_unsupported error and no segment call happens.
    在旧 v1 身份下绝不发起新鲜 SegFormer 调用：每个仍缺失且具备 segformer
    能力的叶子记录稳定 legacy_segformer_protocol_unsupported 错误，且零
    segment 调用。"""
    v1 = EvidencePreprocessing(
        version="greedy-1024-stretch-v1",
        yolo_version=None,
        segformer_version=None,
        segformer_padding_mode=None,
        segformer_rgb_interpolation=None,
        segformer_mask_inverse_interpolation=None,
    )
    seg = _FakeSegmenter(labels={1: "vehicle"})
    execution = _execute(
        _executor(segmenters={"seg_001": seg}, catalog_data=_SEG_CATALOG_DATA,
                  preprocessing=v1),
        plan=_plan(("vehicle",)),
        images={"img1": _region_image((2000, 1024))},
        views=(_view(source_size=(2000, 1024)),),
    )
    assert seg.calls == []
    assert execution.bundle.segments == []
    assert execution.bundle.leaf_states["vehicle"] == "error"
    assert any(
        outcome.leaf_category == "vehicle"
        and outcome.layer == "segformer"
        and outcome.state == "error"
        and outcome.error_code == "legacy_segformer_protocol_unsupported"
        for outcome in execution.outcomes
    )


def test_legacy_v1_preprocessing_keeps_yolo_only_branch_working() -> None:
    """Under the legacy v1 identity the unchanged YOLO phase still runs: v1
    only rejects fresh SegFormer calls, never the YOLO tile path.
    在旧 v1 身份下不变的 YOLO 阶段仍可运行：v1 只拒绝新鲜 SegFormer 调用，
    绝不拒绝 YOLO tile 路径。"""
    v1 = EvidencePreprocessing(
        version="greedy-1024-stretch-v1",
        yolo_version=None,
        segformer_version=None,
        segformer_padding_mode=None,
        segformer_rgb_interpolation=None,
        segformer_mask_inverse_interpolation=None,
    )
    yolo = _FakeYolo(("small_vehicle",))
    execution = _execute(
        _executor(yolo=yolo, preprocessing=v1),
        plan=_plan(("small-vehicle",)),
    )
    assert len(yolo.calls) == 1
    assert execution.bundle.leaf_states["small-vehicle"] == "hit"
    assert execution.bundle.preprocessing_version == "greedy-1024-stretch-v1"
