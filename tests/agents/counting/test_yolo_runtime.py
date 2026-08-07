"""Contract tests for the YOLO model store and OBB counting backend.

YOLO 模型库与 OBB 计数后端契约测试：hash/任务/类别映射校验、惰性缓存、
alias/composite 解析、边界去重、max detections 传递、fake runtime 计数。
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from agents.counting.backends.yolo_model_store import YoloModelStore
from agents.counting.backends.yolo_obb import YoloOBBCountingBackend, _detector_duplicate_pairs
from agents.counting.backends.base import CountingBackendOutcome, CountingRequest
from agents.counting.schema import (
    CountTargetSpec,
    GlobalPointObservation,
    TileSpec,
)
from agents.counting.settings import CountingSettings, YoloDetectorSettings
from agents.errors import (
    DetectorClassMapMismatchError,
    DetectorTaskMismatchError,
    DetectorWeightsHashMismatchError,
    DetectorWeightsMissingError,
)
from data.schema import GroundTruth, ImageRef, UnifiedSample

REPO_ROOT = Path(__file__).resolve().parents[3]

_TARGET = CountTargetSpec(
    canonical_label="car",
    inclusion_rule="visible vehicle",
    exclusion_rule="occluded more than half",
)


def _weights(tmp_path: Path, content: bytes = b"fake weights") -> Path:
    path = tmp_path / "det.pt"
    path.write_bytes(content)
    return path


def _detector(tmp_path: Path, **overrides) -> YoloDetectorSettings:
    values = dict(
        name="det-a",
        enabled=True,
        weights=_weights(tmp_path),
        model_id="m1",
        sha256=hashlib.sha256(b"fake weights").hexdigest(),
        classes=["car", "truck"],
    )
    values.update(overrides)
    return YoloDetectorSettings(**values)


def _detector_without_file(**overrides) -> YoloDetectorSettings:
    """A structurally valid detector whose weights file is never touched.
    结构合法、但权重文件绝不触碰的检测器。"""
    values = dict(
        name="det-a",
        enabled=True,
        weights=Path("/nonexistent/det.pt"),
        model_id="m1",
        sha256="a" * 64,
        classes=["car", "truck"],
    )
    values.update(overrides)
    return YoloDetectorSettings(**values)


class _FakeRuntimeModel:
    """Ultralytics-like fake runtime: no weights, deterministic results.
    类似 Ultralytics 的假运行时：无权重、确定性结果。"""

    task = "obb"
    names = {0: "car", 1: "truck"}

    def __init__(self, results: list[SimpleNamespace] | None = None) -> None:
        self.results = results or []
        self.predict_kwargs_list: list[dict[str, Any]] = []

    def predict(self, **kwargs) -> list[SimpleNamespace]:
        self.predict_kwargs_list.append(dict(kwargs))
        return self.results


def _obb(polygons: list[list[list[float]]], class_ids: list[float], confs: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        obb=SimpleNamespace(xyxyxyxy=polygons, cls=class_ids, conf=confs)
    )


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="counting",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="How many cars?",
        ground_truth=GroundTruth(answers=["2"]),
    )


def _request(tmp_path: Path, image: Image.Image) -> CountingRequest:
    return CountingRequest(
        sample=_sample(),
        image=image,
        target=_TARGET,
        artifact_dir=tmp_path / "run",
    )


def _context() -> object:
    class _Context:
        pass

    return _Context()


# ── 模型库 / model store ──────────────────────────────────────────────────


def test_store_raises_when_weights_missing(tmp_path: Path) -> None:
    detector = _detector(tmp_path)
    detector.weights = tmp_path / "missing.pt"
    with pytest.raises(DetectorWeightsMissingError, match="missing.pt"):
        YoloModelStore(loader=lambda path: _FakeRuntimeModel()).get(detector)


def test_store_raises_on_hash_mismatch(tmp_path: Path) -> None:
    detector = _detector(tmp_path, sha256="b" * 64)
    with pytest.raises(DetectorWeightsHashMismatchError, match="digest mismatch"):
        YoloModelStore(loader=lambda path: _FakeRuntimeModel()).get(detector)


def test_store_raises_on_task_mismatch(tmp_path: Path) -> None:
    model = _FakeRuntimeModel()
    model.task = "detect"
    with pytest.raises(DetectorTaskMismatchError, match="expected task"):
        YoloModelStore(loader=lambda path: model).get(_detector(tmp_path))


def test_store_raises_on_class_map_mismatch(tmp_path: Path) -> None:
    model = _FakeRuntimeModel()
    model.names = {0: "car", 1: "bus"}
    with pytest.raises(DetectorClassMapMismatchError, match="class map"):
        YoloModelStore(loader=lambda path: model).get(_detector(tmp_path))


def test_store_caches_verified_model(tmp_path: Path) -> None:
    loads: list[str] = []
    model = _FakeRuntimeModel()
    store = YoloModelStore(loader=lambda path: loads.append(path) or model)
    detector = _detector(tmp_path)
    first = store.get(detector)
    second = store.get(detector)
    assert first is second is model
    assert loads == [str(detector.weights.resolve())]
    assert store.has(detector.weights, detector.sha256) is True


def test_store_loads_same_key_exactly_once_under_concurrency(tmp_path: Path) -> None:
    """20 concurrent get() calls must load the model exactly once and all
    callers must receive the same object. 20 个并发 get() 必须只加载一次模型，
    且所有调用者获得同一对象。"""
    import threading

    loads: list[str] = []
    model = _FakeRuntimeModel()
    store = YoloModelStore(
        loader=lambda path: loads.append(path) or model
    )
    detector = _detector(tmp_path)
    results: list[Any] = []
    barrier = threading.Barrier(20)

    def _worker() -> None:
        barrier.wait()
        results.append(store.get(detector))

    threads = [threading.Thread(target=_worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert loads == [str(detector.weights.resolve())]
    assert len(results) == 20
    assert all(result is model for result in results)


def test_store_failed_load_can_be_retried(tmp_path: Path) -> None:
    """A first load failure must not poison the cache; a later call may retry
    and succeed. 首次加载失败不得污染缓存；后续调用可重试并成功。"""
    attempts: list[str] = []
    model = _FakeRuntimeModel()

    def _flaky_loader(path: str) -> Any:
        attempts.append(path)
        if len(attempts) == 1:
            raise RuntimeError("transient load failure")
        return model

    store = YoloModelStore(loader=_flaky_loader)
    detector = _detector(tmp_path)
    with pytest.raises(RuntimeError, match="transient"):
        store.get(detector)
    assert store.get(detector) is model
    assert len(attempts) == 2


# ── 类别解析 / class resolution ───────────────────────────────────────────


def _backend(tmp_path: Path, detector: YoloDetectorSettings | None = None) -> YoloOBBCountingBackend:
    return YoloOBBCountingBackend(
        detector or _detector(tmp_path),
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: _FakeRuntimeModel()),
    )


def test_resolve_target_classes_direct_and_plural() -> None:
    backend = YoloOBBCountingBackend(
        _detector_without_file(), counting=CountingSettings(), model_store=YoloModelStore()
    )
    assert backend.resolve_target_classes(_TARGET) == frozenset({"car"})
    plural = CountTargetSpec(canonical_label="cars", inclusion_rule="r", exclusion_rule="e")
    assert backend.resolve_target_classes(plural) == frozenset({"car"})


def test_resolve_target_classes_alias_and_composite() -> None:
    detector = _detector_without_file(
        aliases={"vehicle": "car"},
        composite_targets={"convoy": ["car", "truck"]},
    )
    backend = YoloOBBCountingBackend(
        detector, counting=CountingSettings(), model_store=YoloModelStore()
    )
    alias_target = CountTargetSpec(canonical_label="vehicle", inclusion_rule="r", exclusion_rule="e")
    assert backend.resolve_target_classes(alias_target) == frozenset({"car"})
    composite_target = CountTargetSpec(canonical_label="convoy", inclusion_rule="r", exclusion_rule="e")
    assert backend.resolve_target_classes(composite_target) == frozenset({"car", "truck"})
    unknown = CountTargetSpec(canonical_label="plane", inclusion_rule="r", exclusion_rule="e")
    assert backend.resolve_target_classes(unknown) == frozenset()
    assert backend.supports(unknown) is False


def test_is_available_reflects_detector_enabled(tmp_path: Path) -> None:
    assert _backend(tmp_path).is_available() is True
    disabled = _backend(tmp_path, _detector(tmp_path, enabled=False))
    assert disabled.is_available() is False


def test_trace_profile_has_no_absolute_path(tmp_path: Path) -> None:
    profile = _backend(tmp_path).trace_profile()
    assert profile["detector_name"] == "det-a"
    assert profile["weights_file"] == "det.pt"
    assert str(tmp_path) not in str(profile)


# ── 计数执行 / counting execution ─────────────────────────────────────────


def _single_detection_polygon() -> list[list[float]]:
    return [[50.0, 50.0], [50.0, 150.0], [150.0, 150.0], [150.0, 50.0]]


def test_count_with_fake_runtime(tmp_path: Path) -> None:
    model = _FakeRuntimeModel(
        [_obb([_single_detection_polygon()], [0.0], [0.9])]
    )
    detector = _detector(tmp_path)
    store = YoloModelStore(loader=lambda path: model)
    backend = YoloOBBCountingBackend(detector, counting=CountingSettings(), model_store=store)
    outcome = asyncio.run(
        backend.count(_request(tmp_path, Image.new("RGB", (200, 200), (1, 2, 3))), _context())
    )
    assert isinstance(outcome, CountingBackendOutcome)
    assert outcome.counting.final_count == 1
    assert outcome.counting.status in {"completed", "completed_with_warnings"}
    assert outcome.counting.succeeded_tiles == ["r000_c000"]
    point = outcome.counting.global_points[0]
    assert point.accepted is True
    assert point.provenance is not None
    assert point.provenance.source_class == "car"
    assert point.provenance.weights_sha256 == detector.sha256
    assert outcome.trace["backend_kind"] == "yolo_obb"
    assert outcome.trace["resolved_target_classes"] == ["car"]


def test_count_passes_max_detections_and_parameters(tmp_path: Path) -> None:
    model = _FakeRuntimeModel([_obb([], [], [])])
    detector = _detector(tmp_path, max_detections=42)
    backend = YoloOBBCountingBackend(
        detector,
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: model),
    )
    asyncio.run(
        backend.count(_request(tmp_path, Image.new("RGB", (200, 200), (1, 2, 3))), _context())
    )
    kwargs = model.predict_kwargs_list[0]
    assert kwargs["max_det"] == 42
    assert kwargs["conf"] == detector.confidence
    assert kwargs["iou"] == detector.iou
    assert kwargs["imgsz"] == detector.image_size


def test_count_rejects_unrelated_classes(tmp_path: Path) -> None:
    """Detections of classes outside the resolved target are rejected and
    counted in the trace. 解析目标之外的类别检测被拒绝并计入 trace。"""
    model = _FakeRuntimeModel(
        [_obb([_single_detection_polygon(), _single_detection_polygon()], [0.0, 1.0], [0.9, 0.8])]
    )
    backend = YoloOBBCountingBackend(
        _detector(tmp_path),
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: model),
    )
    outcome = asyncio.run(
        backend.count(_request(tmp_path, Image.new("RGB", (200, 200), (1, 2, 3))), _context())
    )
    assert outcome.counting.final_count == 1  # car only / 仅 car
    assert outcome.trace["unrelated_class_rejected_count"] == 1


def test_count_unsupported_target_raises(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    request = CountingRequest(
        sample=_sample(),
        image=Image.new("RGB", (100, 100)),
        target=CountTargetSpec(canonical_label="plane", inclusion_rule="r", exclusion_rule="e"),
        artifact_dir=tmp_path / "run",
    )
    with pytest.raises(ValueError, match="unsupported target"):
        asyncio.run(backend.count(request, _context()))


def test_count_merges_same_tile_duplicates(tmp_path: Path) -> None:
    """Strongly overlapping detections in the same tile merge via boundary
    dedup. 同一切片内高度重叠的检测经边界去重合并。"""
    polygon_a = [[50.0, 50.0], [50.0, 150.0], [150.0, 150.0], [150.0, 50.0]]
    polygon_b = [[60.0, 60.0], [60.0, 140.0], [140.0, 140.0], [140.0, 60.0]]
    model = _FakeRuntimeModel([_obb([polygon_a, polygon_b], [0.0, 0.0], [0.9, 0.8])])
    backend = YoloOBBCountingBackend(
        _detector(tmp_path),
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: model),
    )
    outcome = asyncio.run(
        backend.count(_request(tmp_path, Image.new("RGB", (200, 200), (1, 2, 3))), _context())
    )
    assert outcome.counting.final_count == 1
    assert len(outcome.counting.merged_groups) == 1
    codes = {record.code for record in outcome.counting.warnings}
    assert "YOLO_DUPLICATE_MERGED" in codes


def test_count_tile_failure_is_recorded(tmp_path: Path) -> None:
    class _ExplodingModel(_FakeRuntimeModel):
        def predict(self, **kwargs):
            raise RuntimeError("runtime boom")

    backend = YoloOBBCountingBackend(
        _detector(tmp_path),
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: _ExplodingModel()),
    )
    outcome = asyncio.run(
        backend.count(_request(tmp_path, Image.new("RGB", (200, 200), (1, 2, 3))), _context())
    )
    assert outcome.counting.failed_tiles == ["r000_c000"]
    assert outcome.counting.status == "failed"
    codes = {record.code for record in outcome.counting.warnings}
    assert "YOLO_TILE_INFERENCE_FAILED" in codes


# ── 边界去重对 / duplicate pair detection ─────────────────────────────────


def _global_point(
    global_id: str,
    tile_id: str,
    x: int,
    y: int,
    *,
    near_boundary: bool,
    accepted: bool = True,
    source_class: str = "car",
) -> GlobalPointObservation:
    from agents.counting.schema import PointProvenance

    return GlobalPointObservation(
        global_id=global_id,
        target="car",
        source_tile_id=tile_id,
        local_id=global_id,
        local_x_norm=x,
        local_y_norm=y,
        local_radius_norm=0,
        global_x_px=x,
        global_y_px=y,
        global_x_norm=x,
        global_y_norm=y,
        radius_px=4.0,
        confidence=0.9,
        ownership_valid=True,
        near_core_boundary=near_boundary,
        accepted=accepted,
        short_evidence="e",
        provenance=PointProvenance(
            source="yolo_obb_center",
            source_class=source_class,
            obb_polygon_global_px=[[x - 4, y - 4], [x - 4, y + 4], [x + 4, y + 4], [x + 4, y - 4]],
        ),
    )


def test_duplicate_pairs_marks_adjacent_boundary_unresolved() -> None:
    from agents.counting.schema import PixelRect

    first_tile = TileSpec(
        tile_id="t0", row=0, col=0,
        crop_global=PixelRect(left=0, top=0, right=100, bottom=100),
        owner_core_global=PixelRect(left=0, top=0, right=50, bottom=100),
        owner_core_local=PixelRect(left=0, top=0, right=50, bottom=100),
        source_width=100, source_height=100, model_input_width=100, model_input_height=100,
    )
    second_tile = TileSpec(
        tile_id="t1", row=0, col=1,
        crop_global=PixelRect(left=50, top=0, right=100, bottom=100),
        owner_core_global=PixelRect(left=50, top=0, right=100, bottom=100),
        owner_core_local=PixelRect(left=0, top=0, right=50, bottom=100),
        source_width=100, source_height=100, model_input_width=100, model_input_height=100,
    )
    # Distance 21px lies between boundary_duplicate_center_px and twice it:
    # too far to merge, close enough to flag unresolved.
    # 距离 21px 位于 center_px 与两倍之间：不足以合并，但足以标记未解决。
    first = _global_point("g0", "t0", 49, 50, near_boundary=True)
    second = _global_point("g1", "t1", 70, 50, near_boundary=True)
    detector = YoloDetectorSettings(
        name="det-a", enabled=True, weights=Path("/nonexistent.pt"), model_id="m",
        sha256="a" * 64, classes=["car"],
        boundary_duplicate_center_px=16.0,
    )
    merged, unresolved = _detector_duplicate_pairs([first, second], [first_tile, second_tile], detector)
    assert merged == []
    assert unresolved == [("g0", "g1")]


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_backend_has_no_qwen_fallback_or_legacy_imports() -> None:
    source = (REPO_ROOT / "agents" / "counting" / "backends" / "yolo_obb.py").read_text(
        encoding="utf-8"
    )
    assert "spacers_agent" not in source
    assert "VRSBench" not in source
    assert "qwen" not in source.casefold().replace("qwen_point", "")


def test_backend_imports_do_not_load_weights() -> None:
    """Importing the YOLO backend must not load any weights.
    导入 YOLO 后端绝不加载任何权重。"""
    import agents.counting.backends.yolo_obb  # noqa: F401
    import agents.counting.backends.yolo_model_store  # noqa: F401

    for heavy in ("ultralytics", "onnxruntime"):
        assert heavy not in __import__("sys").modules, heavy
