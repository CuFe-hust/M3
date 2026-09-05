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
    DetectorWeightsPointerError,
)
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import ObjectDetectionOutput, RuntimeObjectDetectionClient

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
        executable_leaf_categories=(_TARGET.canonical_label,),
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


def test_store_rejects_git_lfs_pointer_before_runtime_load(tmp_path: Path) -> None:
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"a" * 64 + b"\nsize 123\n"
    )
    detector = _detector(tmp_path)
    detector.weights.write_bytes(pointer)
    detector.sha256 = hashlib.sha256(pointer).hexdigest()
    with pytest.raises(DetectorWeightsPointerError, match="actual binary"):
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


def test_resolve_target_classes_raw_alias_without_parent_expansion() -> None:
    detector = _detector_without_file(aliases={"passenger car": "car"})
    backend = YoloOBBCountingBackend(
        detector, counting=CountingSettings(), model_store=YoloModelStore()
    )
    alias_target = CountTargetSpec(canonical_label="passenger-car", inclusion_rule="r", exclusion_rule="e")
    assert backend.resolve_target_classes(alias_target) == frozenset({"car"})
    parent_target = CountTargetSpec(canonical_label="vehicle", inclusion_rule="r", exclusion_rule="e")
    assert backend.resolve_target_classes(parent_target) == frozenset()
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
        executable_leaf_categories=("plane",),
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


def test_all_tiles_failed_propagates_stable_error(tmp_path: Path) -> None:
    """When every tile fails the backend raises DetectorInferenceError instead
    of returning a fake zero result. 所有 tile 均失败时后端抛出
    DetectorInferenceError，而非返回伪造的零结果。"""
    from agents.errors import DetectorInferenceError

    class _ExplodingModel(_FakeRuntimeModel):
        def predict(self, **kwargs):
            raise RuntimeError("runtime boom")

    backend = YoloOBBCountingBackend(
        _detector(tmp_path),
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: _ExplodingModel()),
    )
    with pytest.raises(DetectorInferenceError, match="ALL_YOLO_TILES_FAILED"):
        asyncio.run(
            backend.count(_request(tmp_path, Image.new("RGB", (200, 200), (1, 2, 3))), _context())
        )


def test_partial_tile_failure_returns_partial_with_evidence(tmp_path: Path) -> None:
    """One tile succeeds and one fails: partial result, kept evidence, no
    global error. 一个 tile 成功一个失败：partial 结果、保留证据、无全局错误。"""
    class _PartlyFailingModel(_FakeRuntimeModel):
        def __init__(self) -> None:
            super().__init__([])
            self.calls = 0

        def predict(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first tile boom")
            # Polygon lies inside the second tile's owner core so its
            # evidence is accepted. 多边形位于第二个 tile 的 owner core 内，
            # 使其证据被接受。
            return [_obb([[[200.0, 200.0], [200.0, 900.0], [900.0, 900.0], [900.0, 200.0]]], [0.0], [0.9])]

    backend = YoloOBBCountingBackend(
        _detector(tmp_path),
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: _PartlyFailingModel()),
    )
    outcome = asyncio.run(
        backend.count(_request(tmp_path, Image.new("RGB", (2000, 2000), (1, 2, 3))), _context())
    )
    assert outcome.counting.status == "partial"
    assert "r000_c000" in outcome.counting.failed_tiles
    assert len(outcome.counting.succeeded_tiles) == 8
    assert outcome.counting.final_count >= 1  # evidence from successful tiles kept
    codes = {record.code for record in outcome.counting.warnings}
    assert "YOLO_TILE_INFERENCE_FAILED" in codes


_SENSITIVE_TILE_ERROR = (
    "/home/user/private/model.pt "
    "C:\\\\secret\\\\models\\\\det.onnx "
    "sk-test-secret "
    "Bearer abcdef "
    "data:image/png;base64,AAAA"
)


def test_tile_warning_is_sanitized(tmp_path: Path) -> None:
    """Tile failure warnings must never contain raw exception text, paths,
    credentials, or Base64. tile 失败 warning 绝不包含原始异常文本、路径、
    凭据或 Base64。"""
    class _SensitiveModel(_FakeRuntimeModel):
        def predict(self, **kwargs):
            raise RuntimeError(_SENSITIVE_TILE_ERROR)

    backend = YoloOBBCountingBackend(
        _detector(tmp_path),
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: _SensitiveModel()),
    )
    with pytest.raises(Exception, match="ALL_YOLO_TILES_FAILED"):
        asyncio.run(
            backend.count(_request(tmp_path, Image.new("RGB", (200, 200), (1, 2, 3))), _context())
        )

    # Partial path: warnings surface on the outcome.
    # 部分失败路径：warning 出现在 outcome 上。
    class _SensitivePartial(_FakeRuntimeModel):
        def __init__(self) -> None:
            super().__init__([])
            self.calls = 0

        def predict(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(_SENSITIVE_TILE_ERROR)
            return [_obb([], [], [])]

    backend2 = YoloOBBCountingBackend(
        _detector(tmp_path),
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: _SensitivePartial()),
    )
    outcome = asyncio.run(
        backend2.count(_request(tmp_path, Image.new("RGB", (2000, 2000), (1, 2, 3))), _context())
    )
    warning = next(
        record for record in outcome.counting.warnings
        if record.code == "YOLO_TILE_INFERENCE_FAILED"
    )
    assert "r000_c000" in warning.message
    assert "RuntimeError" in warning.message
    for token in ("/home/user/private", "sk-test-secret", "Bearer abcdef", "base64,AAAA"):
        assert token not in warning.message, token
        assert token not in str(outcome.trace), token


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


# ── 25.6 ONNX provider/device audit / ONNX provider/device 审计 ───────────


def _install_fake_ort(monkeypatch, providers_result: list[str], captured: dict):
    """Install a fake onnxruntime module capturing provider arguments.
    安装捕获 provider 参数的假 onnxruntime 模块。"""
    import sys
    import types

    fake_np = __import__("numpy")
    captured["preload_dlls_calls"] = 0

    class _FakeSession:
        def __init__(self, path, providers=None):
            captured["providers"] = providers

        def get_providers(self):
            return list(providers_result)

        def get_inputs(self):
            return [types.SimpleNamespace(name="input", shape=[1, 3, 1024, 1024])]

        def get_outputs(self):
            return [types.SimpleNamespace(name="output", shape=[1, 5 + 2 + 180])]

        def run(self, *args, **kwargs):
            return [fake_np.zeros((1, 5 + 2 + 180))]

    class _FakeOrt:
        @staticmethod
        def preload_dlls(directory=""):
            captured["preload_dlls_calls"] = captured.get("preload_dlls_calls", 0) + 1

        @staticmethod
        def InferenceSession(path, providers=None):
            return _FakeSession(path, providers)

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt)
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        types.SimpleNamespace(
            resize=lambda *a, **k: None,
            copyMakeBorder=lambda *a, **k: None,
            INTER_LINEAR=1,
            BORDER_CONSTANT=2,
            dnn=types.SimpleNamespace(NMSBoxesRotated=lambda *a, **k: None),
        ),
    )


def _onnx_model(tmp_path: Path, monkeypatch, providers_result, **detector_overrides):
    from agents.counting.backends.yolov5_obb_onnx import YoloV5ObbOnnxModel

    captured: dict = {}
    _install_fake_ort(monkeypatch, providers_result, captured)
    weights = tmp_path / "det.onnx"
    weights.write_bytes(b"fake")
    detector = _detector(tmp_path, **detector_overrides)
    model = YoloV5ObbOnnxModel(
        weights,
        detector.classes,
        device=detector.device,
        require_cuda=detector.require_cuda,
        allow_cpu_fallback=detector.allow_cpu_fallback,
        gpu_mem_limit_bytes=(
            None
            if detector.gpu_mem_limit_gib is None
            else int(detector.gpu_mem_limit_gib * 1024**3)
        ),
    )
    return model, captured


def test_onnx_cuda_device_binding(tmp_path: Path, monkeypatch) -> None:
    model, captured = _onnx_model(
        tmp_path, monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"],
        device="1",
    )
    assert captured["providers"] == [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 1,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_use_max_workspace": "0",
                "do_copy_in_default_stream": "1",
            },
        ),
    ]
    assert model.requested_provider == "CUDAExecutionProvider"
    assert model.requested_device == "1"
    assert model.resolved_provider == "CUDAExecutionProvider"
    assert model.resolved_device == "1"
    assert model.cpu_fallback_used is False
    assert captured["preload_dlls_calls"] == 1


def test_onnx_cuda_device_zero(tmp_path: Path, monkeypatch) -> None:
    model, captured = _onnx_model(
        tmp_path, monkeypatch, ["CUDAExecutionProvider"],
        device="0",
    )
    assert captured["providers"] == [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_use_max_workspace": "0",
                "do_copy_in_default_stream": "1",
            },
        )
    ]


def test_onnx_cuda_provider_receives_explicit_gpu_memory_limit(
    tmp_path: Path, monkeypatch
) -> None:
    _, captured = _onnx_model(
        tmp_path,
        monkeypatch,
        ["CUDAExecutionProvider"],
        device="0",
        gpu_mem_limit_gib=8,
    )

    assert captured["providers"][0][1]["gpu_mem_limit"] == 8 * 1024**3


def test_onnx_cpu_mode_never_requests_cuda(tmp_path: Path, monkeypatch) -> None:
    model, captured = _onnx_model(
        tmp_path, monkeypatch, ["CPUExecutionProvider"],
        device="cpu", require_cuda=False,
    )
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert model.requested_provider == "CPUExecutionProvider"
    assert model.requested_device == "cpu"
    assert model.resolved_provider == "CPUExecutionProvider"
    assert model.resolved_device == "cpu"
    assert captured["preload_dlls_calls"] == 0


def test_onnx_cpu_mode_rejects_unexpected_provider_set(
    tmp_path: Path, monkeypatch
) -> None:
    from agents.errors import DetectorInferenceError

    with pytest.raises(DetectorInferenceError, match="unexpected execution provider"):
        _onnx_model(
            tmp_path,
            monkeypatch,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            device="cpu",
            require_cuda=False,
        )


def test_onnx_cuda_unavailable_without_fallback_fails(tmp_path: Path, monkeypatch) -> None:
    from agents.errors import DetectorInferenceError

    with pytest.raises(DetectorInferenceError, match="CUDAExecutionProvider required"):
        _onnx_model(
            tmp_path, monkeypatch, ["CPUExecutionProvider"],
            device="0", allow_cpu_fallback=False,
        )


def test_onnx_cuda_unavailable_with_fallback_audits_cpu(tmp_path: Path, monkeypatch) -> None:
    model, captured = _onnx_model(
        tmp_path, monkeypatch, ["CPUExecutionProvider"],
        device="0", allow_cpu_fallback=True,
    )
    assert model.cpu_fallback_used is True
    assert model.resolved_provider == "CPUExecutionProvider"
    assert model.resolved_device == "cpu"


def test_onnx_settings_device_contract(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="non-negative"):
        _detector(tmp_path, device="abc")
    with pytest.raises(ValidationError, match="device='cpu'"):
        _detector(tmp_path, device="0", require_cuda=False)
    # Non-negative integers beyond 9 are accepted. / 超过 9 的非负整数可接受。
    detector = _detector(tmp_path, device="10")
    assert detector.device == "10"


# ── 25.7 CPU-only predict / CPU-only 预测契约 ──────────────────────────────


def test_cpu_only_predict_runs_without_fallback_flag(tmp_path: Path, monkeypatch) -> None:
    """Explicit CPU-only mode (require_cuda=False, device=cpu) must predict
    successfully regardless of allow_cpu_fallback. 显式 CPU-only 模式
    （require_cuda=False, device=cpu）必须能成功 predict，与
    allow_cpu_fallback 无关。"""
    import types

    model, captured = _onnx_model(
        tmp_path, monkeypatch, ["CPUExecutionProvider"],
        device="cpu", require_cuda=False, allow_cpu_fallback=False,
    )
    assert captured["providers"] == ["CPUExecutionProvider"]

    # Stub the heavy image path so predict runs without real cv2 math.
    # 桩化重图像路径，使 predict 无需真实 cv2 运算即可运行。
    def _fake_letterbox(image):
        return image, 1.0, (0, 0)

    def _fake_decode(prediction, confidence):
        return []

    def _fake_nms(candidates, confidence, iou, max_det):
        return []

    monkeypatch.setattr(model, "_letterbox", _fake_letterbox)
    monkeypatch.setattr(model, "_decode", _fake_decode)
    monkeypatch.setattr(model, "_nms", _fake_nms)
    model._session.run = lambda *a, **k: [model._np.zeros((1, 5 + 2 + 180))]
    results = model.predict(
        Image.new("RGB", (100, 100)),
        conf=0.2, iou=0.5, imgsz=1024, device="cpu", max_det=100, verbose=False,
    )
    assert len(results) == 1
    assert len(results[0].obb.xyxyxyxy) == 0


def test_predict_device_must_match_initialized_device(tmp_path: Path, monkeypatch) -> None:
    model, _ = _onnx_model(
        tmp_path, monkeypatch, ["CUDAExecutionProvider"],
        device="0",
    )
    with pytest.raises(ValueError, match="differs from initialized"):
        model.predict(
            Image.new("RGB", (100, 100)),
            conf=0.2, iou=0.5, imgsz=1024, device="1", max_det=100, verbose=False,
        )


def test_cuda_mode_predict_rejects_non_integer_device(tmp_path: Path, monkeypatch) -> None:
    model, _ = _onnx_model(
        tmp_path, monkeypatch, ["CUDAExecutionProvider"],
        device="0",
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        model.predict(
            Image.new("RGB", (100, 100)),
            conf=0.2, iou=0.5, imgsz=1024, device="cpu", max_det=100, verbose=False,
        )


def test_identity_error_defined_exactly_once() -> None:
    """MissingModelCacheIdentityError and require_model_cache_identity are
    declared exactly once, in models/base.py; the counting backend import path
    re-exports the very same objects. MissingModelCacheIdentityError 与
    require_model_cache_identity 只在 models/base.py 中声明一次；counting
    backend import 路径重导出同一对象。"""
    import ast

    models_source = (REPO_ROOT / "models" / "base.py").read_text(encoding="utf-8")
    models_tree = ast.parse(models_source)
    class_count = sum(
        1
        for node in models_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MissingModelCacheIdentityError"
    )
    function_count = sum(
        1
        for node in models_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "require_model_cache_identity"
    )
    assert class_count == 1
    assert function_count == 1
    # The counting import path resolves to the same objects. / counting import
    # 路径解析为同一对象。
    from agents.counting.backends import base as counting_base
    from models import base as models_base

    assert (
        counting_base.require_model_cache_identity
        is models_base.require_model_cache_identity
    )
    assert (
        counting_base.MissingModelCacheIdentityError
        is models_base.MissingModelCacheIdentityError
    )


# ── C2 共享检测 seam / shared object-detection seam ───────────────────────


def test_runtime_detection_client_builds_verified_outputs(tmp_path: Path) -> None:
    """ObjectDetectionOutput carries the input size, label, internal
    confidence, pixel frames, logical identity, weight digest, and provider
    audit — never a physical path or tensor. ObjectDetectionOutput 携带输入
    尺寸、标签、内部分数、像素坐标系、逻辑身份、权重摘要与 provider 审计 —
    绝不携带物理路径或 tensor。"""
    from models.base import ModelCacheIdentity

    model = _FakeRuntimeModel(
        [_obb([[[10.0, 10.0], [10.0, 30.0], [30.0, 30.0], [30.0, 10.0]]], [0.0], [0.8])]
    )
    client = RuntimeObjectDetectionClient(
        model, logical_model_id="m1", weights_sha256="ab" * 32
    )
    outputs = client.detect(
        Image.new("RGB", (100, 100)),
        confidence=0.25,
        iou=0.5,
        image_size=640,
        device="cpu",
        max_detections=50,
    )
    assert len(outputs) == 1
    record = outputs[0]
    assert isinstance(record, ObjectDetectionOutput)
    assert record.label == "car"
    assert record.confidence == 0.8
    assert record.xyxy == (10.0, 10.0, 30.0, 30.0)
    assert record.polygon == ((10.0, 10.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0))
    assert record.input_width == 640
    assert record.input_height == 640
    assert record.logical_model_id == "m1"
    assert record.weights_sha256 == "ab" * 32
    assert record.provider_audit["requested_provider"] == ""
    assert record.provider_audit["cpu_fallback_used"] is False
    identity = client.cache_identity
    assert isinstance(identity, ModelCacheIdentity)
    assert identity.model == "m1"
    assert str(tmp_path) not in str(record)


def test_runtime_detection_client_forwards_identical_parameters() -> None:
    """The seam forwards the exact inference parameters the counting backend
    historically passed to the runtime. seam 向运行时转发与计数后端历史完全
    相同的推理参数。"""
    model = _FakeRuntimeModel([_obb([], [], [])])
    client = RuntimeObjectDetectionClient(model, logical_model_id="m1")
    image = Image.new("RGB", (64, 64))
    client.detect(
        image,
        confidence=0.3,
        iou=0.45,
        image_size=512,
        device="0",
        max_detections=9,
    )
    kwargs = model.predict_kwargs_list[0]
    assert set(kwargs) == {"source", "conf", "iou", "imgsz", "device", "max_det", "verbose"}
    assert kwargs["source"] is image
    assert kwargs["conf"] == 0.3
    assert kwargs["iou"] == 0.45
    assert kwargs["imgsz"] == 512
    assert kwargs["device"] == "0"
    assert kwargs["max_det"] == 9
    assert kwargs["verbose"] is False


def test_runtime_detection_client_unknown_class_id_fails_stably() -> None:
    model = _FakeRuntimeModel(
        [_obb([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]], [7.0], [0.9])]
    )
    client = RuntimeObjectDetectionClient(model, logical_model_id="m1")
    with pytest.raises(ValueError, match="YOLO_CLASS_ID_UNKNOWN:7"):
        client.detect(
            Image.new("RGB", (10, 10)),
            confidence=0.2,
            iou=0.5,
            image_size=64,
            device="cpu",
            max_detections=10,
        )


def test_runtime_detection_client_empty_when_no_obb_output() -> None:
    client = RuntimeObjectDetectionClient(_FakeRuntimeModel([]), logical_model_id="m1")
    assert (
        client.detect(
            Image.new("RGB", (10, 10)),
            confidence=0.2,
            iou=0.5,
            image_size=64,
            device="cpu",
            max_detections=10,
        )
        == []
    )


def test_runtime_detection_client_rejects_physical_logical_ids() -> None:
    with pytest.raises(ValueError, match="logical identifier"):
        RuntimeObjectDetectionClient(
            _FakeRuntimeModel(), logical_model_id="/home/user/models/det.pt"
        )


def test_counting_parity_via_shared_seam(tmp_path: Path) -> None:
    """The refactored counting backend still produces the identical outcome
    shape, trace audit keys, and point provenance. 重构后的计数后端仍然产出
    完全相同的 outcome 结构、trace 审计键与点 provenance。"""
    model = _FakeRuntimeModel([_obb([_single_detection_polygon()], [0.0], [0.9])])
    detector = _detector(tmp_path)
    backend = YoloOBBCountingBackend(
        detector,
        counting=CountingSettings(),
        model_store=YoloModelStore(loader=lambda path: model),
    )
    outcome = asyncio.run(
        backend.count(_request(tmp_path, Image.new("RGB", (200, 200), (1, 2, 3))), _context())
    )
    assert outcome.counting.final_count == 1
    for key in (
        "requested_provider",
        "requested_device",
        "actual_providers",
        "resolved_provider",
        "resolved_device",
        "cpu_fallback_used",
    ):
        assert key in outcome.trace, key
    point = outcome.counting.global_points[0]
    assert point.provenance is not None
    assert point.provenance.source_class == "car"
    assert point.provenance.model_id == detector.model_id
    assert point.provenance.weights_sha256 == detector.sha256
    assert str(tmp_path) not in str(outcome.trace)
