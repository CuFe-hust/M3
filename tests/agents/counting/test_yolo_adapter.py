"""Contract tests for the YOLO OBB model adapter.

YOLO OBB 模型适配器契约测试：统一输出（规范点/框 + source class）、惰性
导入、可选依赖缺失专用异常、导入计数包不加载权重。
"""

from __future__ import annotations

import builtins
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from agents.counting.backends.yolo_adapter import (
    OBBDetection,
    OBBModelAdapter,
    UltralyticsOBBModelAdapter,
)
from agents.errors import OptionalDependencyMissingError


def _fake_result(polygons: list[Any], class_ids: list[float], confs: list[float]) -> SimpleNamespace:
    obb = SimpleNamespace(
        xyxyxyxy=polygons,
        cls=class_ids,
        conf=confs,
    )
    return SimpleNamespace(obb=obb)


class _FakeRuntimeModel:
    task = "obb"
    names = {0: "car", 1: "truck"}

    def __init__(self) -> None:
        self.results: list[SimpleNamespace] = []
        self.predict_kwargs: dict[str, Any] = {}

    def predict(self, **kwargs) -> list[SimpleNamespace]:
        self.predict_kwargs = dict(kwargs)
        return self.results


# ── 数据结构与协议 / data structures and protocol ─────────────────────────


def test_obb_detection_is_frozen() -> None:
    import dataclasses

    detection = OBBDetection(
        class_name="car",
        confidence=0.9,
        center_x_px=10.0,
        center_y_px=20.0,
        polygon_xy=((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)),
    )
    assert dataclasses.is_dataclass(detection)
    assert detection.__dataclass_params__.frozen
    assert detection.class_name == "car"


def test_adapter_protocol_shape() -> None:
    assert hasattr(OBBModelAdapter, "predict")
    assert inspect.signature(UltralyticsOBBModelAdapter.predict).parameters["image_size"] is not None


# ── 统一输出 / unified output ──────────────────────────────────────────────


def test_predict_normalizes_ultralytics_output() -> None:
    adapter = UltralyticsOBBModelAdapter()
    model = _FakeRuntimeModel()
    polygon = [[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]]
    model.results = [_fake_result([polygon, polygon], [0.0, 1.0], [0.9, 0.8])]
    adapter._model = model
    detections = adapter.predict(
        Image.new("RGB", (100, 100)),
        image_size=1024,
        confidence=0.2,
        iou=0.5,
        device="cpu",
        max_detections=100,
    )
    assert len(detections) == 2
    first = detections[0]
    assert isinstance(first, OBBDetection)
    assert first.class_name == "car"  # source class from model names / 来自模型 names
    assert first.confidence == 0.9
    assert first.center_x_px == 50.0
    assert first.center_y_px == 50.0
    assert first.polygon_xy == ((0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0))
    assert detections[1].class_name == "truck"


def test_predict_passes_inference_parameters() -> None:
    adapter = UltralyticsOBBModelAdapter()
    model = _FakeRuntimeModel()
    model.results = [_fake_result([], [], [])]
    adapter._model = model
    adapter.predict(
        Image.new("RGB", (100, 100)),
        image_size=640,
        confidence=0.3,
        iou=0.4,
        device="cuda:0",
        max_detections=50,
    )
    assert model.predict_kwargs["imgsz"] == 640
    assert model.predict_kwargs["conf"] == 0.3
    assert model.predict_kwargs["iou"] == 0.4
    assert model.predict_kwargs["device"] == "cuda:0"
    assert model.predict_kwargs["max_det"] == 50


def test_predict_empty_when_no_obb_output() -> None:
    adapter = UltralyticsOBBModelAdapter()
    model = _FakeRuntimeModel()
    model.results = [SimpleNamespace(obb=None)]
    adapter._model = model
    assert adapter.predict(
        Image.new("RGB", (10, 10)),
        image_size=1024,
        confidence=0.2,
        iou=0.5,
        device="cpu",
        max_detections=100,
    ) == []


def test_predict_requires_loaded_model() -> None:
    adapter = UltralyticsOBBModelAdapter()
    with pytest.raises(ValueError, match="load"):
        adapter.predict(
            Image.new("RGB", (10, 10)),
            image_size=1024,
            confidence=0.2,
            iou=0.5,
            device="cpu",
            max_detections=100,
        )


def test_predict_unknown_class_id_fails_explicitly() -> None:
    adapter = UltralyticsOBBModelAdapter()
    model = _FakeRuntimeModel()
    model.results = [_fake_result([[[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]]], [9.0], [0.9])]
    adapter._model = model
    with pytest.raises(ValueError, match="YOLO_CLASS_ID_UNKNOWN"):
        adapter.predict(
            Image.new("RGB", (10, 10)),
            image_size=1024,
            confidence=0.2,
            iou=0.5,
            device="cpu",
            max_detections=100,
        )


# ── 惰性与可选依赖 / lazy loading and optional dependencies ───────────────


def test_ultralytics_missing_raises_specific_error(monkeypatch) -> None:
    adapter = UltralyticsOBBModelAdapter()
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("no ultralytics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(OptionalDependencyMissingError, match="ultralytics"):
        adapter._get_yolo()


def test_load_lazily_instantiates_model(monkeypatch) -> None:
    adapter = UltralyticsOBBModelAdapter()

    class _FakeYOLO:
        def __init__(self, path: str) -> None:
            self.path = path

    monkeypatch.setattr(adapter, "_get_yolo", lambda: _FakeYOLO)
    weight_path = Path("/tmp/fake.pt")
    model = adapter.load(weight_path)
    assert isinstance(model, _FakeYOLO)
    assert model.path == str(weight_path)
    # Cached: a second load reuses the instance. / 缓存：第二次加载复用实例。
    assert adapter.load(Path("/other.pt")) is model


def test_import_counting_backends_does_not_load_runtimes() -> None:
    """Importing the counting package must not load ultralytics/onnxruntime.
    导入计数包绝不加载 ultralytics/onnxruntime。"""
    import agents.counting
    import agents.counting.backends
    import agents.counting.backends.yolo_adapter
    import agents.counting.backends.yolo_model_store
    import agents.counting.backends.yolo_obb
    import agents.counting.backends.yolov5_obb_onnx  # noqa: F401

    for heavy in ("ultralytics", "onnxruntime"):
        assert heavy not in sys.modules, heavy


def test_adapter_has_no_legacy_imports() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "counting" / "backends" / "yolo_adapter.py").read_text(
        encoding="utf-8"
    )
    assert "spacers_agent" not in source
    assert "VRSBench" not in source
