"""Test YOLO OBB backend — model store, weight guard, class matching. / 测试 YOLO OBB 后端。"""

from __future__ import annotations

import pytest

from spacers_agent.agents.counting.backends.yolo_model_store import YoloModelStore
from spacers_agent.schemas import YoloDetectorSettings, CountTargetSpec
from pathlib import Path


def _detector(classes: list[str] | None = None, weight_path: str = "/nonexistent/model.pt") -> YoloDetectorSettings:
    return YoloDetectorSettings(
        name="test_detector", enabled=True, weights=Path(weight_path),
        classes=classes or ["car", "truck"], priority=100,
    )


def _target(label: str = "car") -> CountTargetSpec:
    return CountTargetSpec(canonical_label=label, inclusion_rule="count", exclusion_rule="none")


class TestModelStore:
    """Model store lazy loading and caching. / 模型存储延迟加载与缓存。"""

    def test_has_before_load(self):
        store = YoloModelStore()
        assert not store.has(Path("/nonexistent.pt"))

    def test_missing_weight_raises_before_import(self):
        store = YoloModelStore()
        with pytest.raises(Exception):  # DetectorWeightsMissingError or FileNotFoundError
            store.get(Path("/nonexistent/model.pt"), confidence=0.5, iou=0.5,
                      image_size=1024, device="cpu", max_detections=100)


class TestYoloDetectorSupports:
    """Class matching for YOLO detectors. / YOLO 检测器类别匹配。"""

    def test_exact_class_match(self):
        from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
        backend = YoloOBBCountingBackend(_detector())
        assert backend.supports(_target("car"))

    def test_alias_match(self):
        from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
        detector = _detector()
        detector.aliases = {"automobile": "car"}
        backend = YoloOBBCountingBackend(detector)
        assert backend.supports(_target("automobile"))

    def test_composite_match(self):
        from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
        detector = _detector()
        detector.composite_targets = {"vehicle": ["car", "truck"]}
        backend = YoloOBBCountingBackend(detector)
        assert backend.supports(_target("vehicle"))

    def test_no_match(self):
        from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
        backend = YoloOBBCountingBackend(_detector())
        assert not backend.supports(_target("ship"))


class TestYoloNotAvailable:
    """YOLO backend reports unavailable when disabled or weight missing. / YOLO 后端在禁用或权重缺失时报不可用。"""

    def test_disabled_not_available(self):
        from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
        detector = _detector()
        detector.enabled = False
        backend = YoloOBBCountingBackend(detector)
        assert not backend.is_available()

    def test_weight_missing_not_available(self):
        from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
        backend = YoloOBBCountingBackend(_detector(weight_path="/nonexistent/model.pt"))
        assert not backend.is_available()
