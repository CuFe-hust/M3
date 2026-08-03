"""Test YOLO OBB backend — model store, weight guard, class matching. / 测试 YOLO OBB 后端。"""

from __future__ import annotations

import pytest

import hashlib
from types import SimpleNamespace

from PIL import Image
from spacers_agent.agents.counting.backends.yolo_model_store import YoloModelStore
from spacers_agent.agents.counting.backends.base import CountingRequest
from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
from spacers_agent.agents.errors import DetectorWeightsHashMismatchError, DetectorWeightsMissingError
from spacers_agent.agents.base import AgentContext
from spacers_agent.routing import CallBudgetFactory
from spacers_agent.schemas import YoloDetectorSettings, CountTargetSpec, ImageRef, UnifiedSample
from spacers_agent.settings import AppSettings
from pathlib import Path


def _detector(classes: list[str] | None = None, weight_path: str = "/nonexistent/model.pt", sha256: str = "0" * 64) -> YoloDetectorSettings:
    return YoloDetectorSettings(
        name="test_detector", enabled=True, weights=Path(weight_path),
        classes=classes or ["car", "truck"], priority=100, model_id="test", sha256=sha256,
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
        with pytest.raises(DetectorWeightsMissingError):
            store.get(_detector())

    def test_hash_mismatch_raises_before_loader(self, tmp_path):
        weights = tmp_path / "tiny.pt"
        weights.write_bytes(b"tiny fixture")
        called = False

        def loader(_: str):
            nonlocal called
            called = True
            raise AssertionError("hash mismatch must preclude loading")

        store = YoloModelStore(loader=loader)
        with pytest.raises(DetectorWeightsHashMismatchError):
            store.get(_detector(weight_path=str(weights), sha256="f" * 64))
        assert not called

    def test_verified_model_is_cached_by_path_and_hash(self, tmp_path):
        weights = tmp_path / "tiny.pt"
        weights.write_bytes(b"tiny fixture")
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        calls = 0

        class Model:
            task = "obb"
            names = ["car", "truck"]

        def loader(_: str):
            nonlocal calls
            calls += 1
            return Model()

        detector = _detector(weight_path=str(weights), sha256=digest)
        store = YoloModelStore(loader=loader)
        assert store.get(detector) is store.get(detector)
        assert calls == 1


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

    def test_weight_missing_is_still_plannable_and_fails_at_execution(self):
        from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
        backend = YoloOBBCountingBackend(_detector(weight_path="/nonexistent/model.pt"))
        assert backend.is_available()


@pytest.mark.asyncio
async def test_ship_target_filters_unrelated_detector_classes(tmp_path):
    """Count only ships from mixed fake detector output. / 只从混合假检测输出中统计船只。"""
    weights = tmp_path / "tiny.pt"
    weights.write_bytes(b"fake yolo")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    detector = _detector(classes=["plane", "ship", "small vehicle"], weight_path=str(weights), sha256=digest)

    class Model:
        task = "obb"
        names = ["plane", "ship", "small vehicle"]

        def predict(self, **kwargs):
            obb = SimpleNamespace(
                xyxyxyxy=[
                    [[1, 1], [2, 1], [2, 2], [1, 2]],
                    [[3, 3], [5, 3], [5, 5], [3, 5]],
                    [[6, 6], [7, 6], [7, 7], [6, 7]],
                ],
                cls=[0, 1, 2],
                conf=[0.9, 0.95, 0.8],
            )
            return [SimpleNamespace(obb=obb)]

    backend = YoloOBBCountingBackend(detector, model_store=YoloModelStore(loader=lambda _: Model()))
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)
    sample = UnifiedSample(sample_id="sample", dataset="fixture", split="test", task="counting", images=[ImageRef(image_id="image", path=image_path, role="image")], question="How many ships?")
    context = AgentContext(artifact_dir=tmp_path / "artifacts", settings=AppSettings(), qwen_client=object(), call_budget=CallBudgetFactory().create_for_sample("counting"))
    outcome = await backend.count(CountingRequest(sample=sample, image=Image.open(image_path), target=_target("ship"), artifact_dir=tmp_path / "artifacts"), context)

    assert outcome.counting.final_count == 1
    assert [point.provenance.source_class for point in outcome.counting.global_points] == ["ship"]
    assert outcome.trace["unrelated_class_rejected_count"] == 2
