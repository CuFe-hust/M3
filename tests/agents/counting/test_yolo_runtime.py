"""YOLO planning tests without real weights or Ultralytics imports.
不使用真实权重或 Ultralytics 导入的 YOLO 计划测试。
"""

from pathlib import Path

import pytest

from spacers_agent.agents.counting.backends.qwen_point import QwenPointCountingBackend
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.agents.counting.backends.selector import BackendSelector
from spacers_agent.agents.counting.backends.vrsbench_qwen_count import VRSBenchQwenCountBackend
from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend
from spacers_agent.schemas import CountTargetSpec, ImageRef, UnifiedSample, YoloDetectorSettings, YoloCountingSettings


def _target(label: str) -> CountTargetSpec:
    return CountTargetSpec(canonical_label=label, inclusion_rule="count it", exclusion_rule="none")


def _sample(task: str = "counting", *, dataset: str = "fixture") -> UnifiedSample:
    return UnifiedSample(sample_id="sample", dataset=dataset, split="test", task=task, images=[ImageRef(image_id="image", path=Path("image.png"), role="image")], question="How many ships?")


def _registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(QwenPointCountingBackend(object(), system_prompt="", settings=type("S", (), {"counting": object(), "models": type("M", (), {"qwen": object()})()})()))
    detector = YoloDetectorSettings(name="yolo26s_dota_obb", enabled=True, weights=Path("missing.pt"), model_id="test-yolo", sha256="0" * 64, classes=["plane", "ship", "small vehicle", "large vehicle"], composite_targets={"vehicle": ["small vehicle", "large vehicle"]})
    registry.register(YoloOBBCountingBackend(detector))
    return registry


def test_enabled_yolo_requires_enabled_detector() -> None:
    with pytest.raises(ValueError, match="enabled detector"):
        YoloCountingSettings(enabled=True)


def test_yolo_schema_rejects_unknown_alias_target() -> None:
    with pytest.raises(ValueError, match="unknown class"):
        YoloDetectorSettings(name="detector", weights=Path("x.pt"), model_id="test", sha256="0" * 64, classes=["ship"], aliases={"boat": "plane"})


def test_auto_selects_yolo_for_ship_and_qwen_for_building() -> None:
    selector = BackendSelector(_registry(), default_backend="auto")
    assert selector.plan(_target("ship"), _sample()).primary_backend_name == "yolo26s_dota_obb"
    assert selector.plan(_target("building"), _sample()).primary_backend_name == "qwen_point"


def test_explicit_qwen_ignores_yolo() -> None:
    assert BackendSelector(_registry(), default_backend="qwen_point").plan(_target("ship"), _sample()).primary_backend_name == "qwen_point"


def test_vrsbench_quantity_uses_yolo_with_dedicated_qwen_fallback() -> None:
    registry = _registry()
    registry.register(VRSBenchQwenCountBackend(object(), settings=object(), prompts={"count_proposal": "", "count_localize": ""}))
    sample = _sample("general_vqa", dataset="VRSBench")
    plan = BackendSelector(registry).plan(_target("vehicle"), sample)
    assert plan.primary_backend_name == "yolo26s_dota_obb"
    assert plan.fallback_backend_names == ("vrsbench_qwen_count",)


def test_vrsbench_quantity_explicit_qwen_preserves_dedicated_backend() -> None:
    registry = _registry()
    registry.register(VRSBenchQwenCountBackend(object(), settings=object(), prompts={"count_proposal": "", "count_localize": ""}))
    sample = _sample("general_vqa", dataset="VRSBench")
    plan = BackendSelector(registry, default_backend="qwen_point").plan(_target("vehicle"), sample)
    assert plan.primary_backend_name == "vrsbench_qwen_count"
