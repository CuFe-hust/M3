"""Phase 4 - CountingAgent target selection and status propagation tests.
Phase 4 - CountingAgent 目标选择与状态传播测试。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from spacers_agent.agents.counting.agent import CountingAgent
from spacers_agent.agents.counting.backends.selector import BackendSelector, is_vrsbench_quantity
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.schemas import CountTargetSpec, UnifiedSample, GroundTruth, ImageRef


class _FakeBackend:
    def __init__(self, name: str, priority: int = 0, vehicle: str | None = None) -> None:
        self.name = name
        self.priority = priority
        self._vehicle = vehicle

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec) -> bool:
        if self._vehicle is not None:
            from spacers_agent.vqa_geometry import vrsbench_vehicle_class
            return vrsbench_vehicle_class(target.canonical_label) == self._vehicle
        return True

    async def count(self, *args, **kwargs):
        raise NotImplementedError


def _sample(dataset: str, task: str, question: str, question_type: str = "") -> UnifiedSample:
    return UnifiedSample(
        sample_id="test-sample",
        dataset=dataset,
        split="validation",
        task=task,
        images=[ImageRef(image_id="img", path=Path("/nonexistent.png"), role="image", width=100, height=100)],
        question=question,
        ground_truth=GroundTruth(),
        metadata={"question_type": question_type} if question_type else {},
    )


def test_is_vrsbench_quantity_detects_vehicle_quantity_questions():
    sample = _sample("VRSBench", "general_vqa", "How many large vehicles are visible in the image?", "quantity")
    assert is_vrsbench_quantity(sample)

    sample2 = _sample("other", "general_vqa", "How many large vehicles?", "quantity")
    assert not is_vrsbench_quantity(sample2)

    sample3 = _sample("VRSBench", "general_vqa", "Where is the vehicle?", "position")
    assert not is_vrsbench_quantity(sample3)


def test_backend_selector_selects_qwen_point_for_native_counting():
    registry = BackendRegistry()
    registry.register(_FakeBackend("qwen_point"))
    registry.register(_FakeBackend("vrsbench_qwen_count", vehicle="large-vehicle"))
    selector = BackendSelector(registry)

    sample = _sample("legacy-parity", "counting", "How many buildings?")
    target = CountTargetSpec(canonical_label="building", inclusion_rule="count", exclusion_rule="none")
    selection = selector.select(target, sample)
    assert selection is not None
    assert selection.backend_name == "qwen_point"


def test_backend_selector_selects_vrsbench_for_vehicle_quantity():
    registry = BackendRegistry()
    registry.register(_FakeBackend("qwen_point"))
    registry.register(_FakeBackend("vrsbench_qwen_count", vehicle="large-vehicle"))
    selector = BackendSelector(registry)

    sample = _sample("VRSBench", "general_vqa", "How many large vehicles are visible in the image?", "quantity")
    target = CountTargetSpec(canonical_label="large-vehicle", inclusion_rule="count", exclusion_rule="none")
    selection = selector.select(target, sample)
    assert selection is not None
    assert selection.backend_name == "vrsbench_qwen_count"


def test_backend_selector_uses_yolo_for_vrsbench_vehicle_quantity():
    registry = BackendRegistry()
    registry.register(_FakeBackend("qwen_point"))
    registry.register(_FakeBackend("vrsbench_qwen_count", vehicle="large-vehicle"))
    registry.register(_FakeBackend("yolo26s_dota_obb", priority=100, vehicle="large-vehicle"))
    selector = BackendSelector(registry)

    sample = _sample("VRSBench", "general_vqa", "How many large vehicles are visible in the image?", "quantity")
    target = CountTargetSpec(canonical_label="large-vehicle", inclusion_rule="count", exclusion_rule="none")
    plan = selector.plan(target, sample)
    assert plan is not None
    assert plan.primary_backend_name == "yolo26s_dota_obb"
    assert plan.fallback_backend_names == ("vrsbench_qwen_count",)


def test_backend_selector_uses_generic_qwen_for_non_vehicle_vrsbench_quantity():
    registry = BackendRegistry()
    registry.register(_FakeBackend("qwen_point"))
    registry.register(_FakeBackend("vrsbench_qwen_count", vehicle="large-vehicle"))
    selector = BackendSelector(registry)

    sample = _sample("VRSBench", "general_vqa", "How many ships are visible in the image?", "quantity")
    target = CountTargetSpec(canonical_label="ship", inclusion_rule="count", exclusion_rule="none")
    plan = selector.plan(target, sample)

    assert plan is not None
    assert plan.primary_backend_name == "qwen_point"
    assert plan.reason_codes == ("vrsbench_quantity_generic_target_qwen",)


@pytest.mark.asyncio
async def test_counting_agent_parses_non_vehicle_vrsbench_quantity_target(monkeypatch, tmp_path: Path):
    parsed_target = CountTargetSpec(canonical_label="ship", inclusion_rule="count ships", exclusion_rule="none")
    calls: list[str] = []

    class _Parser:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def parse(self, question: str, **kwargs) -> CountTargetSpec:
            calls.append(question)
            return parsed_target

    class _Budget:
        def reserve_qwen(self) -> None:
            calls.append("reserve_qwen")

    monkeypatch.setattr("spacers_agent.agents.counting.agent.CountTargetParser", _Parser)
    settings = SimpleNamespace(agents=SimpleNamespace(counting=SimpleNamespace(default_backend="auto")))
    agent = CountingAgent(None, {"target": "unused"}, "unused", BackendRegistry(), settings=settings)
    sample = _sample("VRSBench", "general_vqa", "How many ships are visible in the image?", "quantity")
    context = SimpleNamespace(call_budget=_Budget(), artifact_dir=tmp_path)

    assert await agent._target(sample, context) is parsed_target
    assert calls == ["reserve_qwen", "How many ships are visible in the image?"]
