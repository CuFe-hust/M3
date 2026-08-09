"""Contract tests for the counting backend protocol and registry.

计数后端协议与注册表契约测试：稳定注册顺序、重复检测、get/items/
list_available、supports(target, hints)、数据集中性命名、注册表不加载权重、
协议与结果使用新 Schema。
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.counting.backends import (
    BackendPlan,
    BackendRegistry,
    BackendSelection,
    CountingBackendOutcome,
    CountingRequest,
)
from agents.counting.backends.base import CountingBackend
from agents.counting.schema import CountTargetSpec, CountingResult
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, UnifiedSample

REPO_ROOT = Path(__file__).resolve().parents[3]

_TARGET = CountTargetSpec(
    canonical_label="car",
    inclusion_rule="visible vehicle",
    exclusion_rule="occluded more than half",
)


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="counting",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="How many cars?",
        ground_truth=GroundTruth(answers=["3"]),
    )


class _FakeBackend:
    """Protocol-conformant fake backend; construction stands in for weight
    loading. 符合协议的假后端；构造即代表权重加载。"""

    def __init__(
        self,
        name: str,
        *,
        priority: int = 100,
        enabled: bool = True,
        available: bool = True,
        supported_targets: tuple[str, ...] = ("car",),
        load_log: list[str] | None = None,
    ) -> None:
        self.name = name
        self.priority = priority
        self._enabled = enabled
        self._available = available
        self._supported = set(supported_targets)
        self.supports_calls: list[tuple[str, Any]] = []
        if load_log is not None:
            load_log.append(name)

    def is_enabled(self) -> bool:
        return self._enabled

    def is_available(self) -> bool:
        return self._available

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        self.supports_calls.append((target.canonical_label, hints))
        return target.canonical_label in self._supported

    async def count(self, request: CountingRequest, context: object) -> CountingBackendOutcome:
        return CountingBackendOutcome(
            counting=CountingResult(
                sample_id=request.sample.sample_id,
                target=request.target.canonical_label,
                question=request.sample.question,
                source_width=1,
                source_height=1,
                tile_count=1,
                final_count=0,
                status="completed",
            )
        )


# ── 协议 / protocol ───────────────────────────────────────────────────────


def test_backend_protocol_shape() -> None:
    backend = _FakeBackend("qwen_point")
    assert inspect.iscoroutinefunction(backend.count)
    assert inspect.signature(backend.supports).parameters["hints"] is not inspect.Parameter.empty
    assert isinstance(backend.name, str)
    assert isinstance(backend.priority, int)


def test_request_and_outcome_use_new_schema() -> None:
    request = CountingRequest(
        sample=_sample(),
        image=Image.new("RGB", (4, 4)),
        target=_TARGET,
        artifact_dir=Path("/tmp/run"),
    )
    assert isinstance(request.sample, UnifiedSample)
    assert isinstance(request.target, CountTargetSpec)
    outcome = CountingBackendOutcome(
        counting=CountingResult(
            sample_id="s1", target="car", question="Q",
            source_width=1, source_height=1, tile_count=1, final_count=0,
            status="completed",
        )
    )
    assert isinstance(outcome.counting, CountingResult)
    assert outcome.agent_result is None


def test_selection_and_plan_are_neutral_dataclasses() -> None:
    import dataclasses

    selection = BackendSelection(backend_name="qwen_point", reason_codes=("hint",))
    plan = BackendPlan(primary_backend_name="qwen_point", fallback_backend_names=("yolo_obb",))
    assert dataclasses.is_dataclass(selection) and selection.__dataclass_params__.frozen
    assert dataclasses.is_dataclass(plan) and plan.__dataclass_params__.frozen
    assert plan.fallback_backend_names == ("yolo_obb",)


# ── 注册表 / registry ─────────────────────────────────────────────────────


def test_registry_keeps_stable_registration_order() -> None:
    registry = BackendRegistry()
    first = _FakeBackend("qwen_point", priority=10)
    second = _FakeBackend("yolo_obb", priority=20)
    registry.register(first)
    registry.register(second)
    assert registry.all_names() == ["qwen_point", "yolo_obb"]
    assert registry.items() == (first, second)
    assert len(registry) == 2


def test_registry_rejects_duplicate_names() -> None:
    registry = BackendRegistry()
    registry.register(_FakeBackend("qwen_point"))
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(_FakeBackend("qwen_point"))


def test_registry_get_and_unknown_key() -> None:
    registry = BackendRegistry()
    backend = _FakeBackend("qwen_point")
    registry.register(backend)
    assert registry.get("qwen_point") is backend
    with pytest.raises(KeyError, match="Unknown counting backend"):
        registry.get("no-such-backend")


def test_registry_list_available_filters() -> None:
    registry = BackendRegistry()
    available = _FakeBackend("qwen_point", supported_targets=("car", "plane"))
    unavailable = _FakeBackend("yolo_obb", available=False, supported_targets=("car",))
    other_target = _FakeBackend("det_c", supported_targets=("ship",))
    for backend in (available, unavailable, other_target):
        registry.register(backend)
    hits = registry.list_available(_TARGET)
    assert [b.name for b in hits] == ["qwen_point"]
    # exclude_names removes candidates. / exclude_names 排除候选。
    assert registry.list_available(_TARGET, exclude_names=frozenset({"qwen_point"})) == []


def test_registry_list_configured_ignores_runtime_availability() -> None:
    registry = BackendRegistry()
    configured = _FakeBackend("configured", available=False)
    disabled = _FakeBackend("disabled", enabled=False)
    registry.register(configured)
    registry.register(disabled)

    assert registry.list_configured(_TARGET) == [configured]


def test_registry_supports_receives_neutral_hints() -> None:
    registry = BackendRegistry()
    backend = _FakeBackend("qwen_point")
    registry.register(backend)
    registry.list_available(_TARGET, hints={"needs_tiling": True})
    assert backend.supports_calls == [("car", {"needs_tiling": True})]


def test_registry_never_loads_weights() -> None:
    """Registering backends must not trigger weight loading.
    注册后端绝不触发权重加载。"""
    load_log: list[str] = []
    registry = BackendRegistry()
    registry.register(_FakeBackend("qwen_point", load_log=load_log))
    registry.register(_FakeBackend("yolo_obb", load_log=load_log))
    assert load_log == ["qwen_point", "yolo_obb"]  # construction only / 仅构造
    registry.list_available(_TARGET)
    registry.get("qwen_point")
    assert load_log == ["qwen_point", "yolo_obb"]  # unchanged / 未变


# ── 命名 / naming ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_name", ["vrsbench_qwen", "dota_yolo", "xview_det"])
def test_registry_rejects_dataset_embedded_names(bad_name: str) -> None:
    registry = BackendRegistry()
    with pytest.raises(ValueError, match="dataset"):
        registry.register(_FakeBackend(bad_name))


def test_registry_rejects_empty_name() -> None:
    registry = BackendRegistry()
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(_FakeBackend("  "))  # type: ignore[arg-type]


def test_backend_names_are_neutral() -> None:
    for name in ("qwen_point", "yolo_obb", "fused"):
        registry = BackendRegistry()
        registry.register(_FakeBackend(name))
        assert registry.get(name).name == name


# ── 边界 / boundaries ─────────────────────────────────────────────────────


def test_backend_modules_have_no_selection_logic_or_legacy_imports() -> None:
    for relative in ("base.py", "registry.py", "__init__.py"):
        source = (REPO_ROOT / "agents" / "counting" / "backends" / relative).read_text(
            encoding="utf-8"
        )
        assert "spacers_agent" not in source, relative
        assert "BackendSelector" not in source, relative
        assert "VRSBench" not in source, relative
