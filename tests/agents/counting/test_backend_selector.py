"""Contract tests for the counting backend selector.

计数后端选择器契约测试：mode/task/hints/capability 驱动的 plan、无数据集
分支、非计数任务无计划、显式模式回退路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.counting.backends.registry import BackendRegistry
from agents.counting.backends.selector import BackendSelector
from agents.counting.schema import CountTargetSpec, CountingResult

REPO_ROOT = Path(__file__).resolve().parents[3]

_TARGET = CountTargetSpec(
    canonical_label="car",
    inclusion_rule="visible vehicle",
    exclusion_rule="occluded more than half",
)


def _counting(sample_id: str = "s1", **overrides) -> CountingResult:
    values = dict(
        sample_id=sample_id, target="car", question="Q",
        source_width=100, source_height=100, tile_count=1, final_count=0,
        status="completed",
    )
    values.update(overrides)
    return CountingResult(**values)


class _FakeQwenBackend:
    name = "qwen_point"
    kind = "qwen_point"
    priority = 0

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return True

    async def count(self, request, context):
        return __import__("agents.counting.backends.base", fromlist=["CountingBackendOutcome"]).CountingBackendOutcome(
            counting=_counting()
        )


class _FakeYoloBackend:
    name = "det-a"
    kind = "yolo_obb"
    priority = 100

    def __init__(self, available: bool = True, supported: bool = True) -> None:
        self._available = available
        self._supported = supported

    def is_enabled(self) -> bool:
        return self._available

    def is_available(self) -> bool:
        return self._available

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return self._supported

    def trace_profile(self) -> dict[str, object]:
        return {"detector_name": self.name}

    def resolve_target_classes(self, target: CountTargetSpec) -> frozenset[str]:
        return frozenset({"car"})

    async def count(self, request, context):
        return __import__("agents.counting.backends.base", fromlist=["CountingBackendOutcome"]).CountingBackendOutcome(
            counting=_counting()
        )


class _FakeQuantityBackend:
    """Supports targets only under a reliable hint. 仅在可靠 hint 下支持目标。"""

    name = "quantity_proposal"
    kind = "quantity_proposal"
    priority = 5

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return bool(hints and hints.get("quantity_estimation"))

    def resolve_target_classes(self, target: CountTargetSpec) -> frozenset[str]:
        return frozenset({"car"})

    async def count(self, request, context):
        return __import__("agents.counting.backends.base", fromlist=["CountingBackendOutcome"]).CountingBackendOutcome(
            counting=_counting()
        )


def _registry(*backends) -> BackendRegistry:
    registry = BackendRegistry()
    for backend in backends:
        registry.register(backend)
    return registry


def _selector(*backends, default_backend: str = "auto") -> BackendSelector:
    return BackendSelector(_registry(*backends), default_backend=default_backend)


# ── auto 模式 / auto mode ──────────────────────────────────────────────────


def test_auto_prefers_highest_priority_supported_detector() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend(), _FakeQuantityBackend())
    plan = selector.plan(_TARGET, task="counting")
    assert plan is not None
    assert plan.primary_backend_name == "det-a"
    assert plan.fallback_backend_names == ("qwen_point",)
    assert "highest_priority_supported_yolo" in plan.reason_codes


def test_auto_falls_back_to_qwen_without_supported_detector() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend(supported=False))
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "qwen_point"
    assert plan.fallback_backend_names == ()
    assert "no_supported_detector_qwen" in plan.reason_codes


def test_auto_excludes_unavailable_detectors() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend(available=False))
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "qwen_point"


# ── 计划期不可用性 / plan-time unavailability (25.5) ─────────────────────


class _FakeYoloMissingWeights(_FakeYoloBackend):
    """Configured and supported, but its weights are missing at runtime.
    已配置且支持目标，但运行时权重缺失。"""

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True  # runtime readiness verified at count time / 运行时就绪在 count 时验证

    async def count(self, request, context):
        from agents.errors import DetectorWeightsMissingError

        raise DetectorWeightsMissingError(self.name, "det.pt")


def test_unavailable_detector_still_becomes_primary_in_plan() -> None:
    """A configured+supported detector whose weights are missing must still be
    planned as primary so the agent can fall back explicitly at run time.
    已配置+支持但权重缺失的检测器仍必须成为计划主后端，使 Agent 能在运行时
    显式回退。"""
    selector = _selector(_FakeQwenBackend(), _FakeYoloMissingWeights())
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "det-a"
    assert plan.fallback_backend_names == ("qwen_point",)
    assert "explicit_yolo_unsupported_target_qwen" not in plan.reason_codes


def test_explicit_yolo_mode_plans_supported_but_unavailable_detector() -> None:
    selector = _selector(
        _FakeQwenBackend(), _FakeYoloMissingWeights(), default_backend="yolo_obb"
    )
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "det-a"
    assert plan.fallback_backend_names == ("qwen_point",)
    assert "explicit_yolo" in plan.reason_codes


def test_disabled_detector_is_excluded_from_plan() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend(available=False))
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "qwen_point"


# ── 显式模式 / explicit modes ─────────────────────────────────────────────


def test_explicit_qwen_point_mode() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend(), default_backend="qwen_point")
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "qwen_point"
    assert plan.fallback_backend_names == ()
    assert "explicit_qwen_point" in plan.reason_codes


def test_explicit_yolo_mode_with_supported_detector() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend(), default_backend="yolo_obb")
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "det-a"
    assert "explicit_yolo" in plan.reason_codes


def test_explicit_yolo_mode_without_supported_detector() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend(supported=False), default_backend="yolo_obb")
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "qwen_point"
    assert "explicit_yolo_unsupported_target_qwen" in plan.reason_codes


# ── 任务与 hints / tasks and hints ─────────────────────────────────────────


def test_non_counting_task_yields_no_plan() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend())
    assert selector.plan(_TARGET, task="general_vqa") is None
    assert selector.plan(_TARGET, task="caption") is None


def test_fine_grained_counting_is_counting_task() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend())
    assert selector.plan(_TARGET, task="fine_grained_counting") is not None


def test_default_hints_enable_hint_gated_backends() -> None:
    """The quantity backend requires a reliable hint; the selector's default
    neutral hints provide it. quantity 后端需要可靠 hint；选择器的默认中性
    hints 提供之。"""
    selector = _selector(_FakeQwenBackend(), _FakeQuantityBackend())
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "quantity_proposal"


def test_caller_hints_override_defaults() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeQuantityBackend())
    plan = selector.plan(_TARGET, task="counting", hints={})
    assert plan.primary_backend_name == "qwen_point"


def test_select_returns_none_when_primary_unavailable() -> None:
    """When the only detector is unavailable and no fallback backend is
    registered, no plan (and no selection) exists. 唯一检测器不可用且未注册
    任何回退后端时，不存在任何计划与选择。"""
    selector = _selector(_FakeYoloBackend(available=False))
    assert selector.plan(_TARGET, task="counting") is None
    assert selector.select(_TARGET, task="counting") is None
    assert selector.select(_TARGET, task="general_vqa") is None  # non-counting / 非计数


def test_select_returns_single_selection() -> None:
    selector = _selector(_FakeQwenBackend())
    selection = selector.select(_TARGET, task="counting")
    assert selection is not None
    assert selection.backend_name == "qwen_point"
    assert selector.backend_by_name("qwen_point") is not None


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_selector_has_no_dataset_names_or_question_regex() -> None:
    source = (REPO_ROOT / "agents" / "counting" / "backends" / "selector.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "dataset" not in source
    assert "re.search" not in source
    assert "spacers_agent" not in source


# ── 25.6 kind 分类 / kind classification ───────────────────────────────────


def test_quantity_proposal_is_not_a_yolo_candidate() -> None:
    """Quantity proposal never enters the yolo candidate list.
    数量提议绝不进入 yolo 候选列表。"""
    selector = _selector(_FakeQwenBackend(), _FakeQuantityBackend())
    yolo = selector._yolo_candidates(_TARGET, {"quantity_estimation": True})
    assert [backend.name for backend in yolo] == []
    quantity = selector._quantity_candidates(_TARGET, {"quantity_estimation": True})
    assert [backend.name for backend in quantity] == ["quantity_proposal"]


def test_auto_prefers_yolo_over_quantity() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeYoloBackend(), _FakeQuantityBackend())
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "det-a"
    assert plan.fallback_backend_names == ("qwen_point",)


def test_auto_falls_back_to_quantity_without_yolo() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeQuantityBackend())
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "quantity_proposal"
    assert plan.fallback_backend_names == ()
    assert "target_supported_by_quantity_proposal" in plan.reason_codes


def test_auto_falls_back_to_qwen_without_detectors() -> None:
    selector = _selector(_FakeQwenBackend())
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "qwen_point"


def test_explicit_yolo_never_selects_quantity_proposal() -> None:
    selector = _selector(_FakeQwenBackend(), _FakeQuantityBackend(), default_backend="yolo_obb")
    plan = selector.plan(_TARGET, task="counting")
    assert plan.primary_backend_name == "qwen_point"
    assert "explicit_yolo_unsupported_target_qwen" in plan.reason_codes


class _UnknownKindBackend(_FakeQwenBackend):
    name = "mystery"
    kind = "mystery_kind"


def test_unknown_kind_fails_stably() -> None:
    from agents.errors import CountingBackendUnavailableError

    selector = _selector(_FakeQwenBackend(), _UnknownKindBackend())
    with pytest.raises(CountingBackendUnavailableError, match="INVALID_BACKEND_KIND"):
        selector.plan(_TARGET, task="counting")


# ── 25.7 未知 kind 安全 / unknown kind safety ─────────────────────────────


class _UnsafeBackend:
    """A hostile backend carrying path/credential text in name and kind.
    名称与 kind 携带路径/凭据文本的恶意后端。"""

    name = "/home/user/sk-secret"
    kind = "Bearer abcdef data:image/png;base64,AAAA"
    priority = 99

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return True


def test_unknown_kind_error_never_echoes_raw_values() -> None:
    from agents.errors import CountingBackendUnavailableError

    selector = _selector(_FakeQwenBackend(), _UnsafeBackend())
    with pytest.raises(CountingBackendUnavailableError, match="INVALID_BACKEND_KIND") as info:
        selector.plan(_TARGET, task="counting")
    text = str(info.value)
    for token in ("/home/user", "sk-secret", "Bearer abcdef", "base64,AAAA"):
        assert token not in text, token
