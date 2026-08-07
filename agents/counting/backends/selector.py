"""Deterministic and auditable counting-backend planning.

确定且可审计的计数后端执行计划。Selector 只基于 mode、task、CountTargetSpec、
通用 hints 与 backend capability 做选择；backend 类型通过显式 kind 识别，
绝不从名称推断。源码不包含任何数据集名或问题正则。
"""

from __future__ import annotations

from typing import Any

from agents.counting.backends.base import (
    KNOWN_BACKEND_KINDS,
    BackendKind,
    BackendPlan,
    BackendSelection,
)
from agents.counting.backends.registry import BackendRegistry
from agents.counting.schema import CountTargetSpec
from agents.errors import CountingBackendUnavailableError

COUNTING_TASKS = frozenset({"counting", "fine_grained_counting"})

# Neutral hints supplied by default when the caller provides none; they enable
# capability gating without any data-source knowledge.
# 调用方未提供时的中性默认 hints；它们在无需任何数据来源知识的前提下启用
# 能力门控。
_DEFAULT_HINTS: dict[str, Any] = {"quantity_estimation": True}


class BackendSelector:
    """Select a backend plan from configured mode and target capability.
    根据配置模式与目标能力选择后端执行计划。"""

    def __init__(self, registry: BackendRegistry, *, default_backend: str = "auto") -> None:
        self._registry = registry
        self._default_backend = default_backend

    @property
    def default_backend(self) -> str:
        """Configured default backend mode; public read-only access for
        trace/reporting without private-field peeking.
        已配置的默认后端模式；公开只读访问，供 trace/报告使用而无需偷看
        私有字段。"""
        return self._default_backend

    def plan(
        self,
        target: CountTargetSpec,
        *,
        task: str,
        hints: dict[str, Any] | None = None,
    ) -> BackendPlan | None:
        """Plan a primary backend before availability checks can hide a bad
        deployment. Non-counting tasks yield no plan. Auto priority: enabled
        YOLO → enabled quantity proposal → qwen_point.
        在可用性检查隐藏部署错误之前规划主后端。非计数任务不产生计划。
        auto 优先级：已启用 YOLO → 已启用数量提议 → qwen_point。"""
        if task not in COUNTING_TASKS:
            return None
        effective_hints = hints if hints is not None else _DEFAULT_HINTS
        yolo_candidates = self._yolo_candidates(target, effective_hints)
        if self._default_backend == "qwen_point":
            return self._named_plan("qwen_point", (), "explicit_qwen_point", target)
        if self._default_backend == "yolo_obb":
            if not yolo_candidates:
                return self._named_plan(
                    "qwen_point", (), "explicit_yolo_unsupported_target_qwen", target
                )
            selected = max(
                yolo_candidates, key=lambda backend: (backend.priority, backend.name)
            )
            return BackendPlan(
                selected.name,
                ("qwen_point",),
                (
                    f"task_{task}",
                    "explicit_yolo",
                    "highest_priority_supported_yolo",
                ),
                self._target_classes(selected, target),
            )
        if yolo_candidates:
            selected = max(
                yolo_candidates, key=lambda backend: (backend.priority, backend.name)
            )
            return BackendPlan(
                selected.name,
                ("qwen_point",),
                (
                    f"task_{task}",
                    "target_supported_by_yolo",
                    "highest_priority_supported_yolo",
                ),
                self._target_classes(selected, target),
            )
        quantity_candidates = self._quantity_candidates(target, effective_hints)
        if quantity_candidates:
            selected = max(
                quantity_candidates,
                key=lambda backend: (backend.priority, backend.name),
            )
            return BackendPlan(
                selected.name,
                (),
                (
                    f"task_{task}",
                    "target_supported_by_quantity_proposal",
                    "highest_priority_supported_quantity_proposal",
                ),
                self._target_classes(selected, target),
            )
        return self._named_plan("qwen_point", (), "no_supported_detector_qwen", target)

    def select(
        self,
        target: CountTargetSpec,
        *,
        task: str,
        hints: dict[str, Any] | None = None,
    ) -> BackendSelection | None:
        """Return a single selection for callers that do not need a full plan.
        Availability is checked here without changing plan-time semantics.
        为不需要完整计划的调用方返回单一选择。此处检查可用性，但不改变
        计划期语义。"""
        plan = self.plan(target, task=task, hints=hints)
        if plan is None:
            return None
        backend = self.backend_by_name(plan.primary_backend_name)
        if not backend.is_available():
            return None
        return BackendSelection(
            plan.primary_backend_name, plan.reason_codes, plan.target_classes
        )

    def backend_by_name(self, name: str):
        """Resolve a backend through the registry public API.
        通过注册表公开 API 解析后端。"""
        backend = self._registry.get(name)
        _validate_kind(backend)
        return backend

    def _yolo_candidates(
        self,
        target: CountTargetSpec,
        hints: dict[str, Any],
    ) -> list[Any]:
        """Configured, enabled YOLO backends that explicitly support the
        target. 已配置、启用且明确支持目标的 YOLO 后端。"""
        return [
            backend
            for backend in self._registry.items()
            if _validate_kind(backend) == "yolo_obb"
            and backend.is_enabled()
            and backend.supports(target, hints=hints)
        ]

    def _quantity_candidates(
        self,
        target: CountTargetSpec,
        hints: dict[str, Any],
    ) -> list[Any]:
        """Configured, enabled quantity-proposal backends that explicitly
        support the target. 已配置、启用且明确支持目标的数量提议后端。"""
        return [
            backend
            for backend in self._registry.items()
            if _validate_kind(backend) == "quantity_proposal"
            and backend.is_enabled()
            and backend.supports(target, hints=hints)
        ]

    def _named_plan(
        self,
        primary: str,
        fallbacks: tuple[str, ...],
        reason: str,
        target: CountTargetSpec,
    ) -> BackendPlan | None:
        try:
            backend = self.backend_by_name(primary)
        except KeyError:
            return None
        return BackendPlan(primary, fallbacks, (reason,), self._target_classes(backend, target))

    @staticmethod
    def _target_classes(backend: object, target: CountTargetSpec) -> tuple[str, ...]:
        resolve = getattr(backend, "resolve_target_classes", None)
        if callable(resolve):
            return tuple(sorted(resolve(target)))
        return (target.canonical_label,)


def _validate_kind(backend: object) -> BackendKind:
    """Require an explicit, known backend kind; unknown kinds fail with a
    fixed public message that never echoes raw name/kind values.
    要求显式、已知的后端 kind；未知 kind 以固定公共消息失败，绝不回显原始
    name/kind 值。"""
    kind = getattr(backend, "kind", None)
    if kind not in KNOWN_BACKEND_KINDS:
        raise CountingBackendUnavailableError(
            "unknown-target",
            primary_backend="unknown-backend",
            reason_code="INVALID_BACKEND_KIND",
        )
    return kind  # type: ignore[return-value]
