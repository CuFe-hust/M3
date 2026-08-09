"""Deterministic, capability-based counting backend planning.

Planning uses only configured/enabled state, explicit backend kind, target
support, and neutral hints. Runtime availability is deliberately deferred to
the executor. 规划仅使用配置/启用状态、显式 kind、目标能力和中性 hints；运行时
可用性由 Executor 验证。
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

# Kind rank is absolute; backend priority only orders experts of the same kind.
# kind rank 是绝对顺序；backend priority 只在同 kind 专家之间生效。
KIND_RANK: dict[BackendKind, int] = {
    "yolo_obb": 400,
    "semantic_segmentation": 300,
    "quantity_proposal": 200,
    "qwen_point": 100,
}

_DEFAULT_HINTS: dict[str, Any] = {"quantity_estimation": True}


class BackendSelector:
    """Build a stable ordered backend plan from explicit capabilities.
    根据显式能力建立稳定的有序 backend plan。"""

    def __init__(self, registry: BackendRegistry, *, default_backend: str = "auto") -> None:
        self._registry = registry
        self._default_backend = default_backend

    @property
    def default_backend(self) -> str:
        return self._default_backend

    def plan(
        self,
        target: CountTargetSpec,
        *,
        task: str,
        hints: dict[str, Any] | None = None,
    ) -> BackendPlan | None:
        """Plan without consulting transient runtime availability.
        生成计划时不查询临时运行时可用性。"""

        if task not in COUNTING_TASKS:
            return None
        effective_hints = hints if hints is not None else _DEFAULT_HINTS
        ordered = self._ordered_candidates(target, effective_hints)

        if self._default_backend == "qwen_point":
            ordered = [item for item in ordered if _validate_kind(item) == "qwen_point"]
            reason = "explicit_qwen_point"
        elif self._default_backend == "yolo_obb":
            yolo = [item for item in ordered if _validate_kind(item) == "yolo_obb"]
            qwen = [item for item in ordered if _validate_kind(item) == "qwen_point"]
            ordered = [*yolo, *qwen]
            reason = "explicit_yolo" if yolo else "explicit_yolo_unsupported_target_qwen"
        else:
            reason = self._auto_reason(ordered)

        if not ordered:
            return None
        primary = ordered[0]
        return BackendPlan(
            primary_backend_name=primary.name,
            fallback_backend_names=tuple(item.name for item in ordered[1:]),
            reason_codes=(
                f"task_{task}",
                reason,
                "fixed_kind_rank_then_priority_then_name",
            ),
            target_classes=self._target_classes(primary, target),
        )

    def select(
        self,
        target: CountTargetSpec,
        *,
        task: str,
        hints: dict[str, Any] | None = None,
    ) -> BackendSelection | None:
        """Return one currently available primary for legacy callers.
        为旧调用方返回当前可用的 primary。"""

        plan = self.plan(target, task=task, hints=hints)
        if plan is None:
            return None
        backend = self.backend_by_name(plan.primary_backend_name)
        if not backend.is_available():
            return None
        return BackendSelection(
            plan.primary_backend_name,
            plan.reason_codes,
            plan.target_classes,
        )

    def backend_by_name(self, name: str):
        backend = self._registry.get(name)
        _validate_kind(backend)
        return backend

    def _ordered_candidates(
        self,
        target: CountTargetSpec,
        hints: dict[str, Any],
    ) -> list[Any]:
        candidates: list[Any] = []
        for backend in self._registry.items():
            _validate_kind(backend)
            _validate_plan_contract(backend)
            try:
                enabled = backend.is_enabled()
                supported = backend.supports(target, hints=hints)
            except Exception as error:
                raise _invalid_contract_error() from error
            if not isinstance(enabled, bool) or not isinstance(supported, bool):
                raise _invalid_contract_error()
            if enabled and supported:
                candidates.append(backend)
        return sorted(
            candidates,
            key=lambda backend: (
                -KIND_RANK[_validate_kind(backend)],
                -backend.priority,
                backend.name,
            ),
        )

    def _kind_candidates(
        self,
        kind: BackendKind,
        target: CountTargetSpec,
        hints: dict[str, Any],
    ) -> list[Any]:
        return [
            backend
            for backend in self._ordered_candidates(target, hints)
            if _validate_kind(backend) == kind
        ]

    def _yolo_candidates(
        self,
        target: CountTargetSpec,
        hints: dict[str, Any],
    ) -> list[Any]:
        return self._kind_candidates("yolo_obb", target, hints)

    def _quantity_candidates(
        self,
        target: CountTargetSpec,
        hints: dict[str, Any],
    ) -> list[Any]:
        return self._kind_candidates("quantity_proposal", target, hints)

    @staticmethod
    def _auto_reason(ordered: list[Any]) -> str:
        if not ordered:
            return "no_supported_backend"
        return {
            "yolo_obb": "target_supported_by_yolo",
            "semantic_segmentation": "target_supported_by_semantic_segmentation",
            "quantity_proposal": "target_supported_by_quantity_proposal",
            "qwen_point": "no_supported_specialist_qwen",
        }[_validate_kind(ordered[0])]

    @staticmethod
    def _target_classes(backend: object, target: CountTargetSpec) -> tuple[str, ...]:
        resolve = getattr(backend, "resolve_target_classes", None)
        if callable(resolve):
            return tuple(sorted(resolve(target)))
        return (target.canonical_label,)


def _validate_kind(backend: object) -> BackendKind:
    """Require an explicit known kind without echoing hostile values.
    要求显式且已知的 kind，不回显不可信值。"""

    kind = getattr(backend, "kind", None)
    if kind not in KNOWN_BACKEND_KINDS:
        raise CountingBackendUnavailableError(
            "unknown-target",
            primary_backend="unknown-backend",
            reason_code="INVALID_BACKEND_KIND",
        )
    return kind  # type: ignore[return-value]


def _validate_plan_contract(backend: object) -> None:
    if (
        not isinstance(getattr(backend, "name", None), str)
        or not getattr(backend, "name", "").strip()
        or not isinstance(getattr(backend, "priority", None), int)
        or isinstance(getattr(backend, "priority", None), bool)
        or not callable(getattr(backend, "is_enabled", None))
        or not callable(getattr(backend, "supports", None))
    ):
        raise _invalid_contract_error()


def _invalid_contract_error() -> CountingBackendUnavailableError:
    return CountingBackendUnavailableError(
        "unknown-target",
        primary_backend="unknown-backend",
        reason_code="INVALID_BACKEND_CONTRACT",
    )
