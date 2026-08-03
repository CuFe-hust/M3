"""Deterministic and auditable counting-backend planning.
确定且可审计的计数后端执行计划。
"""

from __future__ import annotations

from spacers_agent.agents.counting.backends.base import BackendPlan, BackendSelection
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.schemas import CountTargetSpec, UnifiedSample
from spacers_agent.vqa_geometry import vrsbench_question_subtype


def is_vrsbench_quantity(sample: UnifiedSample) -> bool:
    """Return whether a sample is a VRSBench quantity VQA item.
    返回样本是否为 VRSBench 数量型 VQA 条目。
    """
    return (
        sample.dataset.casefold() == "vrsbench"
        and sample.task == "general_vqa"
        and vrsbench_question_subtype(sample.question, str(sample.metadata.get("question_type", ""))) == "counting"
    )


class BackendSelector:
    """Select a backend plan from configured mode and target capability.
    根据配置模式与目标能力选择后端执行计划。
    """

    def __init__(self, registry: BackendRegistry, *, default_backend: str = "auto") -> None:
        self._registry = registry
        self._default_backend = default_backend

    def plan(self, target: CountTargetSpec, sample: UnifiedSample) -> BackendPlan | None:
        """Plan a primary backend before availability checks can hide a bad deployment.
        在可用性检查隐藏部署错误之前规划主后端。
        """
        if is_vrsbench_quantity(sample):
            return self._named_plan("vrsbench_qwen_count", ("qwen_point",), "vrsbench_quantity_preserved", target)
        if sample.task not in {"counting", "fine_grained_counting"}:
            return None
        if self._default_backend == "qwen_point":
            return self._named_plan("qwen_point", (), "explicit_qwen_point", target)
        yolo_candidates = [
            backend for backend in self._registry.items()
            if backend.name not in {"qwen_point", "vrsbench_qwen_count"} and backend.supports(target)
        ]
        if self._default_backend == "yolo_obb":
            if not yolo_candidates:
                return self._named_plan("qwen_point", (), "explicit_yolo_unsupported_target_qwen", target)
            selected = max(yolo_candidates, key=lambda backend: (backend.priority, backend.name))
            return BackendPlan(selected.name, ("qwen_point",), (f"task_{sample.task}", "explicit_yolo", "highest_priority_supported_detector"), self._target_classes(selected, target))
        if yolo_candidates:
            selected = max(yolo_candidates, key=lambda backend: (backend.priority, backend.name))
            return BackendPlan(selected.name, ("qwen_point",), (f"task_{sample.task}", "target_supported_by_yolo", "highest_priority_supported_detector"), self._target_classes(selected, target))
        return self._named_plan("qwen_point", (), "no_supported_yolo_detector_qwen", target)

    def select(self, target: CountTargetSpec, sample: UnifiedSample | None = None) -> BackendSelection | None:
        """Preserve the legacy selection API for callers outside CountingAgent.
        为 CountingAgent 之外的调用方保留旧选择 API。
        """
        if sample is not None:
            plan = self.plan(target, sample)
            if plan is None:
                return None
            backend = self.backend_by_name(plan.primary_backend_name)
            if not backend.is_available():
                return None
            return BackendSelection(plan.primary_backend_name, plan.reason_codes, plan.target_classes)
        candidates = self._registry.list_available(target)
        if not candidates:
            return None
        selected = max(candidates, key=lambda backend: (backend.priority, backend.name))
        return BackendSelection(selected.name, ("highest_priority_available",), self._target_classes(selected, target))

    def backend(self, selection: BackendSelection):
        return self.backend_by_name(selection.backend_name)

    def backend_by_name(self, name: str):
        """Resolve a backend through the registry public API.
        通过注册表公开 API 解析后端。
        """
        return self._registry.get(name)

    def _named_plan(self, primary: str, fallbacks: tuple[str, ...], reason: str, target: CountTargetSpec) -> BackendPlan | None:
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
