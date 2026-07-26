"""Deterministic counting-backend selection rules.
确定性的计数后端选择规则。
"""

from __future__ import annotations

from spacers_agent.agents.counting.backends.base import BackendSelection
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
        and vrsbench_question_subtype(
            sample.question,
            str(sample.metadata.get("question_type", "")),
        )
        == "counting"
    )


class BackendSelector:
    """Select an active backend by explicit task semantics and stable name.
    按明确的任务语义和稳定名称选择活动后端。
    """

    def __init__(self, registry: BackendRegistry) -> None:
        self._registry = registry

    def select(
        self,
        target: CountTargetSpec,
        sample: UnifiedSample | None = None,
    ) -> BackendSelection | None:
        """Return a supported selection without depending on registration order.
        返回不依赖注册顺序的受支持选择。
        """

        if sample is not None:
            if is_vrsbench_quantity(sample):
                return self._select_named(
                    "vrsbench_qwen_count",
                    target,
                    reason="vrsbench_quantity",
                )
            if sample.task in {"counting", "fine_grained_counting"}:
                return self._select_named("qwen_point", target, reason=f"task_{sample.task}")
            return None

        candidates = self._registry.list_available(target)
        if not candidates:
            return None
        selected = max(candidates, key=lambda backend: (getattr(backend, "priority", 0), backend.name))
        return BackendSelection(
            backend_name=selected.name,
            reason_codes=("highest_priority_available",),
            target_classes=(target.canonical_label,),
        )

    def backend(self, selection: BackendSelection):
        """Resolve a prior selection through the registry public API.
        通过注册表公共 API 解析先前的选择。
        """

        return self._registry.get(selection.backend_name)

    def _select_named(
        self,
        name: str,
        target: CountTargetSpec,
        *,
        reason: str,
    ) -> BackendSelection | None:
        try:
            backend = self._registry.get(name)
        except KeyError:
            return None
        if not backend.is_available() or not backend.supports(target):
            return None
        return BackendSelection(
            backend_name=name,
            reason_codes=(reason,),
            target_classes=(target.canonical_label,),
        )
