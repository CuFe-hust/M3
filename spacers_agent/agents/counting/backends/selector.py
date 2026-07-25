"""Counting backend selector — picks the best backend for a given target.
计数后端选择器 — 为给定目标选择最优后端。
"""

from __future__ import annotations

import logging

from spacers_agent.agents.counting.backends.base import BackendSelection, CountingBackend
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.schemas import CountTargetSpec, UnifiedSample

logger = logging.getLogger(__name__)


class BackendSelector:
    """Select the best counting backend based on target and sample context.
    根据目标和样本上下文选择最优计数后端。
    """

    def __init__(self, registry: BackendRegistry) -> None:
        self._registry = registry

    def select(self, target: CountTargetSpec, sample: UnifiedSample | None = None) -> BackendSelection | None:
        """Return the best matching backend selection, or None. / 返回最优匹配后端选择，或 None。"""
        candidates = self._registry.list_available(target)
        if not candidates:
            return None

        # For VRSBench general_vqa counting, prefer VRSBench-specific backend
        # VRSBench general_vqa 计数优先选择 VRSBench 专用后端
        is_vrsbench_count = (
            sample is not None
            and sample.dataset == "VRSBench"
            and sample.task == "general_vqa"
        )
        vrsbench_candidates = [b for b in candidates if "vrsbench" in b.name] if is_vrsbench_count else []

        selected = vrsbench_candidates[0] if vrsbench_candidates else candidates[0]

        reason = ("vrsbench_count" if vrsbench_candidates else "default_qwen",)
        return BackendSelection(
            backend_name=selected.name,
            reason_codes=reason,
            target_classes=(target.canonical_label,),
        )
