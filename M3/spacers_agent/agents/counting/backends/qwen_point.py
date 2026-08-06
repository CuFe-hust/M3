"""Qwen point-counting backend — wraps PointCountingOrchestrator.
Qwen 点式计数后端 — 封装 PointCountingOrchestrator。
"""

from __future__ import annotations

from spacers_agent.agents.base import AgentContext
from spacers_agent.agents.counting.backends.base import CountingBackendOutcome, CountingRequest
from spacers_agent.agents.counting.point_pipeline import PointCountingOrchestrator
from spacers_agent.schemas import CountTargetSpec


class QwenPointCountingBackend:
    """Default tile-based point counting via Qwen. / 通过 Qwen 的默认 tile 点式计数。"""

    name = "qwen_point"
    priority = 0  # lowest — fallback / 最低 — 兜底

    def __init__(self, client, *, system_prompt: str, seam_prompt: str = "",
                 empty_review_prompt: str = "", settings=None) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._seam_prompt = seam_prompt
        self._empty_review_prompt = empty_review_prompt
        self._settings = settings

    def is_available(self) -> bool:
        return True  # always available / 始终可用

    def supports(self, target: CountTargetSpec) -> bool:
        return True  # handles all counting targets / 处理所有计数目标

    async def count(self, request: CountingRequest, context: AgentContext) -> CountingBackendOutcome:
        is_vrsbench = request.sample.dataset.casefold() == "vrsbench"
        orchestrator = PointCountingOrchestrator(
            self._client,
            counting=self._settings.counting,
            qwen=self._settings.models.qwen,
            system_prompt=self._system_prompt,
            run_dir=request.artifact_dir,
            seam_prompt=self._seam_prompt or None,
            empty_review_prompt=self._empty_review_prompt or None,
            before_qwen_call=context.call_budget.reserve_qwen,
        )
        counting = await orchestrator.count_image(
            request.image, sample_id=request.sample.sample_id,
            question=request.sample.question, target=request.target,
            minimum_scan_depth=(
                self._settings.counting.vrsbench_min_scan_depth if is_vrsbench else 0
            ),
            review_empty=(self._settings.counting.vrsbench_zero_review if is_vrsbench else False),
            upscale_max_side=(
                self._settings.counting.vrsbench_tile_upscale_max_side if is_vrsbench else None
            ),
        )
        return CountingBackendOutcome(counting=counting)
