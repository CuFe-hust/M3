"""Qwen point-counting backend — wraps PointCountingOrchestrator.
Qwen 点式计数后端 — 封装 PointCountingOrchestrator。
"""

from __future__ import annotations

from spacers_agent.agents.counting.backends.base import CountingBackend, CountingRequest
from spacers_agent.counting import PointCountingOrchestrator
from spacers_agent.schemas import CountTargetSpec, CountingResult


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

    async def count(self, request: CountingRequest, context: object) -> CountingResult:
        orchestrator = PointCountingOrchestrator(
            self._client,
            counting=self._settings.counting,
            qwen=self._settings.models.qwen,
            system_prompt=self._system_prompt,
            run_dir=request.artifact_dir,
            seam_prompt=self._seam_prompt or None,
            empty_review_prompt=self._empty_review_prompt or None,
        )
        return await orchestrator.count_image(
            request.image, sample_id=request.sample.sample_id,
            question=request.sample.question, target=request.target,
        )
