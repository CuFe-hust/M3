"""Spatial-relation agent with optional candidate review.

空间关系 Agent — 含可选候选复查。按 SpatialQuerySpec 选择普通/grid prompt，
候选复核最多一次（受 CallBudget 限制），最终调用通用几何后处理；不检查
dataset、不在 Agent 内实现任何数据集评测。
"""

from __future__ import annotations

from dataclasses import replace

from agents.base import AgentContext, AgentExecution
from agents.schema import AgentName, AgentResult
from agents.spatial.candidate_review import SpatialCandidateReviewer
from agents.spatial.geometry import apply_spatial_geometry
from agents.spatial.schema import SpatialQuerySpec, spatial_query_from_metadata
from agents.visual_base import PromptBinding, VisualAgentBase
from data.schema import UnifiedSample
from models.base import VisionLanguageClient


class SpatialAgent(VisualAgentBase):
    """Spatial agent with candidate review. / 含候选复查的空间 Agent。"""

    name: AgentName = "spatial_agent"
    supported_tasks: frozenset[str] = frozenset({"spatial_relation"})

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        prompt: PromptBinding,
        grid_prompt: PromptBinding | None = None,
        review_prompt: str = "",
        review_prompt_version: str = "",
        grid_review_prompt: str = "",
        grid_review_prompt_version: str = "",
        review_max_tokens: int = 128,
        apply_geometry: bool = True,
    ) -> None:
        super().__init__(
            client,
            agent_name=self.name,
            supported_tasks=self.supported_tasks,
            prompt=prompt,
        )
        self._grid_prompt = grid_prompt
        self._apply_geometry = apply_geometry
        self._reviewer = SpatialCandidateReviewer(
            client,
            review_prompt=review_prompt,
            review_prompt_version=review_prompt_version,
            grid_review_prompt=grid_review_prompt,
            grid_review_prompt_version=grid_review_prompt_version,
            review_max_tokens=review_max_tokens,
        )

    def _spatial_spec(self, sample: UnifiedSample) -> SpatialQuerySpec | None:
        return spatial_query_from_metadata(sample.metadata)

    def select_prompt(self, sample: UnifiedSample) -> PromptBinding:
        """Use the grounded grid prompt for grid-position queries.
        九宫格位置查询使用实体定位 Prompt。"""
        spec = self._spatial_spec(sample)
        if (
            spec is not None
            and spec.operation == "grid_position"
            and self._grid_prompt is not None
        ):
            return self._grid_prompt
        return super().select_prompt(sample)

    async def run(
        self,
        sample: UnifiedSample,
        context: AgentContext,
    ) -> AgentExecution:
        """Run the shared visual pipeline, review candidates exactly once, then
        apply the generic geometry post-processing. No dataset branch.
        运行共享视觉管线，候选复核恰好一次，再应用通用几何后处理。无数据集
        分支。"""
        execution = await super().run(sample, context)
        result = execution.payload
        assert isinstance(result, AgentResult)
        spec = self._spatial_spec(sample)
        operation = spec.operation if spec is not None else None
        target_label = spec.target_label if spec is not None else None
        reviewed = await self._reviewer.review(
            sample,
            result,
            context.artifact_dir,
            operation=operation,
            target_label=target_label,
            data_root=context.data_root,
            budget=context.call_budget,
        )
        if self._apply_geometry and spec is not None:
            reviewed = apply_spatial_geometry(spec, reviewed)

        route = (
            f"{type(self).__name__}.run -> VisualAgentBase.run -> complete_json"
        )
        if reviewed.geometry.get("candidate_review_used"):
            route += " -> SpatialCandidateReviewer.review"
        trace = {
            **execution.trace,
            "agent_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "route": route,
            "candidate_review_used": reviewed.geometry.get("candidate_review_used", False),
        }
        return replace(execution, payload=reviewed, trace=trace)
