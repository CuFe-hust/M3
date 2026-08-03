"""Spatial-relation agent with optional candidate review.
空间关系 Agent — 含可选候选复查。
"""

from __future__ import annotations

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.spatial.candidate_review import SpatialCandidateReviewer
from spacers_agent.agents.visual_base import PromptSelection, VisualAgentBase
from spacers_agent.clients.base import VisionLanguageClient
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.schemas import UnifiedSample
from spacers_agent.vqa_geometry import apply_vrsbench_geometry, vrsbench_question_subtype


class SpatialAgent(VisualAgentBase):
    """Spatial agent with candidate review. / 含候选复查的空间 Agent。"""

    name: AgentName = "spatial_agent"
    supported_tasks: frozenset[str] = frozenset({"spatial_relation"})

    def __init__(
        self,
        client: VisionLanguageClient,
        prompt: PromptAsset,
        model: str,
        *,
        grid_prompt: PromptAsset | None = None,
        review_prompt: PromptAsset | None = None,
        grid_review_prompt: PromptAsset | None = None,
        review_max_tokens: int = 128,
        apply_geometry: bool = True,
    ) -> None:
        super().__init__(
            client,
            model,
            agent_name="spatial_expert",
            default_prompt=prompt,
        )
        self._client_ref = client
        self._grid_prompt = grid_prompt
        self._apply_geometry = apply_geometry
        self._reviewer = SpatialCandidateReviewer(
            client,
            model,
            review_prompt=review_prompt.text if review_prompt is not None else "",
            review_prompt_version=review_prompt.version if review_prompt is not None else "",
            grid_review_prompt=grid_review_prompt.text if grid_review_prompt is not None else "",
            grid_review_prompt_version=grid_review_prompt.version if grid_review_prompt is not None else "",
            review_max_tokens=review_max_tokens,
        )

    # ── hooks ─────────────────────────────────────────────────────────────

    def select_prompt(self, sample: UnifiedSample) -> PromptSelection:
        """Use grounded prompt for grid-position questions. / 九宫格位置问题使用实体定位 Prompt。"""
        subtype = vrsbench_question_subtype(
            sample.question,
            str(sample.metadata.get("question_type", "")),
        )
        if subtype == "grid_position" and self._grid_prompt is not None:
            return PromptSelection(text=self._grid_prompt.text, version=self._grid_prompt.version)
        return super().select_prompt(sample)

    # ── main execution / 主执行 ───────────────────────────────────────────

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        result = await super().run(sample, artifact_dir=context.artifact_dir)

        # Review candidates exactly once before the final geometry repair.
        # 在最终几何修复前仅复核一次候选目标。
        reviewed = await self._reviewer.review(sample, result, context.artifact_dir)

        # VRSBench geometry (once, not duplicated)
        # VRSBench 几何（仅一次，不重复）
        question_type = str(sample.metadata.get("question_type", ""))
        if self._apply_geometry and sample.dataset == "VRSBench" and sample.task == "general_vqa":
            reviewed = apply_vrsbench_geometry(sample.question, question_type, reviewed)

        route = f"SpatialAgent.run -> VisualAgentBase.run -> {type(self._client_ref).__name__}.complete_json"
        if reviewed.geometry.get("candidate_review_used"):
            route += " -> SpatialCandidateReviewer.review"

        return AgentExecution(
            agent_name=self.name,
            payload=reviewed,
            result_filename="expert_result.json",
            trace={
                "agent_class": "spacers_agent.agents.spatial.agent.SpatialAgent",
                "route": route,
                "prompt_version": self.select_prompt(sample).version,
                "candidate_review_used": reviewed.geometry.get("candidate_review_used", False),
            },
        )
