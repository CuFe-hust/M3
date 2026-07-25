"""Spatial-relation agent with optional candidate review.
空间关系 Agent — 含可选候选复查。
"""

from __future__ import annotations

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.spatial.candidate_review import SpatialCandidateReviewer
from spacers_agent.agents.visual_base import PromptSelection, VisualAgentBase
from spacers_agent.schemas import UnifiedSample
from spacers_agent.vqa_geometry import vrsbench_question_subtype
from spacers_agent.workflow import apply_vrsbench_geometry  # geometry post-process / 几何后处理


class SpatialAgent(VisualAgentBase):
    """Spatial agent with candidate review. / 含候选复查的空间 Agent。"""

    name: AgentName = "spatial_agent"
    supported_tasks: frozenset[str] = frozenset({"spatial_relation"})

    def __init__(self, client, prompts: dict[str, str], model: str) -> None:
        super().__init__(
            client,
            model,
            agent_name="spatial_expert",
            default_prompt=prompts["spatial"],
            default_prompt_version="spatial-v4",
        )
        self._client_ref = client
        self._grid_prompt = prompts.get("spatial_grid", "")
        self._reviewer = SpatialCandidateReviewer(
            client,
            model,
            review_prompt=prompts.get("spatial_review", ""),
            review_prompt_version="spatial-candidate-review-v2",
            grid_review_prompt=prompts.get("spatial_grid_review", ""),
            grid_review_prompt_version="spatial-candidate-review-v3",
        )

    # ── hooks ─────────────────────────────────────────────────────────────

    def select_prompt(self, sample: UnifiedSample) -> PromptSelection:
        """Use grounded prompt for grid-position questions. / 九宫格位置问题使用实体定位 Prompt。"""
        subtype = vrsbench_question_subtype(
            sample.question,
            str(sample.metadata.get("question_type", "")),
        )
        if subtype == "grid_position" and self._grid_prompt:
            return PromptSelection(text=self._grid_prompt, version="spatial-v5")
        return super().select_prompt(sample)

    async def postprocess(self, sample: UnifiedSample, result):
        """Run candidate review and VRSBench geometry. / 运行候选复查与 VRSBench 几何。"""
        # Candidate review / 候选复查
        reviewed = await self._reviewer.review(sample, result, None)  # artifact_dir via caller

        # VRSBench geometry post-process (applied once, not in review)
        # VRSBench 几何后处理（仅应用一次，不在复查中重复）
        question_type = str(sample.metadata.get("question_type", ""))
        if sample.dataset == "VRSBench" and sample.task == "general_vqa":
            reviewed = apply_vrsbench_geometry(sample.question, question_type, reviewed)

        return reviewed

    # ── main execution / 主执行 ───────────────────────────────────────────

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        result = await super().run(sample, artifact_dir=context.artifact_dir)

        # Post-process with candidate review (pass artifact_dir for review artifacts)
        # 使用候选复查后处理（传递 artifact_dir 用于复查产物）
        reviewed = await self._reviewer.review(sample, result, context.artifact_dir)

        # VRSBench geometry (once, not duplicated)
        # VRSBench 几何（仅一次，不重复）
        question_type = str(sample.metadata.get("question_type", ""))
        if sample.dataset == "VRSBench" and sample.task == "general_vqa":
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
