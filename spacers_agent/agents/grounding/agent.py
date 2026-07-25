"""Grounding agent — self-contained, no dependency on workflow experts.
定位 Agent — 自包含，不依赖 workflow 专家。
"""

from __future__ import annotations

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.visual_base import VisualAgentBase
from spacers_agent.schemas import UnifiedSample


class GroundingAgent(VisualAgentBase):
    """Thin agent over the grounding visual primitive. / 定位视觉原语上的轻量 Agent。"""

    name: AgentName = "grounding_agent"
    supported_tasks: frozenset[str] = frozenset({"grounding"})

    def __init__(self, client, prompts: dict[str, str], model: str) -> None:
        super().__init__(
            client,
            model,
            agent_name="grounding_expert",  # persisted external name / 持久化外部名称
            default_prompt=prompts["general"],
            default_prompt_version="general-vqa-v2",  # TECH DEBT: uses general_vqa prompt / 技术债：使用 general_vqa prompt
        )
        self._client_ref = client

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        result = await super().run(sample, artifact_dir=context.artifact_dir)
        return AgentExecution(
            agent_name=self.name,
            payload=result,
            result_filename="expert_result.json",
            trace={
                "agent_class": "spacers_agent.agents.grounding.agent.GroundingAgent",
                "route": f"GroundingAgent.run -> VisualAgentBase.run -> {type(self._client_ref).__name__}.complete_json",
                "prompt_version": "general-vqa-v2",
                "tech_debt": "grounding uses general_vqa prompt — separate grounding prompt pending",
            },
        )
