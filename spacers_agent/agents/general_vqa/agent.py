"""General VQA agent — self-contained, no dependency on workflow experts.
通用 VQA Agent — 自包含，不依赖 workflow 专家。
"""

from __future__ import annotations

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.visual_base import VisualAgentBase
from spacers_agent.schemas import UnifiedSample


class GeneralVQAAgent(VisualAgentBase):
    """General visual QA agent for open-ended questions. / 开放问题的通用视觉问答 Agent。"""

    name: AgentName = "general_vqa_agent"
    supported_tasks: frozenset[str] = frozenset({
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
        "caption",  # fallback for caption path / caption 兜底路径
    })

    def __init__(self, client, prompts: dict[str, str], model: str) -> None:
        super().__init__(
            client,
            model,
            agent_name="general_vqa_expert",
            default_prompt=prompts["general"],
            default_prompt_version="general-vqa-v2",
        )
        self._client_ref = client

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        result = await super().run(sample, artifact_dir=context.artifact_dir)
        return AgentExecution(
            agent_name=self.name,
            payload=result,
            result_filename="expert_result.json",
            trace={
                "agent_class": "spacers_agent.agents.general_vqa.agent.GeneralVQAAgent",
                "route": f"GeneralVQAAgent.run -> VisualAgentBase.run -> {type(self._client_ref).__name__}.complete_json",
                "prompt_version": "general-vqa-v2",
            },
        )
