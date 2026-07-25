"""Caption agent — dedicated prompt for remote-sensing image captioning.
图像描述 Agent — 遥感图像描述的专用 Prompt。
"""

from __future__ import annotations

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.visual_base import VisualAgentBase
from spacers_agent.schemas import UnifiedSample


class CaptionAgent(VisualAgentBase):
    """Dedicated captioning agent with its own versioned prompt. / 使用自有版本化 Prompt 的专用描述 Agent。"""

    name: AgentName = "caption_agent"
    supported_tasks: frozenset[str] = frozenset({"caption"})

    def __init__(self, client, prompts: dict[str, str], model: str) -> None:
        super().__init__(
            client,
            model,
            agent_name="caption_expert",
            default_prompt=prompts.get("caption", prompts.get("general", "")),
            default_prompt_version="caption-v1",
        )
        self._client_ref = client

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        result = await super().run(sample, artifact_dir=context.artifact_dir)
        return AgentExecution(
            agent_name=self.name,
            payload=result,
            result_filename="expert_result.json",
            trace={
                "agent_class": "spacers_agent.agents.caption.agent.CaptionAgent",
                "route": f"CaptionAgent.run -> VisualAgentBase.run -> {type(self._client_ref).__name__}.complete_json",
                "prompt_version": "caption-v1",
            },
        )
