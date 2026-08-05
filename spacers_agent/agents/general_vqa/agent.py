"""General VQA agent — self-contained, no dependency on workflow experts.
通用 VQA Agent — 自包含，不依赖 workflow 专家。
"""

from __future__ import annotations

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.visual_base import VisualAgentBase
from models.base import VisionLanguageClient
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.schemas import UnifiedSample
from spacers_agent.vqa_geometry import apply_vrsbench_geometry


class GeneralVQAAgent(VisualAgentBase):
    """General visual QA agent for open-ended questions. / 开放问题的通用视觉问答 Agent。"""

    name: AgentName = "general_vqa_agent"
    supported_tasks: frozenset[str] = frozenset({
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
    })

    def __init__(self, client: VisionLanguageClient, prompt: PromptAsset, model: str) -> None:
        super().__init__(
            client,
            model,
            agent_name="general_vqa_expert",
            default_prompt=prompt,
        )
        self._client_ref = client

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        result = await super().run(sample, artifact_dir=context.artifact_dir)
        if sample.dataset == "VRSBench" and sample.task == "general_vqa":
            result = apply_vrsbench_geometry(
                sample.question,
                str(sample.metadata.get("question_type", "")),
                result,
            )
        return AgentExecution(
            agent_name=self.name,
            payload=result,
            result_filename="expert_result.json",
            trace={
                "agent_class": "spacers_agent.agents.general_vqa.agent.GeneralVQAAgent",
                "route": f"GeneralVQAAgent.run -> VisualAgentBase.run -> {type(self._client_ref).__name__}.complete_json",
                "prompt_version": self.select_prompt(sample).version,
            },
        )
