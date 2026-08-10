"""Caption agent — dedicated prompt for remote-sensing image captioning.

图像描述 Agent — 遥感图像描述的专用 Prompt。只支持 caption 一个 task；
复用 VisualAgentBase 的全部请求构造，不在本模块重复任何请求代码，也不
计算任何描述指标。
"""

from __future__ import annotations

from dataclasses import replace

from agents.base import AgentContext, AgentExecution
from agents.schema import AgentName
from agents.visual_base import PromptBinding, VisualAgentBase
from data.schema import UnifiedSample
from models.base import VisionLanguageClient

# English mirror of the baseline caption_v1 prompt's instruction part; the
# base class appends the shared JSON output contract.
# 基线 caption_v1 prompt 指令部分的英文镜像；JSON 输出契约由基类附加。
_DEFAULT_PROMPT_TEXT = (
    "You are a remote-sensing image analyst. Describe the given image in one "
    "concise, factual English sentence. Focus on the scene type, visible "
    "objects, and spatial arrangement. Do not list objects or use bullet "
    "points. Do not mention the coordinate system, image borders, or your "
    "own process."
)

_DEFAULT_PROMPT_VERSION = "caption_v1"


class CaptionAgent(VisualAgentBase):
    """Dedicated captioning agent with its own versioned prompt.
    使用自有版本化 Prompt 的专用描述 Agent。"""

    name: AgentName = "caption_agent"
    supported_tasks: frozenset[str] = frozenset({"caption"})

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        prompt: PromptBinding | None = None,
    ) -> None:
        super().__init__(
            client,
            agent_name=self.name,
            supported_tasks=self.supported_tasks,
            prompt=prompt
            or PromptBinding(text=_DEFAULT_PROMPT_TEXT, version=_DEFAULT_PROMPT_VERSION),
        )

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        """Run the shared pipeline and enrich the trace with a stable agent
        class and route; no request construction happens here.
        运行共享管线并向 trace 增加稳定的 agent class 与 route；本处不做
        任何请求构造。"""
        execution = await super().run(sample, context)
        return replace(
            execution,
            trace={
                **execution.trace,
                "agent_class": f"{type(self).__module__}.{type(self).__qualname__}",
                "route": f"{type(self).__name__}.run -> VisualAgentBase.run -> complete_json",
            },
        )
