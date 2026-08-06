"""Grounding agent — thin agent over the grounding visual primitive.

定位 Agent — 定位视觉原语上的轻量 Agent。只支持 grounding 一个 task；
结果使用统一 0..999 归一化坐标证据（由基类 VisualEvidence 契约强制），
不在本模块计算任何指标。
"""

from __future__ import annotations

from dataclasses import replace

from agents.base import AgentContext, AgentExecution
from agents.schema import AgentName
from agents.visual_base import PromptBinding, VisualAgentBase
from data.schema import UnifiedSample
from models.base import VisionLanguageClient

# Grounding reuses the neutral VQA instruction (the baseline catalog mapped
# grounding to general_vqa_v2); the base class appends the shared JSON output
# contract and enforces 0..999 evidence coordinates.
# 定位复用中性 VQA 指令（基线目录将 grounding 映射到 general_vqa_v2）；
# JSON 输出契约由基类附加，0..999 证据坐标由基类强制。
_DEFAULT_PROMPT_TEXT = (
    "Answer the question concisely from the image. Preserve up to four "
    "representative relevant localized objects as labeled evidence_items; "
    "copy all evidence-item boxes into boxes in the same order. Coordinates "
    "are whole-image 0..999 raster coordinates with the origin at the "
    "top-left, positive x to the right, and positive y downward. A box is one "
    "flat array [x1,y1,x2,y2], never a pair of corner arrays. Use an empty "
    "evidence list only when the answer genuinely has no localizable visual "
    "support. Do not include hidden reasoning."
)

_DEFAULT_PROMPT_VERSION = "general_vqa_v2"


class GroundingAgent(VisualAgentBase):
    """Thin agent over the grounding visual primitive.
    定位视觉原语上的轻量 Agent。"""

    name: AgentName = "grounding_agent"
    supported_tasks: frozenset[str] = frozenset({"grounding"})

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
