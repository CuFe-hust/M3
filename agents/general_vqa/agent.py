"""General VQA agent — thin VisualAgentBase subclass for open-ended QA.

通用 VQA Agent — 开放问答的轻量 VisualAgentBase 子类。覆盖 general_vqa、
scene_classification 与 multiple_choice_vqa 三个 task；选择题载荷包含
choices 与单/多选约束。本模块不做任何专用几何后处理，也不读取 Prompt
文件（提示文本以中性 PromptBinding 注入）。
"""

from __future__ import annotations

from typing import Any

from agents.schema import AgentName
from agents.visual_base import PromptBinding, VisualAgentBase
from data.schema import UnifiedSample
from models.base import VisionLanguageClient

# Neutral default prompt text (English mirror of the baseline general_vqa_v2
# prompt). The repository prompt file is intentionally not read by agents;
# the version string stays aligned with the baseline asset name.
# 中性默认提示文本（基线 general_vqa_v2 prompt 的英文镜像）。Agent 有意不
# 读取仓库 Prompt 文件；版本字符串与基线资产名保持一致。
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


class GeneralVQAAgent(VisualAgentBase):
    """Open-ended / closed-vocabulary visual QA agent.
    开放/闭集词汇视觉问答 Agent。"""

    name: AgentName = "general_vqa_agent"
    supported_tasks: frozenset[str] = frozenset({
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
    })

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

    def build_user_payload(self, sample: UnifiedSample) -> dict[str, Any]:
        """Extend the neutral payload with choice constraints for
        multiple_choice_vqa; other tasks keep the neutral payload unchanged.
        为 multiple_choice_vqa 扩展中性载荷（加入选项与单/多选约束）；其他
        task 保持中性载荷不变。"""
        payload = super().build_user_payload(sample)
        if sample.task == "multiple_choice_vqa":
            constraints = (
                sample.normalization.answer_constraints
                if sample.normalization is not None
                else {}
            )
            payload["choices"] = _extract_choices(constraints)
            payload["allow_multiple"] = bool(constraints.get("allow_multiple", False))
        return payload


def _extract_choices(constraints: dict[str, Any]) -> list[str]:
    """Extract string choices from answer constraints. The constraints are
    answer-domain restrictions (e.g. closed_vocabulary values), never the
    ground truth itself, so nothing is leaked.
    从答案约束提取字符串选项。约束是答案域限制（如 closed_vocabulary
    values），本身并非 ground truth，因此不泄漏任何内容。"""
    for key in ("choices", "values"):
        value = constraints.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
    return []
