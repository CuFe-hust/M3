"""General VQA agent — thin VisualAgentBase subclass for open-ended QA.

通用 VQA Agent — 开放问答的轻量 VisualAgentBase 子类。覆盖 general_vqa、
scene_classification、multiple_choice_vqa 与 spatial_relation 四个 task；
选择题载荷包含 choices 与单/多选约束，输出在 postprocess 中按 choices 约束
校验。spatial_relation 与通用 VQA 共享同一条单次 Qwen 调用路径，不做任何
专用几何后处理，也不读取 Prompt 文件（提示文本以中性 PromptBinding 注入）。
"""

from __future__ import annotations

import re
from typing import Any

from agents.errors import AgentExecutionError
from agents.schema import AgentName, AgentResult
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
        "spatial_relation",
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
            constraints = _choice_constraints(sample)
            payload["choices"] = _extract_choices(constraints)
            payload["allow_multiple"] = bool(constraints.get("allow_multiple", False))
        return payload

    async def postprocess(
        self,
        sample: UnifiedSample,
        result: AgentResult,
    ) -> AgentResult:
        """Enforce the multiple-choice output constraint: a single-choice
        answer must map to exactly one choice; a multi-choice answer must
        contain only choices, deduplicated in stable choice order. Violations
        downgrade status to partial and record answer_constraint_violation.
        强制选择题输出约束：单选答案必须唯一映射到一个选项；多选答案只能
        包含选项、去重并按选项稳定顺序排列。违规将状态降级为 partial 并记录
        answer_constraint_violation。"""
        if sample.task != "multiple_choice_vqa":
            return result
        constraints = _choice_constraints(sample)
        choices = _extract_choices(constraints)
        if not choices:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="multiple_choice_sample_without_choices",
            )
        allow_multiple = bool(constraints.get("allow_multiple", False))
        violation, normalized_answer = _validate_choice_answer(
            result.answer, choices, allow_multiple
        )
        if violation is None:
            if normalized_answer is not None and normalized_answer != result.answer.strip():
                result = result.model_copy(update={"answer": normalized_answer})
            return result
        geometry = dict(result.geometry or {})
        geometry["answer_constraint_violation"] = violation
        return result.model_copy(update={"status": "partial", "geometry": geometry})


def _choice_constraints(sample: UnifiedSample) -> dict[str, Any]:
    """Answer constraints for a multiple-choice sample.
    多选题样本的答案约束。"""
    if sample.normalization is None:
        return {}
    return sample.normalization.answer_constraints


def _normalize_choice(value: str) -> str:
    return value.strip().casefold()


def _choice_text(choice: str) -> str:
    """Strip a leading option letter prefix (A., B), C - ...).
    去除选项首字母前缀（A.、B)、C - ...）。"""
    text = choice.strip()
    match = re.match(r"^([A-Za-z])[.)、\s-]\s*(.*)$", text)
    if match:
        return match.group(2).strip()
    return text


def _match_choice(value: str, choices: list[str]) -> str | None:
    """Match an answer item to a choice: by normalized full text, by the
    prefix-stripped text, or by the leading option letter (A/B/...).
    将答案项匹配到选项：按归一化全文、去前缀文本或选项首字母（A/B/...）。"""
    normalized = _normalize_choice(value)
    for choice in choices:
        if _normalize_choice(choice) == normalized:
            return choice
    for choice in choices:
        if _normalize_choice(_choice_text(choice)) == normalized:
            return choice
    letter = value.strip().upper()
    if len(letter) == 1 and "A" <= letter <= "Z":
        for choice in choices:
            if choice.strip().upper() == letter:
                return choice
            if _normalize_choice(choice).startswith(letter.lower()):
                return choice
    return None


def _validate_choice_answer(
    answer: str,
    choices: list[str],
    allow_multiple: bool,
) -> tuple[str | None, str | None]:
    """Return (violation, normalized_answer); both None when the answer
    satisfies the constraint. 返回（违规描述、规范化答案）；满足约束时两者
    均为 None。"""
    if not allow_multiple:
        if _match_choice(answer, choices) is not None:
            return None, None
        return f"answer {answer!r} does not map to a single choice", None
    parts = [part.strip() for part in re.split(r"[,;，；]", answer) if part.strip()]
    if not parts:
        return "empty multiple-choice answer", None
    matched: list[str] = []
    for part in parts:
        choice = _match_choice(part, choices)
        if choice is None:
            return f"answer item {part!r} is not among the choices", None
        if choice not in matched:
            matched.append(choice)
    # Stable order follows the choice list. / 稳定顺序遵循选项列表。
    ordered = [choice for choice in choices if choice in matched]
    return None, ", ".join(ordered)


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
