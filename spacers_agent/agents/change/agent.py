"""Change-detection agent — self-contained, no dependency on workflow experts.
变化检测 Agent — 自包含，不依赖 workflow 专家。
"""

from __future__ import annotations

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.visual_base import PromptSelection, VisualAgentBase
from spacers_agent.clients.base import VisionLanguageClient
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.routing.budget import CallBudgetExceeded
from spacers_agent.schemas import ExpertResult, UnifiedSample


class ChangeAgent(VisualAgentBase):
    """Thin agent over the change visual primitive. / 变化视觉原语上的轻量 Agent。"""

    name: AgentName = "change_agent"
    supported_tasks: frozenset[str] = frozenset({"change_caption", "change_qa"})

    def __init__(
        self,
        client: VisionLanguageClient,
        prompt: PromptAsset,
        model: str,
        *,
        analysis_prompt: PromptAsset | None = None,
        verification_prompt: PromptAsset | None = None,
    ) -> None:
        super().__init__(
            client,
            model,
            agent_name="change_expert",  # persisted external name / 持久化外部名称
            default_prompt=prompt,
        )
        self._client_ref = client  # for trace
        self._analysis_prompt = analysis_prompt
        self._verification_prompt = verification_prompt

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        if (
            sample.task != "change_caption"
            or self._analysis_prompt is None
            or self._verification_prompt is None
        ):
            result = await super().run(sample, artifact_dir=context.artifact_dir)
            return self._execution(sample, result, call_count=1)

        # Reserve each call only when it is actually needed.
        # 仅在确实需要调用时逐次占用预算。
        if context.call_budget is not None:
            context.call_budget.reserve_qwen()

        analysis = await super().run(
            sample,
            artifact_dir=context.artifact_dir,
            prompt_selection=PromptSelection(
                text=self._analysis_prompt.text,
                version=self._analysis_prompt.version,
            ),
            artifact_subdir="change_expert/analysis",
            request_id_suffix="analysis",
        )
        verification_reasons = _verification_reasons(analysis)
        if not verification_reasons:
            return self._execution(
                sample,
                analysis,
                call_count=1,
                analysis_prompt_version=self._analysis_prompt.version,
                selected_stage="analysis",
                verification_triggered=False,
                verification_reasons=[],
            )

        if context.call_budget is not None:
            try:
                context.call_budget.reserve_qwen()
            except CallBudgetExceeded as error:
                raise CallBudgetExceeded(
                    "Conditional change verification requires one additional Qwen call"
                ) from error
        verification = await super().run(
            sample,
            artifact_dir=context.artifact_dir,
            prompt_selection=PromptSelection(
                text=self._verification_prompt.text,
                version=self._verification_prompt.version,
            ),
            user_payload_updates={
                "first_pass_analysis": {
                    "answer": analysis.answer,
                    "evidence": analysis.evidence,
                    "boxes": analysis.boxes,
                    "evidence_items": [
                        item.model_dump(mode="json")
                        for item in analysis.evidence_items
                    ],
                }
            },
            request_id_suffix="verification",
        )
        selected, selected_stage, verification_guard = _select_verified_result(
            analysis,
            verification,
        )
        return self._execution(
            sample,
            selected,
            call_count=2,
            analysis_prompt_version=self._analysis_prompt.version,
            verification_prompt_version=self._verification_prompt.version,
            selected_stage=selected_stage,
            verification_guard=verification_guard,
            verification_triggered=True,
            verification_reasons=verification_reasons,
        )

    def _execution(
        self,
        sample: UnifiedSample,
        result: ExpertResult,
        *,
        call_count: int,
        analysis_prompt_version: str | None = None,
        verification_prompt_version: str | None = None,
        selected_stage: str = "final",
        verification_guard: str | None = None,
        verification_triggered: bool = False,
        verification_reasons: list[str] | None = None,
    ) -> AgentExecution:
        """Build the persisted execution wrapper for one change result.
        为一次变化检测结果构建持久化执行包装。
        """

        prompt_version = (
            verification_prompt_version
            or analysis_prompt_version
            or self.select_prompt(sample).version
        )
        if analysis_prompt_version is not None:
            stages = [
                {
                    "name": "analysis",
                    "prompt_version": analysis_prompt_version,
                    "artifact_path": "change_expert/analysis",
                }
            ]
            if verification_prompt_version is not None:
                stages.append(
                    {
                        "name": "verification",
                        "prompt_version": verification_prompt_version,
                        "artifact_path": "change_expert",
                    }
                )
        else:
            stages = [
                {
                    "name": "final",
                    "prompt_version": prompt_version,
                    "artifact_path": "change_expert",
                }
            ]
        return AgentExecution(
            agent_name=self.name,
            payload=result,
            result_filename="expert_result.json",
            trace={
                "agent_class": "spacers_agent.agents.change.agent.ChangeAgent",
                "route": f"ChangeAgent.run -> VisualAgentBase.run -> {type(self._client_ref).__name__}.complete_json",
                "prompt_version": prompt_version,
                "image_roles": [ref.role for ref in sample.images],
                "model_call_count": call_count,
                "selected_stage": selected_stage,
                "verification_guard": verification_guard,
                "verification_triggered": verification_triggered,
                "verification_reasons": verification_reasons or [],
                "stages": stages,
            },
        )


def _verification_reasons(result: ExpertResult) -> list[str]:
    """Return deterministic reasons for an optional second visual pass.
    返回触发可选第二次视觉复核的确定性原因。
    """

    reasons: list[str] = []
    if result.status != "completed":
        reasons.append("analysis_status_not_completed")
    combined = " ".join([result.answer, *result.evidence]).casefold()
    if "no visually supported land-cover change" in combined:
        reasons.append("analysis_qualified_no_change")
    if any(
        term in combined
        for term in (
            "possible",
            "possibly",
            "likely",
            "may ",
            "might",
            "appears to",
            "seems",
            "suggests",
            "uncertain",
            "unclear",
            "ambiguous",
            "could be",
        )
    ):
        reasons.append("analysis_contains_uncertainty")
    if (
        not _is_no_change_answer(result.answer)
        and not _has_contrastive_change_evidence(result)
    ):
        reasons.append("positive_without_contrastive_evidence")
    return reasons


def _select_verified_result(
    analysis: ExpertResult,
    verification: ExpertResult,
) -> tuple[ExpertResult, str, str | None]:
    """Reject an unsupported positive override of a no-change analysis.
    拒绝在没有任何证据时将第一阶段的无变化结论改判为有变化。
    """

    if (
        _is_no_change_answer(analysis.answer)
        and not _is_no_change_answer(verification.answer)
    ):
        if _is_appearance_only_positive_change(verification):
            return (
                analysis,
                "analysis",
                "rejected_appearance_only_positive_override",
            )
        if (
            _is_qualified_no_change_answer(analysis.answer)
            and _has_stable_structural_anchor(verification)
        ):
            return verification, "verification", None
        if not _has_contrastive_change_evidence(verification):
            return (
                analysis,
                "analysis",
                "rejected_non_contrastive_positive_override",
            )
    return verification, "verification", None


def _is_qualified_no_change_answer(answer: str) -> bool:
    """Recognize the analysis conclusion that deliberately requests verification."""

    normalized = " ".join(answer.casefold().split())
    return "no visually supported land-cover change" in normalized


def _has_stable_structural_anchor(result: ExpertResult) -> bool:
    """Recognize stable man-made anchors in the answer or evidence."""

    combined = " ".join(
        [result.answer, *result.evidence]
    ).casefold()
    return any(
        term in combined
        for term in (
            "building",
            "house",
            "residential",
            "villa",
            "roof",
            "structure",
            "construction",
            "road",
            "path",
            "parking",
            "paved",
            "pavement",
            "bridge",
            "runway",
            "water body",
            "waterbody",
        )
    )


def _is_appearance_only_positive_change(result: ExpertResult) -> bool:
    """Identify appearance-only vegetation claims without a stable structural anchor.
    识别没有稳定结构锚点、仅由植被外观构成的正向变化结论。
    """

    combined = " ".join(
        [result.answer, *result.evidence]
    ).casefold()
    appearance_terms = (
        "vegetation",
        "green",
        "greener",
        "grass",
        "tree",
        "forest",
        "woodland",
        "canopy",
        "bare ground",
        "brown ground",
        "color",
        "colour",
        "season",
    )
    return (
        any(term in combined for term in appearance_terms)
        and not _has_stable_structural_anchor(result)
    )


def _has_contrastive_change_evidence(result: ExpertResult) -> bool:
    """Require geometry or a localized T1/T2 textual comparison.
    要求几何证据，或同时包含位置及 T1/T2 状态的文字对照。
    """

    if result.boxes or result.evidence_items:
        return True
    return any(_is_contrastive_observation(value) for value in result.evidence)


def _is_contrastive_observation(value: str) -> bool:
    """Reject evidence that merely restates a positive final answer.
    拒绝仅仅换句话复述有变化结论的所谓证据。
    """

    normalized = " ".join(value.casefold().split())
    has_t1 = any(
        token in normalized
        for token in ("t1", "first image", "earlier image", "before image")
    )
    has_t2 = any(
        token in normalized
        for token in ("t2", "second image", "later image", "after image")
    )
    has_location = any(
        token in normalized
        for token in (
            "upper",
            "lower",
            "left",
            "right",
            "center",
            "central",
            "middle",
            "top",
            "bottom",
            "corner",
            "region",
            "area near",
            "along",
            "around",
        )
    )
    return has_t1 and has_t2 and has_location


def _is_no_change_answer(answer: str) -> bool:
    """Recognize common English no-change conclusions without references.
    在不读取参考答案的前提下识别常见英文无变化结论。
    """

    normalized = " ".join(answer.casefold().split())
    if any(
        phrase in normalized
        for phrase in (
            "no change",
            "no significant",
            "no verifiable",
            "no detectable land-cover change",
            "no visible land-cover change",
            "no visually supported land-cover change",
            "scenes are identical",
            "images are identical",
            "same as before",
            "nothing has changed",
        )
    ):
        return True
    return normalized.startswith(
        (
            "the scene remains unchanged",
            "the scenes remain unchanged",
            "the image remains unchanged",
            "the images remain unchanged",
            "the land cover remains unchanged",
            "the landscape remains unchanged",
        )
    )
