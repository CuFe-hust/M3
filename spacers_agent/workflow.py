"""Deprecated import-compatible aliases for the retired workflow module.
已弃用工作流模块的导入兼容别名。

Business execution lives in ``spacers_agent.workflows`` and ``spacers_agent.agents``.
业务执行位于 ``spacers_agent.workflows`` 与 ``spacers_agent.agents``。
"""

from __future__ import annotations

from pathlib import Path

from spacers_agent.agents.base import AgentContext, AgentExecution
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.agents.caption.agent import CaptionAgent
from spacers_agent.agents.change.agent import ChangeAgent
from spacers_agent.agents.counting import evidence as count_evidence
from spacers_agent.agents.counting.target_parser import CountTargetParser, TargetParser
from spacers_agent.agents.general_vqa.agent import GeneralVQAAgent
from spacers_agent.agents.grounding.agent import GroundingAgent
from spacers_agent.agents.spatial import evidence_merge as spatial_evidence
from spacers_agent.agents.spatial.agent import SpatialAgent
from spacers_agent.agents.visual_base import VisualAgentBase
from spacers_agent.clients.base import VisionLanguageClient
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.routing import CallBudgetFactory
from spacers_agent.routing.schemas import normalize_agent_name
from spacers_agent.schemas import ExpertResult, UnifiedSample
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.artifact_writer import atomic_write_json
from spacers_agent.workflows.dataset_runner import DatasetRunner


def _prompt_asset(key: str, text: str, version: str) -> PromptAsset:
    """Adapt a legacy prompt string to an immutable Prompt asset.
    将旧 Prompt 字符串适配为不可变 Prompt 资产。
    """

    return PromptAsset(key=key, path=Path(f"{key}.md"), version=version, text=text)


class VisualExpert(VisualAgentBase):
    """Compatibility adapter for the retired visual-expert public symbol.
    已弃用视觉专家公开符号的兼容适配器。
    """

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str, name: str, prompt_version: str) -> None:
        super().__init__(client, model, agent_name=name, default_prompt=_prompt_asset(name, prompt, prompt_version))


class ChangeExpert(VisualExpert):
    """Compatibility adapter for ChangeAgent imports.
    ChangeAgent 导入的兼容适配器。
    """

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        super().__init__(client, prompt, model, "change_expert", "change-expert-v1")


class GroundingExpert(VisualExpert):
    """Compatibility adapter for GroundingAgent imports.
    GroundingAgent 导入的兼容适配器。
    """

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        super().__init__(client, prompt, model, "grounding_expert", "general-vqa-v2")


class GeneralVQAExpert(VisualExpert):
    """Compatibility adapter for GeneralVQAAgent imports.
    GeneralVQAAgent 导入的兼容适配器。
    """

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        super().__init__(client, prompt, model, "general_vqa_expert", "general-vqa-v2")


class CaptionExpert(VisualExpert):
    """Compatibility adapter for CaptionAgent imports.
    CaptionAgent 导入的兼容适配器。
    """

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        super().__init__(client, prompt, model, "caption_expert", "caption-v1")


class SpatialExpert:
    """Compatibility adapter that delegates to the standalone SpatialAgent.
    委托给独立 SpatialAgent 的兼容适配器。
    """

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str, review_prompt: str = "", grid_prompt: str = "", grid_review_prompt: str = "") -> None:
        self._client = client
        self._agent = SpatialAgent(
            client,
            _prompt_asset("spatial", prompt, "spatial-v4"),
            model,
            grid_prompt=_prompt_asset("spatial_grid", grid_prompt, "spatial-v5") if grid_prompt else None,
            review_prompt=_prompt_asset("spatial_review", review_prompt, "spatial-candidate-review-v4") if review_prompt else None,
            grid_review_prompt=_prompt_asset("spatial_grid_review", grid_review_prompt, "spatial-candidate-review-v5") if grid_review_prompt else None,
            apply_geometry=False,
        )

    async def run(self, sample: UnifiedSample, *, artifact_dir: Path) -> ExpertResult:
        """Delegate one legacy call without recreating spatial-review logic.
        委托一次旧调用而不重新实现空间复查逻辑。
        """

        execution = await self._agent.run(
            sample,
            AgentContext(
                artifact_dir=artifact_dir,
                settings=AppSettings(),
                qwen_client=self._client,
                call_budget=CallBudgetFactory().create_for_sample(sample.task),
            ),
        )
        if not isinstance(execution.payload, ExpertResult):
            raise TypeError(f"SPATIAL_AGENT_RETURNED:{type(execution.payload).__name__}")
        return execution.payload


class WorkflowService:
    """Compatibility facade that delegates expert names to real Agents.
    将专家名称委托给真实 Agent 的兼容门面。
    """

    def __init__(self, client: VisionLanguageClient, prompts: dict[str, str], model: str) -> None:
        self._client = client
        self._registry = AgentRegistry()
        self._registry.register(ChangeAgent(client, _prompt_asset("change", prompts["change"], "change-dual-path-v1"), model))
        self._registry.register(GroundingAgent(client, _prompt_asset("grounding", prompts.get("grounding", prompts["general"]), "general-vqa-v2"), model))
        self._registry.register(SpatialAgent(
            client, _prompt_asset("spatial", prompts["spatial"], "spatial-v4"), model,
            grid_prompt=_prompt_asset("spatial_grid", prompts.get("spatial_grid", ""), "spatial-v5"),
            review_prompt=_prompt_asset("spatial_review", prompts.get("spatial_review", ""), "spatial-candidate-review-v4"),
            grid_review_prompt=_prompt_asset("spatial_grid_review", prompts.get("spatial_grid_review", ""), "spatial-candidate-review-v5"),
            apply_geometry=False,
        ))
        self._registry.register(GeneralVQAAgent(client, _prompt_asset("general", prompts["general"], "general-vqa-v2"), model))
        self._registry.register(CaptionAgent(client, _prompt_asset("caption", prompts["caption"], "caption-v1"), model))

    def get_agent(self, expert_name: str):
        """Return the standalone Agent behind a legacy expert name.
        返回旧专家名称背后的独立 Agent。
        """

        return self._registry.get(normalize_agent_name(expert_name))

    async def execute_agent(self, expert_name: str, sample: UnifiedSample, artifact_dir: Path) -> AgentExecution:
        """Execute a legacy expert name through its compatibility adapter.
        通过兼容适配器执行旧专家名称。
        """

        return await self.get_agent(expert_name).run(
            sample,
            AgentContext(
                artifact_dir=artifact_dir,
                settings=AppSettings(),
                qwen_client=self._client,
                call_budget=CallBudgetFactory().create_for_sample(sample.task),
            ),
        )


_parse_count_answer = count_evidence.parse_count_answer
_recover_count_proposal_header = count_evidence.recover_count_proposal_header
_box_evidence = count_evidence.box_evidence
_accepted_count_evidence = count_evidence.accepted_count_evidence
_merge_count_evidence = count_evidence.merge_count_evidence
_same_count_observation = count_evidence._same_count_observation
_is_tiny_border_fragment = count_evidence.is_tiny_border_fragment
_global_count_point = count_evidence.global_count_point
_needs_spatial_candidate_review = spatial_evidence.needs_candidate_review
_matches_position_target = spatial_evidence.matches_position_target
_position_target_label = spatial_evidence.position_target_label
_position_review_evidence = spatial_evidence.position_review_evidence
_is_status_answer_placeholder = spatial_evidence.is_status_answer_placeholder
_is_corner_anchored_box = spatial_evidence.is_corner_anchored_box
_merge_visual_evidence = spatial_evidence.merge_visual_evidence
_same_visual_observation = spatial_evidence.same_visual_observation
_prefer_candidate_evidence = spatial_evidence.prefer_candidate_evidence
_point_distance = spatial_evidence.point_distance
_maximum_repair_severity = spatial_evidence.maximum_repair_severity
_box_iou = spatial_evidence.box_iou
