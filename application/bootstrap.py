"""Composition root: the only place concrete model clients are created.

组合根：唯一创建具体模型客户端的地方。任何 workflows / agents /
evaluation 不得自行 create model。Qwen 客户端在一次组装中只创建一次；
DeepSeek 客户端仅在注入 api_key 时创建（无 key 即 judge 禁用，回退纯
确定性）。导入本模块绝无副作用：不加载权重、不调用模型。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.base import CallBudget as _CallBudgetProtocol
from agents.caption import CaptionAgent
from agents.change.agent import ChangeAgent
from agents.counting import (
    AgentCountingSettings,
    CountingAgent,
    CountingSettings,
)
from agents.counting.backends import BackendRegistry
from agents.counting.backends.quantity_proposal import QuantityProposalBackend
from agents.counting.backends.qwen_point import QwenPointCountingBackend
from agents.counting.backends.yolo_obb import YoloOBBCountingBackend
from agents.counting.backends.yolo_model_store import YoloModelStore
from agents.counting.expert_catalog import ExpertCatalog
from agents.counting.schema import CountTargetSpec
from agents.counting.settings import CountingTargetStrategy
from agents.general_vqa import GeneralVQAAgent
from agents.grounding import GroundingAgent
from agents.registry import AgentRegistry
from agents.spatial import SpatialAgent
from agents.visual_base import PromptBinding
from application.prompts import PromptCatalog
from application.settings import AppSettings
from data.adapters.base import DatasetAdapter
from data.registry import build_default_registry
from evaluation.judges.deepseek import DeepSeekJudgeClient
from models.base import DenseSemanticClient, VisionLanguageClient
from models.cache import JsonResponseCache
from models.entry import create_model
from reporting.builder import build_report
from reporting.schema import Report
from reporting.visualization import render_counting_overlay
from routing.router import TaskRouter
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudgetFactory
from workflows.dataset_runner import DatasetRunner
from workflows.judge_service import JudgeService
from workflows.run_store import RunStore
from workflows.sample_runner import SampleRunner
from workflows.task_resolver import TaskResolver


@dataclass(frozen=True)
class RuntimeComponents:
    """Every runtime dependency assembled exactly once by the composition
    root. 由组合根恰好组装一次的全体运行时依赖。"""

    qwen_client: VisionLanguageClient
    judge_client: DeepSeekJudgeClient | None
    prompt_catalog: PromptCatalog
    router: TaskRouter
    task_resolver: TaskResolver
    agent_registry: AgentRegistry
    judge_service: JudgeService
    artifact_writer: ArtifactWriter
    call_budget_factory: CallBudgetFactory
    run_store: RunStore
    build_report: Callable[[Path], Report]
    render_overlay: Callable[..., Path]
    dataset_runner_factory: Callable[..., DatasetRunner] = field(
        default=None  # type: ignore[assignment]
    )
    sample_runner_factory: Callable[[Path], SampleRunner] = field(
        default=None  # type: ignore[assignment]
    )


def assemble_runtime(
    settings: AppSettings,
    *,
    project_root: Path,
    qwen_client: VisionLanguageClient | None = None,
    semantic_client: DenseSemanticClient | None = None,
    api_key: str | None = None,
    prompts_root: Path | None = None,
) -> RuntimeComponents:
    """Assemble the full runtime. The Qwen client is created exactly once
    here (or injected for tests); the DeepSeek client is created only when an
    api_key is provided — without a key the judge service degrades to
    deterministic-only. 组装完整运行时。Qwen 客户端在此恰好创建一次（或由
    测试注入）；仅在提供 api_key 时创建 DeepSeek 客户端——无 key 时 judge
    服务退化为纯确定性。"""

    catalog = PromptCatalog(prompts_root or project_root / "prompts")
    service_cache = JsonResponseCache(settings.runs.root / "service" / "cache")
    if qwen_client is None:
        qwen_client = create_model(
            "qwen_transformers",
            settings=settings.models.qwen,
            repair_prompt=catalog["json_repair"],
            cache=service_cache,
        )
    if settings.agents.change.semantic.enabled and semantic_client is None:
        semantic_client = create_model(
            "segformer_transformers",
            settings=settings.models.segformer_isaid,
        )
    if not settings.agents.change.semantic.enabled:
        semantic_client = None
    agent_registry = _build_agent_registry(
        settings,
        catalog,
        qwen_client,
        semantic_client,
    )
    router = TaskRouter()
    task_resolver = TaskResolver(
        qwen_client,
        system_prompt=catalog["task_resolver"],
        confidence_threshold=settings.router.confidence_threshold,
    )
    judge_client = _build_judge_client(settings, catalog, api_key)
    judge_service = JudgeService(
        judge_prompt=catalog["count_judge"],
        judge_prompt_version=catalog.version("count_judge"),
        vqa_judge_prompt=catalog["vqa_judge"],
        vqa_judge_prompt_version=catalog.version("vqa_judge"),
        judge_client=judge_client,
        model_id=settings.models.deepseek.model,
        counting_min_confidence=settings.counting.min_confidence,
    )
    artifact_writer = ArtifactWriter()
    call_budget_factory = CallBudgetFactory(
        default_qwen_calls=settings.router.default_qwen_calls,
        default_deepseek_calls=settings.router.default_deepseek_calls,
    )
    run_store = RunStore(settings.runs.root, project_root)

    def make_sample_runner(data_root: Path) -> SampleRunner:
        return SampleRunner(
            agent_registry,
            router,
            qwen_client,
            artifact_writer,
            call_budget_factory,
            judge_service=judge_service,
            fallback_on_partial=settings.router.fallback_on_partial,
            data_root=data_root,
        )

    def dataset_runner_factory(
        adapter: DatasetAdapter,
        run_dir: Path,
        *,
        judge_policy: str,
        judge_sample_rate: float | None = None,
        data_root: Path,
    ) -> DatasetRunner:
        return DatasetRunner(
            adapter,
            make_sample_runner(data_root),
            run_dir=run_dir,
            artifact_writer=artifact_writer,
            judge_policy=judge_policy,
            judge_sample_rate=judge_sample_rate,
            task_resolver=task_resolver,
            call_budget_factory=call_budget_factory,
        )

    components = RuntimeComponents(
        qwen_client=qwen_client,
        judge_client=judge_client,
        prompt_catalog=catalog,
        router=router,
        task_resolver=task_resolver,
        agent_registry=agent_registry,
        judge_service=judge_service,
        artifact_writer=artifact_writer,
        call_budget_factory=call_budget_factory,
        run_store=run_store,
        build_report=build_report,
        render_overlay=render_counting_overlay,
        dataset_runner_factory=dataset_runner_factory,
        sample_runner_factory=make_sample_runner,
    )
    return components


def _build_agent_registry(
    settings: AppSettings,
    catalog: PromptCatalog,
    qwen_client: VisionLanguageClient,
    semantic_client: DenseSemanticClient | None = None,
) -> AgentRegistry:
    """Register every business agent in stable order; all routable tasks must
    be covered. 按稳定顺序注册全部业务 Agent；所有可路由任务必须有覆盖。"""

    expert_catalog = ExpertCatalog.load(
        Path(__file__).resolve().parents[1]
        / "agents"
        / "counting"
        / "expert_catalog.json"
    )
    backend_registry = _build_backend_registry(
        settings,
        catalog,
        qwen_client,
        expert_catalog=expert_catalog,
    )
    counting_agent = CountingAgent(
        qwen_client,
        target_prompt=catalog["target"],
        backend_registry=backend_registry,
        target_prompt_version=catalog.version("target"),
        default_backend=settings.agents.counting.default_backend,
        fallback_to_qwen_on_unavailable=settings.backend.yolo.fallback_to_qwen_on_unavailable,
        fallback_to_qwen_on_error=settings.backend.yolo.fallback_to_qwen_on_error,
        verify_empty_with_qwen=settings.backend.yolo.verify_empty_with_qwen,
        trust_empty_detection=settings.backend.trust_empty_detection,
        expert_catalog=expert_catalog,
    )
    change_agent = ChangeAgent(
        qwen_client,
        semantic_client=semantic_client,
        prompt=None,
        settings=settings.agents.change,
    )
    grounding_agent = GroundingAgent(qwen_client)
    spatial_agent = SpatialAgent(
        qwen_client,
        prompt=PromptBinding(text=catalog["spatial"], version=catalog.version("spatial")),
        grid_prompt=PromptBinding(
            text=catalog["spatial_grid"], version=catalog.version("spatial_grid")
        ),
        review_prompt=catalog["spatial_review"],
        review_prompt_version=catalog.version("spatial_review"),
        grid_review_prompt=catalog["spatial_grid_review"],
        grid_review_prompt_version=catalog.version("spatial_grid_review"),
        review_max_tokens=settings.models.qwen.spatial_review_max_tokens,
    )
    general_vqa_agent = GeneralVQAAgent(qwen_client)
    caption_agent = CaptionAgent(qwen_client)

    registry = AgentRegistry()
    for agent in (counting_agent, change_agent, grounding_agent, spatial_agent,
                  general_vqa_agent, caption_agent):
        registry.register(agent)
    registry.validate_task_coverage(set(_routable_tasks()))
    return registry


def _build_backend_registry(
    settings: AppSettings,
    catalog: PromptCatalog,
    qwen_client: VisionLanguageClient,
    *,
    expert_catalog: ExpertCatalog | None = None,
) -> BackendRegistry:
    """The Qwen point backend is always registered; YOLO backends only when
    enabled. Qwen 点式后端恒注册；YOLO 后端仅启用时注册。"""

    registry = BackendRegistry()
    strategy_resolver = (
        (lambda target: _target_strategy(expert_catalog, target))
        if expert_catalog is not None
        else None
    )
    registry.register(
        QwenPointCountingBackend(
            qwen_client,
            counting=settings.counting,
            system_prompt=catalog["count_tile"],
            prompt_version=catalog.version("count_tile"),
            empty_review_prompt=catalog["zero_review"],
            empty_review_prompt_version=catalog.version("zero_review"),
            strategy_resolver=strategy_resolver,
        )
    )
    registry.register(
        QuantityProposalBackend(
            qwen_client,
            counting=settings.counting,
            proposal_prompt=catalog["count_localize"],
            localizer_prompt=catalog["count_localize"],
            proposal_prompt_version=catalog.version("count_localize"),
            localizer_prompt_version=catalog.version("count_localize"),
        )
    )
    if settings.backend.yolo.enabled:
        model_store = YoloModelStore()
        for detector in settings.backend.yolo.detectors:
            if detector.enabled:
                registry.register(
                    YoloOBBCountingBackend(
                        detector,
                        counting=settings.counting,
                        model_store=model_store,
                    )
                )
    return registry


def _target_strategy(
    catalog: ExpertCatalog,
    target: CountTargetSpec,
) -> CountingTargetStrategy:
    hints = catalog.target_hints(target).get("hints", ())
    return CountingTargetStrategy.from_hint_names(hints)


def _build_judge_client(
    settings: AppSettings,
    catalog: PromptCatalog,
    api_key: str | None,
) -> DeepSeekJudgeClient | None:
    """No api_key means the judge is disabled: the client is None and the
    judge service records deterministic evaluations only. 无 api_key 即 judge
    禁用：客户端为 None，judge 服务只记录确定性评估。"""

    if not api_key:
        return None
    return DeepSeekJudgeClient(
        settings.models.deepseek,
        api_key=api_key,
        judge_prompt=catalog["count_judge"],
        repair_prompt=catalog["json_repair"],
        cache=JsonResponseCache(settings.runs.root / "service" / "deepseek_cache"),
    )


def _routable_tasks() -> set[str]:
    """All tasks the deterministic router can dispatch. 确定性路由器可分发的
    全部任务。"""

    from routing.policies import POLICIES

    return set(POLICIES)
