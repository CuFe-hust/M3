"""Composition Root — dependency injection for the multi-Agent runtime.
组合根 — 多 Agent 运行时的依赖注入。

Creates clients, registries, agents and wires them together.
``cli.py`` calls ``assemble_runtime()`` and receives a ``RuntimeComponents``
instance; it never knows how individual agents are constructed.
创建客户端、注册表、Agent 并将它们组装在一起。
``cli.py`` 调用 ``assemble_runtime()`` 并接收 ``RuntimeComponents`` 实例；
它无需了解单个 Agent 的构造细节。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.agents.caption.agent import CaptionAgent
from spacers_agent.agents.change.agent import ChangeAgent
from spacers_agent.agents.counting.agent import CountingAgent
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.agents.counting.backends.qwen_point import QwenPointCountingBackend
from spacers_agent.agents.counting.backends.vrsbench_qwen_count import VRSBenchQwenCountBackend
from spacers_agent.agents.general_vqa.agent import GeneralVQAAgent
from spacers_agent.agents.grounding.agent import GroundingAgent
from spacers_agent.agents.spatial.agent import SpatialAgent
from spacers_agent.clients.base import VisionLanguageClient
from spacers_agent.clients.deepseek import DeepSeekJudgeClient
from spacers_agent.dataset_adapters import DatasetAdapter
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing import ROUTES, CallBudgetFactory, TaskRouter
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.artifact_writer import ArtifactWriter
from spacers_agent.workflows.dataset_runner import DatasetRunner
from spacers_agent.workflows.judge_service import JudgeService
from spacers_agent.workflows.sample_runner import SampleRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeComponents:
    """All wired components the CLI needs to run a dataset.
    CLI 运行数据集所需的全部已组装组件。
    """

    qwen_client: VisionLanguageClient
    judge_client: DeepSeekJudgeClient | None
    prompt_catalog: PromptCatalog
    router: TaskRouter
    agent_registry: AgentRegistry
    judge_service: JudgeService
    artifact_writer: ArtifactWriter
    call_budget_factory: CallBudgetFactory
    sample_runner: SampleRunner


def build_agent_registry(
    *,
    settings: AppSettings,
    qwen_client: VisionLanguageClient,
    prompt_catalog: PromptCatalog,
) -> AgentRegistry:
    """Build and populate the agent registry in the prescribed order.
    按指定顺序构建并填充 Agent 注册表。

    Construction order / 构造顺序:
    1. Counting Backend Registry
    2. CountingAgent
    3. ChangeAgent
    4. GroundingAgent
    5. SpatialAgent
    6. GeneralVQAAgent
    7. CaptionAgent
    8. Register all
    9. Validate router-referenced agents all exist
    """

    model = settings.models.qwen.model
    prompts = _load_prompts(prompt_catalog)
    registry = AgentRegistry()

    # 1. Counting backend registry / 计数后端注册表
    backend_registry = _create_counting_backend_registry(settings, qwen_client, prompts)

    # 2-7. Register agents in order / 按顺序注册 Agent
    registry.register(CountingAgent(qwen_client, prompts, model, backend_registry, settings=settings))
    registry.register(ChangeAgent(qwen_client, prompt_catalog.asset("change"), model))
    registry.register(GroundingAgent(qwen_client, prompt_catalog.asset("grounding"), model))
    registry.register(
        SpatialAgent(
            qwen_client,
            prompt_catalog.asset("spatial"),
            model,
            grid_prompt=prompt_catalog.asset("spatial_grid"),
            review_prompt=prompt_catalog.asset("spatial_review"),
            grid_review_prompt=prompt_catalog.asset("spatial_grid_review"),
            review_max_tokens=settings.models.qwen.spatial_review_max_tokens,
        )
    )
    registry.register(GeneralVQAAgent(qwen_client, prompt_catalog.asset("general"), model))
    registry.register(CaptionAgent(qwen_client, prompt_catalog.asset("caption"), model))

    # 9. Validate that all agents referenced by ROUTES exist / 校验 ROUTES 引用的全部 Agent 存在
    _validate_router_coverage(registry)

    logger.info("AgentRegistry built: %d agents — %s", len(registry), list(registry.names()))
    return registry


def assemble_runtime(
    settings: AppSettings,
    *,
    qwen_client: VisionLanguageClient,
    judge_client: DeepSeekJudgeClient | None = None,
    prompt_root: Path | None = None,
    router_prompt: str = "",
) -> RuntimeComponents:
    """Create and wire all agent runtime components without loading models.
    创建并组装全部 Agent 运行时组件，不加载模型。
    """

    if prompt_root is None:
        prompt_root = Path(__file__).resolve().parents[1] / "prompts"

    catalog = PromptCatalog(prompt_root)

    # Build agent registry (includes counting backend registry)
    # 构建 Agent 注册表（包含计数后端注册表）
    agent_registry = build_agent_registry(
        settings=settings,
        qwen_client=qwen_client,
        prompt_catalog=catalog,
    )

    # Router / 路由器
    router = TaskRouter(
        router_client=qwen_client,
        router_prompt=router_prompt or catalog["router"],
    )
    judge_service = JudgeService(
        settings,
        judge_prompt=catalog["count_judge"],
        vqa_judge_prompt=catalog["vqa_judge"],
        repair_prompt=catalog["json_repair"],
        judge_client=judge_client,
    )
    artifact_writer = ArtifactWriter()
    call_budget_factory = CallBudgetFactory()
    sample_runner = SampleRunner(
        settings,
        agent_registry,
        qwen_client,
        catalog,
        router=router,
        judge_service=judge_service,
        artifact_writer=artifact_writer,
        call_budget_factory=call_budget_factory,
    )

    return RuntimeComponents(
        qwen_client=qwen_client,
        judge_client=judge_client,
        prompt_catalog=catalog,
        router=router,
        agent_registry=agent_registry,
        judge_service=judge_service,
        artifact_writer=artifact_writer,
        call_budget_factory=call_budget_factory,
        sample_runner=sample_runner,
    )


def build_dataset_runner(
    runtime: RuntimeComponents,
    *,
    adapter: DatasetAdapter,
    run_dir: Path,
    settings: AppSettings,
    judge_policy: str,
) -> DatasetRunner:
    """Build a dataset runner around the exact injected runtime graph.
    围绕完全相同的注入运行时对象图构建数据集运行器。
    """

    return DatasetRunner(
        adapter,
        runtime.sample_runner,
        run_dir=run_dir,
        settings=settings,
        judge_policy=judge_policy,
    )


def _validate_router_coverage(registry: AgentRegistry) -> None:
    """Ensure every routed Agent is registered.
    确保每个被路由引用的 Agent 均已注册。
    """

    routed_agents = {
        agent_name
        for agent_names in ROUTES.values()
        for agent_name in agent_names
    }
    missing = [agent_name for agent_name in sorted(routed_agents) if not registry.contains(agent_name)]

    if missing:
        raise RuntimeError(
            f"Router references unregistered agents: {missing}; "
            f"registered={list(registry.names())}"
        )


def _load_prompts(catalog: PromptCatalog) -> dict[str, str]:
    """Load all prompt texts via the catalog. / 通过 catalog 加载全部 Prompt 文本。"""
    return {
        "count": catalog["count"],
        "count_zero_review": catalog["zero_review"],
        "count_proposal": catalog["vrsbench_proposal"],
        "count_localize": catalog["vrsbench_localizer"],
        "target": catalog["target"],
        "change": catalog["change"],
        "spatial": catalog["spatial"],
        "spatial_grid": catalog["spatial_grid"],
        "spatial_review": catalog["spatial_review"],
        "spatial_grid_review": catalog["spatial_grid_review"],
        "general": catalog["general"],
        "grounding": catalog["grounding"],
        "caption": catalog["caption"],
        "seam": catalog["seam"],
    }


def _create_counting_backend_registry(
    settings: AppSettings,
    client: VisionLanguageClient | None,
    prompts: dict[str, str] | None,
) -> BackendRegistry:
    """Create counting backend registry without loading optional YOLO models.
    创建计数后端注册表，且不加载可选 YOLO 模型。
    """

    registry = BackendRegistry()

    if client is not None and prompts is not None:
        # Qwen point backend — always available / Qwen 点式后端 — 始终可用
        registry.register(QwenPointCountingBackend(
            client, settings=settings,
            system_prompt=prompts.get("count", ""),
            seam_prompt=prompts.get("seam", ""),
            empty_review_prompt=prompts.get("count_zero_review", ""),
        ))

        # VRSBench quantity backend / VRSBench 数量后端
        registry.register(VRSBenchQwenCountBackend(
            client, settings=settings, prompts=prompts,
        ))

    if settings.backend.yolo.enabled:
        from spacers_agent.agents.counting.backends.yolo_model_store import YoloModelStore
        from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend

        model_store = YoloModelStore()
        for detector in settings.backend.yolo.detectors:
            if detector.enabled:
                registry.register(YoloOBBCountingBackend(detector, model_store=model_store))

    return registry
