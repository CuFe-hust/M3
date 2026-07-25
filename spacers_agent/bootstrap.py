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
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing import ROUTES, TaskRouter
from spacers_agent.settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeComponents:
    """All wired components the CLI needs to run a dataset.
    CLI 运行数据集所需的全部已组装组件。
    """

    agent_registry: AgentRegistry
    router: TaskRouter
    qwen_client: VisionLanguageClient
    judge_client: object | None  # DeepSeekJudgeClient | None
    prompt_catalog: PromptCatalog


def build_agent_registry(
    *,
    settings: AppSettings,
    qwen_client: VisionLanguageClient,
    prompts: dict[str, str],
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
    registry = AgentRegistry()

    # 1. Counting backend registry / 计数后端注册表
    backend_registry = _create_counting_backend_registry(settings, qwen_client, prompts)

    # 2-7. Register agents in order / 按顺序注册 Agent
    registry.register(CountingAgent(qwen_client, prompts, model, backend_registry))
    registry.register(ChangeAgent(qwen_client, prompts, model))
    registry.register(GroundingAgent(qwen_client, prompts, model))
    registry.register(SpatialAgent(qwen_client, prompts, model))
    registry.register(GeneralVQAAgent(qwen_client, prompts, model))
    registry.register(CaptionAgent(qwen_client, prompts, model))

    # 9. Validate that all agents referenced by ROUTES exist / 校验 ROUTES 引用的全部 Agent 存在
    _validate_router_coverage(registry)

    logger.info("AgentRegistry built: %d agents — %s", len(registry), list(registry.names()))
    return registry


def assemble_runtime(
    settings: AppSettings,
    *,
    qwen_client: VisionLanguageClient,
    judge_client: object | None = None,
    prompt_root: Path | None = None,
    router_prompt: str = "",
) -> RuntimeComponents:
    """Create and wire all agent runtime components without loading models.
    创建并组装全部 Agent 运行时组件，不加载模型。
    """

    if prompt_root is None:
        prompt_root = Path(__file__).resolve().parents[1] / "prompts"

    catalog = PromptCatalog(prompt_root)
    prompts = _load_prompts(catalog)

    # Build agent registry (includes counting backend registry)
    # 构建 Agent 注册表（包含计数后端注册表）
    agent_registry = build_agent_registry(
        settings=settings,
        qwen_client=qwen_client,
        prompts=prompts,
    )

    # Router / 路由器
    router = TaskRouter(router_client=qwen_client, router_prompt=router_prompt)

    return RuntimeComponents(
        agent_registry=agent_registry,
        router=router,
        qwen_client=qwen_client,
        judge_client=judge_client,
        prompt_catalog=catalog,
    )


def _validate_router_coverage(registry: AgentRegistry) -> None:
    """Ensure every expert name referenced by ROUTES has a registered agent.
    确保 ROUTES 引用的每个 expert 名都有注册 Agent。
    """

    from spacers_agent.agents.base import LEGACY_AGENT_NAME_ALIASES

    all_expert_names: set[str] = set()
    for experts in ROUTES.values():
        all_expert_names.update(experts)

    missing: list[str] = []
    for expert_name in sorted(all_expert_names):
        agent_name = LEGACY_AGENT_NAME_ALIASES.get(expert_name, expert_name)
        if not registry.contains(agent_name):
            missing.append(f"{expert_name!r} → {agent_name!r}")

    if missing:
        raise RuntimeError(
            f"Router references agents not in registry: {', '.join(missing)}. "
            f"Registered: {list(registry.names())}"
        )


def _load_prompts(catalog: PromptCatalog) -> dict[str, str]:
    """Load all prompt texts via the catalog. / 通过 catalog 加载全部 Prompt 文本。"""
    return {
        "count": catalog["count_tile"],
        "count_zero_review": catalog["count_zero_review"],
        "count_proposal": catalog["count_proposal"],
        "count_localize": catalog["count_localize"],
        "target": catalog["target_parse"],
        "change": catalog["change"],
        "spatial": catalog["spatial"],
        "spatial_grid": catalog["spatial_grid"],
        "spatial_review": catalog["spatial_review"],
        "spatial_grid_review": catalog["spatial_grid_review"],
        "general": catalog["general_vqa"],
        "caption": catalog["caption"],
        "seam": catalog["seam_verify"],
    }


def _create_counting_backend_registry(
    settings: AppSettings,
    client: VisionLanguageClient | None,
    prompts: dict[str, str] | None,
) -> BackendRegistry:
    """Create counting backend registry with QwenPoint + VRSBenchQwenCount backends.
    创建包含 QwenPoint + VRSBenchQwenCount 后端的计数后端注册表。
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

    # YOLO backends — only when enabled + weight file exists
    # YOLO 后端 — 仅在启用且权重文件存在时
    yolo_config = settings.backend.yolo
    if yolo_config.enabled and yolo_config.detectors:
        try:
            from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend  # noqa: PLC0415
        except ImportError:
            logger.warning("YOLO enabled but ultralytics not installed; skipping YOLO backends")
            return registry

        for detector in yolo_config.detectors:
            if not detector.enabled:
                continue
            weight_path = Path(detector.weights)
            if not weight_path.is_file():
                logger.warning("YOLO weight '%s' missing at %s — skipping", detector.name, weight_path)
                continue
            backend = YoloOBBCountingBackend(detector)
            registry.register(backend)
            logger.info("Registered YOLO backend '%s' (pri=%d, classes=%s)", detector.name, detector.priority, detector.classes)

    return registry
