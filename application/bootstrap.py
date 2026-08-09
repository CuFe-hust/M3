"""Composition root: the only place concrete model clients are created.

组合根：唯一创建具体模型客户端的地方。任何 workflows / agents /
evaluation 不得自行 create model。Qwen 客户端在一次组装中只创建一次；
DeepSeek 客户端仅在注入 api_key 时创建（无 key 即 judge 禁用，回退纯
确定性）。导入本模块绝无副作用：不加载权重、不调用模型。
"""

from __future__ import annotations

import json
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
from agents.counting.backends.semantic_segmentation import (
    SemanticSegmentationCountingBackend,
)
from agents.counting.backends.yolo_obb import YoloOBBCountingBackend
from agents.counting.backends.yolo_model_store import YoloModelStore
from agents.counting.expert_catalog import ExpertCatalog, ExpertSpec
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
from models.settings import SegFormerSettings
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


class RuntimeCompositionError(ValueError):
    """A fail-closed composition contract failed without exposing host paths."""


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
    asset_root = Path(__file__).resolve().parents[1]
    expert_catalog = _load_expert_catalog(asset_root)
    segformer_clients = _build_segformer_clients(
        settings,
        expert_catalog,
        project_root=asset_root,
    )
    if settings.agents.change.semantic.enabled and semantic_client is None:
        semantic_client = segformer_clients.get(
            settings.models.segformer_isaid.logical_model_id
        )
        if semantic_client is None:
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
        expert_catalog=expert_catalog,
        segformer_clients=segformer_clients,
        project_root=asset_root,
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
    *,
    expert_catalog: ExpertCatalog | None = None,
    segformer_clients: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> AgentRegistry:
    """Register every business agent in stable order; all routable tasks must
    be covered. 按稳定顺序注册全部业务 Agent；所有可路由任务必须有覆盖。"""

    root = project_root or Path(__file__).resolve().parents[1]
    expert_catalog = expert_catalog or _load_expert_catalog(root)
    backend_registry = _build_backend_registry(
        settings,
        catalog,
        qwen_client,
        expert_catalog=expert_catalog,
        segformer_clients=segformer_clients,
        project_root=root,
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
        prompt=PromptBinding(
            text=catalog["change"], version=catalog.version("change")
        ),
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
    segformer_clients: dict[str, Any] | None = None,
    project_root: Path | None = None,
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
            seam_prompt=catalog["seam"],
            seam_prompt_version=catalog.version("seam"),
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
        for detector in sorted(
            settings.backend.yolo.detectors,
            key=lambda item: item.name,
        ):
            if detector.enabled:
                validated_detector = detector
                if expert_catalog is not None:
                    validated_detector = _catalog_validated_yolo_detector(
                        detector,
                        expert_catalog,
                    )
                registry.register(
                    YoloOBBCountingBackend(
                        validated_detector,
                        counting=settings.counting,
                        model_store=model_store,
                    )
                )
    if expert_catalog is not None:
        root = project_root or Path(__file__).resolve().parents[1]
        clients = segformer_clients or _build_segformer_clients(
            settings,
            expert_catalog,
            project_root=root,
        )
        for expert in _semantic_experts(expert_catalog):
            registry.register(
                SemanticSegmentationCountingBackend(
                    clients[expert.logical_model_id],
                    expert,
                    settings.counting,
                )
            )
    return registry


def _load_expert_catalog(project_root: Path) -> ExpertCatalog:
    return ExpertCatalog.load(
        project_root / "agents" / "counting" / "expert_catalog.json"
    )


def _semantic_experts(catalog: ExpertCatalog) -> tuple[ExpertSpec, ...]:
    # C1 intentionally exposed lookup/candidates only.  Composition needs the
    # complete immutable declarations to instantiate enabled model providers.
    experts = getattr(catalog, "_experts", ())
    return tuple(
        sorted(
            (
                expert
                for expert in experts
                if expert.enabled and expert.kind == "semantic_segmentation"
            ),
            key=lambda expert: expert.backend_name,
        )
    )


def _build_segformer_clients(
    settings: AppSettings,
    catalog: ExpertCatalog,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Create lazy clients from catalog assets, reusing equal logical models."""

    clients: dict[str, Any] = {}
    identities: dict[str, tuple[SegFormerSettings, tuple[tuple[int, str], ...]]] = {}
    for expert in _semantic_experts(catalog):
        runtime = _segformer_runtime_settings(settings, expert, project_root)
        labels = _verified_class_map(expert, project_root)
        identity = (runtime, tuple(labels.items()))
        existing = identities.get(expert.logical_model_id)
        if existing is not None:
            if existing != identity:
                raise RuntimeCompositionError(
                    "SegFormer logical model id maps to inconsistent assets"
                )
            continue
        clients[expert.logical_model_id] = create_model(
            "segformer_transformers",
            settings=runtime,
            id_to_label=labels,
        )
        identities[expert.logical_model_id] = identity
    return clients


def _segformer_runtime_settings(
    settings: AppSettings,
    expert: ExpertSpec,
    project_root: Path,
) -> SegFormerSettings:
    try:
        profile = settings.models.segformer_profile(
            backend_name=expert.backend_name,
            logical_model_id=expert.logical_model_id,
        )
    except ValueError as error:
        raise RuntimeCompositionError(str(error)) from None
    model_dir = _safe_asset(project_root, expert.asset.model_dir)
    class_map = _safe_asset(project_root, expert.asset.class_map)
    weights = _safe_asset(project_root, expert.asset.weights)
    if class_map.parent != model_dir or weights.parent != model_dir:
        raise RuntimeCompositionError("SegFormer assets must share the declared model directory")
    payload = profile.model_dump()
    payload.update(
        {
            "model_path": model_dir,
            "logical_model_id": expert.logical_model_id,
            "weights_filename": weights.name,
            "weights_sha256": expert.asset.sha256,
            "classes_filename": class_map.name,
            "processor_path": None,
            "allow_download": False,
        }
    )
    return SegFormerSettings.model_validate(payload)


def _safe_asset(project_root: Path, reference: str | None) -> Path:
    if reference is None:
        raise RuntimeCompositionError(
            "enabled SegFormer expert has an incomplete asset declaration"
        )
    root = project_root.resolve()
    candidate = (root / Path(reference)).resolve()
    if candidate == root or root not in candidate.parents:
        raise RuntimeCompositionError("expert asset escapes the project root")
    return candidate


def _verified_class_map(expert: ExpertSpec, project_root: Path) -> dict[int, str]:
    path = _safe_asset(project_root, expert.asset.class_map)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        raw = document["id2name"]
        count = document["num_classes"]
        inverse = document["name2id"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        raise RuntimeCompositionError("SegFormer class map is missing or invalid") from None
    if not isinstance(raw, dict) or not raw or count != len(raw):
        raise RuntimeCompositionError("SegFormer class map is inconsistent")
    expected_keys = {str(index) for index in range(len(raw))}
    if set(raw) != expected_keys:
        raise RuntimeCompositionError("SegFormer class ids must be contiguous")
    labels = {index: raw[str(index)] for index in range(len(raw))}
    if any(not isinstance(label, str) or not label.strip() for label in labels.values()):
        raise RuntimeCompositionError("SegFormer class labels must be non-empty strings")
    if inverse != {label: index for index, label in labels.items()}:
        raise RuntimeCompositionError("SegFormer class map inverse differs from id mapping")
    declared = {
        label
        for support in expert.supports.values()
        for label in support.model_labels
    }
    if not declared.issubset(set(labels.values())):
        raise RuntimeCompositionError("SegFormer catalog labels differ from verified class map")
    return labels


def _catalog_validated_yolo_detector(detector: Any, catalog: ExpertCatalog) -> Any:
    try:
        expert = catalog.expert(detector.name)
    except KeyError:
        raise RuntimeCompositionError(
            "enabled YOLO detector is absent from expert catalog"
        ) from None
    if expert.kind != "yolo_obb" or not expert.enabled:
        raise RuntimeCompositionError("YOLO detector catalog declaration is not enabled")
    if expert.logical_model_id != detector.model_id:
        raise RuntimeCompositionError("YOLO logical model id differs from expert catalog")
    backend = YoloOBBCountingBackend(detector, counting=CountingSettings())
    for canonical, support in expert.supports.items():
        if support.counting_mode == "unsupported":
            continue
        target = CountTargetSpec(
            canonical_label=canonical,
            inclusion_rule="include the declared target",
            exclusion_rule="exclude every other object",
        )
        resolved = {value.casefold() for value in backend.resolve_target_classes(target)}
        expected = {value.casefold() for value in support.model_labels}
        if resolved != expected:
            raise RuntimeCompositionError(
                "YOLO catalog model labels differ from detector declarations"
            )
    return detector.model_copy(update={"priority": expert.priority})


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
