"""Composition root: the only place concrete model clients are created.

组合根：唯一创建具体模型客户端的地方。任何 workflows / agents /
evaluation 不得自行 create model。Qwen 客户端在一次组装中只创建一次；
DeepSeek 客户端仅在注入 api_key 时创建（无 key 即 judge 禁用，回退纯
确定性）。导入本模块绝无副作用：不加载权重、不调用模型。
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.base import CallBudget as _CallBudgetProtocol
from agents.base import VisualPlanBindings
from agents.caption import CaptionAgent
from agents.change.agent import ChangeAgent
from agents.change.perception import SemanticExpertBinding
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
from agents.counting.settings import CountingTargetStrategy, YoloDetectorSettings
from agents.counting.target_parser import CountTargetResolver
from agents.evidence_catalog import CatalogCategoryError, EvidenceCatalog
from agents.general_vqa import GeneralVQAAgent
from agents.general_vqa.evidence.executor import (
    EvidencePolicy,
    ObjectEvidenceExecutor,
)
from agents.general_vqa.evidence.schema import EvidencePreprocessing
from agents.grounding import GroundingAgent
from agents.grounding.evidence import (
    GroundingEvidenceExecutor,
    GroundingEvidencePolicy,
)
from agents.registry import AgentRegistry
from agents.schema import COUNTING_TASKS, GENERAL_VQA_AGENT_TASKS
from agents.visual_base import PromptBinding
from application.prompts import PromptCatalog
from application.settings import (
    AppSettings,
    VisualDetectorSettings,
    VisualEvidencePreprocessSettings,
)
from data.adapters.base import DatasetAdapter
from data.registry import build_default_registry
from evaluation.judges.deepseek import DeepSeekJudgeClient
from models.base import (
    DenseSemanticClient,
    LearnedChangeClient,
    ModelCacheIdentity,
    ObjectDetectionOutput,
    RuntimeObjectDetectionClient,
    SemanticMaskClient,
    VisionLanguageClient,
    hash_class_names,
    require_model_cache_identity,
)
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
from workflows.schema import EvidencePreprocessingIdentity, VQA_ASSISTANCE_SCOPE
from workflows.visual_planner import (
    VisualTaskPlanner,
)


@dataclass(frozen=True)
class RuntimeComponents:
    """Every runtime dependency assembled exactly once by the composition
    root. 由组合根恰好组装一次的全体运行时依赖。"""

    qwen_client: VisionLanguageClient
    judge_client: DeepSeekJudgeClient | None
    prompt_catalog: PromptCatalog
    router: TaskRouter
    agent_registry: AgentRegistry
    judge_service: JudgeService
    artifact_writer: ArtifactWriter
    call_budget_factory: CallBudgetFactory
    run_store: RunStore
    build_report: Callable[[Path], Report]
    render_overlay: Callable[..., Path]
    # Canonical v5 planner and shared evidence bindings for every fresh entry.
    # 所有新鲜入口共用的 v5 规划器与视觉证据绑定。
    visual_task_planner: VisualTaskPlanner
    visual_bindings: VisualPlanBindings
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
    learned_change_client: LearnedChangeClient | None = None,
    api_key: str | None = None,
    prompts_root: Path | None = None,
) -> RuntimeComponents:
    """Assemble the full runtime. The Qwen client is created exactly once
    here (or injected for tests); the DeepSeek client is created only when an
    api_key is provided — without a key the judge service degrades to
    deterministic-only. 组装完整运行时。Qwen 客户端在此恰好创建一次（或由
    测试注入）；仅在提供 api_key 时创建 DeepSeek 客户端——无 key 时 judge
    服务退化为纯确定性。"""

    catalog = _load_prompt_catalog(project_root, prompts_root)
    service_cache = JsonResponseCache(settings.runs.root / "service" / "cache")
    if qwen_client is None:
        qwen_client = create_model(
            "qwen_transformers",
            settings=settings.models.qwen,
            repair_prompt=catalog["json_repair"],
            cache=service_cache,
        )
    asset_root = _expert_asset_root(project_root)
    expert_catalog = _load_expert_catalog(asset_root)
    evidence_catalog = _load_evidence_catalog(asset_root)
    segformer_clients = _build_segformer_clients(
        settings,
        expert_catalog,
        project_root=asset_root,
    )
    # Verified semantic-mask clients for the VQA evidence service, keyed by
    # stable settings binding; counting-enabled clients are reused so one
    # runtime assembly creates one client per logical model id.
    # 供 VQA evidence 服务使用的已验证 semantic-mask clients，按稳定 settings
    # binding 键控；复用 counting 启用的 client，使一次 runtime assembly 每个
    # 逻辑模型 id 只创建一个 client。
    vqa_segmenter_clients = _build_vqa_segmenter_clients(
        settings,
        expert_catalog,
        segformer_clients,
        evidence_catalog=evidence_catalog,
        project_root=asset_root,
    )
    # One audited YOLO model store per runtime assembly, shared by the
    # counting backends and the feature-flagged visual evidence services
    # (14A3 C9). No weights are loaded by constructing the store itself.
    # 每次 runtime assembly 一个可审计 YOLO 模型 store，计数后端与特性开关式
    # 视觉证据服务共享（14A3 C9）。仅构造 store 本身不加载任何权重。
    model_store = YoloModelStore()
    # Every fresh sample uses the same visual-only planner. The deprecated
    # feature flag is deliberately ignored for new execution.
    # 每条新鲜样本都使用同一个纯视觉规划器；废弃的 feature flag 对新执行无效。
    visual_task_planner, visual_bindings = _build_visual_task_planning(
        settings,
        catalog,
        qwen_client,
        model_store,
        expert_catalog=expert_catalog,
        segmenter_clients=vqa_segmenter_clients,
        evidence_catalog=evidence_catalog,
        project_root=asset_root,
    )
    change_semantic_experts = _build_change_semantic_bindings(
        settings,
        expert_catalog,
        segformer_clients,
        project_root=asset_root,
    )
    if learned_change_client is None:
        learned_change_client = _build_learned_change_client(
            settings=settings,
            project_root=asset_root,
            semantic_experts=change_semantic_experts,
        )
    if settings.agents.change.semantic.enabled and semantic_client is None:
        semantic_client = (
            change_semantic_experts[0].client if change_semantic_experts else None
        )
    if not settings.agents.change.semantic.enabled:
        semantic_client = None
    agent_registry = _build_agent_registry(
        settings,
        catalog,
        qwen_client,
        semantic_client,
        learned_change_client=learned_change_client,
        expert_catalog=expert_catalog,
        segformer_clients=segformer_clients,
        project_root=asset_root,
        model_store=model_store,
        semantic_experts=change_semantic_experts,
    )
    router = TaskRouter()
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
            visual_bindings=visual_bindings,
        )

    def dataset_runner_factory(
        adapter: DatasetAdapter,
        run_dir: Path,
        *,
        judge_policy: str,
        judge_sample_rate: float | None = None,
        data_root: Path,
        planning_mode: str = "visual-task-plan-v5",
        evidence_preprocessing: EvidencePreprocessingIdentity | None = None,
        vqa_assistance_scope: str | None = None,
    ) -> DatasetRunner:
        return DatasetRunner(
            adapter,
            make_sample_runner(data_root),
            run_dir=run_dir,
            artifact_writer=artifact_writer,
            judge_policy=judge_policy,
            judge_sample_rate=judge_sample_rate,
            call_budget_factory=call_budget_factory,
            visual_task_planner=visual_task_planner,
            planning_mode=planning_mode,
            data_root=data_root,
            evidence_preprocessing=evidence_preprocessing,
            vqa_assistance_scope=vqa_assistance_scope,
        )

    components = RuntimeComponents(
        qwen_client=qwen_client,
        judge_client=judge_client,
        prompt_catalog=catalog,
        router=router,
        agent_registry=agent_registry,
        judge_service=judge_service,
        artifact_writer=artifact_writer,
        call_budget_factory=call_budget_factory,
        run_store=run_store,
        build_report=build_report,
        render_overlay=render_counting_overlay,
        dataset_runner_factory=dataset_runner_factory,
        sample_runner_factory=make_sample_runner,
        visual_task_planner=visual_task_planner,
        visual_bindings=visual_bindings,
    )
    return components


def _build_agent_registry(
    settings: AppSettings,
    catalog: PromptCatalog,
    qwen_client: VisionLanguageClient,
    semantic_client: DenseSemanticClient | None = None,
    learned_change_client: LearnedChangeClient | None = None,
    *,
    expert_catalog: ExpertCatalog | None = None,
    segformer_clients: dict[str, Any] | None = None,
    project_root: Path | None = None,
    model_store: YoloModelStore | None = None,
    semantic_experts: tuple[SemanticExpertBinding, ...] = (),
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
        model_store=model_store,
    )
    counting_agent = CountingAgent(
        qwen_client,
        target_resolver=CountTargetResolver(
            evidence_catalog=_load_evidence_catalog(root),
            expert_catalog=expert_catalog,
        ),
        backend_registry=backend_registry,
        default_backend=settings.agents.counting.default_backend,
        fallback_on_backend_unavailable=settings.counting.fallback_on_backend_unavailable,
        fallback_on_backend_error=settings.counting.fallback_on_backend_error,
        verify_empty_detection=settings.counting.verify_empty_detection,
        verify_empty_semantic=settings.counting.verify_empty_semantic,
        trust_empty_detection=settings.counting.trust_empty_detection,
        multi_detector_enabled=settings.counting.multi_detector_enabled,
        max_selected_detector_experts=settings.counting.max_selected_detector_experts,
        min_successful_detector_experts=settings.counting.min_successful_detector_experts,
        ensemble_iou_threshold=settings.counting.ensemble_iou_threshold,
        ensemble_center_distance_ratio=settings.counting.ensemble_center_distance_ratio,
        ensemble_singleton_high_confidence=settings.counting.ensemble_singleton_high_confidence,
        unresolved_ensemble_policy=settings.counting.unresolved_ensemble_policy,
        expert_catalog=expert_catalog,
    )
    change_agent = ChangeAgent(
        qwen_client,
        semantic_client=semantic_client,
        learned_change_client=learned_change_client,
        semantic_experts=semantic_experts,
        prompt=PromptBinding(
            text=catalog["change"], version=catalog.version("change")
        ),
        building_rescue_prompt=PromptBinding(
            text=catalog["change_building_rescue"],
            version=catalog.version("change_building_rescue"),
        ),
        settings=settings.agents.change,
    )
    grounding_agent = GroundingAgent(qwen_client)
    general_vqa_agent = GeneralVQAAgent(qwen_client)
    caption_agent = CaptionAgent(qwen_client)

    registry = AgentRegistry()
    for agent in (counting_agent, change_agent, grounding_agent,
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
    model_store: YoloModelStore | None = None,
) -> BackendRegistry:
    """The Qwen point backend is always registered; YOLO backends only when
    enabled. All enabled detectors share the assembly's single model store.
    Qwen 点式后端恒注册；YOLO 后端仅启用时注册。全部启用检测器共享本次组装
    的单一模型 store。"""

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
            disagreement_prompt=catalog["count_disagreement"],
            disagreement_prompt_version=catalog.version("count_disagreement"),
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
        model_store = model_store or YoloModelStore()
        for detector in sorted(
            settings.backend.yolo.detectors,
            key=lambda item: item.name,
        ):
            if detector.enabled:
                validated_detector = _resolve_yolo_detector(detector, project_root)
                if expert_catalog is not None:
                    validated_detector = _catalog_validated_yolo_detector(
                        validated_detector,
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
        for expert in _enabled_semantic_specs(expert_catalog):
            registry.register(
                SemanticSegmentationCountingBackend(
                    clients[expert.logical_model_id],
                    expert,
                    settings.counting,
                )
            )
    return registry


def _load_prompt_catalog(
    project_root: Path,
    explicit: Path | None,
) -> PromptCatalog:
    """Resolve and validate active prompts without exposing host paths."""

    if explicit is not None:
        try:
            return PromptCatalog(explicit)
        except (OSError, UnicodeError, KeyError):
            raise RuntimeCompositionError(
                "explicit prompt metadata is unavailable"
            ) from None

    package_root = Path(__file__).resolve().parents[1]
    seen: set[Path] = set()
    for candidate in (project_root / "prompts", package_root / "prompts"):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            return PromptCatalog(candidate)
        except (OSError, UnicodeError, KeyError):
            continue
    raise RuntimeCompositionError("prompt metadata is unavailable")


def _resolve_yolo_detector(
    detector: YoloDetectorSettings,
    project_root: Path | None,
) -> YoloDetectorSettings:
    """Canonicalize physical weights once at the composition boundary."""

    weights = detector.weights
    if not weights.is_absolute():
        root = project_root or Path(__file__).resolve().parents[1]
        weights = (root / weights).resolve()
    return detector.model_copy(update={"weights": weights})


def _load_expert_catalog(project_root: Path) -> ExpertCatalog:
    return ExpertCatalog.load(
        project_root / "agents" / "counting" / "expert_catalog.json",
        asset_root=project_root,
    )


def _enabled_semantic_specs(catalog: ExpertCatalog) -> tuple[ExpertSpec, ...]:
    return catalog.experts(
        kinds=frozenset({"semantic_segmentation"}),
        enabled_only=True,
    )


def _build_change_semantic_bindings(
    settings: AppSettings,
    catalog: ExpertCatalog,
    clients: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> tuple[SemanticExpertBinding, ...]:
    """Bind only catalog-verified semantic experts into Change runtime."""

    semantic_settings = settings.agents.change.semantic
    if not semantic_settings.enabled:
        return ()
    candidates = sorted(
        (
            expert
            for expert in _enabled_semantic_specs(catalog)
            if expert.change_semantics is not None
            and expert.change_semantics.enabled
        ),
        key=lambda expert: (-expert.priority, expert.backend_name),
    )
    if not semantic_settings.multi_expert_enabled:
        candidates = candidates[:1]
    else:
        candidates = candidates[: semantic_settings.max_experts]
    bindings: list[SemanticExpertBinding] = []
    for expert in candidates:
        semantic = expert.change_semantics
        assert semantic is not None
        try:
            client = clients[expert.logical_model_id]
        except KeyError:
            raise RuntimeCompositionError(
                "Change semantic expert client is unavailable"
            ) from None
        labels = (
            _verified_class_map(expert, project_root)
            if project_root is not None
            else {}
        )
        bindings.append(
            SemanticExpertBinding(
                expert_id=expert.backend_name,
                logical_model_id=expert.logical_model_id,
                priority=expert.priority,
                participation=semantic.participation,
                role=semantic.role,
                neutral_labels=frozenset(semantic.neutral_model_labels),
                transient_labels=frozenset(semantic.transient_model_labels),
                persistent_labels=frozenset(semantic.persistent_model_labels),
                client=client,
                structural_labels=frozenset(semantic.structural_model_labels),
                landcover_candidate_labels=frozenset(
                    semantic.landcover_candidate_model_labels
                ),

                rescue_model_labels=frozenset(semantic.rescue_model_labels),
                rescue_strategy=semantic.rescue_strategy,
                class_names=tuple(labels.values()),
                class_names_sha256=hash_class_names(tuple(labels.values())),
                weights_sha256=expert.asset.sha256,

            )
        )
    return tuple(bindings)


def _build_learned_change_client(
    *,
    settings: AppSettings,
    project_root: Path,
    semantic_experts: tuple[SemanticExpertBinding, ...],
) -> LearnedChangeClient | None:
    """Auto-assemble a ChangeHead only when the feature is explicitly enabled."""

    learned_settings = settings.agents.change.learned_change
    if not learned_settings.enabled:
        return None
    from models.change_head.checkpoint import (
        ChangeHeadCheckpointError,
        load_change_head_checkpoint,
    )
    from models.change_head.fingerprint import build_change_input_pipeline_fingerprint
    from models.change_head.runtime import (
        ChangeHeadRuntimeError,
        TorchLearnedChangeClient,
        validate_change_head_runtime_compatibility,
    )
    try:
        learned_settings.validate_runtime_configuration()
        checkpoint = load_change_head_checkpoint(
            (learned_settings.checkpoint_dir
             if learned_settings.checkpoint_dir is not None
             and learned_settings.checkpoint_dir.is_absolute()
             else project_root / learned_settings.checkpoint_dir)  # type: ignore[operator]
        )
        identities = {
            binding.expert_id: require_model_cache_identity(
                binding.client, component=f"change semantic expert {binding.expert_id}"
            )
            for binding in semantic_experts
        }
        fingerprint, _ = build_change_input_pipeline_fingerprint(
            settings=settings.agents.change,
            semantic_client_identities=identities,
        )
        validate_change_head_runtime_compatibility(
            manifest=checkpoint.manifest,
            semantic_experts=semantic_experts,
            pipeline_fingerprint=fingerprint,
            strict=learned_settings.strict_contract,
        )
        return TorchLearnedChangeClient(
            checkpoint,
            device=learned_settings.device,
        )
    except (ChangeHeadCheckpointError, ChangeHeadRuntimeError, ValueError) as error:
        code = getattr(error, "code", None) or "LEARNED_CHANGE_INFERENCE_FAILED"
        if learned_settings.mode == "required" or learned_settings.failure_policy == "fail":
            raise RuntimeCompositionError(code) from None
        return None


def _expert_asset_root(project_root: Path) -> Path:
    """Prefer the project root; installed wheels use package resources.
    优先使用项目根；安装后的 wheel 使用包资源。"""

    marker = Path("agents") / "counting" / "expert_catalog.json"
    if (project_root / marker).is_file():
        return project_root
    package_root = Path(__file__).resolve().parents[1]
    if (package_root / marker).is_file():
        return package_root
    raise RuntimeCompositionError("counting expert metadata is unavailable")


def _build_segformer_clients(
    settings: AppSettings,
    catalog: ExpertCatalog,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Create lazy clients from catalog assets, reusing equal logical models."""

    clients: dict[str, Any] = {}
    identities: dict[str, tuple[SegFormerSettings, tuple[tuple[int, str], ...]]] = {}
    for expert in _enabled_semantic_specs(catalog):
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


def _build_vqa_segmenter_clients(
    settings: AppSettings,
    catalog: ExpertCatalog,
    counting_clients: Mapping[str, Any],
    *,
    evidence_catalog: EvidenceCatalog,
    project_root: Path,
) -> dict[str, SemanticMaskClient]:
    """Verified semantic-mask clients for the VQA evidence service, keyed by
    the stable settings binding. The binding is a logical backend name, never
    a checkpoint path: it must resolve to a verified semantic-segmentation
    expert. Counting-enabled clients are reused by logical model id so one
    runtime assembly creates one client per logical id; disabled bindings get
    their own lazy client. Every declared binding fails closed at assembly
    when the catalog or verification does not match.
    供 VQA evidence 服务使用的已验证 semantic-mask clients，按稳定 settings
    binding 键控。binding 是逻辑 backend 名而非 checkpoint 路径：必须解析到
    已验证 semantic-segmentation expert。按逻辑模型 id 复用 counting 启用
    client，使一次 runtime assembly 每个逻辑 id 只创建一个 client；禁用
    binding 各自获得惰性 client。任何声明的 binding 在目录或验证不匹配时于
    组装期严格失败。"""

    clients: dict[str, SemanticMaskClient] = {}
    created: dict[str, SemanticMaskClient] = {}
    for binding in settings.visual_planning.segmenters:
        try:
            expert = catalog.expert(binding)
        except KeyError:
            raise RuntimeCompositionError(
                "visual segmenter binding is absent from the expert catalog"
            ) from None
        if expert.kind != "semantic_segmentation":
            raise RuntimeCompositionError(
                "visual segmenter binding is not a semantic segmentation expert"
            )
        if expert.verification.class_map != "verified":
            raise RuntimeCompositionError(
                "visual segmenter binding has an unverified class map"
            )
        required_labels = _vqa_segformer_labels_for_binding(
            evidence_catalog, binding
        )
        runtime = _segformer_runtime_settings(settings, expert, project_root)
        labels = _verified_class_map(
            expert,
            project_root,
            expected_version=settings.visual_planning.segmenters[binding].class_map_version,
            required_labels=required_labels,
        )
        client = counting_clients.get(expert.logical_model_id)
        if client is None:
            client = created.get(expert.logical_model_id)
            if client is None:
                client = create_model(
                    "segformer_transformers",
                    settings=runtime,
                    id_to_label=labels,
                )
                created[expert.logical_model_id] = client
        clients[binding] = client
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
    catalog_model_dir = _safe_asset(project_root, expert.asset.model_dir)
    class_map = _safe_asset(project_root, expert.asset.class_map)
    weights = _safe_asset(project_root, expert.asset.weights)
    if class_map.parent != catalog_model_dir or weights.parent != catalog_model_dir:
        raise RuntimeCompositionError("SegFormer assets must share the declared model directory")
    configured_model_dir = profile.model_path
    model_dir = (
        catalog_model_dir
        if configured_model_dir == Path(expert.asset.model_dir)
        else _resolve_runtime_model_dir(project_root, configured_model_dir)
    )
    payload = profile.model_dump()
    payload.update(
        {
            "model_path": model_dir,
            "logical_model_id": expert.logical_model_id,
            "weights_filename": weights.name,
            "weights_sha256": expert.asset.sha256,
            "classes_filename": class_map.name,
            "allow_download": False,
        }
    )
    return SegFormerSettings.model_validate(payload)


def _resolve_runtime_model_dir(project_root: Path, configured: Path) -> Path:
    """Resolve a deploy-time path without treating it as model identity.
    解析部署时路径，但不把它当作模型身份。"""

    return configured if configured.is_absolute() else (project_root / configured).resolve()


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


def _verified_class_map(
    expert: ExpertSpec,
    project_root: Path,
    *,
    expected_version: str | None = None,
    required_labels: frozenset[str] = frozenset(),
) -> dict[int, str]:
    """Validate the immutable class-map identity before any client is exposed.
    在任何 client 暴露前校验不可变 class map 身份。

    The catalog digest, the class-map verification metadata, and the VQA raw
    labels must agree. This is intentionally composition-time validation; the
    runtime remains responsible for hashing the actual weight file on load.
    catalog digest、class map verification metadata 与 VQA raw labels 必须一致。
    这是组合期校验；实际权重文件的哈希仍由 runtime 在加载时负责。
    """
    path = _safe_asset(project_root, expert.asset.class_map)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        raw = document["id2name"]
        count = document["num_classes"]
        inverse = document["name2id"]
        verification = document["verification"]
        verified_date = verification["verified_date"]
        checkpoint_sha256 = verification["checkpoint_sha256"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        raise RuntimeCompositionError("SegFormer class map is missing or invalid") from None
    if (
        not isinstance(verified_date, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", verified_date) is None
    ):
        raise RuntimeCompositionError("SegFormer class map verification version is invalid")
    actual_version = f"verified-{verified_date}"
    if expected_version is not None and expected_version != actual_version:
        raise RuntimeCompositionError("SegFormer class map version differs from settings")
    if (
        not isinstance(checkpoint_sha256, str)
        or checkpoint_sha256 != expert.asset.sha256
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None
    ):
        raise RuntimeCompositionError("SegFormer class map checkpoint digest differs from asset")
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
    if not required_labels.issubset(set(labels.values())):
        raise RuntimeCompositionError("SegFormer VQA raw labels differ from verified class map")
    declared = {
        label
        for support in expert.supports.values()
        for label in support.model_labels
    }
    if not declared.issubset(set(labels.values())):
        raise RuntimeCompositionError("SegFormer catalog labels differ from verified class map")
    return labels


def _vqa_segformer_labels_for_binding(
    catalog: EvidenceCatalog,
    binding: str,
) -> frozenset[str]:
    """Collect every VQA raw label owned by one stable segmenter binding.
    收集一个稳定 segmenter binding 所拥有的全部 VQA 原始标签。"""

    labels: set[str] = set()
    for leaf in catalog.executable_leaves_for_task("general_vqa"):
        if (
            catalog.capability_enabled(leaf, "segformer")
            and catalog.leaf_segformer_binding(leaf) == binding
        ):
            labels.update(catalog.leaf_segformer_labels(leaf) or ())
    return frozenset(labels)


def _catalog_validated_yolo_detector(detector: Any, catalog: ExpertCatalog) -> Any:
    expected_kinds = {
        "obb": "yolo_obb",
        "detect": "yolo_detect",
    }
    try:
        expected_kind = expected_kinds[detector.task]
    except (AttributeError, KeyError):
        raise RuntimeCompositionError("unsupported YOLO detector task") from None
    try:
        expert = catalog.expert(detector.name)
    except KeyError:
        raise RuntimeCompositionError(
            "enabled YOLO detector is absent from expert catalog"
        ) from None
    if expert.kind != expected_kind or not expert.enabled:
        raise RuntimeCompositionError("YOLO detector catalog declaration is not enabled")
    if expert.logical_model_id != detector.model_id:
        raise RuntimeCompositionError("YOLO logical model id differs from expert catalog")
    if expert.asset.sha256 != detector.sha256:
        raise RuntimeCompositionError("YOLO weight digest differs from expert catalog")
    if expert.priority != detector.priority:
        raise RuntimeCompositionError("YOLO priority differs from expert catalog")
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
    return detector


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


def _build_visual_task_planning(
    settings: AppSettings,
    catalog: PromptCatalog,
    qwen_client: VisionLanguageClient,
    model_store: YoloModelStore,
    *,
    expert_catalog: ExpertCatalog,
    segmenter_clients: Mapping[str, SemanticMaskClient],
    evidence_catalog: EvidenceCatalog,
    project_root: Path,
) -> tuple[VisualTaskPlanner, VisualPlanBindings]:
    """Assemble the always-on v5 planner and shared evidence bindings.
    组装始终启用的 v5 规划器与共享证据绑定。"""
    planner_settings = settings.visual_planning.planner
    if planner_settings.catalog_version != evidence_catalog.catalog_version:
        raise RuntimeCompositionError(
            "visual task planning catalog version differs from the evidence catalog asset"
        )
    if planner_settings.task_prompt_version != catalog.version("visual_task_plan"):
        raise RuntimeCompositionError(
            "visual task planning prompt version differs from the prompt catalog"
        )
    bindings = _build_visual_bindings(
        settings,
        catalog,
        evidence_catalog,
        qwen_client,
        model_store,
        segmenter_clients=segmenter_clients,
        project_root=project_root,
    )
    # Runtime availability is task-specific. Counting specialists remain
    # independent from VQA/Grounding evidence service availability.
    # 运行时能力按 task 分开；counting 专家不受 VQA/Grounding 证据服务开关影响。
    counting_leaves = _enabled_counting_catalog_leaves(
        settings, evidence_catalog, expert_catalog, task="counting"
    )
    fine_grained_counting_leaves = _enabled_counting_catalog_leaves(
        settings,
        evidence_catalog,
        expert_catalog,
        task="fine_grained_counting",
    )
    # The four GeneralVQAAgent tasks share one immutable runtime capability
    # set, computed once from the same availability filter. When the VQA
    # evidence service is unavailable the filter already yields empty tuples
    # for all four tasks. 四个 GeneralVQAAgent task 共享同一份不可变运行时能力
    # 集合，由同一个可用性过滤器计算一次。VQA evidence 服务不可用时该过滤器
    # 对四个 task 都产出空元组。
    vqa_leaves = _vqa_executable_leaves(
        settings,
        evidence_catalog,
        segmenter_clients,
        project_root=project_root,
    )
    executable_categories_by_task = {
        "counting": counting_leaves,
        "fine_grained_counting": fine_grained_counting_leaves,
        **{task: vqa_leaves for task in GENERAL_VQA_AGENT_TASKS},
        "grounding": (
            evidence_catalog.executable_leaves_for_task("grounding")
            if bindings.grounding_evidence is not None
            else ()
        ),
    }
    planner = VisualTaskPlanner(
        qwen_client,
        system_prompt=catalog["visual_task_plan"],
        prompt_version=planner_settings.task_prompt_version,
        catalog=evidence_catalog,
        executable_categories_by_task=executable_categories_by_task,
        max_side=planner_settings.preview_max_side,
        roi_quantum=planner_settings.roi_quantum,
        roi_coordinate_frame=planner_settings.roi_coordinate_frame,
        roi_materialization_policy=planner_settings.roi_materialization_policy,
        large_image_policy=planner_settings.large_image_policy,
        vqa_assistance_scope=VQA_ASSISTANCE_SCOPE,
    )
    return planner, bindings


def _enabled_counting_catalog_leaves(
    settings: AppSettings,
    catalog: EvidenceCatalog,
    expert_catalog: ExpertCatalog,
    *,
    task: str,
) -> tuple[str, ...]:
    """Return verified leaves backed by an enabled counting specialist.
    返回当前已启用 counting specialist 能支撑的已验证叶子。"""

    if task not in COUNTING_TASKS:
        raise RuntimeCompositionError(
            f"unsupported counting planner task: {task}"
        )

    enabled_detectors = tuple(
        detector
        for detector in settings.backend.yolo.detectors
        if settings.backend.yolo.enabled and detector.enabled
    )
    yolo_specs = expert_catalog.experts(
        kinds=frozenset({"yolo_obb", "yolo_detect"}), enabled_only=True
    )
    semantic_specs = expert_catalog.experts(
        kinds=frozenset({"semantic_segmentation"}), enabled_only=True
    )
    enabled: list[str] = []
    for leaf in catalog.executable_leaves_for_task(task):
        yolo_ready = catalog.capability_enabled(leaf, "yolo") and any(
            detector.model_id == expert.logical_model_id
            and leaf in expert.supports
            and expert.supports[leaf].counting_mode == "native_detection"
            and bool(
                {label.casefold() for label in expert.supports[leaf].model_labels}
                & {label.casefold() for label in detector.classes}
            )
            for detector in enabled_detectors
            for expert in yolo_specs
        )
        semantic_ready = catalog.capability_enabled(leaf, "segformer") and any(
            leaf in expert.supports
            and expert.supports[leaf].counting_mode == "connected_components"
            and expert.verification.class_map == "verified"
            for expert in semantic_specs
        )
        if yolo_ready or semantic_ready:
            enabled.append(leaf)
    return tuple(enabled)


def _build_visual_bindings(
    settings: AppSettings,
    catalog: PromptCatalog,
    evidence_catalog: EvidenceCatalog,
    qwen_client: VisionLanguageClient,
    model_store: YoloModelStore,
    *,
    segmenter_clients: Mapping[str, SemanticMaskClient],
    project_root: Path,
) -> VisualPlanBindings:
    """Shared evidence bindings for the canonical visual planner: the VQA object
    evidence executor and the grounding evidence executor, assembled from the
    frozen settings policies. 规范视觉规划器共享的证据绑定：VQA object-evidence
    执行器与 grounding 证据执行器，由冻结的 settings 策略组装。"""

    return VisualPlanBindings(
        vqa_evidence=_build_vqa_evidence_service(
            settings,
            evidence_catalog,
            model_store,
            segmenter_clients=segmenter_clients,
            project_root=project_root,
        ),
        grounding_evidence=_build_grounding_evidence_service(
            settings,
            catalog,
            evidence_catalog,
            qwen_client,
            model_store,
            project_root=project_root,
        ),
    )


def _load_evidence_catalog(project_root: Path) -> EvidenceCatalog:
    """Load the shared closed evidence catalog asset; failures stay stable
    and never expose host paths. 加载共享封闭证据目录资产；失败保持稳定且不
    暴露主机路径。"""
    try:
        return EvidenceCatalog.from_file(
            project_root / "agents" / "evidence_catalog.json"
        )
    except (CatalogCategoryError, OSError) as exc:
        raise RuntimeCompositionError(
            "visual evidence catalog is unavailable"
        ) from exc


class _LazyObjectDetectionClient:
    """Defer YOLO weight loading to the first inference so composition never
    touches weights (AGENTS.md lazy-loading contract). The shared store
    validates digest/task/class map on first use; load failures surface at
    runtime inside the executor's per-ROI failure seam as stable type names,
    never as host paths. 将 YOLO 权重加载推迟到首次推理，使组合期绝不触碰权
    重（AGENTS.md 惰性加载契约）。共享 store 在首次使用时校验摘要/任务/类别
    映射；加载失败在运行时执行器逐 ROI 失败 seam 内以稳定类型名呈现，绝不携
    带主机路径。"""

    def __init__(
        self,
        model_store: YoloModelStore,
        detector: YoloDetectorSettings,
    ) -> None:
        self._model_store = model_store
        self._detector = detector
        self._delegate: RuntimeObjectDetectionClient | None = None
        self._lock = threading.Lock()

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        """Logical identity without loading weights. 无需加载权重即可得到的
        逻辑身份。"""
        return ModelCacheIdentity(
            model=self._detector.model_id,
            generation={"weights_sha256": self._detector.sha256},
            client_version="yolo-detection-runtime-v1",
        )

    def detect(
        self,
        image: Any,
        *,
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
        max_detections: int,
    ) -> list[ObjectDetectionOutput]:
        if self._delegate is None:
            with self._lock:
                if self._delegate is None:
                    self._delegate = RuntimeObjectDetectionClient(
                        self._model_store.get(self._detector),
                        logical_model_id=self._detector.model_id,
                        weights_sha256=self._detector.sha256,
                    )
        return self._delegate.detect(
            image,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
            device=device,
            max_detections=max_detections,
        )


def _build_vqa_evidence_service(
    settings: AppSettings,
    evidence_catalog: EvidenceCatalog,
    model_store: YoloModelStore,
    *,
    segmenter_clients: Mapping[str, SemanticMaskClient],
    project_root: Path,
) -> ObjectEvidenceExecutor | None:
    """Assemble the VQA object-evidence executor from the frozen capability
    settings. Three modes, each fail-closed:

    - no enabled capability            -> None (plans fail at runtime);
    - detector only                    -> YOLO-only executor;
    - segmenter(s) only                -> SegFormer-only executor;
    - detector + segmenter(s)          -> combined executor.

    The detector policy is assembled only when fully calibrated, and a
    calibrated policy requires an enabled detector; an enabled segmenter
    binding must resolve to a verified client from the runtime inventory —
    no production default is ever invented.
    按冻结的能力设置组装 VQA object-evidence 执行器，三种模式均严格失败：

    - 无启用能力 -> None（计划在运行时失败）；
    - 仅检测器 -> YOLO-only 执行器；
    - 仅分割器 -> SegFormer-only 执行器；
    - 检测器+分割器 -> 组合执行器。

    检测器策略仅在完整校准时组装，且已校准策略必须有启用检测器；启用的
    segmenter binding 必须解析到运行时清单中的已验证 client——绝不杜撰
    生产默认值。"""

    policy = _resolved_evidence_policy(settings.visual_planning.detectors)
    yolo_client = None
    yolo_device = None
    yolo_image_size = None
    if policy is not None:
        detector = _first_enabled_detector(settings, project_root)
        if detector is None:
            raise RuntimeCompositionError(
                "calibrated VQA detector policy requires an enabled detector"
            )
        # The detector is wired lazily: composition never loads weights; the
        # first inference goes through the shared audited store.
        # 检测器惰性接线：组合期绝不加载权重；首次推理经共享审计 store。
        yolo_client = _LazyObjectDetectionClient(model_store, detector)
        yolo_device = detector.device
        yolo_image_size = detector.image_size
    enabled_segmenters = {
        binding: entry
        for binding, entry in settings.visual_planning.segmenters.items()
        if entry.enabled
    }
    unresolved = sorted(
        binding for binding in enabled_segmenters if binding not in segmenter_clients
    )
    if unresolved:
        raise RuntimeCompositionError(
            "enabled visual segmenter binding has no verified runtime client: "
            + ", ".join(unresolved)
        )
    if yolo_client is None and not enabled_segmenters:
        return None
    return ObjectEvidenceExecutor(
        catalog=evidence_catalog,
        policy=(
            None
            if policy is None
            else EvidencePolicy(
                confidence_threshold=policy.confidence_threshold,
                nms_iou_threshold=policy.nms_iou_threshold,
                max_detections=policy.max_detections,
            )
        ),
        yolo_client=yolo_client,
        yolo_device=yolo_device,
        yolo_image_size=yolo_image_size,
        segmenter_clients={
            binding: segmenter_clients[binding] for binding in enabled_segmenters
        },
        preprocessing=_evidence_preprocessing(settings.visual_planning.preprocessing),
    )


def _build_grounding_evidence_service(
    settings: AppSettings,
    catalog: PromptCatalog,
    evidence_catalog: EvidenceCatalog,
    qwen_client: VisionLanguageClient,
    model_store: YoloModelStore,
    *,
    project_root: Path,
) -> GroundingEvidenceExecutor:
    """Assemble the grounding evidence seam (14A3 C9). An all-None policy is
    the explicit uncalibrated state (YOLO phase off, final Qwen free boxes);
    a fully calibrated policy requires an enabled detector, otherwise the
    assembly fails closed. 组装 grounding 证据 seam（14A3 C9）。全 None 策略是
    显式未校准状态（YOLO 阶段关闭，最终 Qwen 自由补框）；完整校准策略必须
    有启用检测器，否则组装严格失败。"""

    policy = _resolved_evidence_policy(settings.visual_planning.detectors)
    yolo_client = None
    yolo_device = None
    yolo_image_size = None
    # The executor policy defaults to the explicit all-None uncalibrated
    # state; a calibrated policy supplies the frozen values instead and
    # requires an enabled detector — an uncalibrated policy never wires a
    # detector it will not use, and the wiring stays lazy so composition
    # never loads weights. 执行器策略默认显式全 None 未校准状态；已校准策略
    # 改提供冻结值并要求启用检测器——未校准策略绝不接线用不到的检测器，接线
    # 保持惰性使组合期绝不加载权重。
    executor_policy = GroundingEvidencePolicy()
    if policy is not None:
        executor_policy = GroundingEvidencePolicy(
            confidence_threshold=policy.confidence_threshold,
            nms_iou_threshold=policy.nms_iou_threshold,
            max_detections=policy.max_detections,
        )
        detector = _first_enabled_detector(settings, project_root)
        if detector is None:
            raise RuntimeCompositionError(
                "calibrated grounding detector policy requires an enabled detector"
            )
        yolo_client = _LazyObjectDetectionClient(model_store, detector)
        yolo_device = detector.device
        yolo_image_size = detector.image_size
    try:
        return GroundingEvidenceExecutor(
            catalog=evidence_catalog,
            qwen_client=qwen_client,
            prompt=PromptBinding(
                text=catalog["grounding"], version=catalog.version("grounding")
            ),
            policy=executor_policy,
            yolo_client=yolo_client,
            yolo_device=yolo_device,
            yolo_image_size=yolo_image_size,
        )
    except ValueError as exc:
        raise RuntimeCompositionError(str(exc)) from exc


def _resolved_evidence_policy(
    detectors: dict[str, VisualDetectorSettings],
) -> VisualDetectorSettings | None:
    """Resolve the single global detector policy from the per-label settings:
    zero calibrated entries mean uncalibrated (None), exactly one fully
    calibrated entry is the global policy, and any partial or multiple
    calibration is ambiguous and fails closed. 从逐标签设置解析单一全局检测
    策略：零条已校准条目表示未校准（None），恰好一条完整校准条目即全局策略，
    部分或多条校准即歧义并严格失败。"""

    calibrated = [
        entry
        for entry in detectors.values()
        if entry.confidence_threshold is not None
        or entry.nms_iou_threshold is not None
        or entry.max_detections is not None
    ]
    if not calibrated:
        return None
    if len(calibrated) > 1:
        raise RuntimeCompositionError(
            "multiple calibrated detector policies cannot form one global policy"
        )
    entry = calibrated[0]
    if (
        entry.confidence_threshold is None
        or entry.nms_iou_threshold is None
        or entry.max_detections is None
    ):
        raise RuntimeCompositionError(
            "partially calibrated detector policy is not frozen"
        )
    return entry


def _first_enabled_detector(
    settings: AppSettings,
    project_root: Path,
) -> YoloDetectorSettings | None:
    """Deterministic single detector for the evidence services: the first
    enabled detector in stable name order, resolved against the project root.
    证据服务的确定性单检测器：按稳定名称顺序的第一个启用检测器，相对项目根
    解析。"""

    enabled = sorted(
        (detector for detector in settings.backend.yolo.detectors if detector.enabled),
        key=lambda item: item.name,
    )
    if not enabled:
        return None
    return _resolve_yolo_detector(enabled[0], project_root)


def _evidence_preprocessing(
    pre: VisualEvidencePreprocessSettings,
) -> EvidencePreprocessing:
    """Mirror the frozen settings identity into the agents-local contract;
    agents never import application settings, so every value is injected
    explicitly and verified by the shared Literal versions.
    将冻结的 settings 身份镜像进 agents 局部契约；agents 绝不导入 application
    settings，因此每个值都显式注入，并由共享 Literal 版本互相校验。"""

    return EvidencePreprocessing(
        version=pre.version,
        tile_size=pre.tile_size,
        partition_policy=pre.partition_policy,
        remainder_resize=pre.remainder_resize,
        rgb_interpolation=pre.rgb_interpolation,
        mask_inverse_interpolation=pre.mask_inverse_interpolation,
        max_tile_concurrency=pre.max_tile_concurrency,
    )


def _vqa_executable_leaves(
    settings: AppSettings,
    catalog: EvidenceCatalog,
    segmenter_clients: Mapping[str, SemanticMaskClient],
    *,
    project_root: Path,
) -> tuple[str, ...]:
    """Deterministic runtime-availability filter over the catalog's
    general_vqa leaves, keeping catalog order. A leaf is executable when a
    calibrated detector's classes intersect its YOLO labels, or its frozen
    segmenter binding is enabled with a verified runtime client. A non-None
    service alone does not imply that all 26 categories are executable.
    对目录 general_vqa 叶子做确定性的运行时能力过滤，保持目录顺序。叶子在
    已校准检测器类别与其 YOLO 标签相交、或其冻结 segmenter binding 已启用且
    存在已验证运行时 client 时才算可执行。服务非 None 本身不代表 26 类全部
    可执行。"""

    yolo_ready = _resolved_evidence_policy(settings.visual_planning.detectors) is not None
    detector = _first_enabled_detector(settings, project_root) if yolo_ready else None
    detector_labels = (
        {label.casefold() for label in detector.classes} if detector is not None else set()
    )
    enabled_segmenters = {
        binding for binding, entry in settings.visual_planning.segmenters.items() if entry.enabled
    }
    executable: list[str] = []
    for leaf in catalog.executable_leaves_for_task("general_vqa"):
        leaf_yolo_ready = (
            yolo_ready
            and detector is not None
            and catalog.capability_enabled(leaf, "yolo")
            and bool(
                detector_labels
                & {label.casefold() for label in catalog.leaf_yolo_labels(leaf)}
            )
        )
        binding = catalog.leaf_segformer_binding(leaf)
        leaf_segformer_ready = (
            binding is not None
            and binding in enabled_segmenters
            and binding in segmenter_clients
            and catalog.capability_enabled(leaf, "segformer")
        )
        if leaf_yolo_ready or leaf_segformer_ready:
            executable.append(leaf)
    return tuple(executable)


def _routable_tasks() -> set[str]:
    """All tasks the deterministic router can dispatch. 确定性路由器可分发的
    全部任务。"""

    from routing.policies import POLICIES

    return set(POLICIES)
