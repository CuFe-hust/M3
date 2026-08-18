"""Composition root: the only place concrete model clients are created.

组合根：唯一创建具体模型客户端的地方。任何 workflows / agents /
evaluation 不得自行 create model。Qwen 客户端在一次组装中只创建一次；
DeepSeek 客户端仅在注入 api_key 时创建（无 key 即 judge 禁用，回退纯
确定性）。导入本模块绝无副作用：不加载权重、不调用模型。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.base import CallBudget as _CallBudgetProtocol
from agents.base import VisualPlanBindings
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
from agents.counting.settings import CountingTargetStrategy, YoloDetectorSettings
from agents.counting.target_parser import CountTargetResolver
from agents.evidence_catalog import CatalogCategoryError, EvidenceCatalog
from agents.general_vqa import GeneralVQAAgent
from agents.general_vqa.evidence.executor import (
    EvidencePolicy,
    ObjectEvidenceExecutor,
)
from agents.grounding import GroundingAgent
from agents.grounding.evidence import (
    GroundingEvidenceExecutor,
    GroundingEvidencePolicy,
)
from agents.registry import AgentRegistry
from agents.schema import COUNTING_TASKS
from agents.visual_base import PromptBinding
from application.prompts import PromptCatalog
from application.settings import AppSettings, VisualDetectorSettings
from data.adapters.base import DatasetAdapter
from data.registry import build_default_registry
from evaluation.judges.deepseek import DeepSeekJudgeClient
from models.base import (
    DenseSemanticClient,
    LearnedChangeClient,
    ModelCacheIdentity,
    ObjectDetectionOutput,
    RuntimeObjectDetectionClient,
    VisionLanguageClient,
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
    segformer_clients = _build_segformer_clients(
        settings,
        expert_catalog,
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
        learned_change_client=learned_change_client,
        expert_catalog=expert_catalog,
        segformer_clients=segformer_clients,
        project_root=asset_root,
        model_store=model_store,
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
        expert_catalog=expert_catalog,
    )
    change_agent = ChangeAgent(
        qwen_client,
        semantic_client=semantic_client,
        learned_change_client=learned_change_client,
        prompt=PromptBinding(
            text=catalog["change"], version=catalog.version("change")
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
    project_root: Path,
) -> tuple[VisualTaskPlanner, VisualPlanBindings]:
    """Assemble the always-on v5 planner and shared evidence bindings.
    组装始终启用的 v5 规划器与共享证据绑定。"""
    evidence_catalog = _load_evidence_catalog(project_root)
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
    executable_categories_by_task = {
        "counting": counting_leaves,
        "fine_grained_counting": fine_grained_counting_leaves,
        "general_vqa": (
            evidence_catalog.executable_leaves_for_task("general_vqa")
            if bindings.vqa_evidence is not None
            else ()
        ),
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
        kinds=frozenset({"yolo_obb"}), enabled_only=True
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
    project_root: Path,
) -> VisualPlanBindings:
    """Shared evidence bindings for the canonical visual planner: the VQA object
    evidence executor and the grounding evidence executor, assembled from the
    frozen settings policies. 规范视觉规划器共享的证据绑定：VQA object-evidence
    执行器与 grounding 证据执行器，由冻结的 settings 策略组装。"""

    return VisualPlanBindings(
        vqa_evidence=_build_vqa_evidence_service(
            settings, evidence_catalog, model_store, project_root=project_root
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
    project_root: Path,
) -> ObjectEvidenceExecutor | None:
    """Assemble the VQA object-evidence executor only when the settings
    declare a fully calibrated global detector policy; otherwise the service
    stays absent and object_evidence_vqa plans fail closed at runtime. An
    enabled segmenter without a frozen model binding fails closed at
    assembly — no production default is ever invented.
    仅在 settings 声明完整校准的全局检测策略时组装 VQA object-evidence
    执行器；否则服务保持缺失，object_evidence_vqa 计划在运行时严格失败。
    已启用分割器但没有冻结的模型绑定时在组装时严格失败——绝不杜撰生产默认值。"""

    if any(
        entry.enabled for entry in settings.visual_planning.segmenters.values()
    ):
        raise RuntimeCompositionError(
            "visual segmenter calibration is declared but its model binding is not frozen"
        )
    policy = _resolved_evidence_policy(settings.visual_planning.detectors)
    if policy is None:
        return None
    detector = _first_enabled_detector(settings, project_root)
    if detector is None:
        raise RuntimeCompositionError(
            "calibrated VQA detector policy requires an enabled detector"
        )
    # The detector is wired lazily: composition never loads weights; the
    # first inference goes through the shared audited store.
    # 检测器惰性接线：组合期绝不加载权重；首次推理经共享审计 store。
    yolo_client = _LazyObjectDetectionClient(model_store, detector)
    return ObjectEvidenceExecutor(
        catalog=evidence_catalog,
        policy=EvidencePolicy(
            confidence_threshold=policy.confidence_threshold,
            nms_iou_threshold=policy.nms_iou_threshold,
            max_detections=policy.max_detections,
        ),
        yolo_client=yolo_client,
        yolo_device=detector.device,
        yolo_image_size=detector.image_size,
        segformer_client=None,
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


def _routable_tasks() -> set[str]:
    """All tasks the deterministic router can dispatch. 确定性路由器可分发的
    全部任务。"""

    from routing.policies import POLICIES

    return set(POLICIES)
