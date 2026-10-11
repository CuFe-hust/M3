"""Contract tests for the composition root: single Qwen creation, judge
disabled without a key, route coverage, and side-effect-free imports.

组合根契约测试：Qwen 只创建一次、无 key 时 judge 禁用、路由覆盖与无副作用
导入。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.change.settings import ChangeSemanticSettings
from agents.counting.expert_catalog import ExpertCatalog, ExpertCatalogError
from agents.counting.schema import CountTargetSpec
from agents.counting.settings import YoloDetectorSettings
from agents.evidence_catalog import EvidenceCatalog
import application.bootstrap as bootstrap_module
from application.bootstrap import (
    RuntimeCompositionError,
    _build_backend_registry,
    _build_change_semantic_bindings,
    _build_segformer_clients,
    _catalog_validated_yolo_detector,
    _enabled_counting_catalog_leaves,
    _resolve_yolo_detector,
    assemble_runtime,
)
from application.prompts import PromptCatalog
from application.settings import AppSettings, load_settings
from models.base import ModelCacheIdentity
from models.settings import (
    ModelSettings,
    QwenAdapterBindings,
    QwenAdapterSettings,
)


class _FakeQwenClient:
    """Protocol-compatible fake with a stable cache identity.
    带稳定缓存身份的协议兼容 fake。"""

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake",
            generation={"temperature": 0.0},
            client_version="1",
        )

    async def complete_json(self, **kwargs: Any) -> Any:
        raise AssertionError("bootstrap must not call the model")


class _FakeBoundEngine:
    """Path-free fake engine proving composition binding selection.
    不含路径的 fake engine，用于证明组合绑定选择。"""

    def __init__(self, settings: ModelSettings) -> None:
        self.clients = {
            name: _NamedFakeQwenClient(name)
            for name in ("base", *settings.qwen_adapters)
        }
        self.bind_calls: list[str] = []
        self.runtime_identity = {
            "base_model_id": settings.qwen.effective_cache_model_id,
            "base_revision": settings.qwen.revision,
            "client_version": "fake-multi-v1",
            "adapters": {
                name: {
                    "logical_id": adapter.logical_id,
                    "revision": adapter.revision,
                    "peft_version": "test",
                }
                for name, adapter in settings.qwen_adapters.items()
                if adapter.enabled
            },
        }

    def bind(self, name: str) -> "_NamedFakeQwenClient":
        self.bind_calls.append(name)
        return self.clients[name]


class _NamedFakeQwenClient(_FakeQwenClient):
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-base",
            generation={"adapter": self.name},
            client_version="fake-multi-v1",
        )


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "agents" / "counting" / "expert_catalog.json"


def test_bootstrap_does_not_access_catalog_private_storage() -> None:
    source = (REPO_ROOT / "application" / "bootstrap.py").read_text(encoding="utf-8")

    assert "catalog._experts" not in source
    assert "getattr(catalog" not in source


def test_change_bindings_are_catalog_driven_and_deterministic() -> None:
    settings = AppSettings()
    catalog = ExpertCatalog.load(CATALOG_PATH)
    client = object()

    bindings = _build_change_semantic_bindings(
        settings,
        catalog,
        {
            "SegFormer-MiT-B2:iSAID:local": client,
            "SegFormer-MiT-B2:OpenEarthMap:local": client,
        },
    )

    assert [binding.expert_id for binding in bindings] == [
        "segmenter_mitb2_001",
        "segmenter_oem_001",
    ]
    assert bindings[0].client is client
    assert bindings[0].persistent_labels == frozenset(
        {"storage_tank", "Swimming_pool", "Harbor", "tennis_court", "Ground_Track_Field", "Soccer_ball_field", "baseball_diamond", "Bridge", "basketball_court", "Roundabout"}
    )
    assert bindings[1].role == "persistent_landcover"
    assert bindings[1].participation == "rescue"
    assert bindings[1].neutral_labels == frozenset({"background"})
    assert bindings[1].persistent_labels == frozenset({"building"})
    assert bindings[1].rescue_model_labels == frozenset({"building"})
    assert bindings[1].rescue_strategy == "building_footprint_delta"


def _settings(tmp_path: Path) -> AppSettings:
    from application.settings import RunSettings

    return AppSettings(runs=RunSettings(root=tmp_path / "runs"))


def _assemble(tmp_path: Path, **kwargs: Any) -> Any:
    return assemble_runtime(
        _settings(tmp_path),
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        **kwargs,
    )


def test_assemble_runtime_with_injected_qwen(tmp_path: Path) -> None:
    components = _assemble(tmp_path, qwen_client=_FakeQwenClient())
    assert components.qwen_client is not None
    assert len(components.agent_registry) == 5
    assert components.agent_registry.names() == (
        "counting_agent",
        "change_agent",
        "grounding_agent",
        "general_vqa_agent",
        "caption_agent",
    )
    assert components.prompt_catalog is not None
    assert components.visual_task_planner is not None
    assert components.judge_service is not None
    assert components.dataset_runner_factory is not None
    change_agent = components.agent_registry.get("change_agent")
    runtime_prompt = getattr(change_agent, "_prompt")
    assert runtime_prompt.text == components.prompt_catalog["change"]
    assert runtime_prompt.version == components.prompt_catalog.version("change")
    assert components.prompt_catalog.asset("change").path in (
        components.prompt_catalog.snapshot_paths()
    )


def test_runtime_wires_planner_agents_and_nested_services_by_binding(
    tmp_path: Path,
) -> None:
    digest_a, digest_b = "a" * 64, "b" * 64
    settings = _settings(tmp_path)
    models = settings.models.model_copy(
        update={
            "qwen_adapters": {
                "adapter-a": QwenAdapterSettings(
                    path=Path("unused/a"),
                    logical_id="adapter-a-v1",
                    revision=digest_a,
                ),
                "adapter-b": QwenAdapterSettings(
                    path=Path("unused/b"),
                    logical_id="adapter-b-v1",
                    revision=digest_b,
                ),
            },
            "qwen_adapter_bindings": QwenAdapterBindings(
                planner="adapter-a",
                counting="adapter-a",
                change="adapter-b",
                grounding="adapter-b",
                general_vqa="adapter-a",
                caption="adapter-b",
            ),
        }
    )
    settings = settings.model_copy(update={"models": models})
    engine = _FakeBoundEngine(models)

    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=engine,  # type: ignore[arg-type]
    )

    assert engine.bind_calls == [
        "adapter-a",
        "adapter-a",
        "adapter-b",
        "adapter-b",
        "adapter-a",
        "adapter-b",
    ]
    assert getattr(components.visual_task_planner, "_client").name == "adapter-a"
    expected = {
        "counting_agent": "adapter-a",
        "change_agent": "adapter-b",
        "grounding_agent": "adapter-b",
        "general_vqa_agent": "adapter-a",
        "caption_agent": "adapter-b",
    }
    for agent_name, binding in expected.items():
        assert getattr(components.agent_registry.get(agent_name), "_client").name == binding
        assert components.qwen_clients[agent_name].name == binding
    counting = components.agent_registry.get("counting_agent")
    backend_registry = getattr(getattr(counting, "_selector"), "_registry")
    assert getattr(backend_registry.get("qwen_point"), "_client").name == "adapter-a"
    assert getattr(backend_registry.get("quantity_proposal"), "_client").name == (
        "adapter-a"
    )
    grounding_evidence = components.visual_bindings.grounding_evidence
    assert grounding_evidence is not None
    assert getattr(grounding_evidence, "_qwen_client").name == "adapter-b"
    assert components.qwen_runtime_identity["bindings"]["change"] == {
        "catalog_name": "adapter-b",
        "logical_id": "adapter-b-v1",
        "revision": digest_b,
    }


def test_runtime_uses_project_prompt_root_when_present(tmp_path: Path) -> None:
    components = assemble_runtime(
        _settings(tmp_path),
        project_root=REPO_ROOT,
        qwen_client=_FakeQwenClient(),
    )

    assert components.prompt_catalog.asset("count_tile").path.parent == (
        REPO_ROOT / "prompts"
    )


def test_runtime_uses_packaged_prompt_root_when_project_prompts_missing(
    tmp_path: Path,
) -> None:
    components = assemble_runtime(
        _settings(tmp_path),
        project_root=tmp_path / "arbitrary-cwd",
        qwen_client=_FakeQwenClient(),
    )

    assert components.prompt_catalog.asset("count_tile").path.parent == (
        REPO_ROOT / "prompts"
    )


def test_explicit_invalid_prompt_root_fails_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "private" / "missing-prompts"

    with pytest.raises(
        RuntimeCompositionError,
        match="explicit prompt metadata is unavailable",
    ) as raised:
        assemble_runtime(
            _settings(tmp_path),
            project_root=REPO_ROOT,
            prompts_root=invalid,
            qwen_client=_FakeQwenClient(),
        )

    assert str(invalid) not in str(raised.value)


def test_bootstrap_registers_quantity_proposal_backend(tmp_path: Path) -> None:
    registry = _build_backend_registry(
        _settings(tmp_path),
        PromptCatalog(REPO_ROOT / "prompts"),
        _FakeQwenClient(),
    )
    assert registry.all_names() == [
        "qwen_point",
        "quantity_proposal",
    ]
    assert registry.get("quantity_proposal").kind == "quantity_proposal"


def test_enabled_segformer_experts_are_registered_lazily(tmp_path: Path) -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)
    registry = _build_backend_registry(
        _settings(tmp_path),
        PromptCatalog(REPO_ROOT / "prompts"),
        _FakeQwenClient(),
        expert_catalog=catalog,
        project_root=REPO_ROOT,
    )

    assert registry.all_names() == [
        "qwen_point",
        "quantity_proposal",
        "segmenter_mitb2_001",
        "segmenter_oem_001",
    ]
    backend = registry.get("segmenter_mitb2_001")
    assert backend.kind == "semantic_segmentation"
    assert getattr(backend, "_client").loaded is False
    oem_backend = registry.get("segmenter_oem_001")
    assert oem_backend.kind == "semantic_segmentation"
    assert getattr(oem_backend, "_client").loaded is False


def test_segformer_assembly_uses_verified_map_and_never_predicts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    client = object()

    def fake_create_model(name: str, **kwargs: Any) -> object:
        calls.append((name, kwargs))
        return client

    monkeypatch.setattr("application.bootstrap.create_model", fake_create_model)
    clients = _build_segformer_clients(
        _settings(tmp_path),
        ExpertCatalog.load(CATALOG_PATH),
        project_root=REPO_ROOT,
    )

    assert clients == {
        "SegFormer-MiT-B2:iSAID:local": client,
        "SegFormer-MiT-B2:OpenEarthMap:local": client,
    }
    assert [name for name, _ in calls] == [
        "segformer_transformers",
        "segformer_transformers",
    ]
    runtime = calls[0][1]["settings"]
    assert runtime.allow_download is False
    assert runtime.model_path == REPO_ROOT / "models" / "segformer_mitb2_isaid"
    assert calls[0][1]["id_to_label"][3] == "Small_Vehicle"


def test_segformer_runtime_profile_can_override_physical_model_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []
    settings = _settings(tmp_path)
    external = tmp_path / "mounted" / "segformer"
    profile = settings.models.segformer_isaid.model_copy(
        update={"model_path": external, "device": "cpu"}
    )
    settings = settings.model_copy(
        update={
            "models": settings.models.model_copy(
                update={"segformer_experts": {"segmenter_mitb2_001": profile}}
            )
        }
    )

    def fake_create_model(name: str, **kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("application.bootstrap.create_model", fake_create_model)
    _build_segformer_clients(
        settings,
        ExpertCatalog.load(CATALOG_PATH),
        project_root=REPO_ROOT,
    )

    runtime = calls[0]["settings"]
    assert runtime.model_path == external
    assert runtime.device == "cpu"
    assert runtime.logical_model_id == "SegFormer-MiT-B2:iSAID:local"
    assert runtime.weights_sha256 == (
        "f8e60686ec41160b5cbc494e8a3c1d28a92f7afdd41708c7b77e3d5793908b9a"
    )


def test_segformer_runtime_override_must_keep_catalog_identity(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = settings.models.segformer_isaid.model_copy(
        update={
            "model_path": tmp_path / "mounted",
            "logical_model_id": "different-logical-model",
        }
    )
    settings = settings.model_copy(
        update={
            "models": settings.models.model_copy(
                update={"segformer_experts": {"segmenter_mitb2_001": profile}}
            )
        }
    )

    with pytest.raises(RuntimeCompositionError, match="logical model id"):
        _build_segformer_clients(
            settings,
            ExpertCatalog.load(CATALOG_PATH),
            project_root=REPO_ROOT,
        )


def test_multiple_segformer_backends_register_stably_and_reuse_client(
    tmp_path: Path,
    monkeypatch,
) -> None:
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    semantic = next(
        expert
        for expert in document["experts"]
        if expert["backend_name"] == "segmenter_mitb2_001"
    )
    duplicate = json.loads(json.dumps(semantic))
    duplicate["backend_name"] = "segmenter_mitb2_002"
    document["experts"].append(duplicate)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    catalog = ExpertCatalog.load(path, asset_root=REPO_ROOT)
    clients_created: list[str] = []

    def fake_create_model(name: str, **kwargs: Any) -> object:
        clients_created.append(name)
        return object()

    monkeypatch.setattr("application.bootstrap.create_model", fake_create_model)
    registry = _build_backend_registry(
        _settings(tmp_path),
        PromptCatalog(REPO_ROOT / "prompts"),
        _FakeQwenClient(),
        expert_catalog=catalog,
        project_root=REPO_ROOT,
    )

    assert registry.all_names()[-3:] == [
        "segmenter_mitb2_001",
        "segmenter_mitb2_002",
        "segmenter_oem_001",
    ]
    assert clients_created == ["segformer_transformers", "segformer_transformers"]
    assert getattr(registry.get("segmenter_mitb2_001"), "_client") is getattr(
        registry.get("segmenter_mitb2_002"), "_client"
    )


def test_duplicate_catalog_backend_name_fails_before_registration(tmp_path: Path) -> None:
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    document["experts"].append(document["experts"][1])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        ExpertCatalog.load(path)


def test_segformer_catalog_label_mismatch_fails_without_absolute_path(
    tmp_path: Path,
) -> None:
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    semantic = next(
        expert
        for expert in document["experts"]
        if expert["backend_name"] == "segmenter_mitb2_001"
    )
    semantic["supports"]["small-vehicle"]["model_labels"] = ["not_a_real_label"]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ExpertCatalogError) as raised:
        ExpertCatalog.load(path, asset_root=REPO_ROOT)
    assert str(REPO_ROOT) not in str(raised.value)


def test_yolo_catalog_class_mismatch_fails_fast(tmp_path: Path) -> None:
    settings = load_settings(REPO_ROOT / "configs" / "yolo.example.yaml", environ={})
    detector = settings.backend.yolo.detectors[0].model_copy(
        update={
                "classes": [
                    value for value in settings.backend.yolo.detectors[0].classes
                    if value != "small-vehicle"
                ]
        }
    )
    yolo = settings.backend.yolo.model_copy(update={"detectors": [detector]})
    settings = settings.model_copy(
        update={"backend": settings.backend.model_copy(update={"yolo": yolo})}
    )

    with pytest.raises(RuntimeCompositionError, match="model labels differ"):
        _build_backend_registry(
            settings,
            PromptCatalog(REPO_ROOT / "prompts"),
            _FakeQwenClient(),
            expert_catalog=ExpertCatalog.load(CATALOG_PATH),
            project_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "different-model", "logical model id"),
        ("sha256", "0" * 64, "weight digest"),
        ("priority", 99, "priority"),
    ],
)
def test_yolo_catalog_identity_mismatch_fails_fast(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    settings = load_settings(REPO_ROOT / "configs" / "yolo.example.yaml", environ={})
    detector = settings.backend.yolo.detectors[0].model_copy(update={field: value})
    settings = settings.model_copy(
        update={
            "backend": settings.backend.model_copy(
                update={
                    "yolo": settings.backend.yolo.model_copy(
                        update={"detectors": [detector]}
                    )
                }
            )
        }
    )

    with pytest.raises(RuntimeCompositionError, match=message):
        _build_backend_registry(
            settings,
            PromptCatalog(REPO_ROOT / "prompts"),
            _FakeQwenClient(),
            expert_catalog=ExpertCatalog.load(CATALOG_PATH),
            project_root=REPO_ROOT,
        )


def test_relative_yolo_weights_resolve_against_project_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    detector = settings.backend.yolo.detectors[0]
    project_root = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    project_root.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = _resolve_yolo_detector(detector, project_root)

    assert resolved.weights == (
        project_root / "models" / "yolo_obb" / "dota_v2_yolo11m_obb_best.pt"
    ).resolve()
    assert elsewhere not in resolved.weights.parents


def test_absolute_yolo_weights_are_preserved(tmp_path: Path) -> None:
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    external = tmp_path / "mounted-models" / "detector.pt"
    detector = settings.backend.yolo.detectors[0].model_copy(
        update={"weights": external}
    )

    resolved = _resolve_yolo_detector(detector, tmp_path / "repo")

    assert resolved.weights == external


def test_yolo_runtime_path_does_not_affect_catalog_identity(tmp_path: Path) -> None:
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    detector = settings.backend.yolo.detectors[0].model_copy(
        update={"weights": tmp_path / "external" / "detector.pt"}
    )
    catalog = ExpertCatalog.load(CATALOG_PATH, asset_root=REPO_ROOT)

    validated = _catalog_validated_yolo_detector(detector, catalog)

    assert validated.weights == detector.weights
    assert validated.name == "detector_obb_csl_001"


@pytest.mark.parametrize(
    ("detector_index", "task", "expected_message"),
    [
        (0, "detect", "catalog declaration is not enabled"),
        (1, "detect", "catalog declaration is not enabled"),
    ],
)
def test_yolo_catalog_kind_matches_detector_task(
    detector_index: int,
    task: str,
    expected_message: str,
) -> None:
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    detector = settings.backend.yolo.detectors[detector_index].model_copy(
        update={"task": task}
    )

    with pytest.raises(RuntimeCompositionError, match=expected_message):
        _catalog_validated_yolo_detector(
            detector,
            ExpertCatalog.load(CATALOG_PATH, asset_root=REPO_ROOT),
        )


def test_yolo_catalog_rejects_unknown_detector_task() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    detector = settings.backend.yolo.detectors[0].model_copy(
        update={"task": "segment"}
    )

    with pytest.raises(RuntimeCompositionError, match="unsupported YOLO detector task"):
        _catalog_validated_yolo_detector(
            detector,
            ExpertCatalog.load(CATALOG_PATH, asset_root=REPO_ROOT),
        )


def test_both_configured_yolo_detectors_register_with_matching_catalog() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    registry = _build_backend_registry(
        settings,
        PromptCatalog(REPO_ROOT / "prompts"),
        _FakeQwenClient(),
        expert_catalog=ExpertCatalog.load(CATALOG_PATH, asset_root=REPO_ROOT),
        project_root=REPO_ROOT,
    )

    assert registry.all_names()[:4] == [
        "qwen_point",
        "quantity_proposal",
        "detector_obb_csl_001",
        "detector_obb_ultralytics_001",
    ]


def test_route_coverage_after_assembly(tmp_path: Path) -> None:
    """Every routable task must resolve to a registered agent after assembly.
    组装后每个可路由任务都必须解析到已注册 Agent。"""
    from routing.policies import POLICIES
    from routing.router import TaskRouter

    components = _assemble(tmp_path, qwen_client=_FakeQwenClient())
    router = TaskRouter()
    for task in POLICIES:
        decision = router.route(task)
        assert components.agent_registry.contains(decision.primary_agent)
        for fallback in decision.fallback_agents:
            assert components.agent_registry.contains(fallback)


def test_composed_auto_plan_uses_catalog_and_full_fixed_priority_chain(
    tmp_path: Path,
) -> None:
    settings = load_settings(REPO_ROOT / "configs" / "yolo.example.yaml", environ={})
    settings = settings.model_copy(
        update={"runs": settings.runs.model_copy(update={"root": tmp_path / "runs"})}
    )
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    agent = components.agent_registry.get("counting_agent")
    selector = getattr(agent, "_selector")
    catalog = getattr(agent, "_expert_catalog")
    target = CountTargetSpec(
        canonical_label="small-vehicle",
        inclusion_rule="include visible small vehicles",
        exclusion_rule="exclude every other object",
    )
    hints = {"quantity_estimation": True, **catalog.target_hints(target)}

    plan = selector.plan(
        target, task="counting",
        executable_leaf_categories=("small-vehicle",), hints=hints,
    )

    assert plan is not None
    assert plan.primary_backend_name == "detector_obb_csl_001"
    assert plan.fallback_backend_names == (
        "segmenter_mitb2_001",
        "quantity_proposal",
        "qwen_point",
    )

    vehicle = target.model_copy(update={"canonical_label": "vehicle"})
    vehicle_hints = {
        "quantity_estimation": True,
        **catalog.target_hints(vehicle),
    }
    vehicle_plan = selector.plan(
        vehicle, task="counting",
        executable_leaf_categories=("small-vehicle", "large-vehicle"),
        hints=vehicle_hints,
    )
    assert plan.ensemble_backend_names == ()
    assert vehicle_plan is not None
    assert vehicle_plan.primary_backend_name == "detector_obb_csl_001"
    assert vehicle_plan.fallback_backend_names == (
        "segmenter_mitb2_001", "quantity_proposal", "qwen_point",
    )

    aircraft = target.model_copy(update={"canonical_label": "aircraft"})
    aircraft_hints = {
        "quantity_estimation": True,
        **catalog.target_hints(aircraft),
    }
    aircraft_plan = selector.plan(
        aircraft, task="counting",
        executable_leaf_categories=("plane", "helicopter"), hints=aircraft_hints,
    )
    assert aircraft_plan is not None
    assert aircraft_plan.primary_backend_name == "detector_obb_csl_001"
    assert aircraft_plan.fallback_backend_names == (
        "segmenter_mitb2_001", "qwen_point",
    )


def test_local_inventory_selects_vrsbench_then_dota_for_shared_labels(
    tmp_path: Path,
) -> None:
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    settings = settings.model_copy(
        update={"runs": settings.runs.model_copy(update={"root": tmp_path / "runs"})}
    )
    components = assemble_runtime(
        settings,
        project_root=REPO_ROOT,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    injected_identity = components.qwen_clients["counting_agent"].cache_identity
    assert "adapter" not in injected_identity.generation_payload()
    agent = components.agent_registry.get("counting_agent")
    selector = getattr(agent, "_selector")
    catalog = getattr(agent, "_expert_catalog")

    def plan_for(label: str, leaves: tuple[str, ...]):
        target = CountTargetSpec(
            canonical_label=label,
            inclusion_rule="include the declared target",
            exclusion_rule="exclude every other object",
        )
        return selector.plan(
            target,
            task="counting",
            executable_leaf_categories=leaves,
            hints={"quantity_estimation": True, **catalog.target_hints(target)},
        )

    shared = plan_for("plane", ("plane",))
    assert shared is not None
    assert shared.selected_detector_expert_names == (
        "detector_obb_ultralytics_001",
        "detector_obb_csl_001",
    )
    assert shared.fallback_backend_names == (
        "segmenter_mitb2_001",
        "qwen_point",
    )

    airport = plan_for("airport", ("airport",))
    assert airport is not None
    assert airport.selected_detector_expert_names == (
        "detector_obb_ultralytics_001",
        "detector_obb_csl_001",
    )


def test_composed_schema_default_plan_uses_segformer_or_qwen_only(
    tmp_path: Path,
) -> None:
    components = _assemble(tmp_path, qwen_client=_FakeQwenClient())
    agent = components.agent_registry.get("counting_agent")
    selector = getattr(agent, "_selector")
    catalog = getattr(agent, "_expert_catalog")

    swimming_pool = CountTargetSpec(
        canonical_label="swimming-pool",
        inclusion_rule="include each visible pool",
        exclusion_rule="exclude non-pool regions",
    )
    pool_hints = {
        "quantity_estimation": True,
        **catalog.target_hints(swimming_pool),
    }
    pool_plan = selector.plan(
        swimming_pool, task="counting",
        executable_leaf_categories=("swimming-pool",), hints=pool_hints,
    )
    assert pool_plan is not None
    assert pool_plan.primary_backend_name == "segmenter_mitb2_001"
    assert pool_plan.fallback_backend_names == ("qwen_point",)

    crane = swimming_pool.model_copy(update={"canonical_label": "crane"})
    crane_hints = {"quantity_estimation": True, **catalog.target_hints(crane)}
    crane_plan = selector.plan(
        crane, task="counting",
        executable_leaf_categories=("container-crane",), hints=crane_hints,
    )
    assert crane_plan is not None
    assert crane_plan.primary_backend_name == "qwen_point"
    assert crane_plan.fallback_backend_names == ()


def test_judge_disabled_without_api_key(tmp_path: Path) -> None:
    components = _assemble(tmp_path, qwen_client=_FakeQwenClient())
    assert components.judge_client is None
    assert components.judge_service.judge_client is None


def test_judge_enabled_with_api_key(tmp_path: Path) -> None:
    components = _assemble(
        tmp_path, qwen_client=_FakeQwenClient(), api_key="test-api-key"
    )
    assert components.judge_client is not None
    assert components.judge_service.judge_client is not None


def test_qwen_created_exactly_once(tmp_path: Path, monkeypatch) -> None:
    """The composition root creates the Qwen client exactly once per
    assembly — never per sample. 组合根每次组装恰好创建一次 Qwen 客户端——
    绝不逐样本创建。"""
    calls = []

    def fake_create_model(name, **kwargs):
        calls.append(name)
        return _FakeQwenClient()

    monkeypatch.setattr("application.bootstrap.create_model", fake_create_model)
    components = _assemble(tmp_path)
    assert calls == [
        "qwen3_5_multi_adapter",
        "segformer_transformers",
        "segformer_transformers",
    ]
    assert components.qwen_client is not None
    # Injecting a client must never trigger creation. / 注入客户端绝不触发创建。
    _assemble(tmp_path, qwen_client=_FakeQwenClient())
    assert calls == [
        "qwen3_5_multi_adapter",
        "segformer_transformers",
        "segformer_transformers",
        "segformer_transformers",
        "segformer_transformers",
    ]


def test_counting_segformer_is_reused_when_change_semantic_is_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    dense_client = object()

    def fake_create_model(name, **kwargs):
        calls.append((name, kwargs.get("settings")))
        if name == "qwen3_5_multi_adapter":
            return _FakeQwenClient()
        if name == "segformer_transformers":
            return dense_client
        raise AssertionError(name)

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "agents": settings.agents.model_copy(
                update={
                    "change": settings.agents.change.model_copy(
                        update={"semantic": ChangeSemanticSettings(enabled=True)}
                    )
                }
            )
        }
    )
    monkeypatch.setattr("application.bootstrap.create_model", fake_create_model)

    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
    )

    assert [name for name, _ in calls] == [
        "qwen3_5_multi_adapter",
        "segformer_transformers",
        "segformer_transformers",
    ]
    assert calls[1][1].logical_model_id == settings.models.segformer_isaid.logical_model_id
    assert calls[1][1].model_path == REPO_ROOT / "models" / "segformer_mitb2_isaid"
    change_agent = components.agent_registry.get("change_agent")
    assert getattr(change_agent, "_semantic_client") is dense_client


def test_disabled_change_semantic_ignores_injected_dense_client(tmp_path: Path) -> None:
    injected = object()
    settings = _settings(tmp_path)
    settings.agents.change.semantic = ChangeSemanticSettings(enabled=False)

    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
        semantic_client=injected,
    )

    change_agent = components.agent_registry.get("change_agent")
    assert getattr(change_agent, "_semantic_client") is None


def test_learned_change_client_is_composition_root_injection_only(tmp_path: Path) -> None:
    learned = object()
    components = _assemble(
        tmp_path,
        qwen_client=_FakeQwenClient(),
        learned_change_client=learned,
    )
    change_agent = components.agent_registry.get("change_agent")
    assert getattr(change_agent, "_learned_change_client") is learned


def test_import_application_has_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    """Importing the application package must not create models, read configs,
    or touch the filesystem. 导入 application 包不得创建模型、读取配置或触碰
    文件系统。"""
    import importlib

    def boom_create_model(*args, **kwargs):
        raise AssertionError("import must not create models")

    monkeypatch.setattr("application.bootstrap.create_model", boom_create_model)
    monkeypatch.setattr("application.settings.load_settings", boom_create_model)
    application = importlib.import_module("application")
    importlib.reload(application)
    # Source-level guarantee: model creation only lives inside bootstrap.
    # 源码级保证：模型创建只存在于 bootstrap 内部。
    for module_name in ("__init__", "settings", "prompts", "runtime"):
        source = (
            Path(__file__).resolve().parents[2] / "application" / f"{module_name}.py"
        ).read_text(encoding="utf-8")
        assert "create_model(" not in source, module_name
        assert "complete_json" not in source, module_name


# ── visual planning composition (14A3 C9) / 视觉规划组合装配 ─────────────


def _visual_settings(
    tmp_path: Path,
    *,
    visual_planning: dict[str, Any] | None = None,
    yolo: dict[str, Any] | None = None,
) -> AppSettings:
    payload: dict[str, Any] = dict(
        runs={"root": tmp_path / "runs"},
    )
    if visual_planning is not None:
        payload["visual_planning"] = visual_planning
    if yolo is not None:
        payload["backend"] = {"yolo": yolo}
    return AppSettings(**payload)


def _calibrated_detector(
    tmp_path: Path,
    name: str = "fake-det",
    classes: list[str] | None = None,
) -> YoloDetectorSettings:
    """A structurally valid detector whose weights file does not exist, so
    the shared store fails closed at assembly. 结构合法但权重文件不存在的检测
    器，使共享 store 在组装时严格失败。"""
    return YoloDetectorSettings(
        name=name,
        enabled=True,
        weights=tmp_path / f"{name}.onnx",
        model_id=f"fake:{name}:v1",
        sha256="0" * 64,
        classes=classes or ["plane"],
        require_cuda=False,
        device="cpu",
    )


class _FakeSegmenterClient:
    """Marker semantic-mask client; composition must never call it.
    标记 semantic-mask client；组装期绝不调用它。"""


def _record_segmenter_creations(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    """Replace bootstrap create_model with a recorder returning marker
    clients, proving composition creates no real model and no weights load.
    用记录器替换 bootstrap create_model 并返回标记 client，证明组装期不创建
    真实模型、不加载任何权重。"""

    created: list[tuple[str, dict[str, Any]]] = []

    def _create(name: str, **kwargs: Any) -> Any:
        created.append((name, kwargs))
        return _FakeSegmenterClient()

    monkeypatch.setattr("application.bootstrap.create_model", _create)
    return created


def test_visual_planning_is_always_v5_and_injected(tmp_path: Path) -> None:
    """Fresh composition always injects the v5 planner bindings.
    新鲜组合始终注入 v5 规划器绑定。"""
    components = _assemble(tmp_path, qwen_client=_FakeQwenClient())
    runner = components.sample_runner_factory(data_root=tmp_path)
    assert components.visual_task_planner is not None
    assert runner.visual_bindings is components.visual_bindings


def test_visual_planning_uses_v5_with_uncalibrated_bindings(
    tmp_path: Path,
) -> None:
    """The v5 planner is always assembled; uncalibrated evidence stays closed.
    v5 规划器始终组装；未校准的证据能力保持关闭。"""
    settings = _visual_settings(tmp_path, visual_planning={})
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    runner = components.sample_runner_factory(data_root=tmp_path)
    assert runner.visual_bindings is components.visual_bindings
    planner = components.visual_task_planner
    assert planner is not None
    # The settings-declared versions must bind the real prompt/catalog assets.
    # settings 声明版本必须绑定真实 prompt/catalog 资产。
    assert planner.prompt_version == components.prompt_catalog.version("visual_task_plan")
    assert planner._catalog.catalog_version == (
        settings.visual_planning.planner.catalog_version
    )
    binding = json.loads(planner.system_prompt.split("planner_binding=", 1)[1])
    assert len(binding["canonical_leaf_categories"]) == 26
    assert "vehicle" not in binding["canonical_leaf_categories"]
    assert binding["parent_expansions"]["vehicle"] == [
        "small-vehicle", "large-vehicle"
    ]
    assert len(binding["task_executable_categories"]["counting"]) == 13
    assert components.visual_bindings is not None
    assert components.visual_bindings.vqa_evidence is None
    assert components.visual_bindings.grounding_evidence is not None
    grounding = components.visual_bindings.grounding_evidence
    assert grounding._policy.yolo_enabled is False
    # Uncalibrated means the YOLO phase is off: no detector is wired at all,
    # so nothing is ever loaded for it. 未校准即 YOLO 阶段关闭：完全不接线检测
    # 器，因此永远不会为它加载任何东西。
    assert grounding._yolo_client is None


def test_visual_planning_uncalibrated_binding_fails_closed_for_all_four_vqa_tasks(
    tmp_path: Path,
) -> None:
    """With the VQA evidence service unavailable, every GeneralVQAAgent task
    publishes an empty executable binding — the planner fails closed for all
    four tasks, not just general_vqa. VQA evidence 服务不可用时，每个
    GeneralVQAAgent task 都发布空可执行绑定——planner 对全部四个 task 严格
    fail closed，而非只对 general_vqa。"""
    settings = _visual_settings(tmp_path, visual_planning={})
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    assert components.visual_bindings.vqa_evidence is None
    binding = json.loads(
        components.visual_task_planner.system_prompt.split("planner_binding=", 1)[1]
    )
    for task in (
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
        "spatial_relation",
    ):
        assert binding["task_executable_categories"][task] == []


def test_visual_planning_binds_four_vqa_tasks_to_shared_capabilities(
    tmp_path: Path, monkeypatch
) -> None:
    """With a full runtime profile, the four GeneralVQAAgent tasks expose the
    identical executable leaves from one shared immutable capability set, and
    the frozen VQA assistance scope is bound into the planner identity.
    完整 runtime profile 下，四个 GeneralVQAAgent task 暴露同一份共享不可变
    能力集合的可执行叶子，且冻结的 VQA assistance scope 进入 planner 身份。"""
    _record_segmenter_creations(monkeypatch)

    class _FakeStore:
        def get(self, detector: YoloDetectorSettings) -> Any:
            raise AssertionError("composition must not load weights")

    monkeypatch.setattr("application.bootstrap.YoloModelStore", _FakeStore)
    monkeypatch.setattr(
        "application.bootstrap._catalog_validated_yolo_detector",
        lambda detector, _catalog: detector,
    )
    catalog = EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")
    yolo_labels = [
        label
        for leaf in catalog.executable_leaves_for_task("general_vqa")
        for label in catalog.leaf_yolo_labels(leaf)
    ]
    detector = _calibrated_detector(
        tmp_path, name="detector_obb_csl_001", classes=yolo_labels
    )
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "detectors": {
                "detector_obb_csl_001": {
                    "confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.5,
                    "max_detections": 5,
                },
            },
            "segmenters": {
                "segmenter_oem_001": {
                    "enabled": True,
                    "class_map_version": "verified-2026-08-20",
                },
            },
        },
        yolo={"enabled": True, "detectors": [detector]},
    )
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    planner = components.visual_task_planner
    binding = json.loads(planner.system_prompt.split("planner_binding=", 1)[1])
    shared = binding["task_executable_categories"]["general_vqa"]
    assert len(shared) == len(catalog.executable_leaves_for_task("general_vqa"))
    for task in ("scene_classification", "multiple_choice_vqa", "spatial_relation"):
        assert binding["task_executable_categories"][task] == shared
    assert (
        planner.planning_parameters["vqa_assistance_scope"]
        == "general-vqa-agent-tasks-v1"
    )


def test_counting_planner_leaves_require_real_specialist_support(tmp_path: Path) -> None:
    evidence = EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")
    experts = ExpertCatalog.load(CATALOG_PATH, asset_root=REPO_ROOT)

    semantic_only = _visual_settings(
        tmp_path,
        yolo={"enabled": False, "detectors": []},
    )
    semantic_leaves = _enabled_counting_catalog_leaves(
        semantic_only, evidence, experts, task="counting"
    )
    assert "small-vehicle" in semantic_leaves
    assert "bridge" not in semantic_leaves
    assert "harbor" not in semantic_leaves

    yolo_enabled = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    yolo_leaves = _enabled_counting_catalog_leaves(
        yolo_enabled, evidence, experts, task="counting"
    )
    assert "bridge" in yolo_leaves
    assert "harbor" in yolo_leaves


def test_counting_planner_leaves_are_independent_per_counting_task(
    tmp_path: Path,
) -> None:
    data = json.loads(
        (REPO_ROOT / "agents" / "evidence_catalog.json").read_text(encoding="utf-8")
    )
    data["catalog_version"] = "task-split-test-v1"
    data["task_capabilities"]["counting"] = [
        "small-vehicle", "large-vehicle"
    ]
    data["task_capabilities"]["fine_grained_counting"] = ["small-vehicle"]
    evidence = EvidenceCatalog(data)
    experts = ExpertCatalog.load(CATALOG_PATH, asset_root=REPO_ROOT)
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})

    counting = _enabled_counting_catalog_leaves(
        settings, evidence, experts, task="counting"
    )
    fine = _enabled_counting_catalog_leaves(
        settings, evidence, experts, task="fine_grained_counting"
    )
    assert counting == ("small-vehicle", "large-vehicle")
    assert fine == ("small-vehicle",)

    with pytest.raises(RuntimeCompositionError, match="unsupported counting planner task"):
        _enabled_counting_catalog_leaves(
            settings, evidence, experts, task="general_vqa"
        )


def test_visual_planner_binding_keeps_counting_task_capabilities_separate(
    tmp_path: Path, monkeypatch,
) -> None:
    data = json.loads(
        (REPO_ROOT / "agents" / "evidence_catalog.json").read_text(encoding="utf-8")
    )
    data["catalog_version"] = "task-split-binding-v1"
    data["task_capabilities"]["counting"] = [
        "small-vehicle", "large-vehicle"
    ]
    data["task_capabilities"]["fine_grained_counting"] = ["small-vehicle"]
    evidence = EvidenceCatalog(data)
    monkeypatch.setattr(
        bootstrap_module, "_load_evidence_catalog", lambda _root: evidence
    )
    settings = _visual_settings(
        tmp_path,
        visual_planning={"planner": {"catalog_version": evidence.catalog_version}},
        yolo={"enabled": False, "detectors": []},
    )
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    planner = components.visual_task_planner
    binding = json.loads(planner.system_prompt.split("planner_binding=", 1)[1])
    by_task = binding["task_executable_categories"]
    assert by_task["counting"] == ["small-vehicle", "large-vehicle"]
    assert by_task["fine_grained_counting"] == ["small-vehicle"]


def test_visual_planning_catalog_version_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    """A settings catalog version that drifts from the evidence catalog asset
    must fail closed at assembly. settings 的 catalog 版本与证据目录资产漂移
    时必须在组装时严格失败。"""
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "planner": {"catalog_version": "bogus-catalog-v1"},
        },
    )
    with pytest.raises(RuntimeCompositionError):
        assemble_runtime(
            settings,
            project_root=tmp_path,
            prompts_root=REPO_ROOT / "prompts",
            qwen_client=_FakeQwenClient(),
        )


def test_visual_planning_prompt_version_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    """A settings prompt version that drifts from the prompt catalog binding
    must fail closed at assembly. settings 的 prompt 版本与 prompt catalog 绑
    定漂移时必须在组装时严格失败。"""
    settings = _visual_settings(
        tmp_path,
        visual_planning={"planner": {"task_prompt_version": "v99"}},
    )
    with pytest.raises(RuntimeCompositionError):
        assemble_runtime(
            settings,
            project_root=tmp_path,
            prompts_root=REPO_ROOT / "prompts",
            qwen_client=_FakeQwenClient(),
        )


def test_visual_planning_partial_calibration_fails_closed(tmp_path: Path) -> None:
    """A partially calibrated detector policy cannot form a global policy and
    must fail closed instead of inventing defaults. 部分校准的检测策略无法构成
    全局策略，必须严格失败而不是杜撰默认值。"""
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "detectors": {"small_vehicle": {"confidence_threshold": 0.5}},
        },
    )
    with pytest.raises(RuntimeCompositionError):
        assemble_runtime(
            settings,
            project_root=tmp_path,
            prompts_root=REPO_ROOT / "prompts",
            qwen_client=_FakeQwenClient(),
        )


def test_visual_planning_multiple_calibration_fails_closed(tmp_path: Path) -> None:
    """Multiple differing calibrated policies are ambiguous for the single
    global executor policy and must fail closed. 多条不同已校准策略对单一全局
    执行器策略构成歧义，必须严格失败。"""
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "detectors": {
                "small_vehicle": {
                    "confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.5,
                    "max_detections": 5,
                },
                "large_vehicle": {
                    "confidence_threshold": 0.6,
                    "nms_iou_threshold": 0.4,
                    "max_detections": 3,
                },
            },
        },
    )
    with pytest.raises(RuntimeCompositionError):
        assemble_runtime(
            settings,
            project_root=tmp_path,
            prompts_root=REPO_ROOT / "prompts",
            qwen_client=_FakeQwenClient(),
        )


def test_visual_planning_calibrated_without_detector_fails_closed(
    tmp_path: Path,
) -> None:
    """A calibrated VQA policy without any enabled detector is inconsistent
    and must fail closed. 已校准 VQA 策略没有任何启用检测器即不一致，必须严格
    失败。"""
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "detectors": {
                "small_vehicle": {
                    "confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.5,
                    "max_detections": 5,
                }
            },
        },
    )
    with pytest.raises(RuntimeCompositionError):
        assemble_runtime(
            settings,
            project_root=tmp_path,
            prompts_root=REPO_ROOT / "prompts",
            qwen_client=_FakeQwenClient(),
        )


def test_visual_planning_enabled_segmenter_fails_closed(tmp_path: Path) -> None:
    """An enabled segmenter whose model binding is not frozen must fail closed
    at assembly instead of silently ignoring the declared capability.
    已启用分割器但模型绑定未冻结时必须在组装时严格失败，而不是静默忽略已声明
    能力。"""
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "segmenters": {
                "building": {"enabled": True, "class_map_version": "isaid-v2"}
            },
        },
    )
    with pytest.raises(RuntimeCompositionError):
        assemble_runtime(
            settings,
            project_root=tmp_path,
            prompts_root=REPO_ROOT / "prompts",
            qwen_client=_FakeQwenClient(),
        )


def test_visual_planning_segmenter_only_composes_segformer_only_executor(
    tmp_path: Path, monkeypatch
) -> None:
    """A VQA-enabled segmenter binding (the OEM expert is disabled for
    counting) composes a SegFormer-only executor: no YOLO client, no detector
    policy, the missing client created lazily by the composition root, and
    the planner publishes only the eight OEM leaves.
    仅启用 segmenter（OEM expert 在 counting 中禁用）时组装 SegFormer-only
    执行器：无 YOLO client、无检测策略，缺失 client 由组合根惰性创建，
    planner 只发布 8 个 OEM 叶子。"""
    created = _record_segmenter_creations(monkeypatch)
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "segmenters": {
                "segmenter_oem_001": {
                    "enabled": True,
                    "class_map_version": "verified-2026-08-20",
                },
            },
            "preprocessing": {"max_tile_concurrency": 2},
        },
    )
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    vqa = components.visual_bindings.vqa_evidence
    assert vqa is not None
    assert vqa._yolo_client is None
    assert vqa._policy is None
    assert set(vqa._segmenter_clients) == {"segmenter_oem_001"}
    # The frozen preprocessing identity is mirrored into the agents-local
    # contract, including the overridden concurrency bound; fresh composition
    # defaults to the combined yolo-v1-segformer-pad-v1 identity.
    # 冻结预处理身份镜像进 agents 局部契约，包括被覆盖的并发上限；新鲜组装
    # 默认使用 yolo-v1-segformer-pad-v1 组合身份。
    assert vqa._preprocessing.version == "yolo-v1-segformer-pad-v1"
    assert vqa._preprocessing.tile_size == 1024
    assert vqa._preprocessing.max_tile_concurrency == 2
    assert vqa._preprocessing.yolo_version == "greedy-1024-stretch-v1"
    assert vqa._preprocessing.segformer_version == "pad-multiple-1024-resize-square-v1"
    assert vqa._preprocessing.segformer_padding_mode == "constant-black-right-bottom"
    assert vqa._preprocessing.segformer_rgb_interpolation == "lanczos"
    assert vqa._preprocessing.segformer_mask_inverse_interpolation == "nearest"
    # The counting-disabled OEM expert gets its own lazy client; no real
    # model was constructed and nothing was loaded.
    # counting 中禁用的 OEM expert 获得独立惰性 client；未构造真实模型、
    # 未加载任何权重。
    assert any(
        name == "segformer_transformers"
        and kwargs["settings"].logical_model_id
        == "SegFormer-MiT-B2:OpenEarthMap:local"
        for name, kwargs in created
    )
    planner = components.visual_task_planner
    binding = json.loads(planner.system_prompt.split("planner_binding=", 1)[1])
    assert binding["task_executable_categories"]["general_vqa"] == [
        "bareland",
        "rangeland",
        "developed-space",
        "road",
        "tree",
        "water",
        "agriculture-land",
        "building",
    ]


def test_visual_planning_oem_class_map_version_is_consumed_at_composition(
    tmp_path: Path,
) -> None:
    """A stale OEM class-map version fails before planner publication.
    过期的 OEM class-map 版本必须在 planner 发布前于组合期失败。"""
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "segmenters": {
                "segmenter_oem_001": {
                    "enabled": True,
                    "class_map_version": "verified-2026-08-06",
                },
            },
        },
    )
    with pytest.raises(RuntimeCompositionError, match="class map version"):
        assemble_runtime(
            settings,
            project_root=tmp_path,
            prompts_root=REPO_ROOT / "prompts",
            qwen_client=_FakeQwenClient(),
        )


def test_visual_planning_oem_raw_labels_are_checked_at_composition(
    tmp_path: Path, monkeypatch
) -> None:
    """VQA catalog labels are checked even when counting supports are empty.
    即使 counting supports 为空，也必须在组合期校验 VQA catalog 原始标签。"""
    data = json.loads(
        (REPO_ROOT / "agents" / "evidence_catalog.json").read_text(encoding="utf-8")
    )
    data["catalog_version"] = "oem-raw-label-check-v1"
    data["leaves"]["building"]["segformer_labels"] = ["not-a-class-map-label"]
    monkeypatch.setattr(
        bootstrap_module,
        "_load_evidence_catalog",
        lambda _root: EvidenceCatalog(data),
    )
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "planner": {"catalog_version": "oem-raw-label-check-v1"},
            "segmenters": {
                "segmenter_oem_001": {
                    "enabled": True,
                    "class_map_version": "verified-2026-08-20",
                },
            },
        },
    )
    with pytest.raises(RuntimeCompositionError, match="VQA raw labels"):
        assemble_runtime(
            settings,
            project_root=tmp_path,
            prompts_root=REPO_ROOT / "prompts",
            qwen_client=_FakeQwenClient(),
        )


def test_segformer_class_map_checkpoint_digest_is_verified_before_client_creation(
    tmp_path: Path,
) -> None:
    """Class-map metadata cannot declare a different checkpoint digest.
    class-map metadata 不得声明与 expert asset 不同的 checkpoint digest。"""
    expert = ExpertCatalog.load(CATALOG_PATH, asset_root=REPO_ROOT).expert(
        "segmenter_oem_001"
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    class_map = json.loads(
        (REPO_ROOT / "models" / "segformer_mitb2_oem" / "classes.json").read_text(
            encoding="utf-8"
        )
    )
    class_map["verification"]["checkpoint_sha256"] = "0" * 64
    (model_dir / "classes.json").write_text(
        json.dumps(class_map), encoding="utf-8"
    )
    local_expert = expert.model_copy(
        update={
            "asset": expert.asset.model_copy(
                update={
                    "model_dir": "model",
                    "class_map": "model/classes.json",
                    "weights": "model/model.safetensors",
                }
            )
        }
    )
    with pytest.raises(RuntimeCompositionError, match="checkpoint digest"):
        bootstrap_module._verified_class_map(local_expert, tmp_path)


def test_visual_planning_detector_plus_segmenter_composes_combined_executor(
    tmp_path: Path, monkeypatch
) -> None:
    """A calibrated detector plus an enabled segmenter composes the combined
    executor, and the planner publishes exactly the intersect-checked YOLO
    leaf plus the eight OEM leaves, without publishing unavailable leaves.
    已校准检测器加启用 segmenter 组装组合执行器，planner 恰好发布经标签相交
    校验的 YOLO 叶子加 8 个 OEM 叶子，不发布当前不可用的其他叶子。"""
    _record_segmenter_creations(monkeypatch)

    class _FakeStore:
        def get(self, detector: YoloDetectorSettings) -> Any:
            raise AssertionError("composition must not load weights")

    monkeypatch.setattr("application.bootstrap.YoloModelStore", _FakeStore)
    monkeypatch.setattr(
        "application.bootstrap._catalog_validated_yolo_detector",
        lambda detector, _catalog: detector,
    )
    detector = _calibrated_detector(tmp_path, name="small_vehicle", classes=["plane"])
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "detectors": {
                "small_vehicle": {
                    "confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.5,
                    "max_detections": 5,
                },
            },
            "segmenters": {
                "segmenter_oem_001": {
                    "enabled": True,
                    "class_map_version": "verified-2026-08-20",
                },
            },
        },
        yolo={"enabled": True, "detectors": [detector]},
    )
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    vqa = components.visual_bindings.vqa_evidence
    assert vqa is not None
    assert vqa._yolo_client is not None
    assert vqa._policy is not None
    assert vqa._policy.confidence_threshold == 0.5
    assert vqa._policy.nms_iou_threshold == 0.5
    assert vqa._policy.max_detections == 5
    assert set(vqa._segmenter_clients) == {"segmenter_oem_001"}
    planner = components.visual_task_planner
    binding = json.loads(planner.system_prompt.split("planner_binding=", 1)[1])
    assert binding["task_executable_categories"]["general_vqa"] == [
        "plane",
        "bareland",
        "rangeland",
        "developed-space",
        "road",
        "tree",
        "water",
        "agriculture-land",
        "building",
    ]


def test_visual_planning_full_runtime_capabilities_publish_all_catalog_leaves(
    tmp_path: Path, monkeypatch
) -> None:
    """All 26 leaves are published when, and only because, the injected
    runtime profile really covers all 18 YOLO leaves plus all eight OEM
    leaves. A non-None service alone is not the capability assertion.
    只有注入的 runtime profile 实际覆盖 18 个 YOLO 叶子和 8 个 OEM 叶子时，
    才发布全部 26 类；service 非空本身不是能力断言。"""
    _record_segmenter_creations(monkeypatch)

    class _FakeStore:
        def get(self, detector: YoloDetectorSettings) -> Any:
            raise AssertionError("composition must not load weights")

    monkeypatch.setattr("application.bootstrap.YoloModelStore", _FakeStore)
    monkeypatch.setattr(
        "application.bootstrap._catalog_validated_yolo_detector",
        lambda detector, _catalog: detector,
    )
    catalog = EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")
    yolo_labels = [
        label
        for leaf in catalog.executable_leaves_for_task("general_vqa")
        for label in catalog.leaf_yolo_labels(leaf)
    ]
    detector = _calibrated_detector(
        tmp_path, name="detector_obb_csl_001", classes=yolo_labels
    )
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "detectors": {
                "detector_obb_csl_001": {
                    "confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.5,
                    "max_detections": 5,
                },
            },
            "segmenters": {
                "segmenter_oem_001": {
                    "enabled": True,
                    "class_map_version": "verified-2026-08-20",
                },
            },
        },
        yolo={"enabled": True, "detectors": [detector]},
    )
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    binding = json.loads(
        components.visual_task_planner.system_prompt.split("planner_binding=", 1)[1]
    )
    assert binding["task_executable_categories"]["general_vqa"] == list(
        catalog.executable_leaves_for_task("general_vqa")
    )


def test_visual_planning_yolo_service_never_publishes_unmatched_leaves(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A calibrated detector whose classes intersect no catalog leaf still
    composes a YOLO-only service, but the planner publishes no executable
    leaves: a non-None service never means all 26 categories are runnable.
    已校准检测器类别不匹配任何目录叶子时仍组装 YOLO-only 服务，但 planner 不
    发布任何可执行叶子：服务非 None 绝不意味着 26 类全部可运行。"""
    detector = _calibrated_detector(
        tmp_path, name="small_vehicle", classes=["zzz-not-in-catalog"]
    )
    monkeypatch.setattr(
        "application.bootstrap._catalog_validated_yolo_detector",
        lambda detector, _catalog: detector,
    )
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "detectors": {
                "small_vehicle": {
                    "confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.5,
                    "max_detections": 5,
                },
            },
        },
        yolo={"enabled": True, "detectors": [detector]},
    )
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    assert components.visual_bindings.vqa_evidence is not None
    planner = components.visual_task_planner
    binding = json.loads(planner.system_prompt.split("planner_binding=", 1)[1])
    assert binding["task_executable_categories"]["general_vqa"] == []


def test_visual_planning_yolo_stays_lazy_until_first_inference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Composition must never load YOLO weights (AGENTS.md lazy-loading
    contract): assembly wires a lazy client without touching the store, the
    first detect() goes through the shared audited store, and a missing
    weights failure surfaces only then — as the stable domain error, never
    with a host path. 组合期绝不加载 YOLO 权重（AGENTS.md 惰性加载契约）：装
    配只接线惰性客户端而不触碰 store，首次 detect() 才经共享审计 store；权重
    缺失的失败只在那时呈现——以稳定领域错误出现，绝不携带主机路径。"""

    get_calls: list[str] = []

    class _FakeStore:
        def get(self, detector: YoloDetectorSettings) -> Any:
            get_calls.append(detector.name)
            from agents.errors import DetectorWeightsMissingError

            raise DetectorWeightsMissingError(detector.name, "weights.onnx")

    monkeypatch.setattr("application.bootstrap.YoloModelStore", _FakeStore)
    monkeypatch.setattr(
        "application.bootstrap._catalog_validated_yolo_detector",
        lambda detector, _catalog: detector,
    )
    detector = _calibrated_detector(tmp_path, name="small_vehicle")
    settings = _visual_settings(
        tmp_path,
        visual_planning={
            "detectors": {
                "small_vehicle": {
                    "confidence_threshold": 0.5,
                    "nms_iou_threshold": 0.5,
                    "max_detections": 5,
                }
            },
        },
        yolo={"enabled": True, "detectors": [detector]},
    )
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=_FakeQwenClient(),
    )
    planner = components.visual_task_planner
    assert planner is not None and components.visual_bindings is not None
    assert get_calls == []  # assembly never loads / 组合期绝不加载
    vqa = components.visual_bindings.vqa_evidence
    assert vqa is not None
    with pytest.raises(Exception) as exc_info:
        vqa._yolo_client.detect(  # type: ignore[attr-defined]
            "unused",
            confidence=0.5,
            iou=0.5,
            image_size=1024,
            device="cpu",
            max_detections=5,
        )
    assert get_calls == ["small_vehicle"]  # first inference loads once / 首次推理才加载
    # Only the stable error type name surfaces; never the host path.
    # 只呈现稳定错误类型名；绝不携带主机路径。
    assert "DetectorWeightsMissingError" in type(exc_info.value).__name__
    assert str(tmp_path) not in str(exc_info.value)
