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
from application.bootstrap import (
    RuntimeCompositionError,
    _build_backend_registry,
    _build_segformer_clients,
    assemble_runtime,
)
from application.prompts import PromptCatalog
from application.settings import AppSettings, load_settings
from models.base import ModelCacheIdentity


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


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "agents" / "counting" / "expert_catalog.json"


def test_bootstrap_does_not_access_catalog_private_storage() -> None:
    source = (REPO_ROOT / "application" / "bootstrap.py").read_text(encoding="utf-8")

    assert "_experts" not in source
    assert "getattr(catalog" not in source


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
    assert len(components.agent_registry) == 6
    assert components.agent_registry.names() == (
        "counting_agent",
        "change_agent",
        "grounding_agent",
        "spatial_agent",
        "general_vqa_agent",
        "caption_agent",
    )
    assert components.prompt_catalog is not None
    assert components.task_resolver is not None
    assert components.judge_service is not None
    assert components.dataset_runner_factory is not None
    change_agent = components.agent_registry.get("change_agent")
    runtime_prompt = getattr(change_agent, "_prompt")
    assert runtime_prompt.text == components.prompt_catalog["change"]
    assert runtime_prompt.version == components.prompt_catalog.version("change")
    assert components.prompt_catalog.asset("change").path in (
        components.prompt_catalog.snapshot_paths()
    )


def test_bootstrap_registers_quantity_proposal_backend(tmp_path: Path) -> None:
    registry = _build_backend_registry(
        _settings(tmp_path),
        PromptCatalog(REPO_ROOT / "prompts"),
        _FakeQwenClient(),
    )
    assert registry.all_names() == ["qwen_point", "quantity_proposal"]
    assert registry.get("quantity_proposal").kind == "quantity_proposal"


def test_enabled_segformer_is_registered_lazily_and_oem_is_not(tmp_path: Path) -> None:
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
    ]
    backend = registry.get("segmenter_mitb2_001")
    assert backend.kind == "semantic_segmentation"
    assert getattr(backend, "_client").loaded is False
    assert "segmenter_oem_001" not in registry.all_names()


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

    assert clients == {"SegFormer-MiT-B2:iSAID:local": client}
    assert [name for name, _ in calls] == ["segformer_transformers"]
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

    assert registry.all_names()[-2:] == [
        "segmenter_mitb2_001",
        "segmenter_mitb2_002",
    ]
    assert clients_created == ["segformer_transformers"]
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
        update={"composite_targets": {"vehicle": ["small vehicle"]}}
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

    plan = selector.plan(target, task="counting", hints=hints)

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
    vehicle_plan = selector.plan(vehicle, task="counting", hints=vehicle_hints)
    assert vehicle_plan is not None
    assert vehicle_plan.primary_backend_name == "detector_obb_csl_001"
    assert vehicle_plan.fallback_backend_names == (
        "segmenter_mitb2_001",
        "quantity_proposal",
        "qwen_point",
    )

    aircraft = target.model_copy(update={"canonical_label": "aircraft"})
    aircraft_hints = {
        "quantity_estimation": True,
        **catalog.target_hints(aircraft),
    }
    aircraft_plan = selector.plan(aircraft, task="counting", hints=aircraft_hints)
    assert aircraft_plan is not None
    assert aircraft_plan.primary_backend_name == "detector_obb_csl_001"
    assert aircraft_plan.fallback_backend_names == (
        "segmenter_mitb2_001",
        "qwen_point",
    )


def test_composed_default_plan_uses_segformer_then_qwen_or_qwen_only(
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
    pool_plan = selector.plan(swimming_pool, task="counting", hints=pool_hints)
    assert pool_plan is not None
    assert pool_plan.primary_backend_name == "segmenter_mitb2_001"
    assert pool_plan.fallback_backend_names == ("qwen_point",)

    crane = swimming_pool.model_copy(update={"canonical_label": "crane"})
    crane_hints = {"quantity_estimation": True, **catalog.target_hints(crane)}
    crane_plan = selector.plan(crane, task="counting", hints=crane_hints)
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
    assert calls == ["qwen_transformers", "segformer_transformers"]
    assert components.qwen_client is not None
    # Injecting a client must never trigger creation. / 注入客户端绝不触发创建。
    _assemble(tmp_path, qwen_client=_FakeQwenClient())
    assert calls == [
        "qwen_transformers",
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
        if name == "qwen_transformers":
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
        "qwen_transformers",
        "segformer_transformers",
    ]
    assert calls[1][1].logical_model_id == settings.models.segformer_isaid.logical_model_id
    assert calls[1][1].model_path == REPO_ROOT / "models" / "segformer_mitb2_isaid"
    change_agent = components.agent_registry.get("change_agent")
    assert getattr(change_agent, "_semantic_client") is dense_client


def test_disabled_change_semantic_ignores_injected_dense_client(tmp_path: Path) -> None:
    injected = object()

    components = _assemble(
        tmp_path,
        qwen_client=_FakeQwenClient(),
        semantic_client=injected,
    )

    change_agent = components.agent_registry.get("change_agent")
    assert getattr(change_agent, "_semantic_client") is None


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
