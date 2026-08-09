"""Contract tests for the composition root: single Qwen creation, judge
disabled without a key, route coverage, and side-effect-free imports.

组合根契约测试：Qwen 只创建一次、无 key 时 judge 禁用、路由覆盖与无副作用
导入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.change.settings import ChangeSemanticSettings
from application.bootstrap import _build_backend_registry, assemble_runtime
from application.prompts import PromptCatalog
from application.settings import AppSettings
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


def test_bootstrap_registers_quantity_proposal_backend(tmp_path: Path) -> None:
    registry = _build_backend_registry(
        _settings(tmp_path),
        PromptCatalog(REPO_ROOT / "prompts"),
        _FakeQwenClient(),
    )
    assert registry.all_names() == ["qwen_point", "quantity_proposal"]
    assert registry.get("quantity_proposal").kind == "quantity_proposal"


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
    assert calls == ["qwen_transformers"]
    assert components.qwen_client is not None
    # Injecting a client must never trigger creation. / 注入客户端绝不触发创建。
    _assemble(tmp_path, qwen_client=_FakeQwenClient())
    assert calls == ["qwen_transformers"]


def test_segformer_is_created_only_when_change_semantic_is_enabled(
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
    assert calls[1][1] is settings.models.segformer_isaid
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
