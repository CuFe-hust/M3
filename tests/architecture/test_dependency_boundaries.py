"""Enforce dependency boundaries — no YOLO on import, no CLI from agents, etc.
强制依赖边界 — import 时不加载 YOLO，Agent 不 import CLI 等。
"""

from __future__ import annotations

import sys


# ── YOLO lazy loading / YOLO 延迟加载 ──────────────────────────────────────


def test_agents_import_does_not_load_ultralytics():
    """Importing spacers_agent.agents must NOT import ultralytics."""
    # Remove ultralytics from sys.modules if already loaded
    ultralytics_loaded_before = "ultralytics" in sys.modules
    # Re-import agents to verify
    import spacers_agent.agents  # noqa: F401
    assert "ultralytics" not in sys.modules or ultralytics_loaded_before, (
        "spacers_agent.agents imported ultralytics — YOLO must be lazy-loaded"
    )


def test_bootstrap_import_does_not_load_ultralytics():
    """Importing spacers_agent.bootstrap must NOT import ultralytics."""
    ultralytics_loaded_before = "ultralytics" in sys.modules
    import spacers_agent.bootstrap  # noqa: F401
    assert "ultralytics" not in sys.modules or ultralytics_loaded_before, (
        "spacers_agent.bootstrap imported ultralytics — YOLO must be lazy-loaded"
    )


def test_yolo_backend_import_does_not_load_ultralytics():
    """Importing the yolo_obb module must NOT load ultralytics (only the class)."""
    ultralytics_loaded_before = "ultralytics" in sys.modules
    from spacers_agent.agents.counting.backends.yolo_obb import YoloOBBCountingBackend  # noqa: F401
    assert "ultralytics" not in sys.modules or ultralytics_loaded_before, (
        "Importing YoloOBBCountingBackend imported ultralytics — model must be lazy-loaded"
    )


# ── Agent ↔ CLI boundary / Agent ↔ CLI 边界 ─────────────────────────────


def test_agent_modules_do_not_import_cli():
    """No agent module imports from cli.py or CLI-related modules."""
    import ast
    from pathlib import Path

    agents_root = Path(__file__).resolve().parents[2] / "spacers_agent" / "agents"
    violations: list[str] = []

    for py_file in agents_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "cli" in module.lower() and "client" not in module.lower():
                    violations.append(f"{py_file.name}: from {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "cli" in alias.name.lower() and "client" not in alias.name.lower():
                        violations.append(f"{py_file.name}: import {alias.name}")

    assert not violations, (
        "Agent modules must not import CLI:\n" + "\n".join(violations)
    )


def test_non_counting_agent_modules_do_not_import_legacy_workflow():
    """Cut-over non-counting Agents must not depend on transitional workflow.
    已切换的非计数 Agent 不得依赖过渡期 workflow。
    """

    import ast
    from pathlib import Path

    agents_root = Path(__file__).resolve().parents[2] / "spacers_agent" / "agents"
    violations: list[str] = []
    for py_file in agents_root.rglob("*.py"):
        if "counting" in py_file.parts:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "spacers_agent.workflow":
                violations.append(py_file.as_posix())
            elif isinstance(node, ast.Import):
                if any(alias.name == "spacers_agent.workflow" for alias in node.names):
                    violations.append(py_file.as_posix())

    assert not violations, "Agent modules import legacy workflow:\n" + "\n".join(violations)


def test_count_image_enters_the_composed_runtime():
    """The direct counting CLI must use the production composition root.
    直接计数 CLI 必须使用生产组合根。
    """

    from pathlib import Path

    command_path = Path(__file__).resolve().parents[2] / "spacers_agent" / "commands" / "count_image.py"
    source = command_path.read_text(encoding="utf-8")
    assert "assemble_runtime" in source
    assert "CountingWorkflow" not in source
    assert "PointCountingOrchestrator" not in source


def test_workflow_module_is_a_thin_compatibility_layer():
    """The old workflow import module must not define a DatasetRunner implementation.
    旧 workflow 导入模块不得定义 DatasetRunner 实现。
    """

    import ast
    from pathlib import Path

    workflow_path = Path(__file__).resolve().parents[2] / "spacers_agent" / "workflow.py"
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"))
    defined_classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "DatasetRunner" not in defined_classes


def test_router_does_not_import_yolo_classes():
    """TaskRouter and its submodules must not import YOLO classes."""
    import ast
    from pathlib import Path

    routing_root = Path(__file__).resolve().parents[2] / "spacers_agent" / "routing"
    violations: list[str] = []

    for py_file in routing_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "yolo" in module.lower():
                    violations.append(f"{py_file.name}: from {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "yolo" in alias.name.lower():
                        violations.append(f"{py_file.name}: import {alias.name}")

    assert not violations, (
        "Routing modules must not import YOLO classes:\n" + "\n".join(violations)
    )


# ── Registry / 注册表 ──────────────────────────────────────────────────────


def test_agent_registry_empty_construction_does_nothing():
    """AgentRegistry() creates an empty registry with no side effects."""
    from spacers_agent.agents.registry import AgentRegistry
    reg = AgentRegistry()
    assert reg.names() == ()


def test_agent_names_are_unique():
    """All registered agents must have unique names."""
    from spacers_agent.agents.change.agent import ChangeAgent
    from spacers_agent.agents.grounding.agent import GroundingAgent
    from spacers_agent.agents.spatial.agent import SpatialAgent
    from spacers_agent.agents.general_vqa.agent import GeneralVQAAgent
    from spacers_agent.agents.caption.agent import CaptionAgent
    from spacers_agent.agents.counting.agent import CountingAgent

    agents = [
        ChangeAgent,
        GroundingAgent,
        SpatialAgent,
        GeneralVQAAgent,
        CaptionAgent,
        CountingAgent,
    ]
    names = [cls.name for cls in agents if hasattr(cls, "name")]
    assert len(names) == len(set(names)), f"Duplicate agent names: {names}"


# ── AgentExecution / AgentExecution ─────────────────────────────────────────


def test_agent_execution_rejects_absolute_path():
    """result_filename must not be an absolute path."""
    from spacers_agent.agents.base import AgentExecution
    import pytest

    with pytest.raises(ValueError, match="plain basename"):
        AgentExecution(
            agent_name="change_agent",
            payload=None,
            result_filename="/etc/passwd",
        )


def test_agent_execution_rejects_parent_ref():
    """result_filename must not contain '..'."""
    from spacers_agent.agents.base import AgentExecution
    import pytest

    with pytest.raises(ValueError, match="not contain '..'"):
        AgentExecution(
            agent_name="change_agent",
            payload=None,
            result_filename="../expert_result.json",
        )


def test_agent_execution_accepts_valid_filenames():
    """result_filename allows counting_result.json and expert_result.json."""
    from spacers_agent.agents.base import AgentExecution

    for name in ("counting_result.json", "expert_result.json"):
        exec_result = AgentExecution(
            agent_name="change_agent",
            payload=None,
            result_filename=name,
        )
        assert exec_result.result_filename == name


def test_agent_context_no_api_keys():
    """AgentContext must not leak API keys through __repr__ or dict conversion."""
    from spacers_agent.agents.base import AgentContext
    from pathlib import Path

    ctx = AgentContext(
        artifact_dir=Path("/tmp/test"),
        settings=None,
        qwen_client=None,
        call_budget=None,
    )
    repr_str = repr(ctx)
    assert "sk-" not in repr_str.lower()
    assert "api_key" not in repr_str.lower()


def test_agent_execution_trace_rejects_sensitive_keys():
    """AgentExecution trace validation rejects sensitive keys (api_key etc)."""
    from spacers_agent.agents.base import AgentExecution
    import pytest

    with pytest.raises(ValueError, match="sensitive key"):
        AgentExecution(
            agent_name="change_agent",
            payload=None,
            result_filename="expert_result.json",
            trace={"authorization": "Bearer sk-fake-key-for-test-only"},
        )
