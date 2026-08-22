"""Package-discovery guards: the wheel must ship every architecture package.

打包发现守卫：wheel 必须包含 import boundary 定义的全部顶层包（含
routing），CI 编译与 wheel smoke 必须覆盖 routing。
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IMPORT_RULES = REPO_ROOT / "architecture" / "import_rules.json"


def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def _workflow() -> str:
    return (
        REPO_ROOT / ".github" / "workflows" / "offline-tests.yml"
    ).read_text(encoding="utf-8")


def _architecture_top_level_packages() -> set[str]:
    """Return package roots from the import-boundary contract, not a file list."""
    rules = json.loads(IMPORT_RULES.read_text(encoding="utf-8"))
    return {package.split(".", 1)[0] for package in rules["internal_packages"]}


def test_routing_is_in_package_discovery() -> None:
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "routing*" in include, f"routing* missing from packages.find include: {include}"


def test_every_architecture_package_is_discovered() -> None:
    """Every import-boundary package must appear in package discovery."""
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    for package in sorted(_architecture_top_level_packages()):
        assert f"{package}*" in include, f"{package}* missing from packages.find include"


def test_ci_compileall_covers_routing() -> None:
    workflow = _workflow()
    assert (
        "python -m compileall data models agents routing workflows evaluation "
        "reporting application tests" in workflow
    )


def test_wheel_smoke_script_imports_routing() -> None:
    workflow = _workflow()
    assert "import routing" in workflow
    assert "from routing import" in workflow
    assert "TaskRouter().route(\"counting\")" in workflow
    assert "from agents.counting import CountingAgent, CountingResult" in workflow


def test_wheel_smoke_runs_outside_source_tree() -> None:
    workflow = _workflow()
    assert "cd /tmp" in workflow
    assert "wheel-import-contract: PASS" in workflow
    assert "counting-wheel-runtime: PASS" in workflow
    assert "counting-wheel-assets: PASS" in workflow


def test_ci_declares_build_dependency_without_stale_migration_extra() -> None:
    dev = _pyproject()["project"]["optional-dependencies"]["dev"]
    workflow = _workflow()

    assert any(item.startswith("build>=") for item in dev)
    assert ".[dev,change]" in workflow
    assert ".[dev,migration,change]" not in workflow


def test_required_counting_metadata_is_packaged_without_large_weights() -> None:
    config = _pyproject()["tool"]["setuptools"]
    package_data = config["package-data"]

    assert "prompts*" in config["packages"]["find"]["include"]
    assert "counting/expert_catalog.json" in package_data["agents"]
    assert set(package_data["models"]) >= {
        "segformer_mitb2_isaid/classes.json",
        "segformer_mitb2_isaid/config.json",
        "segformer_mitb2_isaid/preprocessor_config.json",
    }
    assert "*.md" in package_data["prompts"]
    serialized = json.dumps(package_data)
    assert ".safetensors" not in serialized
    assert ".onnx" not in serialized
    assert ".pt" not in serialized
