"""Package-discovery guards: the wheel must ship every implemented package.

打包发现守卫：wheel 必须包含全部已实现顶层包（含 routing），CI 编译与
wheel smoke 必须覆盖 routing，且 package discovery 与 implementation status
保持一致。
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def _workflow() -> str:
    return (
        REPO_ROOT / ".github" / "workflows" / "offline-tests.yml"
    ).read_text(encoding="utf-8")


def _status() -> dict:
    return json.loads(
        (REPO_ROOT / "architecture" / "implementation_status.json").read_text(
            encoding="utf-8"
        )
    )


def _implemented_top_level_packages() -> set[str]:
    packages: set[str] = set()
    for rel in _status()["implemented_files"]:
        top = rel.split("/", 1)[0]
        if top not in {"scripts", "main.py"}:
            packages.add(top)
    return packages


def test_routing_is_in_package_discovery() -> None:
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "routing*" in include, f"routing* missing from packages.find include: {include}"


def test_every_implemented_package_is_discovered() -> None:
    """Every implemented top-level package must appear in package discovery.
    每个已实现顶层包都必须出现在 package discovery 中。"""
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    implemented = _implemented_top_level_packages()
    for package in sorted(implemented):
        assert f"{package}*" in include, f"{package}* missing from packages.find include"


def test_ci_compileall_covers_routing() -> None:
    workflow = _workflow()
    assert "python -m compileall data models agents routing workflows evaluation tests" in workflow


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
