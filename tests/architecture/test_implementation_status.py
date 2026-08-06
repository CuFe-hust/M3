"""Implementation-status guard: implemented files must exist and be non-empty;
pending files must never be imported or exported by production code.

实施状态守卫：implemented 文件必须存在且非空；pending 文件不得被生产代码
import，也不得从任何 __init__.py 导出；仓库内不允许出现未批准的 0 字节文件。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = REPO_ROOT / "architecture" / "implementation_status.json"
WHITELIST_PATH = REPO_ROOT / "architecture" / "allowed_python_files.txt"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "build", "dist", "tmp"}


def _status() -> dict:
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))


def _whitelist_patterns() -> list[str]:
    patterns = []
    for line in WHITELIST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _production_py_files() -> list[Path]:
    files = []
    for path in REPO_ROOT.rglob("*.py"):
        parts = path.relative_to(REPO_ROOT).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        if parts[0] == "tests":
            continue
        files.append(path)
    return sorted(files)


def test_status_json_is_valid() -> None:
    status = _status()
    for key in ("completed_tasks", "implemented_files", "pending_files", "reference_commit"):
        assert key in status, f"missing key {key}"
    assert status["reference_commit"] == "ec962eb87c3ad0b8c1502efcbd08db0daec48868"
    assert isinstance(status["implemented_files"], list)
    assert isinstance(status["pending_files"], list)


def test_implemented_files_exist_and_are_nonempty() -> None:
    for rel in _status()["implemented_files"]:
        path = REPO_ROOT / rel
        assert path.is_file(), f"implemented file missing: {rel}"
        assert path.stat().st_size > 0, f"implemented file is empty: {rel}"


def test_pending_and_implemented_are_disjoint() -> None:
    status = _status()
    assert not set(status["implemented_files"]) & set(status["pending_files"])


def test_pending_files_are_not_imported_by_production_code() -> None:
    pending = set(_status()["pending_files"])
    if not pending:
        return
    pending_modules = {rel[:-3].replace("/", ".") for rel in pending}
    violations = []
    for path in _production_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name == module or alias.name.startswith(module + ".")
                           for module in pending_modules):
                        violations.append(f"{path.as_posix()}: imports pending {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if any(node.module == module or node.module.startswith(module + ".")
                       for module in pending_modules):
                    violations.append(f"{path.as_posix()}: imports pending {node.module}")
    assert not violations, "pending files must not be imported:\n" + "\n".join(violations)


def test_task3_files_are_implemented() -> None:
    implemented = set(_status()["implemented_files"])
    assert "data/schema.py" in implemented
    assert "data/__init__.py" in implemented


def test_no_unapproved_empty_python_files() -> None:
    """Every empty .py must be explicitly whitelisted and must not be an implemented file.
    每个 0 字节 .py 必须显式列入白名单，且不得是 implemented 文件。"""
    patterns = _whitelist_patterns()
    implemented = set(_status()["implemented_files"])
    violations = []
    for path in REPO_ROOT.rglob("*.py"):
        parts = path.relative_to(REPO_ROOT).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        if path.stat().st_size > 0:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in implemented:
            violations.append(f"{rel}: implemented file is empty")
        if rel not in patterns:
            violations.append(f"{rel}: empty file not in whitelist")
    assert not violations, "unapproved empty Python files:\n" + "\n".join(violations)
