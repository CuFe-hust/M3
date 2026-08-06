"""Implementation-status guard with exact bidirectional coverage.

实施状态守卫（双向精确覆盖）：
- implemented_files 中的文件必须存在且非空；
- 实际存在的生产 .py 必须全部被 declared（implemented ∪ pending）声明；
- pending 文件不得被生产代码 import（绝对与相对导入均检测）；
- 仓库内不允许出现未批准的 0 字节文件。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = REPO_ROOT / "architecture" / "implementation_status.json"
WHITELIST_PATH = REPO_ROOT / "architecture" / "allowed_python_files.txt"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "build", "dist", "tmp"}
PRODUCTION_ROOTS = (
    "data",
    "models",
    "agents",
    "routing",
    "workflows",
    "evaluation",
    "reporting",
    "application",
)


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
    """All existing non-empty production .py files (excluding tests/scripts).
    实际存在的全部非空生产 .py（排除 tests/scripts）。"""
    files = []
    for root in PRODUCTION_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.relative_to(base).parts):
                continue
            files.append(path)
    main = REPO_ROOT / "main.py"
    if main.is_file():
        files.append(main)
    return sorted(files)


def _actual_production_relative_paths() -> set[str]:
    return {path.relative_to(REPO_ROOT).as_posix() for path in _production_py_files()}


def _declared_paths() -> set[str]:
    status = _status()
    return set(status["implemented_files"]) | set(status["pending_files"])


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


def test_implementation_status_exact_coverage() -> None:
    """actual production files == declared (implemented ∪ pending), with
    separate error listings for each direction.
    实际生产文件必须与 declared 精确相等；两个方向分别列出错误。"""
    actual = _actual_production_relative_paths()
    declared = _declared_paths()
    undeclared = sorted(actual - declared)
    missing = sorted(declared - actual)
    assert not undeclared, f"undeclared actual production files: {undeclared}"
    assert not missing, f"declared but missing files: {missing}"
    assert actual == declared


def _pending_modules() -> set[str]:
    pending = set(_status()["pending_files"])
    return {rel[:-3].replace("/", ".") for rel in pending}


def test_pending_files_are_not_imported_by_production_code() -> None:
    """Pending modules must not be imported, absolutely or relatively.
    pending 模块不得被 import（绝对与相对导入均检测）。"""
    pending_modules = _pending_modules()
    violations = []
    for path in _production_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        package_parts = list(path.relative_to(REPO_ROOT).parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name == module or alias.name.startswith(module + ".")
                           for module in pending_modules):
                        violations.append(f"{path.as_posix()}: imports pending {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    if any(node.module == module or node.module.startswith(module + ".")
                           for module in pending_modules):
                        violations.append(f"{path.as_posix()}: imports pending {node.module}")
                elif node.level > 0:
                    base = package_parts[: len(package_parts) - (node.level - 1)]
                    parts = list(base)
                    if node.module:
                        parts += node.module.split(".")
                    candidate = ".".join(parts)
                    if any(candidate == module or candidate.startswith(module + ".")
                           for module in pending_modules):
                        violations.append(f"{path.as_posix()}: relative import of pending {candidate}")
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
