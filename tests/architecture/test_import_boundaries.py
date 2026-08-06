"""Enforce the import dependency DAG for the new top-level packages.

使用 AST 解析 import（不执行模块），按 architecture/import_rules.json 的允许
依赖集合校验每个新包文件；同一包内互引、标准库与第三方导入不受约束。
扫描只覆盖已存在且非空的实现文件，不扫描空壳。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPO_ROOT / "architecture" / "import_rules.json"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "build", "dist", "tmp"}
NEW_PACKAGE_DIRS = (
    "application",
    "data",
    "models",
    "agents",
    "routing",
    "workflows",
    "evaluation",
    "reporting",
)


def _rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _iter_imports(tree: ast.AST):
    """Yield full module paths, not just top-level names.
    产出完整模块路径，而非仅顶层名。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module


def _is_allowed(module: str, allow: list[str]) -> bool:
    for entry in allow:
        if module == entry or module.startswith(entry + "."):
            return True
    return False


def _glob_to_regex(pattern: str) -> re.Pattern:
    chunks = pattern.split("**")
    parts = []
    for index, chunk in enumerate(chunks):
        parts.append(re.escape(chunk).replace(r"\*", r"[^.]*"))
        if index < len(chunks) - 1:
            parts.append(".*")
    return re.compile("^" + "".join(parts) + "$")


def _files_for_package(package: str) -> list[Path]:
    if package == "main":
        main = REPO_ROOT / "main.py"
        return [main] if main.is_file() and main.stat().st_size > 0 else []
    root = REPO_ROOT / package
    if not root.is_dir():
        return []
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.stat().st_size > 0:
            files.append(path)
    return sorted(files)


def test_import_rules_file_is_valid_json() -> None:
    rules = _rules()
    assert "packages" in rules and "internal_packages" in rules
    for package in ("data", "models", "agents", "routing", "workflows",
                    "evaluation", "reporting", "application", "main"):
        assert package in rules["packages"], f"missing allow rules for {package}"


def test_internal_packages_exclude_legacy_packages() -> None:
    internal = set(_rules()["internal_packages"])
    assert "spacers_agent" not in internal and "eval" not in internal


def test_every_new_package_respects_its_allowed_dependencies() -> None:
    rules = _rules()
    internal = set(rules["internal_packages"])
    violations = []
    for package in NEW_PACKAGE_DIRS + ("main",):
        allow = rules["packages"][package]["allow"]
        for path in _files_for_package(package):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for module in _iter_imports(tree):
                module_top = module.split(".")[0]
                if module_top not in internal:
                    continue
                if not _is_allowed(module, allow):
                    violations.append(
                        f"{path.as_posix()}: imports {module} (allowed: {allow})"
                    )
    assert not violations, "Dependency violations:\n" + "\n".join(violations)


def test_forbidden_patterns_are_enforced() -> None:
    rules = _rules()
    forbidden = rules.get("forbidden_patterns", {})
    violations = []
    for package in NEW_PACKAGE_DIRS:
        patterns = [(pattern, _glob_to_regex(pattern)) for pattern in forbidden.get(package, [])]
        if not patterns:
            continue
        for path in _files_for_package(package):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for module in _iter_imports(tree):
                if any(regex.match(module) for _, regex in patterns):
                    violations.append(f"{path.as_posix()}: imports forbidden {module}")
    assert not violations, "Forbidden import patterns:\n" + "\n".join(violations)


def test_application_may_import_every_new_package() -> None:
    rules = _rules()
    allow = set(rules["packages"]["application"]["allow"])
    for package in NEW_PACKAGE_DIRS:
        assert package in allow, f"application must be allowed to import {package}"


def test_main_imports_only_application() -> None:
    rules = _rules()
    allow = rules["packages"]["main"]["allow"]
    assert allow == ["application"], "main.py must import only application"
    files = _files_for_package("main")
    if not files:
        return  # main.py is not implemented yet. / main.py 尚未实现。
    internal = set(rules["internal_packages"])
    tree = ast.parse(files[0].read_text(encoding="utf-8"))
    for module in _iter_imports(tree):
        if module.split(".")[0] in internal:
            assert _is_allowed(module, allow), f"main.py imports forbidden {module}"
