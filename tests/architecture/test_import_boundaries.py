"""Enforce the import dependency DAG for the new top-level packages.

使用 AST 解析 import（不执行模块），按 architecture/import_rules.json 的允许
依赖集合校验每个新包文件；同一包内互引、标准库与第三方导入不受约束。
"""

from __future__ import annotations

import ast
import json
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
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def _is_allowed(module_top: str, allow: list[str]) -> bool:
    for entry in allow:
        if module_top == entry or module_top.startswith(entry + "."):
            return True
    return False


def _files_for_package(package: str) -> list[Path]:
    if package == "main":
        return [REPO_ROOT / "main.py"]
    root = REPO_ROOT / package
    return [p for p in root.rglob("*.py") if not any(part in SKIP_DIRS for part in p.relative_to(root).parts)]


def test_import_rules_file_is_valid_json() -> None:
    rules = _rules()
    assert "packages" in rules and "internal_packages" in rules
    for package in ("data", "models", "agents", "routing", "workflows",
                    "evaluation", "reporting", "application", "main"):
        assert package in rules["packages"], f"missing allow rules for {package}"


def test_every_new_package_respects_its_allowed_dependencies() -> None:
    rules = _rules()
    internal = set(rules["internal_packages"])
    violations = []
    for package in NEW_PACKAGE_DIRS + ("main",):
        allow = rules["packages"][package]["allow"]
        for path in _files_for_package(package):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for module_top in _iter_imports(tree):
                if module_top not in internal:
                    continue
                if not _is_allowed(module_top, allow):
                    violations.append(
                        f"{path.as_posix()}: imports {module_top} (allowed: {allow})"
                    )
    assert not violations, "Dependency violations:\n" + "\n".join(violations)


def test_application_may_import_every_new_package() -> None:
    rules = _rules()
    allow = set(rules["packages"]["application"]["allow"])
    for package in NEW_PACKAGE_DIRS:
        assert package in allow, f"application must be allowed to import {package}"


def test_main_imports_only_application() -> None:
    rules = _rules()
    allow = rules["packages"]["main"]["allow"]
    assert allow == ["application"], "main.py must import only application"
    tree = ast.parse((REPO_ROOT / "main.py").read_text(encoding="utf-8"))
    internal = set(rules["internal_packages"])
    for module_top in _iter_imports(tree):
        if module_top in internal:
            assert _is_allowed(module_top, allow), f"main.py imports forbidden {module_top}"
