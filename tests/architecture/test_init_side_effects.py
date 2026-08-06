"""`__init__.py` files must only re-export; no side effects at import time.

`__init__.py` 只允许：模块 docstring、import / from-import（禁止 import *）、
__all__ 赋值、简单版本字符串常量、TYPE_CHECKING 块（块内只允许 import）。
不允许函数/类定义、条件注册、文件读取、模型加载、网络调用、Registry 实例化。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "build", "dist", "tmp"}
LEGACY_DIRS = ("spacers_agent", "eval")


def _init_files() -> list[Path]:
    files = []
    for path in REPO_ROOT.rglob("__init__.py"):
        parts = path.relative_to(REPO_ROOT).parts
        if any(part in SKIP_DIRS or part in LEGACY_DIRS for part in parts):
            continue
        files.append(path)
    return sorted(files)


def _check_top_level(node: ast.AST, path: Path, violations: list[str]) -> None:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return  # module docstring / 模块 docstring
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            violations.append(f"{path.as_posix()}: import *")
        return
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        name = target.id if isinstance(target, ast.Name) else None
        value = node.value
        if name == "__all__" and isinstance(value, ast.List) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts
        ):
            return
        if name == "__version__" and isinstance(value, ast.Constant) and isinstance(value.value, str):
            return
        violations.append(f"{path.as_posix()}: disallowed assignment to {name!r}")
        return
    if isinstance(node, ast.If):
        test = node.test
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            for child in node.body:
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    if isinstance(child, ast.ImportFrom) and any(a.name == "*" for a in child.names):
                        violations.append(f"{path.as_posix()}: TYPE_CHECKING import *")
                    continue
                violations.append(f"{path.as_posix()}: TYPE_CHECKING block contains {type(child).__name__}")
            if node.orelse:
                violations.append(f"{path.as_posix()}: TYPE_CHECKING else branch")
            return
        violations.append(f"{path.as_posix()}: conditional block {type(node.test).__name__}")
        return
    violations.append(f"{path.as_posix()}: top-level {type(node).__name__}")


def test_init_files_have_no_import_time_side_effects() -> None:
    violations = []
    for path in _init_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            _check_top_level(node, path, violations)
    assert not violations, "`__init__.py` import-time side effects:\n" + "\n".join(violations)


def test_no_function_or_class_definitions_in_init_files() -> None:
    violations = []
    for path in _init_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                violations.append(f"{path.as_posix()}: defines {type(node).__name__} {node.name}")
    assert not violations, "`__init__.py` must not define functions or classes:\n" + "\n".join(violations)
