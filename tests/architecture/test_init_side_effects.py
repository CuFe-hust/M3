"""`__init__.py` files must only re-export; no side effects at import time.

`__init__.py` 只允许导出：函数/类体外不得有调用、文件读取、模型加载、网络
调用、条件导入或 `import *`。旧包（spacers_agent/eval）不受本测试约束。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "build", "dist", "tmp"}
LEGACY_DIRS = ("spacers_agent", "eval")
FORBIDDEN_STATEMENTS = (
    ast.Call,
    ast.Try,
    ast.If,
    ast.With,
    ast.For,
    ast.While,
    ast.Delete,
    ast.AugAssign,
    ast.Raise,
    ast.Assert,
    ast.Global,
    ast.Nonlocal,
    ast.Yield,
)
ALLOWED_VALUE_NODES = (ast.Name, ast.Attribute, ast.Constant, ast.List, ast.Tuple, ast.Subscript)


def _value_is_export_only(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_value_is_export_only(item) for item in node.elts)
    return isinstance(node, ALLOWED_VALUE_NODES)


def _init_files() -> list[Path]:
    files = []
    for path in REPO_ROOT.rglob("__init__.py"):
        parts = path.relative_to(REPO_ROOT).parts
        if any(part in SKIP_DIRS or part in LEGACY_DIRS for part in parts):
            continue
        files.append(path)
    return sorted(files)


def test_init_files_have_no_import_time_side_effects() -> None:
    violations = []
    for path in _init_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue  # module docstring / 模块 docstring
            if isinstance(node, (ast.Pass, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # definitions are inert / 定义为惰性
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                    violations.append(f"{path.as_posix()}: import *")
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value if isinstance(node, ast.Assign) else node.value
                if value is None:
                    continue  # bare annotation / 纯注解
                if isinstance(value, ast.AST) and not _value_is_export_only(value):
                    violations.append(
                        f"{path.as_posix()}: assignment value {type(value).__name__}"
                    )
                continue
            violations.append(f"{path.as_posix()}: top-level {type(node).__name__}")
    assert not violations, "`__init__.py` import-time side effects:\n" + "\n".join(violations)


def test_every_new_package_has_an_init_file() -> None:
    top_level = set()
    for path in REPO_ROOT.rglob("*.py"):
        parts = path.relative_to(REPO_ROOT).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        top_level.add(parts[0])
    for package in sorted(top_level):
        if package in {"tests"} or package.startswith("."):
            continue
        if (REPO_ROOT / package).is_dir():
            assert (REPO_ROOT / package / "__init__.py").is_file(), (
                f"{package}/ must ship __init__.py"
            )
