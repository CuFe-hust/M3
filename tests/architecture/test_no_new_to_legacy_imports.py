"""New code must never import the legacy packages (spacers_agent, eval).

新代码（新顶层包、main.py、tests）不得导入旧包；旧包自身内部不受此约束。
迁移期旧包允许存在，但新包不得反向依赖旧包。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "build", "dist", "tmp"}
LEGACY_TOP_LEVEL = ("spacers_agent", "eval")
SCAN_DIRS = (
    "application",
    "data",
    "models",
    "agents",
    "routing",
    "workflows",
    "evaluation",
    "reporting",
    "tests",
)


def _iter_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def _scan_files() -> list[Path]:
    files = [REPO_ROOT / "main.py"]
    for directory in SCAN_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not any(part in SKIP_DIRS for part in path.relative_to(root).parts):
                files.append(path)
    return sorted(files)


def test_new_code_never_imports_legacy_packages() -> None:
    violations = []
    for path in _scan_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module_top in _iter_imports(tree):
            if module_top in LEGACY_TOP_LEVEL:
                violations.append(f"{path.as_posix()}: imports {module_top}")
    assert not violations, "New code must not import legacy packages:\n" + "\n".join(violations)
