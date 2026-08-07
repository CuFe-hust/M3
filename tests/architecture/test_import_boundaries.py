"""Enforce the import dependency DAG for the new top-level packages.

使用 AST 解析 import（不执行模块），按 architecture/import_rules.json 的允许
依赖集合校验每个新包文件；同一包内互引、标准库与第三方导入不受约束。
path_rules 比 package 规则更具体：匹配文件路径的 path rule 优先，否则回退到
package 规则；future judge/metrics 文件不存在时不影响测试。扫描只覆盖已存在
且非空的实现文件，不扫描空壳。领域层只能依赖模型协议（models.base/
models.images），具体模型实现（models.entry / models.qwen_transformers /
models.qwen3_*）只允许 composition root（application）选择。
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


def _resolve_allow(rules: dict, package: str, relative_path: str) -> list[str]:
    """Path rules take precedence over package rules; without a matching path
    rule the package rule is the fallback. path rule 优先于 package 规则；没有
    匹配的 path rule 时回退到 package 规则。"""
    for pattern in rules.get("path_rules", {}):
        if _glob_to_regex(pattern).match(relative_path):
            return rules["path_rules"][pattern]["allow"]
    return rules["packages"][package]["allow"]


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
    path_rules = rules.get("path_rules", {})
    assert isinstance(path_rules, dict)
    for pattern, rule in path_rules.items():
        assert "allow" in rule, f"path rule {pattern} must carry allow"


def test_internal_packages_exclude_legacy_packages() -> None:
    internal = set(_rules()["internal_packages"])
    assert "spacers_agent" not in internal and "eval" not in internal


def test_every_new_package_respects_its_allowed_dependencies() -> None:
    rules = _rules()
    internal = set(rules["internal_packages"])
    violations = []
    for package in NEW_PACKAGE_DIRS + ("main",):
        for path in _files_for_package(package):
            relative = path.relative_to(REPO_ROOT).as_posix()
            allow = _resolve_allow(rules, package, relative)
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


# ── path rule 优先级 / path-rule precedence ────────────────────────────────


def test_path_rules_take_precedence_over_package_rules() -> None:
    """A matching path rule wins over the package rule; unmatched files fall
    back to the package rule. 匹配的 path rule 优先于 package 规则；未匹配
    文件回退到 package 规则。"""
    rules = _rules()
    metrics = _resolve_allow(rules, "evaluation", "evaluation/metrics/counting.py")
    judges = _resolve_allow(rules, "evaluation", "evaluation/judges/base.py")
    fallback = _resolve_allow(rules, "evaluation", "evaluation/records.py")
    # The metrics path rule forbids models even though nothing else would.
    # metrics path rule 禁止 models。
    assert not _is_allowed("models.base", metrics)
    # The judges path rule re-opens only the approved model contracts.
    # judges path rule 只开放批准的模型契约。
    assert _is_allowed("models.base", judges)
    # Uncovered files fall back to the package rule (which has no models).
    # 未覆盖文件回退到 package 规则（无 models）。
    assert fallback == rules["packages"]["evaluation"]["allow"]
    assert not _is_allowed("models.base", fallback)


def test_future_judge_paths_are_covered_without_existing_on_disk() -> None:
    """A future judge/metrics file resolves through its path rule even before
    the file exists; absent files never fail the scan. 未来 judge/metrics 文件
    即使不存在也按其 path rule 解析；不存在的文件绝不导致扫描失败。"""
    rules = _rules()
    assert _is_allowed("models.base", _resolve_allow(rules, "evaluation", "evaluation/judges/__init__.py"))
    assert _is_allowed("models.cache", _resolve_allow(rules, "evaluation", "evaluation/judges/deepseek.py"))
    assert not _is_allowed("models.base", _resolve_allow(rules, "evaluation", "evaluation/metrics/aggregate.py"))
    # No judge implementation file exists yet, and none is required.
    # 尚不存在任何 judge 实现文件，也不要求存在。
    assert not (REPO_ROOT / "evaluation" / "judges" / "deepseek.py").is_file()


# ── 领域层只依赖模型契约 / domain layers depend on model contracts only ───


def test_agents_never_import_concrete_model_implementations() -> None:
    rules = _rules()
    allow = rules["packages"]["agents"]["allow"]
    assert _is_allowed("models.base", allow)
    assert _is_allowed("models.images", allow)
    for forbidden in (
        "models.qwen_transformers",
        "models.entry",
        "models.qwen3_vl",
        "models.qwen3_vl.baseline",
        "models.qwen3_5",
        "models.qwen3_5.model",
    ):
        assert not _is_allowed(forbidden, allow), forbidden


def test_routing_never_imports_models() -> None:
    rules = _rules()
    allow = rules["packages"]["routing"]["allow"]
    assert not _is_allowed("models", allow)
    assert not _is_allowed("models.base", allow)
    assert not _is_allowed("models.entry", allow)


def test_workflows_import_model_contracts_only() -> None:
    rules = _rules()
    allow = rules["packages"]["workflows"]["allow"]
    assert _is_allowed("models.base", allow)
    for forbidden in ("models.qwen_transformers", "models.entry", "models.qwen3_vl.baseline"):
        assert not _is_allowed(forbidden, allow), forbidden


def test_evaluation_metrics_have_no_model_dependency() -> None:
    rules = _rules()
    allow = _resolve_allow(rules, "evaluation", "evaluation/metrics/counting.py")
    for forbidden in ("models", "models.base", "models.cache", "models.settings"):
        assert not _is_allowed(forbidden, allow), forbidden


def test_evaluation_judges_may_import_approved_model_contracts() -> None:
    rules = _rules()
    allow = _resolve_allow(rules, "evaluation", "evaluation/judges/deepseek.py")
    for approved in ("models.base", "models.cache", "models.settings"):
        assert _is_allowed(approved, allow), approved
    for forbidden in ("models.qwen_transformers", "models.entry", "models.qwen3_vl.baseline"):
        assert not _is_allowed(forbidden, allow), forbidden


def test_reporting_never_imports_agent_implementations() -> None:
    rules = _rules()
    allow = rules["packages"]["reporting"]["allow"]
    assert not _is_allowed("agents.caption.agent", allow)
    assert not _is_allowed("agents.counting.agent", allow)
    pattern = _glob_to_regex("agents.*.agent")
    assert pattern.match("agents.caption.agent")
    assert pattern.match("agents.counting.agent")
    assert not pattern.match("agents.schema")


def test_application_remains_composition_root() -> None:
    """Only the composition root may select concrete model implementations.
    只有 composition root 可以选择具体模型实现。"""
    rules = _rules()
    allow = rules["packages"]["application"]["allow"]
    for package in NEW_PACKAGE_DIRS:
        assert package in allow, f"application must be allowed to import {package}"
    assert _is_allowed("models.entry", allow)
    assert _is_allowed("models.qwen_transformers", allow)
    # No other package may import models.entry. / 其他包都不得 import models.entry。
    for package in ("agents", "workflows", "evaluation", "routing", "reporting"):
        assert not _is_allowed("models.entry", rules["packages"][package]["allow"]), package


def test_forbidden_patterns_are_enforced() -> None:
    rules = _rules()
    forbidden = rules.get("forbidden_patterns", {})
    internal = set(rules["internal_packages"])
    violations = []
    for package in NEW_PACKAGE_DIRS:
        patterns = [(pattern, _glob_to_regex(pattern)) for pattern in forbidden.get(package, [])]
        if not patterns:
            continue
        for path in _files_for_package(package):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for module in _iter_imports(tree):
                # Forbidden patterns constrain internal packages only; stdlib
                # modules (e.g. dataclasses) can never violate them.
                # forbidden pattern 只约束内部包；标准库模块绝不构成违规。
                if module.split(".")[0] not in internal:
                    continue
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
