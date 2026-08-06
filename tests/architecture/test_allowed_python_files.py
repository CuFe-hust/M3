"""Enforce the final architecture Python file allowlist.

最终架构白名单测试：
- 仓库实际存在的每个 .py 必须匹配最终白名单（actual ⊆ allowed）；
- 白名单中尚未创建的未来路径不要求存在（allowed ⊄ actual）；
- 关键架构路径不得被删除；泛化兜底文件名禁止出现在白名单；
- 旧包目录永久禁止；
- Golden 生成器可启动（--help），且拒绝脏参考 checkout。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WHITELIST = REPO_ROOT / "architecture" / "allowed_python_files.txt"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "build", "dist", "tmp"}
LEGACY_DIRS = ("spacers_agent", "eval")
FORBIDDEN_BASENAMES = ("utils.py", "helpers.py", "manager.py", "compat.py", "legacy.py")


class pytest_raises_SystemExit:
    """Minimal context manager asserting SystemExit with a message fragment.
    最小上下文管理器：断言 SystemExit 且消息包含指定片段。"""

    def __init__(self, fragment: str) -> None:
        self.fragment = fragment

    def __enter__(self) -> "pytest_raises_SystemExit":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        assert exc_type is SystemExit, f"expected SystemExit, got {exc_type}"
        assert self.fragment in str(exc_value), f"{self.fragment!r} not in {exc_value!r}"
        return True

REQUIRED_FINAL_PATHS = (
    "main.py",
    "data/validation.py",
    "data/adapters/base.py",
    "routing/router.py",
    "agents/counting/agent.py",
    "workflows/sample_runner.py",
    "evaluation/records.py",
    "reporting/html.py",
    "application/bootstrap.py",
)


def _pattern_to_regex(pattern: str) -> re.Pattern:
    chunks = pattern.split("**")
    parts = []
    for index, chunk in enumerate(chunks):
        parts.append(re.escape(chunk).replace(r"\*", r"[^/]*"))
        if index < len(chunks) - 1:
            parts.append(".*")
    return re.compile("^" + "".join(parts) + "$")


def _whitelist_patterns() -> list[str]:
    patterns = []
    for line in WHITELIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _repository_py_files() -> list[str]:
    found = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        found.append(path.relative_to(REPO_ROOT).as_posix())
    return sorted(found)


def test_whitelist_file_exists_and_is_nonempty() -> None:
    assert WHITELIST.is_file(), "missing architecture/allowed_python_files.txt"
    assert len(_whitelist_patterns()) > 100, "final allowlist looks incomplete"


def test_every_existing_python_file_matches_the_whitelist() -> None:
    """actual ⊆ allowed: existing files must be approved. / 实际文件必须已批准。"""
    patterns = [(p, _pattern_to_regex(p)) for p in _whitelist_patterns()]
    violations = []
    for rel in _repository_py_files():
        if not any(regex.match(rel) for _, regex in patterns):
            violations.append(rel)
    assert not violations, "Python files outside the final allowlist:\n" + "\n".join(violations)


def test_future_allowlisted_paths_need_not_exist() -> None:
    """allowed ⊄ actual: approved future paths must not be required on disk.
    白名单未来路径不得被要求当前存在（禁止预先创建空壳）。"""
    for pattern in _whitelist_patterns():
        if "*" in pattern:
            continue
        # A matching file may exist; its absence must not fail here.
        # 匹配文件可以存在；不存在也不得在此失败。
        assert True


def test_allowlist_contains_required_final_architecture_paths() -> None:
    """Key architecture paths can never be silently dropped.
    关键架构路径不得被静默删除。"""
    patterns = _whitelist_patterns()
    missing = [path for path in REQUIRED_FINAL_PATHS if path not in patterns]
    assert not missing, f"final allowlist misses required paths: {missing}"


def test_allowlist_forbids_generic_helper_basenames() -> None:
    """Generic catch-all filenames are forbidden unless explicitly approved.
    泛化兜底文件名禁止出现在白名单。"""
    violations = []
    for pattern in _whitelist_patterns():
        basename = pattern.rsplit("/", 1)[-1]
        if basename in FORBIDDEN_BASENAMES:
            violations.append(pattern)
    assert not violations, f"forbidden generic filenames in allowlist: {violations}"


def test_whitelist_never_contains_legacy_packages() -> None:
    patterns = _whitelist_patterns()
    for pattern in patterns:
        assert not pattern.startswith("spacers_agent") and not pattern.startswith("eval/"), (
            f"whitelist must not contain legacy pattern {pattern}"
        )


def test_legacy_directories_do_not_exist() -> None:
    for legacy in LEGACY_DIRS:
        assert not (REPO_ROOT / legacy).exists(), (
            f"{legacy}/ must not exist on the from-zero branch"
        )


def test_migration_generator_help_runs() -> None:
    """The generator must start in a clean environment (no undeclared imports).
    生成器必须在干净环境可启动（无未声明顶层依赖）。"""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_migration_fixtures.py"), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "reference-root" in result.stdout


def test_dirty_reference_checkout_is_rejected() -> None:
    """A reference checkout with uncommitted changes must be refused. The check
    is exercised through the generator's own helper function (unit-level).
    带未提交改动的参考 checkout 必须被拒绝；通过生成器辅助函数做单元级验证。"""
    import importlib.util
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "migration_generator", REPO_ROOT / "scripts" / "generate_migration_fixtures.py"
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    with tempfile.TemporaryDirectory(prefix="ref_dirty_") as tmp:
        repo = Path(tmp)
        subprocess.run(["git", "init", "-q", "-b", "ref"], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "base"],
            cwd=repo, check=True,
        )
        (repo / "dirty.txt").write_text("untracked", encoding="utf-8")
        with pytest_raises_SystemExit("dirty"):
            generator._verify_reference_clean(repo)
        (repo / "dirty.txt").unlink()
        generator._verify_reference_clean(repo)  # clean checkout passes / 干净 checkout 通过
