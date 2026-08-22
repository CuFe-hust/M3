"""Repository hygiene guards.

保留与路径审批无关的仓库卫生检查：旧包目录禁止重新出现，迁移生成器
必须可启动，并且必须拒绝带未提交改动的参考 checkout。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIRS = ("spacers_agent", "eval")


class pytest_raises_SystemExit:
    """Minimal context manager asserting SystemExit with a message fragment."""

    def __init__(self, fragment: str) -> None:
        self.fragment = fragment

    def __enter__(self) -> "pytest_raises_SystemExit":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        assert exc_type is SystemExit, f"expected SystemExit, got {exc_type}"
        assert self.fragment in str(exc_value), f"{self.fragment!r} not in {exc_value!r}"
        return True


def test_legacy_directories_do_not_exist() -> None:
    for legacy in LEGACY_DIRS:
        assert not (REPO_ROOT / legacy).exists(), (
            f"{legacy}/ must not exist on the from-zero branch"
        )


def test_migration_generator_help_runs() -> None:
    """The generator must start in a clean environment (no undeclared imports)."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_migration_fixtures.py"), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "reference-root" in result.stdout


def test_dirty_reference_checkout_is_rejected() -> None:
    """A reference checkout with uncommitted changes must be refused."""
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
        generator._verify_reference_clean(repo)
