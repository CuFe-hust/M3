"""Enforce the Python file whitelist and the final-mode legacy deletion.

清单外新增 .py 必须失败；最终模式（M3_ARCH_FINAL=1）下旧包目录必须不存在。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WHITELIST = REPO_ROOT / "architecture" / "allowed_python_files.txt"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "build", "dist", "tmp"}
LEGACY_DIRS = ("spacers_agent", "eval")
FINAL_MODE_ENV = "M3_ARCH_FINAL"


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
    assert len(_whitelist_patterns()) > 10, "whitelist looks empty"


def test_every_python_file_matches_the_whitelist() -> None:
    patterns = [(p, _pattern_to_regex(p)) for p in _whitelist_patterns()]
    violations = []
    for rel in _repository_py_files():
        if not any(regex.match(rel) for _, regex in patterns):
            violations.append(rel)
    assert not violations, "Python files outside the whitelist:\n" + "\n".join(violations)


def test_whitelist_contains_legacy_globs_during_migration() -> None:
    patterns = _whitelist_patterns()
    for legacy in LEGACY_DIRS:
        assert any(pattern == f"{legacy}/**/*.py" for pattern in patterns), (
            f"whitelist must keep {legacy}/**/*.py during migration"
        )


def test_final_mode_requires_legacy_directories_absent() -> None:
    if os.environ.get(FINAL_MODE_ENV) != "1":
        return
    for legacy in LEGACY_DIRS:
        assert not (REPO_ROOT / legacy).exists(), (
            f"final mode: {legacy}/ must be deleted (M3_ARCH_FINAL=1)"
        )
