"""Enforce the Python file whitelist; the from-zero branch forbids legacy dirs.

清单外新增 .py 必须失败；本分支从零重建，旧包目录必须直接不存在
（无需环境开关）。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WHITELIST = REPO_ROOT / "architecture" / "allowed_python_files.txt"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "build", "dist", "tmp"}
LEGACY_DIRS = ("spacers_agent", "eval")


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
    assert len(_whitelist_patterns()) > 5, "whitelist looks empty"


def test_every_python_file_matches_the_whitelist() -> None:
    patterns = [(p, _pattern_to_regex(p)) for p in _whitelist_patterns()]
    violations = []
    for rel in _repository_py_files():
        if not any(regex.match(rel) for _, regex in patterns):
            violations.append(rel)
    assert not violations, "Python files outside the whitelist:\n" + "\n".join(violations)


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
