"""Explicit artifact canonicalization for runtime parity comparisons.
用于运行时等价比较的显式产物规范化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


_IGNORED_KEYS = frozenset({"updated_at", "inference_seconds"})
_PATH_KEYS = frozenset({"artifact_dir", "path", "result_path", "sample_file"})
_TRACE_KEYS = frozenset(
    {
        "entrypoint",
        "route",
        "router_used",
        "task_type",
        "judge_status",
        "execution_task",
        "official_question_type",
        "prompt_version",
        "geometry",
        "fallback_used",
        "primary_reason",
    }
)


def canonicalize_artifact(
    value: Any,
    *,
    run_root: Path | None = None,
    project_root: Path | None = None,
    artifact_kind: str | None = None,
) -> Any:
    """Normalize only approved volatile fields while preserving unknown data.
    仅规范化获准的易变字段，同时保留未知数据。
    """

    roots = tuple(
        (label, root.resolve().as_posix())
        for label, root in (("<RUN_ROOT>", run_root), ("<PROJECT_ROOT>", project_root))
        if root is not None
    )

    def visit(current: Any, *, parent_kind: str | None = None) -> Any:
        if isinstance(current, dict):
            source = current
            if parent_kind == "trace":
                source = {key: item for key, item in current.items() if key in _TRACE_KEYS}
            normalized: dict[str, Any] = {}
            for key, item in source.items():
                if key in _IGNORED_KEYS:
                    continue
                if key == "agent_class" and isinstance(item, str):
                    normalized[key] = item.rsplit(".", 1)[-1]
                elif key in _PATH_KEYS and item is not None:
                    normalized[key] = normalize_path(
                        str(item),
                        roots,
                        anchor="samples/" if key == "result_path" else None,
                    )
                else:
                    normalized[key] = visit(item)
            return normalized
        if isinstance(current, list):
            return [visit(item) for item in current]
        if isinstance(current, Path):
            return normalize_path(str(current), roots)
        return current

    return visit(value, parent_kind=artifact_kind)


def normalize_path(
    value: str,
    roots: tuple[tuple[str, str], ...],
    *,
    anchor: str | None = None,
) -> str:
    """Normalize separators and replace only explicitly supplied roots.
    规范化路径分隔符，并且仅替换显式提供的根目录。
    """

    normalized = value.replace("\\", "/")
    for label, root in roots:
        if normalized.casefold().startswith(root.casefold()):
            return label + normalized[len(root) :]
    if anchor is not None:
        index = normalized.casefold().find(anchor.casefold())
        if index >= 0:
            return "<RUN_ROOT>/" + normalized[index:]
    return normalized
