"""Invoke the team standard evaluator without duplicating its metric rules.
调用团队统一评分器，避免在项目内复制其指标规则。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def default_standard_report_path(result_path: Path) -> Path:
    """Return the adjacent report path used by the HTML integration.
    返回供 HTML 集成自动发现的相邻报告路径。
    """

    return result_path.with_suffix(".standard.json")


def run_standard_evaluation(
    result_path: Path,
    *,
    tool_dir: Path,
    output_path: Path | None = None,
    python_executable: str | Path = sys.executable,
    extra_args: Sequence[str] = (),
) -> dict[str, Any]:
    """Run eval_standard/evaluate.py and validate its persisted JSON report.
    运行 eval_standard/evaluate.py，并校验其持久化 JSON 报告。
    """

    result_path = result_path.expanduser().resolve()
    tool_dir = tool_dir.expanduser().resolve()
    evaluate_path = tool_dir / "evaluate.py"
    if not result_path.is_file():
        raise FileNotFoundError(f"Canonical prediction file not found: {result_path}")
    if not evaluate_path.is_file():
        raise FileNotFoundError(f"Standard evaluator entry point not found: {evaluate_path}")
    destination = (output_path or default_standard_report_path(result_path)).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [str(python_executable), str(evaluate_path), str(result_path), "--output", str(destination), *extra_args]
    completed = subprocess.run(command, cwd=tool_dir, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Standard evaluator failed with exit code {completed.returncode}.")
    if not destination.is_file():
        raise RuntimeError(f"Standard evaluator did not create its report: {destination}")
    report = json.loads(destination.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Standard evaluator report must be a JSON object.")
    return report
