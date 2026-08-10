"""Explicit external standard-evaluator seam: invoke the team's separately
maintained evaluator without duplicating its metric rules.

显式外部标准评估器 seam：调用团队独立维护的评估器，不在本仓复制其指标
规则。本模块只负责：canonical result → <tool-dir>/evaluate.py →
*.standard.json → 校验 JSON 对象。subprocess 调用显式 shell=False；
任何失败（入口缺失/退出码非零/未产出报告/非对象 JSON）都以稳定错误失败，
绝不静默替代为旧版指标。绝不复活旧 eval.audit_report。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def default_standard_report_path(result_path: Path) -> Path:
    """Return the adjacent report path: <result-stem>.standard.json.
    返回相邻报告路径：<result-stem>.standard.json。"""

    return result_path.with_suffix(".standard.json")


def run_standard_evaluation(
    result_path: Path,
    *,
    tool_dir: Path,
    output_path: Path | None = None,
    python_executable: str | Path = sys.executable,
    extra_args: Sequence[str] = (),
) -> dict[str, Any]:
    """Run <tool-dir>/evaluate.py on the canonical result and validate its
    persisted JSON report. 运行 <tool-dir>/evaluate.py 处理 canonical 结果
    并校验其持久化 JSON 报告。

    - the result file and the evaluator entry point must exist
    - the subprocess runs with shell=False and no network expectations
    - a nonzero exit, a missing report, or a non-object report fails stably
    - result 文件与评估器入口必须存在
    - 子进程以 shell=False 运行，无网络依赖
    - 退出码非零、报告缺失或非对象报告稳定失败
    """

    resolved_result = result_path.expanduser().resolve()
    resolved_tool = tool_dir.expanduser().resolve()
    evaluate_path = resolved_tool / "evaluate.py"
    if not resolved_result.is_file():
        raise FileNotFoundError("canonical result file does not exist")
    if not evaluate_path.is_file():
        raise FileNotFoundError("standard evaluator entry point does not exist")
    destination = (
        output_path or default_standard_report_path(resolved_result)
    ).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable),
        str(evaluate_path),
        str(resolved_result),
        "--output",
        str(destination),
        *extra_args,
    ]
    completed = subprocess.run(command, cwd=resolved_tool, shell=False, check=False)
    if completed.returncode != 0:
        raise RuntimeError("standard evaluator failed with a nonzero exit code")
    if not destination.is_file():
        raise RuntimeError("standard evaluator did not create its report")
    try:
        report = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("standard evaluator report is invalid JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("standard evaluator report must be a JSON object")
    return report
