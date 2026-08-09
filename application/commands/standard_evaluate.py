"""Public `standard-evaluate` CLI command: run the external standard evaluator.

公开 `standard-evaluate` CLI 命令：运行外部团队标准评估器。结果经
evaluation.standard.adapter 校验为 JSON 对象后输出；仅当结果文件关联当前
run（位于 runs.root 下）时刷新统一报告。绝不复活旧 eval.audit_report；
本命令无模型调用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from application.settings import load_settings
from evaluation.standard.adapter import (
    default_standard_report_path,
    run_standard_evaluation,
)
from reporting.builder import build_report

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130

_DEFAULT_TOOL_DIR = Path("~/eval_standard")


def run_standard_evaluate(args: argparse.Namespace) -> int:
    """Run the external evaluator, print the validated report, and refresh
    the unified report when the result belongs to a current run.
    运行外部评估器、输出校验后的报告；结果属于当前 run 时刷新统一报告。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        result_path = Path(args.result).expanduser().resolve()
        tool_dir = (
            Path(args.tool_dir).expanduser().resolve()
            if args.tool_dir
            else _DEFAULT_TOOL_DIR.expanduser()
        )
        output_path = (
            Path(args.output).expanduser().resolve() if args.output else None
        )
        python_executable = args.python or sys.executable
        report = run_standard_evaluation(
            result_path,
            tool_dir=tool_dir,
            output_path=output_path,
            python_executable=python_executable,
        )
        report_path = (
            output_path or default_standard_report_path(result_path)
        ).expanduser().resolve()
        run_id = _associated_run_id(settings, result_path)
        if run_id is not None:
            # Refresh the unified report only for a current run; the report
            # builder is read-only and never touches the run artifacts.
            # 仅对当前 run 刷新统一报告；报告构建器只读，绝不触碰 run 产物。
            build_report(settings.runs.root / run_id)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as error:
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}"}),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    print(
        json.dumps(
            {
                "status": "ok",
                "report": report,
                "report_path": report_path.as_posix(),
                "run_id": run_id,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK


def _associated_run_id(settings: object, result_path: Path) -> str | None:
    """Return the run id when the result file lives under the runs root;
    None otherwise (no report refresh). 结果文件位于 runs root 下时返回
    run id；否则返回 None（不刷新报告）。"""

    try:
        relative = result_path.relative_to(settings.runs.root.resolve())
    except ValueError:
        return None
    segments = relative.parts
    if not segments:
        return None
    candidate = settings.runs.root / segments[0]
    if not candidate.is_dir():
        return None
    return segments[0]
