"""Public command-line interface for standardized M3-RS evaluations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from m3rs_eval.config import ConfigError, load_config
from m3rs_eval.history import rebuild_history
from m3rs_eval.orchestrator import ResumeMismatch, run_evaluation
from m3rs_eval.preflight import run_doctor
from m3rs_eval.registry import MetricRegistry
from m3rs_eval.reporting import build_report_context, render_report_context_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="m3rs_eval")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--config", required=True, type=Path)

    run = commands.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--mode", required=True, choices=("smoke", "full"))
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--run-id")

    rebuild = commands.add_parser("rebuild-table")
    rebuild.add_argument("--project-root", type=Path, default=None)
    prepare = commands.add_parser("prepare-report")
    prepare.add_argument("--project-root", type=Path, default=None)
    target = prepare.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--latest-compatible", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "rebuild-table":
        return _rebuild_table(Path(args.project_root) if args.project_root else Path.cwd())
    if args.command == "prepare-report":
        return _prepare_report(
            Path(args.project_root) if args.project_root else Path.cwd(),
            args.run_id,
            args.latest_compatible,
        )
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            report = run_doctor(config)
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            return report.exit_code
        if bool(args.resume) != bool(args.run_id):
            parser.error("run requires --resume and --run-id together")
        outcome = run_evaluation(
            config,
            args.mode,
            args.limit,
            args.run_id if args.resume else None,
        )
        print(json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2))
        return outcome.exit_code
    except (ConfigError, ResumeMismatch, ValueError, OSError) as error:
        print(f"m3rs_eval: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"m3rs_eval: unexpected failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


def _rebuild_table(project_root: Path) -> int:
    """Rebuild the canonical history CSVs and the core Excel scorecard."""
    runs_root = project_root / "runs"
    history_root = project_root / "history"
    registry_path = project_root / "registry" / "metrics.yaml"
    output_xlsx = project_root / "评测表.xlsx"
    if not runs_root.is_dir():
        print(f"m3rs_eval: no runs directory: {runs_root}", file=sys.stderr)
        return 3
    if not registry_path.is_file():
        print(f"m3rs_eval: no metric registry: {registry_path}", file=sys.stderr)
        return 3
    try:
        history = rebuild_history(runs_root, history_root)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[2] / "tools" / "build_workbook.py"),
                str(history_root),
                str(registry_path),
                str(output_xlsx),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=300,
        )
        if completed.returncode != 0:
            print(f"m3rs_eval: workbook build failed: {completed.stderr.strip()}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "history": history.to_dict(),
                    "workbook": str(output_xlsx),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError) as error:
        print(f"m3rs_eval: {error}", file=sys.stderr)
        return 2


def _prepare_report(project_root: Path, run_id: str | None, latest: bool) -> int:
    """Deterministically build the report context for one run and render it."""
    runs_root = project_root / "runs"
    history_root = project_root / "history"
    registry_path = project_root / "registry" / "metrics.yaml"
    if not runs_root.is_dir():
        print(f"m3rs_eval: no runs directory: {runs_root}", file=sys.stderr)
        return 3
    if not registry_path.is_file():
        print(f"m3rs_eval: no metric registry: {registry_path}", file=sys.stderr)
        return 3
    try:
        history = rebuild_history(runs_root, history_root)
        if latest:
            if not history.ranked_run_ids:
                print(
                    "m3rs_eval: no eligible full history runs to report",
                    file=sys.stderr,
                )
                return 3
            run_id = history.ranked_run_ids[-1]
        registry = MetricRegistry.load(registry_path)
        context = build_report_context(str(run_id), history, registry)
        if "error" in context:
            print(f"m3rs_eval: {context['error']}", file=sys.stderr)
            return 3
        reports_root = project_root / "reports"
        reports_root.mkdir(parents=True, exist_ok=True)
        json_path = reports_root / f"report_context_{run_id}.json"
        markdown_path = reports_root / f"report_context_{run_id}.md"
        json_path.write_text(
            json.dumps(context, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        markdown_path.write_text(
            render_report_context_markdown(context),
            encoding="utf-8",
            newline="\n",
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "json": str(json_path),
                    "markdown": str(markdown_path),
                    "compatible_runs": len(context.get("compatible_history", [])),
                    "incompatible_runs": len(context.get("incompatible_runs", [])),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError) as error:
        print(f"m3rs_eval: {error}", file=sys.stderr)
        return 2
