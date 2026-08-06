"""Deterministic report context builder and markdown renderer.

Builds structured context from history and metrics registry; renders
evidence-only markdown tables with Chinese labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from m3rs_eval.comparability import compare_runs
from m3rs_eval.contracts import MetricRecord, RunManifest, read_jsonl
from m3rs_eval.history import HistoryIndex
from m3rs_eval.registry import MetricRegistry


@dataclass(frozen=True)
class _ContextRun:
    """Internal representation of a run for context building."""
    manifest: RunManifest
    metrics: list[MetricRecord]
    coverage: dict[str, Any]
    failures: list[dict[str, Any]]


def build_report_context(
    run_id: str, history: HistoryIndex, registry: MetricRegistry
) -> dict[str, Any]:
    """Build a deterministic report context dictionary for *run_id*.

    The context includes the current run summary, all quality-comparable
    prior runs with their metrics, the latest baseline, historical best
    per metric (direction-aware), and an appendix of incompatible runs.
    """
    candidate = _load_run(history.runs_root / run_id)
    if candidate is None:
        return {"error": f"run {run_id} not found in {history.runs_root}"}

    prior_runs = _load_prior_runs(history.runs_root, history.ranked_run_ids)
    # Exclude candidate from prior list
    prior_runs = [r for r in prior_runs if r.manifest.run_id != run_id]

    # Determine compatible and incompatible runs
    compatible: list[_ContextRun] = []
    incompatible: list[dict[str, Any]] = []
    for prior in prior_runs:
        comp = compare_runs(candidate.manifest, prior.manifest)
        if comp.quality_comparable:
            compatible.append(prior)
        else:
            incompatible.append({
                "run_id": prior.manifest.run_id,
                "status": prior.manifest.status,
                "mode": prior.manifest.mode,
                "quality_comparable": comp.quality_comparable,
                "resource_comparable": comp.resource_comparable,
                "quality_reasons": comp.quality_reasons,
                "resource_reasons": comp.resource_reasons,
            })

    # Compatible runs are already in time order (by ranked_run_ids, which is sorted)
    # Sort by created_at for robustness
    compatible.sort(key=lambda r: (r.manifest.created_at, r.manifest.run_id))

    # Latest baseline (most recent compatible run)
    latest_baseline = compatible[-1] if compatible else None

    # Best by metric
    best_by_metric = _compute_best_by_metric(compatible, registry)

    # Build context
    context = {
        "run_id": run_id,
        "current_run": _summarize_run(candidate),
        "compatible_history": [_summarize_run_with_metrics(r, registry) for r in compatible],
        "latest_baseline": _summarize_run_with_metrics(latest_baseline, registry) if latest_baseline else None,
        "best_by_metric": best_by_metric,
        "incompatible_runs": incompatible,
    }

    # Compute deltas against latest baseline
    if latest_baseline is not None:
        context["deltas"] = _compute_deltas(candidate, latest_baseline, registry)
    else:
        context["deltas"] = {}

    return context


def render_report_context_markdown(context: dict[str, Any]) -> str:
    """Render a report context dictionary as Chinese-labelled Markdown tables."""
    lines: list[str] = []

    if "error" in context:
        lines.append(f"### 错误\n\n{context['error']}\n")
        return "\n".join(lines)

    # Current run summary
    current = context.get("current_run", {})
    lines.append("## 当前运行摘要\n")
    lines.append(f"| 字段 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| run_id | {current.get('run_id', '')} |")
    lines.append(f"| 状态 | {current.get('status', '')} |")
    lines.append(f"| 模式 | {current.get('mode', '')} |")
    lines.append(f"| 可参与历史 | {current.get('eligible_for_history', '')} |")
    lines.append(f"| 协议 | {current.get('protocol_id', '')} |")
    lines.append(f"| 创建时间 | {current.get('created_at', '')} |")
    lines.append("")

    # Metrics table for current run
    current_metrics = current.get("metrics", [])
    if current_metrics:
        lines.append("### 当前运行指标\n")
        lines.append(_metrics_table_header())
        for m in current_metrics:
            lines.append(_metrics_table_row(m))
        lines.append("")

    # Latest baseline
    baseline = context.get("latest_baseline")
    if baseline:
        lines.append("## 最近兼容基线\n")
        lines.append(f"| 字段 | 值 |")
        lines.append(f"|------|----|")
        lines.append(f"| run_id | {baseline.get('run_id', '')} |")
        lines.append(f"| 状态 | {baseline.get('status', '')} |")
        lines.append(f"| 创建时间 | {baseline.get('created_at', '')} |")
        lines.append("")

        baseline_metrics = baseline.get("metrics", [])
        if baseline_metrics:
            lines.append("### 基线指标\n")
            lines.append(_metrics_table_header())
            for m in baseline_metrics:
                lines.append(_metrics_table_row(m))
            lines.append("")

    # Deltas
    deltas = context.get("deltas", {})
    if deltas:
        lines.append("### 相对基线的指标变化\n")
        lines.append("| metric_id | 基线值 | 当前值 | raw_delta | improvement |")
        lines.append("|-----------|--------|--------|-----------|-------------|")
        for metric_id, delta_info in deltas.items():
            lines.append(
                f"| {metric_id} "
                f"| {delta_info.get('baseline_value', '')} "
                f"| {delta_info.get('current_value', '')} "
                f"| {delta_info.get('raw_delta', '')} "
                f"| {delta_info.get('improvement', '')} |"
            )
        lines.append("")

    # Compatible history
    compat_history = context.get("compatible_history", [])
    if compat_history:
        lines.append("## 全部兼容历史运行\n")
        lines.append("| run_id | 创建时间 | 可选指标数 |")
        lines.append("|--------|----------|-----------|")
        for rh in compat_history:
            lines.append(
                f"| {rh.get('run_id', '')} "
                f"| {rh.get('created_at', '')} "
                f"| {len(rh.get('metrics', []))} |"
            )
        lines.append("")

    # Historical best
    best = context.get("best_by_metric", {})
    if best:
        lines.append("## 历史最佳\n")
        lines.append("| metric_id | 最佳 run_id | 最佳值 | 方向 |")
        lines.append("|-----------|-------------|--------|------|")
        for metric_id, info in best.items():
            lines.append(
                f"| {metric_id} "
                f"| {info.get('run_id', '')} "
                f"| {info.get('value', '')} "
                f"| {info.get('direction', '')} |"
            )
        lines.append("")

    # Incompatible runs
    incompatible = context.get("incompatible_runs", [])
    if incompatible:
        lines.append("## 不可比运行附录\n")
        lines.append("| run_id | 原因 |")
        lines.append("|--------|------|")
        for inc in incompatible:
            reasons = "; ".join(inc.get("quality_reasons", []))
            lines.append(f"| {inc.get('run_id', '')} | {reasons} |")
        lines.append("")

    return "\n".join(lines)


def _metrics_table_header() -> str:
    return "| metric_id | 值 | n_samples | n_failures |"


def _metrics_table_row(metric: dict[str, Any]) -> str:
    return (
        f"| {metric.get('metric_id', '')} "
        f"| {metric.get('value_canonical', '')} "
        f"| {metric.get('n_samples', '')} "
        f"| {metric.get('n_failures', '')} |"
    )


def _load_run(run_dir: Path) -> _ContextRun | None:
    """Load a single run package into a _ContextRun."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = RunManifest.from_dict(manifest_raw)
    except Exception:
        return None

    metrics: list[MetricRecord] = []
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.is_file():
        try:
            metrics = read_jsonl(metrics_path, MetricRecord)
        except Exception:
            pass

    coverage: dict[str, Any] = {}
    coverage_path = run_dir / "coverage.json"
    if coverage_path.is_file():
        try:
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    failures: list[dict[str, Any]] = []
    failures_path = run_dir / "failures.jsonl"
    if failures_path.is_file():
        try:
            lines = failures_path.read_text(encoding="utf-8").splitlines()
            failures = [json.loads(line) for line in lines if line.strip()]
        except Exception:
            pass

    return _ContextRun(manifest=manifest, metrics=metrics, coverage=coverage, failures=failures)


def _load_prior_runs(runs_root: Path, ranked_run_ids: list[str]) -> list[_ContextRun]:
    """Load all ranked runs from the runs root."""
    runs: list[_ContextRun] = []
    for rid in ranked_run_ids:
        run = _load_run(runs_root / rid)
        if run is not None:
            runs.append(run)
    return runs


def _summarize_run(run: _ContextRun) -> dict[str, Any]:
    """Create a summary dictionary for a run."""
    return {
        "run_id": run.manifest.run_id,
        "status": run.manifest.status,
        "mode": run.manifest.mode,
        "protocol_id": run.manifest.protocol_id,
        "created_at": run.manifest.created_at,
        "eligible_for_history": run.manifest.eligible_for_history,
        "metrics": [
            {
                "metric_id": m.metric_id,
                "value_canonical": m.value_canonical,
                "availability": m.availability,
                "n_samples": m.n_samples,
                "n_failures": m.n_failures,
                "provenance": m.provenance,
            }
            for m in run.metrics
        ],
        "coverage": run.coverage,
        "failure_count": len(run.failures),
    }


def _summarize_run_with_metrics(
    run: _ContextRun | None, registry: MetricRegistry
) -> dict[str, Any] | None:
    if run is None:
        return None
    summary = _summarize_run(run)
    return summary


def _compute_best_by_metric(
    compatible_runs: list[_ContextRun], registry: MetricRegistry
) -> dict[str, dict[str, Any]]:
    """For each metric_id across all compatible runs, find the best value.

    Direction is taken from the registry: "higher" means larger is better,
    "lower" means smaller is better.
    """
    best: dict[str, dict[str, Any]] = {}
    for run in compatible_runs:
        for m in run.metrics:
            if m.value_canonical is None or m.availability != "available":
                continue
            metric_id = m.metric_id
            try:
                definition = registry.require(metric_id)
                direction = definition.direction
            except Exception:
                direction = "higher"

            if metric_id not in best:
                best[metric_id] = {
                    "run_id": run.manifest.run_id,
                    "value": m.value_canonical,
                    "direction": direction,
                }
            else:
                current = best[metric_id]["value"]
                if direction == "higher":
                    if m.value_canonical > current:
                        best[metric_id] = {
                            "run_id": run.manifest.run_id,
                            "value": m.value_canonical,
                            "direction": direction,
                        }
                else:  # "lower"
                    if m.value_canonical < current:
                        best[metric_id] = {
                            "run_id": run.manifest.run_id,
                            "value": m.value_canonical,
                            "direction": direction,
                        }
    return best


def _compute_deltas(
    candidate: _ContextRun,
    baseline: _ContextRun,
    registry: MetricRegistry,
) -> dict[str, dict[str, Any]]:
    """Compute raw_delta and improvement for each shared metric."""
    deltas: dict[str, dict[str, Any]] = {}
    baseline_metrics = {m.metric_id: m for m in baseline.metrics if m.value_canonical is not None}
    for cm in candidate.metrics:
        if cm.metric_id not in baseline_metrics or cm.value_canonical is None:
            continue
        bm = baseline_metrics[cm.metric_id]

        raw_delta = cm.value_canonical - bm.value_canonical

        # Determine direction
        try:
            definition = registry.require(cm.metric_id)
            direction = definition.direction
        except Exception:
            direction = "higher"

        # improvement: positive when change is in the good direction
        if direction == "higher":
            improvement = raw_delta  # higher is better, so positive delta means improvement
        else:  # "lower"
            improvement = -raw_delta  # lower is better, so negative delta means improvement

        deltas[cm.metric_id] = {
            "baseline_value": bm.value_canonical,
            "current_value": cm.value_canonical,
            "raw_delta": raw_delta,
            "improvement": improvement,
        }
    return deltas
