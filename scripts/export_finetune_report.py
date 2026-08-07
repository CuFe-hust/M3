"""Export training curves and VRSBench test statistics into a report.
自动导出训练曲线与 VRSBench 测试结果统计。

The training side reads the Hugging Face ``trainer_state.json`` written by
``scripts/finetune_qwen3vl_merger_lora.py`` (train loss, learning rate,
gradient norm, and eval loss per step). The test side reads every
``*.summary.json`` in the evaluation output directory produced by
``scripts/evaluate_qwen3vl_merger_lora.py`` and re-analyzes the corresponding
sample/prediction JSONL for answer-length statistics and failure reasons.
Outputs are a Markdown report, a machine-readable JSON payload, a CSV table
of per-step metrics, and (when matplotlib is available) a PNG chart.
训练侧读取 ``scripts/finetune_qwen3vl_merger_lora.py`` 写出的 Hugging Face
``trainer_state.json``（每步的训练 loss、学习率、梯度范数与 eval loss）。
测试侧读取 ``scripts/evaluate_qwen3vl_merger_lora.py`` 在评测输出目录写出的
全部 ``*.summary.json``，并重新分析对应的 sample/prediction JSONL，
统计答案长度分布与失败原因。输出包括 Markdown 报告、机器可读 JSON、
每步指标 CSV 表，以及（装有 matplotlib 时）PNG 曲线图。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_DIR = Path("outputs/finetune/qwen3-vl-8b-merger-lora")
DEFAULT_EVAL_DIR = Path("outputs/eval/qwen3-vl-8b-merger-lora")
DEFAULT_REPORT_PATH = Path("outputs/reports/qwen3-vl-8b-merger-lora/report.md")


def build_parser() -> argparse.ArgumentParser:
    """Build the export CLI. / 构建导出 CLI。"""
    parser = argparse.ArgumentParser(
        description=(
            "Export merger-LoRA training curves and VRSBench test statistics "
            "into Markdown/JSON/CSV and optional PNG charts."
        )
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Training output directory containing trainer_state.json.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Evaluation output directory containing *.summary.json; "
        "skipped when absent.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report output path; JSON/CSV/PNG sidecars share its stem.",
    )
    parser.add_argument(
        "--title",
        default="Qwen3-VL-8B Merger-LoRA Training & Test Report",
        help="Report title used in Markdown and JSON.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip PNG chart rendering even when matplotlib is installed.",
    )
    parser.add_argument(
        "--chart-dpi",
        type=int,
        default=150,
        help="PNG chart resolution in dots per inch; default 150.",
    )
    return parser


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON file, failing with a useful message.
    读取一个 UTF-8 JSON 文件，失败时给出明确信息。
    """
    if not path.is_file():
        raise SystemExit(f"Required JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {path}: {error}") from error


def load_trainer_state(train_dir: Path) -> dict[str, Any]:
    """Load and validate the Trainer state from a training output directory.
    从训练输出目录加载并校验 Trainer state。
    """
    state = read_json(train_dir / "trainer_state.json")
    if not isinstance(state, dict):
        raise SystemExit(f"{train_dir}/trainer_state.json must contain a JSON object.")
    return state


def split_log_history(
    log_history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split Trainer log entries into train-only and eval-only rows.
    将 Trainer 日志条目拆分为仅训练与仅验证两类。
    """
    train_rows = [entry for entry in log_history if "loss" in entry]
    eval_rows = [entry for entry in log_history if "eval_loss" in entry]
    return train_rows, eval_rows


def build_metric_rows(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge train/eval log entries into one row per step for CSV/PNG export.
    将训练/验证日志条目按 step 合并为每步一行，用于 CSV/PNG 导出。
    """
    by_step: dict[int, dict[str, Any]] = {}
    for entry in log_history:
        step = int(entry.get("step", 0))
        row = by_step.setdefault(
            step,
            {
                "step": step,
                "epoch": None,
                "loss": None,
                "learning_rate": None,
                "grad_norm": None,
                "eval_loss": None,
                "eval_runtime": None,
                "eval_samples_per_second": None,
            },
        )
        row["epoch"] = entry.get("epoch")
        if "loss" in entry:
            row["loss"] = entry["loss"]
            row["learning_rate"] = entry.get("learning_rate")
            row["grad_norm"] = entry.get("grad_norm")
        if "eval_loss" in entry:
            row["eval_loss"] = entry["eval_loss"]
            row["eval_runtime"] = entry.get("eval_runtime")
            row["eval_samples_per_second"] = entry.get("eval_samples_per_second")
    return [by_step[step] for step in sorted(by_step)]


def series_summary(values: list[float], digits: int = 6) -> dict[str, float | int]:
    """Summarize a numeric series: count/min/max/mean/last.
    汇总数值序列：count/min/max/mean/last。
    """
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), digits),
        "max": round(max(values), digits),
        "mean": round(statistics.fmean(values), digits),
        "last": round(values[-1], digits),
    }


def summarize_metrics(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build compact loss summaries for the report.
    为报告构建紧凑的 loss 汇总。
    """
    train_losses = [float(row["loss"]) for row in train_rows if row.get("loss") is not None]
    eval_losses = [float(row["eval_loss"]) for row in eval_rows if row.get("eval_loss") is not None]
    learning_rates = [
        float(row["learning_rate"]) for row in train_rows if row.get("learning_rate") is not None
    ]
    return {
        "train_loss": series_summary(train_losses),
        "eval_loss": series_summary(eval_losses),
        "learning_rate": series_summary(learning_rates),
    }


def analyze_predictions(jsonl_path: Path) -> dict[str, Any]:
    """Re-analyze a canonical sample/prediction JSONL for statistics.
    重新分析规范化 sample/prediction JSONL 以生成统计。
    """
    stats: dict[str, Any] = {
        "path": str(jsonl_path),
        "total": 0,
        "succeeded": 0,
        "failed": 0,
        "empty_predictions": 0,
        "error_types": {},
        "tasks": {},
    }
    if not jsonl_path.is_file():
        stats["path_missing"] = True
        return stats
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON at {jsonl_path}:{line_number}: {error}") from error
            sample = payload.get("sample") or {}
            prediction = payload.get("prediction") or {}
            task = sample.get("task_type") or prediction.get("task_type") or "unknown"
            task_stats = stats["tasks"].setdefault(
                task,
                {
                    "total": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "empty_predictions": 0,
                    "answer_chars": [],
                    "error_types": {},
                },
            )
            stats["total"] += 1
            task_stats["total"] += 1
            text = prediction.get("text") or ""
            error = prediction.get("meta", {}).get("error")
            if error:
                stats["failed"] += 1
                task_stats["failed"] += 1
                error_type = error.split(":", 1)[0].strip() or "UNKNOWN"
                stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1
                task_stats["error_types"][error_type] = (
                    task_stats["error_types"].get(error_type, 0) + 1
                )
                continue
            stats["succeeded"] += 1
            task_stats["succeeded"] += 1
            if not text:
                stats["empty_predictions"] += 1
                task_stats["empty_predictions"] += 1
            task_stats["answer_chars"].append(len(text))
    for task_stats in stats["tasks"].values():
        task_stats["answer_chars"] = series_summary(task_stats["answer_chars"], digits=2)
    return stats


def load_eval_summaries(eval_dir: Path) -> list[dict[str, Any]]:
    """Load every evaluation summary plus its prediction statistics.
    加载每个评测摘要及其预测统计。
    """
    if not eval_dir.is_dir():
        return []
    reports: list[dict[str, Any]] = []
    for summary_path in sorted(eval_dir.glob("*.summary.json")):
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            raise SystemExit(f"{summary_path} must contain a JSON object.")
        predictions_path = summary.get("output_path")
        reports.append(
            {
                "summary_path": str(summary_path),
                "summary": summary,
                "prediction_stats": analyze_predictions(Path(predictions_path))
                if predictions_path
                else analyze_predictions(summary_path.with_suffix(".jsonl")),
            }
        )
    return reports


def _fmt(value: Any) -> str:
    """Format one report cell without trailing zeros.
    格式化一个报告单元格，去掉多余的尾零。
    """
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_training_section(data: dict[str, Any]) -> list[str]:
    """Render the training part of the Markdown report.
    渲染 Markdown 报告的训练部分。
    """
    state = data["train"]["state"]
    summary = data["train"]["metrics"]["summary"]
    lines = ["## Training", ""]
    state_rows = [
        ("Global step", state.get("global_step")),
        ("Epoch", state.get("epoch")),
        ("Max steps", state.get("max_steps")),
        ("Num train epochs", state.get("num_train_epochs")),
        ("Best eval metric", state.get("best_metric")),
        ("Best checkpoint", state.get("best_model_checkpoint")),
        ("Train batch size", state.get("train_batch_size")),
        ("Eval batch size", state.get("eval_batch_size")),
    ]
    for key, value in state_rows:
        lines.append(f"- **{key}:** {_fmt(value)}")
    lines.append("")
    if summary["train_loss"]["count"]:
        lines.append(
            f"- Train loss: last {summary['train_loss']['last']}, "
            f"min {summary['train_loss']['min']}, mean {summary['train_loss']['mean']} "
            f"over {summary['train_loss']['count']} logged steps."
        )
    if summary["eval_loss"]["count"]:
        lines.append(
            f"- Eval loss: last {summary['eval_loss']['last']}, "
            f"min {summary['eval_loss']['min']}, mean {summary['eval_loss']['mean']} "
            f"over {summary['eval_loss']['count']} logged evals."
        )
    lines += [
        "",
        f"Per-step metrics CSV: `{data['train']['csv_path']}`",
    ]
    if data["train"]["chart_path"]:
        lines.append(f"Training curves PNG: `{data['train']['chart_path']}`")
    else:
        lines.append(
            "Training curves PNG: not generated (matplotlib unavailable or no log rows)."
        )
    return lines


def render_test_section(test_report: dict[str, Any]) -> list[str]:
    """Render one evaluation summary into the Markdown report.
    将一条评测摘要渲染进 Markdown 报告。
    """
    summary = test_report["summary"]
    prediction_stats = test_report["prediction_stats"]
    summary_path = Path(test_report["summary_path"])
    lines = [f"### {summary_path.name}", ""]
    lines.append(f"- Model: `{_fmt(summary.get('model_id'))}`")
    lines.append(f"- Adapter: `{_fmt(summary.get('adapter_path'))}`")
    lines.append(f"- Data root: `{_fmt(summary.get('data_root'))}`")
    lines.append(f"- Summary file: `{summary_path}`")
    lines.append("")
    results = summary.get("results") or {}
    lines.append(
        "| task | total | succeeded | failed | exact_match | "
        "mean_inference_s | empty_preds | answer_chars(mean) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for task in sorted(results):
        task_stats = results[task]
        pred_task = prediction_stats["tasks"].get(task, {})
        chars = pred_task.get("answer_chars", {})
        lines.append(
            f"| {task} | {_fmt(task_stats.get('total'))} "
            f"| {_fmt(task_stats.get('succeeded'))} "
            f"| {_fmt(task_stats.get('failed'))} "
            f"| {_fmt(task_stats.get('exact_match'))} "
            f"| {_fmt(task_stats.get('mean_inference_seconds'))} "
            f"| {pred_task.get('empty_predictions', 0)} "
            f"| {_fmt(chars.get('mean'))} |"
        )
    lines.append("")
    total = prediction_stats
    if prediction_stats.get("path_missing"):
        lines.append(
            f"- Prediction JSONL not found: `{prediction_stats['path']}`; "
            "per-sample statistics are unavailable."
        )
    else:
        lines.append(
            f"- Prediction JSONL total: {total['total']}; "
            f"succeeded {total['succeeded']}; failed {total['failed']}; "
            f"empty predictions {total['empty_predictions']}."
        )
    error_types = prediction_stats["error_types"]
    if error_types:
        lines.append("- Failure reasons:")
        for error_type, count in sorted(error_types.items(), key=lambda item: -item[1]):
            lines.append(f"  - `{error_type}`: {count}")
    else:
        lines.append("- Failure reasons: none.")
    return lines


def render_markdown(data: dict[str, Any]) -> str:
    """Render the full Markdown report.
    渲染完整 Markdown 报告。
    """
    lines = [f"# {data['title']}", ""]
    lines.append(f"- Generated at: {data['generated_at']}")
    lines.append(f"- Train dir: `{data['train']['dir']}`")
    lines.append(f"- Eval dir: `{data['eval_dir']}`")
    lines.append("")
    lines += render_training_section(data)
    lines += ["", "## Test Results", ""]
    test_reports = data["tests"]
    if not test_reports:
        lines.append("No evaluation summaries found; the test section is empty.")
    else:
        for test_report in test_reports:
            lines += render_test_section(test_report)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_charts(
    metric_rows: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> Path | None:
    """Render train loss / learning rate / eval loss PNG when matplotlib exists.
    装有 matplotlib 时渲染 train loss / learning rate / eval loss PNG。
    """
    if not metric_rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    steps = [row["step"] for row in metric_rows]
    has_eval = any(row.get("eval_loss") is not None for row in metric_rows)
    figure, axes = plt.subplots(
        1, 3 if has_eval else 2, figsize=(15 if has_eval else 10, 4.5)
    )
    axis_iter = iter(axes) if has_eval else iter([axes[0], axes[1]])

    train_losses = [row.get("loss") for row in metric_rows]
    axis = next(axis_iter)
    axis.plot(steps, train_losses, marker="o", markersize=2, linewidth=1)
    axis.set_title("Train Loss")
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)

    learning_rates = [row.get("learning_rate") for row in metric_rows]
    axis = next(axis_iter)
    axis.plot(steps, learning_rates, marker="o", markersize=2, linewidth=1)
    axis.set_title("Learning Rate")
    axis.set_xlabel("Step")
    axis.set_ylabel("LR")
    axis.grid(True, alpha=0.3)

    if has_eval:
        eval_losses = [row.get("eval_loss") for row in metric_rows]
        axis = next(axis_iter)
        axis.plot(steps, eval_losses, marker="o", markersize=2, linewidth=1)
        axis.set_title("Eval Loss")
        axis.set_xlabel("Step")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.3)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    figure.savefig(tmp_path, dpi=dpi, bbox_inches="tight", format="png")
    tmp_path.replace(output_path)
    plt.close(figure)
    return output_path


def write_text_atomic(text: str, path: Path) -> None:
    """Atomically write a UTF-8 text file.
    原子写入一个 UTF-8 文本文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Atomically write one JSON object.
    原子写入一个 JSON 对象。
    """
    write_text_atomic(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        path,
    )


def write_csv(metric_rows: list[dict[str, Any]], path: Path) -> None:
    """Write per-step metrics as CSV with stable column order.
    按稳定列顺序将每步指标写为 CSV。
    """
    columns = [
        "step",
        "epoch",
        "loss",
        "learning_rate",
        "grad_norm",
        "eval_loss",
        "eval_runtime",
        "eval_samples_per_second",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    tmp_path.replace(path)


def build_payload(
    args: argparse.Namespace,
    state: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    test_reports: list[dict[str, Any]],
    csv_path: Path,
    chart_path: Path | None,
) -> dict[str, Any]:
    """Assemble the machine-readable JSON payload.
    组装机器可读的 JSON 载荷。
    """
    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)
    report_path = Path(args.report_path)
    selected_state = {
        key: state.get(key)
        for key in (
            "global_step",
            "epoch",
            "max_steps",
            "num_train_epochs",
            "best_metric",
            "best_model_checkpoint",
            "train_batch_size",
            "eval_batch_size",
        )
    }
    return {
        "title": args.title,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train": {
            "dir": str(train_dir),
            "state": selected_state,
            "metrics": {
                "rows": metric_rows,
                "summary": summarize_metrics(train_rows, eval_rows),
            },
            "csv_path": str(csv_path),
            "chart_path": str(chart_path) if chart_path else None,
        },
        "eval_dir": str(eval_dir) if eval_dir.is_dir() else None,
        "tests": test_reports,
        "artifacts": {
            "report": str(report_path),
            "json": str(report_path.with_suffix(".json")),
            "csv": str(csv_path),
            "chart": str(chart_path) if chart_path else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Export the report; returns process exit code.
    导出报告；返回进程退出码。
    """
    args = build_parser().parse_args(argv)
    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)
    report_path = Path(args.report_path)

    state = load_trainer_state(train_dir)
    log_history = state.get("log_history")
    if not isinstance(log_history, list):
        raise SystemExit(f"{train_dir}/trainer_state.json has no log_history list.")
    train_rows, eval_rows = split_log_history(log_history)
    metric_rows = build_metric_rows(log_history)
    test_reports = load_eval_summaries(eval_dir)

    csv_path = report_path.with_suffix(".csv")
    write_csv(metric_rows, csv_path)
    chart_path = None
    if not args.no_charts:
        chart_path = render_charts(
            metric_rows,
            report_path.with_name(report_path.stem + "_training_curves.png"),
            args.chart_dpi,
        )

    payload = build_payload(
        args=args,
        state=state,
        metric_rows=metric_rows,
        train_rows=train_rows,
        eval_rows=eval_rows,
        test_reports=test_reports,
        csv_path=csv_path,
        chart_path=chart_path,
    )
    write_json_atomic(payload, report_path.with_suffix(".json"))
    write_text_atomic(render_markdown(payload), report_path)

    print(f"Report written to {report_path}")
    print(f"Metrics CSV written to {csv_path}")
    if chart_path:
        print(f"Training curves PNG written to {chart_path}")
    if not train_rows:
        print("Warning: no training log rows found in trainer_state.json.")
    if not test_reports:
        print(f"Warning: no *.summary.json found under {eval_dir}; test section is empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
