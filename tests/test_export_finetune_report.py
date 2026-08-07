"""Unit tests for the training-report export script.
训练报告导出脚本的单元测试。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import export_finetune_report as reporter  # noqa: E402


def test_split_log_history_separates_train_and_eval() -> None:
    history = [
        {"step": 0, "loss": 2.5, "learning_rate": 1e-4, "epoch": 0.0},
        {"step": 10, "loss": 1.8, "learning_rate": 1e-4, "epoch": 0.01},
        {"step": 10, "eval_loss": 1.9, "eval_runtime": 3.2, "epoch": 0.01},
    ]
    train_rows, eval_rows = reporter.split_log_history(history)
    assert len(train_rows) == 2
    assert len(eval_rows) == 1
    assert eval_rows[0]["eval_loss"] == 1.9


def test_build_metric_rows_merges_by_step() -> None:
    history = [
        {"step": 10, "loss": 1.8, "learning_rate": 1e-4, "grad_norm": 0.5, "epoch": 0.01},
        {"step": 10, "eval_loss": 1.9, "eval_runtime": 3.2, "epoch": 0.01},
        {"step": 20, "loss": 1.5, "learning_rate": 9e-5, "epoch": 0.02},
    ]
    rows = reporter.build_metric_rows(history)
    assert [row["step"] for row in rows] == [10, 20]
    assert rows[0]["loss"] == 1.8
    assert rows[0]["eval_loss"] == 1.9
    assert rows[0]["learning_rate"] == 1e-4
    assert rows[1]["eval_loss"] is None


def test_series_summary_empty_and_values() -> None:
    assert reporter.series_summary([]) == {"count": 0}
    summary = reporter.series_summary([1.0, 2.0, 3.0], digits=3)
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["mean"] == 2.0
    assert summary["last"] == 3.0
    assert summary["count"] == 3


def test_analyze_predictions_counts_failures_lengths_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "vrsbench_test.jsonl"
    lines = [
        {
            "sample": {"id": "a", "task_type": "caption"},
            "prediction": {"text": "A bridge over water.", "meta": {}},
        },
        {
            "sample": {"id": "b", "task_type": "vqa"},
            "prediction": {"text": "Highway toll station", "meta": {}},
        },
        {
            "sample": {"id": "c", "task_type": "caption"},
            "prediction": {"text": "", "meta": {"error": "RuntimeError: OOM"}},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )
    stats = reporter.analyze_predictions(path)
    assert stats["total"] == 3
    assert stats["succeeded"] == 2
    assert stats["failed"] == 1
    assert stats["error_types"] == {"RuntimeError": 1}
    caption = stats["tasks"]["caption"]
    assert caption["succeeded"] == 1
    assert caption["answer_chars"]["mean"] == len("A bridge over water.")


def test_render_markdown_contains_training_and_test_sections() -> None:
    payload = {
        "title": "Report",
        "generated_at": "2026-08-07T00:00:00+00:00",
        "train": {
            "dir": "outputs/finetune",
            "state": {"global_step": 10, "best_metric": None},
            "metrics": {
                "summary": {
                    "train_loss": {"count": 2, "min": 1.5, "max": 2.0, "mean": 1.75, "last": 1.5},
                    "eval_loss": {"count": 1, "min": 1.6, "max": 1.6, "mean": 1.6, "last": 1.6},
                    "learning_rate": {"count": 0},
                }
            },
            "csv_path": "metrics.csv",
            "chart_path": None,
        },
        "eval_dir": "outputs/eval",
        "tests": [],
    }
    markdown = reporter.render_markdown(payload)
    assert "## Training" in markdown
    assert "Train loss: last 1.5" in markdown
    assert "No evaluation summaries found" in markdown


def test_main_writes_report_artifacts_without_charts(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    train_dir.mkdir()
    eval_dir.mkdir()
    (train_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 20,
                "epoch": 0.02,
                "max_steps": 100,
                "log_history": [
                    {"step": 10, "loss": 2.0, "learning_rate": 1e-4, "epoch": 0.01},
                    {"step": 10, "eval_loss": 2.1, "eval_runtime": 3.0, "epoch": 0.01},
                    {"step": 20, "loss": 1.5, "learning_rate": 9e-5, "epoch": 0.02},
                ],
            }
        ),
        encoding="utf-8",
    )
    predictions = eval_dir / "vrsbench_test.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "sample": {"id": "a", "task_type": "vqa"},
                "prediction": {"text": "Windmill", "meta": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_dir / "vrsbench_test.summary.json").write_text(
        json.dumps(
            {
                "model_id": "fake-model",
                "adapter_path": None,
                "data_root": "/tmp/vrsbench",
                "tasks": ["vqa"],
                "output_path": str(predictions),
                "results": {
                    "vqa": {
                        "total": 1,
                        "succeeded": 1,
                        "failed": 0,
                        "exact_match": 1.0,
                        "mean_inference_seconds": 0.4,
                    }
                },
                "total_records": 1,
                "total_failures": 0,
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.md"
    assert (
        reporter.main(
            [
                "--train-dir",
                str(train_dir),
                "--eval-dir",
                str(eval_dir),
                "--report-path",
                str(report_path),
                "--no-charts",
            ]
        )
        == 0
    )
    markdown = report_path.read_text(encoding="utf-8")
    assert "## Training" in markdown
    assert "exact_match" in markdown
    assert "## Test Results" in markdown

    payload = json.loads(report_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["train"]["metrics"]["summary"]["train_loss"]["last"] == 1.5
    assert payload["tests"][0]["summary"]["results"]["vqa"]["exact_match"] == 1.0
    assert payload["tests"][0]["prediction_stats"]["succeeded"] == 1

    with report_path.with_suffix(".csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["step"] == "10"
    assert rows[0]["eval_loss"] == "2.1"


def test_main_fails_when_trainer_state_is_missing(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit) as error:
        reporter.main(
            [
                "--train-dir",
                str(empty_dir),
                "--eval-dir",
                str(tmp_path / "missing-eval"),
                "--report-path",
                str(tmp_path / "report.md"),
                "--no-charts",
            ]
        )
    assert "trainer_state.json" in str(error.value)


def test_render_charts_writes_png_when_matplotlib_available(tmp_path: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib is not installed")
    rows = [
        {"step": 0, "loss": 2.0, "learning_rate": 1e-4},
        {"step": 10, "loss": 1.5, "learning_rate": 9e-5, "eval_loss": 1.8},
    ]
    chart_path = tmp_path / "training_curves.png"
    assert reporter.render_charts(rows, chart_path, dpi=80) == chart_path
    assert chart_path.stat().st_size > 0


def test_render_charts_returns_none_without_rows(tmp_path: Path) -> None:
    assert reporter.render_charts([], tmp_path / "none.png", dpi=80) is None
