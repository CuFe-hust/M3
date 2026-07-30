"""Tests for the persisted LEVIR-CC comparison report.
已持久化 LEVIR-CC 对比报告测试。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from eval.change_comparison_report import build_change_comparison_report


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_change_report_compares_stages_and_copies_images(tmp_path: Path) -> None:
    image_a = tmp_path / "source" / "a.png"
    image_b = tmp_path / "source" / "b.png"
    image_a.parent.mkdir()
    Image.new("RGB", (8, 8), "red").save(image_a)
    Image.new("RGB", (8, 8), "green").save(image_b)

    baseline_path = tmp_path / "baseline.jsonl"
    baseline_path.write_text(
        json.dumps(
            {
                "sample": {"id": "7", "answers": ["no change"]},
                "prediction": {"text": "A building appeared."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sample_dir = tmp_path / "run" / "samples" / "7"
    _write_json(
        sample_dir / "sample.json",
        {
            "sample_id": "7",
            "images": [
                {"path": str(image_a), "role": "t1"},
                {"path": str(image_b), "role": "t2"},
            ],
            "ground_truth": {"answers": ["no change"], "raw": {"adapter_version": "1"}},
        },
    )
    _write_json(sample_dir / "change_expert" / "analysis" / "parsed.json", {"answer": "No change.", "evidence": []})
    _write_json(sample_dir / "change_expert" / "parsed.json", {"answer": "No change.", "evidence": []})
    _write_json(sample_dir / "expert_result.json", {"answer": "No change."})
    _write_json(
        sample_dir / "agent_trace.json",
        {"selected_stage": "verification", "verification_guard": None, "inference_seconds": 2.5},
    )
    metric_values = {name: 0.1 for name in ("BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr")}
    metrics_path = tmp_path / "metrics.json"
    _write_json(
        metrics_path,
        {
            "baseline_metrics": metric_values,
            "agent_metrics": {name: 0.2 for name in metric_values},
            "metric_deltas": {name: 0.1 for name in metric_values},
        },
    )
    manifest_path = tmp_path / "samples.json"
    _write_json(manifest_path, [{"id": "7", "changeflag": 0}])

    html_path, summary_path = build_change_comparison_report(
        baseline_path,
        tmp_path / "run",
        tmp_path / "report" / "comparison.html",
        metrics_path,
        manifest_path,
    )

    report = html_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "样本 01 · ID 7" in report
    assert "Agent 辅助判断：正确" in report
    assert "全文严格一致不是 LEVIR-CC" in report
    assert summary["auxiliary_changeflag_accuracy"]["baseline"] == 0.0
    assert summary["auxiliary_changeflag_accuracy"]["agent"] == 1.0
    assert summary["total_inference_seconds"] == 2.5
    assert len(list((html_path.parent / "images").glob("*.png"))) == 2


def test_change_report_supports_skipped_conditional_verification(
    tmp_path: Path,
) -> None:
    image_a = tmp_path / "source" / "a.png"
    image_b = tmp_path / "source" / "b.png"
    image_a.parent.mkdir()
    Image.new("RGB", (8, 8), "red").save(image_a)
    Image.new("RGB", (8, 8), "green").save(image_b)

    baseline_path = tmp_path / "baseline.jsonl"
    baseline_path.write_text(
        json.dumps(
            {
                "sample": {"id": "8", "answers": ["no change"]},
                "prediction": {"text": "No change."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sample_dir = tmp_path / "run" / "samples" / "8"
    _write_json(
        sample_dir / "sample.json",
        {
            "sample_id": "8",
            "images": [
                {"path": str(image_a), "role": "t1"},
                {"path": str(image_b), "role": "t2"},
            ],
            "ground_truth": {
                "answers": ["no change"],
                "raw": {"adapter_version": "1"},
            },
        },
    )
    _write_json(
        sample_dir / "change_expert" / "analysis" / "parsed.json",
        {"answer": "No change.", "evidence": []},
    )
    _write_json(sample_dir / "expert_result.json", {"answer": "No change."})
    _write_json(
        sample_dir / "agent_trace.json",
        {
            "selected_stage": "analysis",
            "verification_triggered": False,
            "verification_reasons": [],
            "verification_guard": None,
            "inference_seconds": 1.0,
        },
    )
    metric_values = {
        name: 0.1
        for name in (
            "BLEU_1",
            "BLEU_2",
            "BLEU_3",
            "BLEU_4",
            "METEOR",
            "ROUGE_L",
            "CIDEr",
        )
    }
    metrics_path = tmp_path / "metrics.json"
    _write_json(
        metrics_path,
        {
            "baseline_metrics": metric_values,
            "agent_metrics": metric_values,
            "metric_deltas": {name: 0.0 for name in metric_values},
        },
    )
    manifest_path = tmp_path / "samples.json"
    _write_json(manifest_path, [{"id": "8", "changeflag": 0}])

    html_path, summary_path = build_change_comparison_report(
        baseline_path,
        tmp_path / "run",
        tmp_path / "report" / "comparison.html",
        metrics_path,
        manifest_path,
    )

    report = html_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "未触发条件核验。" in report
    assert "未触发（沿用第一阶段）" in report
    assert summary["verification_trigger_distribution"] == {"False": 1}
