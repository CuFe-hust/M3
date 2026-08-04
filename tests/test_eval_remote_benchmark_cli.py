"""Tests for the remote benchmark evaluation CLI.
远端基准评测命令行测试。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_remote_benchmark import (
    _build_combined_results,
    _create_run_dir,
    _limit_samples,
    _short_label,
    _write_metrics_csv,
    parse_args,
    plot_figures,
)


def test_parse_args_defaults() -> None:
    args = parse_args(
        [
            "--model-type",
            "qwen3-vl-4b",
            "--model-path",
            "models/InternVL3_5-8B",
            "--dataset-root",
            "/data/评测数据集",
        ]
    )
    assert args.model_type == "qwen3-vl-4b"
    assert args.lora_path is None
    assert args.datasets == ["vrsbench", "mme_real_rs", "xlrs", "levir_cc"]
    assert args.limit is None
    assert args.local_files_only is True
    assert args.skip_figures is False
    assert args.seed == 42
    assert args.deepseek_proxy is False
    assert args.deepseek_api_key is None
    assert args.deepseek_model == "deepseek-chat"
    assert args.deepseek_base_url == "https://api.deepseek.com/v1"


def test_parse_args_deepseek_switch() -> None:
    args = parse_args(
        [
            "--model-type",
            "qwen3-vl-4b",
            "--model-path",
            "models/InternVL3_5-8B",
            "--dataset-root",
            "/data/评测数据集",
            "--deepseek-proxy",
            "--deepseek-model",
            "deepseek-v4-flash",
            "--deepseek-base-url",
            "https://api.deepseek.com",
            "--deepseek-api-key",
            "test-key",
        ]
    )
    assert args.deepseek_proxy is True
    assert args.deepseek_api_key == "test-key"
    assert args.deepseek_model == "deepseek-v4-flash"
    assert args.deepseek_base_url == "https://api.deepseek.com"


def test_limit_samples_uses_seed() -> None:
    samples = list(range(20))
    first = _limit_samples(samples, 5, seed=42)
    assert len(first) == 5
    assert _limit_samples(samples, 5, seed=42) == first
    assert _limit_samples(samples, 5, seed=7) != first
    assert _limit_samples(samples, None, seed=42) == samples
    assert samples == list(range(20))


def test_parse_args_rejects_unknown_model_type() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-type",
                "llama-3b",
                "--model-path",
                "models/InternVL3_5-8B",
                "--dataset-root",
                "/data",
            ]
        )


def test_short_label() -> None:
    assert _short_label("vrsbench_caption") == "VRS-cap"
    assert _short_label("levir_cc_change_caption") == "LEVIR-cap"
    assert _short_label("custom_label") == "custom_label"


def test_create_run_dir_creates_unique(tmp_path: Path) -> None:
    first = _create_run_dir(tmp_path)
    second = _create_run_dir(tmp_path)
    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_write_metrics_csv(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    all_metrics = {
        "vrsbench_vqa": {"metrics": {"metric": "exact_match_accuracy", "score": 0.5, "total": 2}, "samples": 2},
        "vrsbench_caption": {"metrics": {"CIDEr": 0.25, "BLEU_1": 0.5}, "samples": 1},
    }
    _write_metrics_csv(path, all_metrics)
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "label,metric,value"
    assert "vrsbench_vqa,score,0.5" in rows
    assert "vrsbench_caption,CIDEr,0.25" in rows


def _record(task_type: str, sample_id: str) -> dict:
    return {
        "sample": {"id": sample_id, "task_type": task_type, "answers": ["A"]},
        "prediction": {"answer": "A", "text": "A"},
    }


def test_build_combined_results(monkeypatch) -> None:
    seen_ids: list[str] = []

    def fake_evaluate(records, **kwargs):
        seen_ids.extend(str(record["sample"]["id"]) for record in records)
        task_type = records[0]["sample"]["task_type"]
        return {"metric": "fake", "task_type": task_type, "total": len(records), "score": 1.0}

    monkeypatch.setattr("scripts.eval_remote_benchmark.evaluate_records", fake_evaluate)
    records_by_label = {
        "vrsbench_vqa": [_record("vqa", "a"), _record("vqa", "b")],
        "xlrs_vqa_lite": [_record("vqa", "c")],
        "vrsbench_caption": [_record("caption", "d")],
        "levir_cc_change_caption": [_record("change_caption", "e")],
    }
    dataset_labels = {
        "vrsbench": ["vrsbench_vqa", "vrsbench_caption"],
        "xlrs": ["xlrs_vqa_lite"],
        "levir_cc": ["levir_cc_change_caption"],
    }

    combined = _build_combined_results(records_by_label, dataset_labels)

    assert combined["datasets"]["vrsbench"]["labels"] == ["vrsbench_vqa", "vrsbench_caption"]
    assert combined["datasets"]["vrsbench"]["samples"] == 3
    assert set(combined["datasets"]["vrsbench"]["task_type_metrics"]) == {"vqa", "caption"}
    assert combined["overall"]["samples"] == 5
    assert set(combined["overall"]["task_type_metrics"]) == {"vqa", "caption"}
    assert combined["overall"]["task_type_metrics"]["vqa"]["total"] == 3
    assert combined["overall"]["task_type_metrics"]["caption"]["total"] == 2
    assert "vrsbench:a" in seen_ids
    assert "levir_cc:e" in seen_ids


def test_plot_figures_exports_all(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    all_metrics = {
        "vrsbench_vqa": {"metrics": {"metric": "exact_match_accuracy", "score": 0.6}, "samples": 10},
        "vrsbench_caption": {"metrics": {"BLEU_1": 0.4, "BLEU_2": 0.3, "BLEU_3": 0.2, "BLEU_4": 0.1, "METEOR": 0.2, "ROUGE_L": 0.3, "CIDEr": 0.35}, "samples": 10},
        "vrsbench_grounding": {"metrics": {"mean_iou": 0.45}, "samples": 10},
    }
    per_sample = {
        "vrsbench_vqa": [
            {"id": str(i), "task_type": "vqa", "question_type": "object category", "difficulty": "", "latency_seconds": 1.0, "correct": i % 2 == 0, "iou": None}
            for i in range(10)
        ],
        "vrsbench_grounding": [
            {"id": str(i), "task_type": "grounding", "question_type": "", "difficulty": "", "latency_seconds": 1.0, "correct": None, "iou": (i % 10) / 10}
            for i in range(10)
        ],
        "vrsbench_caption": [
            {"id": str(i), "task_type": "caption", "question_type": "", "difficulty": "", "latency_seconds": 1.0, "correct": None, "iou": None}
            for i in range(10)
        ],
    }
    figures = plot_figures(all_metrics, per_sample, tmp_path)
    assert set(figures) == {
        "metrics_overview",
        "caption_scores",
        "vqa_accuracy_by_type",
        "grounding_iou_histogram",
    }
    for path in figures.values():
        assert Path(path).is_file()


def test_plot_figures_manifest_json(tmp_path: Path) -> None:
    manifest = tmp_path / "figures_manifest.json"
    manifest.write_text(json.dumps({"metrics_overview": "x.png"}), encoding="utf-8")
    assert json.loads(manifest.read_text(encoding="utf-8")) == {"metrics_overview": "x.png"}
