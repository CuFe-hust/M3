from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

import eval.metrics as metrics_module
import main as baseline_main
from data.schema import CanonicalPrediction, CanonicalSample
from eval.audit_report import AuditReportWriter, build_audit_report, report_dir_for_result, write_deepseek_audit
from eval.metrics import evaluate_records


def _sample(sample_id: str) -> CanonicalSample:
    return CanonicalSample(
        id=sample_id,
        task_type="vqa",
        images=[Image.new("RGB", (8, 8), color="navy")],
        prompt="Answer using only the image. What is visible?",
        answers=["harbor"],
        meta={"question": "What is visible?"},
    )


def _prediction(sample_id: str, answer: str = "harbor") -> CanonicalPrediction:
    return CanonicalPrediction(
        id=sample_id,
        task_type="vqa",
        text=answer,
        answer=answer,
        meta={"raw_text": answer},
    )


def test_audit_report_persists_images_html_csv_and_deepseek_details(tmp_path: Path) -> None:
    result_path = tmp_path / "vrsbench_vqa.jsonl"
    result_path.write_text("{}\n", encoding="utf-8")
    result_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "completed_samples": 2,
                "model": {"id": "local-qwen", "dtype": "bfloat16", "max_new_tokens": 64},
                "model_load_seconds": 1.5,
                "inference_seconds": 2.5,
            }
        ),
        encoding="utf-8",
    )
    with AuditReportWriter(result_path, max_samples=2) as writer:
        writer.capture(_sample("q-1"), _prediction("q-1"), 0.1)
        writer.capture(_sample("q-2"), _prediction("q-2", "a harbor"), 0.2)

    metrics_path = result_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "metric": "exact_match_accuracy",
                "correct": 1,
                "total": 2,
                "score": 0.5,
                "deepseek_proxy": {"score": 1.0, "evaluated": 2},
            }
        ),
        encoding="utf-8",
    )
    audit_path = report_dir_for_result(result_path) / "deepseek_audit.jsonl"
    write_deepseek_audit(
        audit_path,
        [
            {
                "sample_id": sample_id,
                "score": 1.0,
                "raw_content": '{"score": 1}',
                "raw_api_response": {"choices": [{"message": {"content": '{"score": 1}'}}]},
                "duration_seconds": 0.25,
                "attempts": 1,
                "token_usage": {"total_tokens": 12},
                "error": None,
            }
            for sample_id in ("q-1", "q-2")
        ],
    )

    html_path = build_audit_report(result_path, metrics_path, audit_path)

    assert html_path is not None and html_path.is_file()
    report_html = html_path.read_text(encoding="utf-8")
    assert "Qwen 原始回复" in report_html
    assert "DeepSeek 完整原始 API 响应" in report_html
    assert "local-qwen" in report_html
    assert report_html.count('<article class="card">') == 2
    assert len(list((report_dir_for_result(result_path) / "images").glob("*.png"))) == 1
    with (report_dir_for_result(result_path) / "samples.csv").open(encoding="utf-8-sig", newline="") as file:
        assert len(list(csv.DictReader(file))) == 2


def test_deepseek_proxy_collects_auditable_raw_response(monkeypatch) -> None:
    body = {
        "choices": [{"message": {"content": '{"score": 1}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(body).encode("utf-8")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(metrics_module, "urlopen", lambda request, timeout: FakeResponse())
    audit: list[dict[str, object]] = []
    records = [
        {
            "sample": _sample("q-1").serializable(),
            "prediction": _prediction("q-1").serializable(),
        }
    ]

    metrics = evaluate_records(records, use_deepseek=True, deepseek_audit=audit)

    assert metrics["deepseek_proxy"]["score"] == 1.0
    assert audit[0]["score"] == 1.0
    assert audit[0]["raw_api_response"] == body
    assert audit[0]["raw_content"] == '{"score": 1}'
    assert audit[0]["attempts"] == 1
    assert audit[0]["token_usage"] == body["usage"]
    assert "request_payload" in audit[0]


def test_infer_prints_default_report_absolute_path(tmp_path: Path, monkeypatch, capsys) -> None:
    sample = _sample("q-1")

    class FakeModel:
        def predict(self, current: CanonicalSample) -> CanonicalPrediction:
            return _prediction(current.id)

    monkeypatch.setattr(baseline_main, "load_samples", lambda dataset_name, data_root: iter([sample]))
    baseline_main._infer_target(
        "vrsbench_vqa",
        tmp_path / "data",
        tmp_path / "output",
        FakeModel(),
        limit=1,
        overwrite=False,
        config={"model": {"id": "local-qwen"}, "report": {"enabled": True, "max_samples": 20}},
        model_load_seconds=1.0,
    )

    output = capsys.readouterr().out
    expected = (tmp_path / "output" / "vrsbench_vqa.report" / "report.html").resolve()
    assert f"Saved default audit report to {expected}" in output
    assert expected.is_file()
