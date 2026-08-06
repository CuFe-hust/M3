from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from data.schema import CanonicalPrediction, CanonicalSample
from eval.audit_report import AuditReportWriter, build_audit_report
from eval.standard_adapter import default_standard_report_path, run_standard_evaluation


def test_standard_adapter_invokes_external_evaluator(tmp_path: Path) -> None:
    result_path = tmp_path / "predictions.jsonl"
    result_path.write_text('{"sample": {}, "prediction": {}}\n', encoding="utf-8")
    tool_dir = tmp_path / "eval_standard"
    tool_dir.mkdir()
    (tool_dir / "evaluate.py").write_text(
        """import argparse, json
p = argparse.ArgumentParser()
p.add_argument('input')
p.add_argument('--output', required=True)
a = p.parse_args()
json.dump({'primary_metric': 'open_vqa_accuracy', 'primary_value': 0.75, 'score': 75.0}, open(a.output, 'w'))
""",
        encoding="utf-8",
    )

    report = run_standard_evaluation(result_path, tool_dir=tool_dir, python_executable=sys.executable)

    assert report["score"] == 75.0
    assert default_standard_report_path(result_path).is_file()


def test_audit_html_discovers_adjacent_standard_report(tmp_path: Path) -> None:
    result_path = tmp_path / "predictions.jsonl"
    sample = CanonicalSample(id="1", task_type="vqa", images=[Image.new("RGB", (8, 8))], prompt="q", answers=["a"])
    prediction = CanonicalPrediction(id="1", task_type="vqa", text="a", answer="a")
    with AuditReportWriter(result_path, max_samples=1) as writer:
        writer.capture(sample, prediction, 0.1)
    result_path.write_text(
        json.dumps({"sample": sample.serializable(), "prediction": prediction.serializable()}) + "\n",
        encoding="utf-8",
    )
    default_standard_report_path(result_path).write_text(
        json.dumps({"primary_metric": "open_vqa_accuracy", "primary_value": 0.75, "score": 75.0}),
        encoding="utf-8",
    )

    html_path = build_audit_report(result_path)

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert "Standard evaluation" in html
    assert "primary_metric=open_vqa_accuracy" in html
    assert "score=75.0" in html
