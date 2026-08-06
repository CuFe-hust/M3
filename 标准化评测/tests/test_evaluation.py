from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from m3rs_eval.contracts import PredictionRecord, RequestRecord
from m3rs_eval.evaluation import (
    EvaluationError,
    MetricContext,
    align_predictions,
    evaluate_materialization,
    make_metric_record,
    read_prediction_evidence,
    render_official_scorer_argv,
    run_official_scorer,
)


def request(sample_id: str, expected_output: str = "choice") -> RequestRecord:
    return RequestRecord.from_dict(
        {
            "sample_id": sample_id,
            "dataset": "fixture",
            "benchmark_version": "fixture-v1",
            "split": "test",
            "task": "vqa" if expected_output != "boxes" else "grounding",
            "images": ["fixture.png"],
            "prompt": "Answer the request.",
            "expected_output": expected_output,
            "request_hash": f"hash-{sample_id}",
        }
    )


def prediction(sample_id: str, value: str = "A") -> PredictionRecord:
    return PredictionRecord.from_dict(
        {"sample_id": sample_id, "status": "ok", "prediction": value}
    )


def write_valid_predictions(materialization, path):
    rows = []
    for line in materialization.requests_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        row = {"sample_id": item["sample_id"], "status": "ok"}
        if item["expected_output"] == "boxes":
            row["boxes"] = [[0, 0, 9, 9]]
        else:
            row["prediction"] = "A"
        rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_alignment_keeps_every_expected_request_and_missing_prediction_as_failure():
    result = align_predictions(
        [request("a"), request("b")],
        [prediction("a")],
    )

    assert result.expected == 2
    assert result.valid == 1
    assert result.valid_expected == 1
    assert result.expected_failures == 1
    assert result.extraneous_predictions == 0
    assert [row.request.sample_id for row in result.rows] == ["a", "b"]
    assert result.rows[1].failure.reason == "missing_prediction"
    assert not result.complete


def test_unknown_prediction_is_extraneous_failure_without_inflating_expected_denominator():
    result = align_predictions(
        [request("a")],
        [prediction("a"), prediction("unknown")],
    )

    assert result.expected == 1
    assert result.valid_expected == 1
    assert result.expected_failures == 0
    assert result.extraneous_predictions == 1
    assert result.failure_count == 1
    assert result.failures[0].reason == "unknown_sample_id"
    assert not result.complete


def test_duplicate_predictions_invalidate_the_expected_row_and_are_not_silently_dropped():
    result = align_predictions(
        [request("a")],
        [prediction("a", "A"), prediction("a", "B")],
    )

    assert result.expected == 1
    assert result.valid_expected == 0
    assert result.expected_failures == 1
    assert result.duplicate_predictions == 2
    assert result.extraneous_predictions == 0
    assert result.rows[0].prediction is None
    assert result.rows[0].failure.reason == "duplicate_prediction"
    assert result.rows[0].failure.prediction_indexes == (1, 2)


@pytest.mark.parametrize(
    ("raw_prediction", "reason"),
    [
        (
            {
                "sample_id": "a",
                "status": "error",
                "error_code": "inference_error",
                "error": "model failed",
            },
            "explicit_error_prediction",
        ),
        (
            {
                "sample_id": "a",
                "status": "error",
                "error_code": "timeout",
                "error": "deadline exceeded",
            },
            "timed_out_prediction",
        ),
        ({"sample_id": "a", "status": "ok", "prediction": ""}, "malformed_prediction"),
        ({"sample_id": "a", "status": "ok", "prediction": "A"}, "output_shape_mismatch"),
    ],
)
def test_alignment_structures_error_malformed_and_shape_failures(raw_prediction, reason):
    expected_output = "boxes" if reason == "output_shape_mismatch" else "choice"
    result = align_predictions([request("a", expected_output)], [raw_prediction])

    assert result.valid_expected == 0
    assert result.expected_failures == 1
    assert result.rows[0].failure.reason == reason
    json.dumps(result.rows[0].failure.to_dict())


def test_malformed_jsonl_row_is_failure_evidence_and_does_not_hide_missing_expected(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        '{"sample_id":"a","status":"ok","prediction":"A"}\nnot-json\n',
        encoding="utf-8",
    )

    evidence = read_prediction_evidence(path)
    result = align_predictions([request("a"), request("b")], evidence)

    assert result.expected == 2
    assert result.valid_expected == 1
    assert result.expected_failures == 1
    assert result.extraneous_predictions == 1
    assert {failure.reason for failure in result.failures} == {
        "malformed_prediction",
        "missing_prediction",
    }
    assert all(json.dumps(failure.to_dict()) for failure in result.failures)


def metric_context(source_log_path: str = "logs/mme-scorer.json") -> MetricContext:
    return MetricContext(
        run_id="run-20260805",
        recorded_at="2026-08-05T12:00:00+08:00",
        protocol_id="official_full_v1",
        benchmark_version="mme-realworld-rs:Remote_Sensing",
        source_log_path=source_log_path,
    )


def test_metric_factory_requires_registered_id_and_canonical_range(registry):
    with pytest.raises(EvaluationError, match="unknown metric_id"):
        make_metric_record(
            registry,
            metric_context(),
            "not.registered",
            0.5,
            n_samples=2,
            n_failures=0,
            provenance="supplemental",
        )

    with pytest.raises(EvaluationError, match="not applicable"):
        make_metric_record(
            registry,
            metric_context(),
            "levir.caption.no_change.cider_d",
            0.0,
            n_samples=2,
            n_failures=0,
            provenance="official",
        )

    with pytest.raises(EvaluationError, match="canonical range"):
        make_metric_record(
            registry,
            metric_context(),
            "mme_rs.avg",
            50.0,
            n_samples=2,
            n_failures=0,
            provenance="official",
        )


def test_metric_factory_copies_stable_context_registry_dimensions_and_relative_log(registry):
    record = make_metric_record(
        registry,
        metric_context(),
        "mme_rs.avg",
        0.5,
        n_samples=6,
        n_failures=1,
        provenance="official",
    )

    assert record.run_id == "run-20260805"
    assert record.recorded_at == "2026-08-05T12:00:00+08:00"
    assert record.protocol_id == "official_full_v1"
    assert record.benchmark_version == "mme-realworld-rs:Remote_Sensing"
    assert record.dataset == "MME-RealWorld-RS"
    assert record.task == "multiple_choice_vqa"
    assert record.slice == "all"
    assert record.source_log_path == "logs/mme-scorer.json"
    assert record.provenance == "official"
    assert record.ci95_low is None and record.ci95_high is None

    with pytest.raises(EvaluationError, match="relative"):
        make_metric_record(
            registry,
            metric_context("C:/secret/scorer.json"),
            "mme_rs.avg",
            0.5,
            n_samples=6,
            n_failures=0,
            provenance="official",
        )


def test_metric_factory_uses_wilson_only_for_sample_binomial_rates(registry):
    record = make_metric_record(
        registry,
        metric_context(),
        "mme_rs.acc.color",
        2 / 3,
        n_samples=3,
        n_failures=0,
        provenance="supplemental",
        binomial_successes=2,
    )

    assert (record.ci95_low, record.ci95_high) == pytest.approx(
        (0.2076596008, 0.9385080553)
    )


def test_metric_factory_bootstraps_only_when_sample_observations_exist(registry):
    first = make_metric_record(
        registry,
        metric_context(),
        "vrs.grounding.hbb.all.mean_iou",
        0.5,
        n_samples=4,
        n_failures=0,
        provenance="supplemental",
        bootstrap_observations=[0.0, 0.0, 1.0, 1.0],
    )
    second = make_metric_record(
        registry,
        metric_context(),
        "vrs.grounding.hbb.all.mean_iou",
        0.5,
        n_samples=4,
        n_failures=0,
        provenance="supplemental",
        bootstrap_observations=[0.0, 0.0, 1.0, 1.0],
    )

    assert (first.ci95_low, first.ci95_high) == (second.ci95_low, second.ci95_high)
    assert first.ci95_low is not None and first.ci95_high is not None


def test_evaluate_materialization_aligns_once_and_keeps_denominators_separate(
    mme_rs_adapter, registry, tmp_path
):
    materialization = mme_rs_adapter.materialize("full", None, tmp_path / "materialized")
    requests = [
        RequestRecord.from_dict(json.loads(line))
        for line in materialization.requests_path.read_text(encoding="utf-8").splitlines()
    ]
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"sample_id": item.sample_id, "status": "ok", "prediction": "A"}
        for item in requests[:-1]
    ]
    rows.append({"sample_id": "unknown-extra", "status": "ok", "prediction": "A"})
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = evaluate_materialization(
        mme_rs_adapter,
        materialization,
        predictions,
        registry,
        context=metric_context("logs/mme/evaluation.json"),
        log_dir=tmp_path / "logs",
    )
    records = {record.metric_id: record for record in result.metric_records}

    assert result.coverage["expected_requests"] == 6
    assert result.coverage["valid_expected_predictions"] == 5
    assert result.coverage["expected_prediction_failures"] == 1
    assert result.coverage["extraneous_predictions"] == 1
    assert result.coverage["failure_count"] == 2
    assert result.status == "incomplete"
    assert records["mme_rs.avg"].n_samples == 6
    assert records["mme_rs.avg"].value_canonical == pytest.approx(5 / 6)
    assert {failure.reason for failure in result.failures} == {
        "missing_prediction",
        "unknown_sample_id",
    }


def test_official_scorer_bridge_uses_exact_placeholders_and_copies_version_evidence(
    vrsbench_adapter, registry, tmp_path
):
    materialization = vrsbench_adapter.materialize("smoke", 1, tmp_path / "materialized")
    predictions = tmp_path / "predictions.jsonl"
    write_valid_predictions(materialization, predictions)
    script = tmp_path / "fixture_scorer.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "requests, references, predictions, output = sys.argv[1:5]\n"
        "assert pathlib.Path(requests).is_file()\n"
        "assert pathlib.Path(references).is_file()\n"
        "assert pathlib.Path(predictions).is_file()\n"
        "pathlib.Path(output).write_text(json.dumps({"
        "'scorer_version':'fixture-official-v1','metrics':[{'metric_id':'vrs.caption.bleu_4',"
        "'value_canonical':0.25,'n_samples':1,'n_failures':0}]}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    config = replace(
        vrsbench_adapter.config,
        official_scorer_output=None,
        official_scorer_command=(
            sys.executable,
            str(script),
            "{requests_jsonl}",
            "{references_jsonl}",
            "{predictions_jsonl}",
            "{output_json}",
        ),
        official_scorer_working_directory=tmp_path,
        official_scorer_timeout_seconds=5,
        official_scorer_expected_version="fixture-official-v1",
    )

    result = run_official_scorer(
        config, materialization, predictions, registry, tmp_path / "logs", "vrsbench"
    )

    assert result.scores[0].metric_id == "vrs.caption.bleu_4"
    assert result.raw_output_path.is_file()
    assert result.version_path.read_text(encoding="utf-8") == "fixture-official-v1\n"
    assert result.scope_manifest_path.is_file()
    assert len(result.scope_manifest_hash) == 64
    command_evidence = json.loads(result.command_path.read_text(encoding="utf-8"))
    assert command_evidence["shell"] is False
    assert command_evidence["timed_out"] is False

    with pytest.raises(EvaluationError, match="unknown or nonexact"):
        render_official_scorer_argv(
            (sys.executable, "{requests_jsonl}.suffix"), materialization, predictions, tmp_path / "x.json"
        )


def test_official_scorer_bridge_fails_closed_but_preserves_malformed_raw_output(
    levir_adapter, registry, tmp_path
):
    materialization = levir_adapter.materialize("full", None, tmp_path / "materialized")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"not":"the scorer contract"}', encoding="utf-8")
    config = replace(
        levir_adapter.config,
        official_scorer_output=malformed,
    )

    with pytest.raises(EvaluationError, match="malformed official scorer output"):
        run_official_scorer(
            config, materialization, predictions, registry, tmp_path / "logs", "levir_cc"
        )

    assert (tmp_path / "logs" / "levir_cc" / "official_scores.raw.json").read_text(
        encoding="utf-8"
    ) == malformed.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("metric_id", "n_samples", "n_failures", "version", "message"),
    [
        ("levir.caption.all.bleu_4", 2, 0, "vrs-caption-fixture-output-v1", "not allowed"),
        ("vrs.caption.bleu_4", 3, 0, "vrs-caption-fixture-output-v1", "n_samples"),
        ("vrs.caption.bleu_4", 2, 0, "wrong-version", "version"),
    ],
)
def test_official_scorer_rejects_foreign_scope_counts_and_version(
    vrsbench_adapter, registry, tmp_path, metric_id, n_samples, n_failures, version, message
):
    materialization = vrsbench_adapter.materialize("full", None, tmp_path / "materialized")
    predictions = tmp_path / "predictions.jsonl"
    write_valid_predictions(materialization, predictions)
    scorer_output = tmp_path / "scores.json"
    scorer_output.write_text(
        json.dumps(
            {
                "scorer_version": version,
                "metrics": [
                    {
                        "metric_id": metric_id,
                        "value_canonical": 0.5,
                        "n_samples": n_samples,
                        "n_failures": n_failures,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    vrsbench_adapter.config = replace(
        vrsbench_adapter.config,
        official_scorer_output=scorer_output,
        official_scorer_expected_version="vrs-caption-fixture-output-v1",
    )

    result = evaluate_materialization(
        vrsbench_adapter,
        materialization,
        predictions,
        registry,
        context=metric_context("logs/vrs/evaluation.json"),
        log_dir=tmp_path / "logs",
    )

    scorer_failure = next(f for f in result.failures if f.reason == "official_scorer_failed")
    assert message in scorer_failure.detail
    assert result.status == "incomplete"


def test_official_scorer_rejects_failure_count_below_aligned_scope_failures(
    vrsbench_adapter, registry, tmp_path
):
    materialization = vrsbench_adapter.materialize("full", None, tmp_path / "materialized")
    requests = [json.loads(line) for line in materialization.requests_path.read_text(encoding="utf-8").splitlines()]
    rows = []
    skipped = False
    for item in requests:
        if item["task"] == "caption" and not skipped:
            skipped = True
            continue
        row = {"sample_id": item["sample_id"], "status": "ok"}
        row["boxes" if item["expected_output"] == "boxes" else "prediction"] = (
            [[0, 0, 9, 9]] if item["expected_output"] == "boxes" else "A"
        )
        rows.append(row)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = evaluate_materialization(
        vrsbench_adapter,
        materialization,
        predictions,
        registry,
        context=metric_context("logs/vrs/evaluation.json"),
        log_dir=tmp_path / "logs",
    )

    failure = next(f for f in result.failures if f.reason == "official_scorer_failed")
    assert "n_failures" in failure.detail


def test_xlrs_scorer_manifest_and_inputs_are_full_only(
    xlrs_bench_adapter, registry, tmp_path
):
    materialization = xlrs_bench_adapter.materialize("full", None, tmp_path / "materialized")
    predictions = tmp_path / "predictions.jsonl"
    write_valid_predictions(materialization, predictions)

    result = evaluate_materialization(
        xlrs_bench_adapter,
        materialization,
        predictions,
        registry,
        context=metric_context("logs/xlrs/evaluation.json"),
        log_dir=tmp_path / "logs",
    )

    manifest_path = tmp_path / "logs" / "xlrs_bench" / "scorer_scope_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filtered_requests = manifest_path.parent / manifest["combined_inputs"]["requests_path"]
    rows = [json.loads(line) for line in filtered_requests.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {row["variant"] for row in rows} == {"full"}
    assert {row["task"] for row in rows} == {"caption", "grounding"}
    assert result.coverage["scorer_scope_manifest_hash"] == manifest["manifest_hash"]
    assert len(manifest["scopes"]) == 4


def test_required_metric_reconciliation_marks_missing_metric_and_authoritative_status(
    vrsbench_adapter, registry, tmp_path
):
    materialization = vrsbench_adapter.materialize("full", None, tmp_path / "materialized")
    predictions = tmp_path / "predictions.jsonl"
    write_valid_predictions(materialization, predictions)
    protocol = deepcopy(vrsbench_adapter.protocol)
    protocol["datasets"]["vrsbench"]["required_metric_ids"].append("vrs.caption.bleu_1")
    vrsbench_adapter.protocol = protocol

    result = evaluate_materialization(
        vrsbench_adapter,
        materialization,
        predictions,
        registry,
        context=metric_context("logs/vrs/evaluation.json"),
        log_dir=tmp_path / "logs",
    )
    records = {record.metric_id: record for record in result.metric_records}

    assert records["vrs.caption.bleu_1"].availability == "missing"
    assert any(
        failure.reason == "required_metric_unavailable"
        and failure.raw["metric_id"] == "vrs.caption.bleu_1"
        for failure in result.failures
    )
    assert result.status == "incomplete"
    assert result.coverage["status"] == result.status
