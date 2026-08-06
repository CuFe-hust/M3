from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from m3rs_eval.contracts import (
    ContractError,
    MetricRecord,
    PredictionRecord,
    RequestRecord,
    RunManifest,
    read_jsonl,
    validate_persisted_record,
    write_jsonl,
)


def test_prediction_requires_matching_status_fields():
    with pytest.raises(ContractError):
        PredictionRecord.from_dict({"sample_id": "x", "status": "error", "prediction": "A"})

    with pytest.raises(ContractError, match="error_code"):
        PredictionRecord.from_dict({"sample_id": "x", "status": "error", "error": "failed"})


@pytest.mark.parametrize(
    "error_code", ["timeout", "inference_error", "parse_error", "cancelled", "unknown"]
)
def test_prediction_error_code_is_machine_readable_and_round_trips(error_code):
    record = PredictionRecord.from_dict(
        {
            "sample_id": "x",
            "status": "error",
            "error_code": error_code,
            "error": "descriptive text",
        }
    )

    assert record.error_code == error_code
    assert record.to_dict()["error_code"] == error_code


def test_prediction_error_code_rejects_unknown_taxonomy_and_ok_rows():
    with pytest.raises(ContractError, match="error_code"):
        PredictionRecord.from_dict(
            {"sample_id": "x", "status": "error", "error_code": "hung", "error": "timeout"}
        )
    with pytest.raises(ContractError, match="error_code"):
        PredictionRecord.from_dict(
            {"sample_id": "x", "status": "ok", "prediction": "A", "error_code": "unknown"}
        )


def test_jsonl_rejects_duplicate_prediction_sample_ids_by_default(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        '{"sample_id":"x","status":"ok","prediction":"A"}\n' * 2,
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="duplicate sample_id: x"):
        read_jsonl(path, PredictionRecord)


def test_jsonl_reports_the_line_of_malformed_records(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ContractError, match="line 1"):
        read_jsonl(path, PredictionRecord)


def test_prediction_round_trip_preserves_optional_evidence(tmp_path):
    record = PredictionRecord.from_dict(
        {
            "sample_id": "x",
            "status": "ok",
            "prediction": "A",
            "latency_ms": 42.5,
            "raw_output": "The answer is A.",
            "trace": {"request_id": "req-1"},
        }
    )
    path = tmp_path / "predictions.jsonl"

    write_jsonl(path, [record])

    assert read_jsonl(path, PredictionRecord) == [record]


def test_every_record_and_schema_reject_unknown_fields():
    cases = [
        (RequestRecord, _request_payload()),
        (PredictionRecord, _prediction_payload()),
        (MetricRecord, _metric_payload()),
        (RunManifest, _manifest_payload()),
    ]

    for model_type, payload in cases:
        with pytest.raises(ContractError, match="unexpected"):
            validate_persisted_record({**payload, "unexpected": True}, model_type)
        with pytest.raises(ContractError, match="unknown field"):
            model_type.from_dict({**payload, "unexpected": True})


@pytest.mark.parametrize(
    "changes",
    [
        {"ci95_low": 0.2},
        {"ci95_low": 0.8, "ci95_high": 0.2},
        {"n_samples": 2, "n_failures": 3},
    ],
)
def test_metric_schema_and_dataclass_reject_the_same_cross_field_cases(changes):
    payload = {**_metric_payload(), **changes}

    with pytest.raises(ContractError):
        validate_persisted_record(payload, MetricRecord)
    with pytest.raises(ContractError):
        MetricRecord.from_dict(payload)


def test_metric_schema_and_dataclass_accept_a_valid_record():
    payload = {**_metric_payload(), "ci95_low": 0.2, "ci95_high": 0.8}

    validate_persisted_record(payload, MetricRecord)
    assert MetricRecord.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize("source_log_path", ["/absolute/log.json", "C:/logs/x.json", "logs/../x.json", "logs\\..\\x.json"])
def test_metric_source_log_path_is_run_relative_in_schema_and_dataclass(source_log_path):
    payload = {**_metric_payload(), "source_log_path": source_log_path}

    with pytest.raises(ContractError):
        validate_persisted_record(payload, MetricRecord)
    with pytest.raises(ContractError):
        MetricRecord.from_dict(payload)


def test_available_metric_requires_positive_sample_count_in_schema_and_dataclass():
    payload = {**_metric_payload(), "n_samples": 0, "n_failures": 0}

    with pytest.raises(ContractError):
        validate_persisted_record(payload, MetricRecord)
    with pytest.raises(ContractError):
        MetricRecord.from_dict(payload)


def test_metric_record_schema_version_is_explicit_and_legacy_is_rejected():
    payload = _metric_payload()
    assert payload["record_schema_version"] == 2
    assert MetricRecord.from_dict(payload).record_schema_version == 2

    legacy = dict(payload)
    del legacy["record_schema_version"]
    with pytest.raises(ContractError, match="legacy.*regenerate"):
        MetricRecord.from_dict(legacy)
    with pytest.raises(ContractError):
        validate_persisted_record(legacy, MetricRecord)


@pytest.mark.parametrize("availability", ["missing", "not_applicable", "failed"])
def test_unavailable_metric_requires_null_value_and_no_interval(availability):
    payload = {
        **_metric_payload(),
        "availability": availability,
        "value_canonical": None,
    }

    validate_persisted_record(payload, MetricRecord)
    assert MetricRecord.from_dict(payload).availability == availability

    with pytest.raises(ContractError):
        MetricRecord.from_dict({**payload, "value_canonical": 0.0})
    with pytest.raises(ContractError):
        validate_persisted_record({**payload, "ci95_low": 0.0, "ci95_high": 1.0}, MetricRecord)


def test_metric_availability_is_required_by_schema_and_dataclass():
    payload = _metric_payload()
    del payload["availability"]

    with pytest.raises(ContractError):
        validate_persisted_record(payload, MetricRecord)
    with pytest.raises(ContractError):
        MetricRecord.from_dict(payload)


def test_metric_provenance_is_required_and_strict():
    payload = _metric_payload()
    del payload["provenance"]
    with pytest.raises(ContractError):
        validate_persisted_record(payload, MetricRecord)
    with pytest.raises(ContractError):
        MetricRecord.from_dict(payload)

    with pytest.raises(ContractError):
        MetricRecord.from_dict({**_metric_payload(), "provenance": "approximation"})


def test_jsonl_read_uses_persisted_metric_validation(tmp_path):
    payload = {**_metric_payload(), "ci95_low": 0.8, "ci95_high": 0.2}
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match="line 1"):
        read_jsonl(path, MetricRecord)


def test_jsonl_write_uses_persisted_metric_validation(tmp_path):
    payload = {**_metric_payload(), "n_samples": 2, "n_failures": 3}
    invalid_record = MetricRecord(**payload)

    with pytest.raises(ContractError, match="schema validation"):
        write_jsonl(tmp_path / "metrics.jsonl", [invalid_record])


def _request_payload():
    return {
        "sample_id": "request-1",
        "dataset": "vrsbench",
        "benchmark_version": "v1",
        "split": "test",
        "task": "vqa",
        "images": ["image.tif"],
        "prompt": "Which option is correct?",
        "expected_output": "choice",
        "request_hash": "sha256",
    }


def _prediction_payload():
    return {"sample_id": "prediction-1", "status": "ok", "prediction": "A"}


def _metric_payload():
    return {
        "record_schema_version": 2,
        "run_id": "run-1",
        "metric_id": "mme_rs.avg",
        "availability": "available",
        "provenance": "official",
        "value_canonical": 0.5,
        "n_samples": 4,
        "n_failures": 1,
        "recorded_at": "2026-08-04T00:00:00+08:00",
        "protocol_id": "official_full_v1",
        "benchmark_version": "v1",
    }


def _manifest_payload():
    return {
        "run_id": "run-1",
        "status": "created",
        "mode": "full",
        "protocol_id": "official_full_v1",
        "created_at": "2026-08-04T00:00:00+08:00",
        "config_hash": "abc",
        "request_manifest_hash": "def",
        "eligible_for_history": False,
        "protocol_hash": "protocol-hash",
        "command_hash": "command-hash",
        "system_version_hash": "system-version-hash",
        "metadata": {},
    }
