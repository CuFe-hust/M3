from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import load_only_run_manifest
from m3rs_eval.config import load_config
from m3rs_eval.contracts import MetricRecord, RequestRecord, read_jsonl
from m3rs_eval.orchestrator import ResumeMismatch, run_evaluation


DATASETS = ("levir_cc", "vrsbench", "xlrs_bench", "mme_rs")
ROOT_FILES = {
    "run_manifest.json",
    "resolved_config.redacted.yaml",
    "metrics.jsonl",
    "resource_metrics.json",
    "coverage.json",
    "failures.jsonl",
    "qc_summary.json",
    "report_context.json",
}


def _run_cli(cli_runner, config_factory, behavior="ok", mode="smoke", limit="2", **kwargs):
    config_path, output_root = config_factory(behavior, **kwargs)
    args = ["run", "--config", config_path, "--mode", mode]
    if limit is not None:
        args.extend(["--limit", limit])
    return cli_runner(*args, output_root=output_root), config_path


def test_smoke_run_builds_evidence_but_not_formal_history(cli_runner, config_factory):
    result, _ = _run_cli(cli_runner, config_factory)

    assert result.returncode == 0, result.stderr
    run_dir, manifest = load_only_run_manifest(result.output_root)
    assert manifest["status"] == "complete"
    assert manifest["eligible_for_history"] is False
    assert ROOT_FILES <= {path.name for path in run_dir.iterdir() if path.is_file()}
    assert {path.name for path in (run_dir / "requests").glob("*.jsonl")} == {
        f"{dataset}.jsonl" for dataset in DATASETS
    }
    assert {path.name for path in (run_dir / "predictions").glob("*.jsonl")} == {
        f"{dataset}.jsonl" for dataset in DATASETS
    }
    for dataset in DATASETS:
        request_text = (run_dir / "requests" / f"{dataset}.jsonl").read_text(encoding="utf-8")
        assert "fixture_prediction" not in request_text
        assert (run_dir / "logs" / dataset / "command_result.json").is_file()
    metrics = read_jsonl(run_dir / "metrics.jsonl", MetricRecord)
    assert metrics
    assert all(
        record.source_log_path is None
        or (not Path(record.source_log_path).is_absolute() and ".." not in Path(record.source_log_path).parts)
        for record in metrics
    )


def test_full_fixture_run_is_complete_but_ineligible(cli_runner, config_factory):
    result, _ = _run_cli(cli_runner, config_factory, mode="full", limit=None)

    assert result.returncode == 0, result.stderr
    _, manifest = load_only_run_manifest(result.output_root)
    assert manifest["status"] == "complete"
    assert manifest["eligible_for_history"] is False


@pytest.mark.parametrize("behavior", ["missing", "malformed", "duplicate", "error"])
def test_prediction_failures_are_incomplete(cli_runner, config_factory, behavior):
    result, _ = _run_cli(cli_runner, config_factory, behavior=behavior)

    assert result.returncode == 3, result.stderr
    run_dir, manifest = load_only_run_manifest(result.output_root)
    assert manifest["status"] == "incomplete"
    assert (run_dir / "failures.jsonl").read_text(encoding="utf-8").strip()
    coverage = json.loads((run_dir / "coverage.json").read_text(encoding="utf-8"))
    assert any(row["status"] == "incomplete" for row in coverage["datasets"].values())


def test_nonzero_system_command_is_failed_and_preserves_logs(cli_runner, config_factory):
    result, _ = _run_cli(cli_runner, config_factory, behavior="nonzero")

    assert result.returncode == 1
    run_dir, manifest = load_only_run_manifest(result.output_root)
    assert manifest["status"] == "failed"
    command = json.loads(
        (run_dir / "logs" / "levir_cc" / "command_result.json").read_text(encoding="utf-8")
    )
    assert command["returncode"] == 17
    assert (run_dir / "logs" / "levir_cc" / "system.stderr.log").is_file()


def test_timed_out_system_command_is_failed(cli_runner, config_factory):
    result, _ = _run_cli(cli_runner, config_factory, behavior="timeout")

    assert result.returncode == 1
    run_dir, manifest = load_only_run_manifest(result.output_root)
    assert manifest["status"] == "failed"
    command = json.loads(
        (run_dir / "logs" / "levir_cc" / "command_result.json").read_text(encoding="utf-8")
    )
    assert command["timed_out"] is True


def test_resolved_config_and_command_evidence_are_redacted(cli_runner, config_factory):
    secret = "do-not-persist-this-secret"
    result, _ = _run_cli(cli_runner, config_factory, secret=secret)

    assert result.returncode == 0, result.stderr
    run_dir, _ = load_only_run_manifest(result.output_root)
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in run_dir.rglob("*")
        if path.is_file()
    )
    assert secret not in persisted
    resolved = (run_dir / "resolved_config.redacted.yaml").read_text(encoding="utf-8")
    assert "API_TOKEN" not in resolved
    command = json.loads(
        (run_dir / "logs" / "levir_cc" / "command_result.json").read_text(encoding="utf-8")
    )
    assert isinstance(command["argv"], list)


def test_doctor_cli_errors_return_two_and_warnings_return_zero(
    cli_runner, config_factory, tmp_path
):
    config_path, output_root = config_factory()
    warning = cli_runner("doctor", "--config", config_path, output_root=output_root)
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [valid", encoding="utf-8")
    error = cli_runner("doctor", "--config", bad, output_root=tmp_path)

    assert warning.returncode == 0
    assert "GIT_UNAVAILABLE" in warning.stdout
    assert error.returncode == 2


def test_deferred_commands_have_strict_shapes_and_do_not_claim_success(cli_runner, tmp_path):
    missing = cli_runner("prepare-report")
    conflicting = cli_runner(
        "prepare-report", "--run-id", "run-1", "--latest-compatible"
    )
    report = cli_runner("prepare-report", "--run-id", "run-1", "--project-root", tmp_path)
    rebuild = cli_runner("rebuild-table", "--project-root", tmp_path)

    assert missing.returncode == 2
    assert conflicting.returncode == 2
    assert report.returncode != 0 and "run" in report.stderr.lower()
    assert rebuild.returncode != 0 and "runs" in rebuild.stderr.lower()


def test_run_resume_flags_must_be_used_together(cli_runner, config_factory):
    config_path, output_root = config_factory()

    without_resume = cli_runner(
        "run", "--config", config_path, "--mode", "smoke", "--run-id", "run-1",
        output_root=output_root,
    )
    without_id = cli_runner(
        "run", "--config", config_path, "--mode", "smoke", "--resume",
        output_root=output_root,
    )

    assert without_resume.returncode == 2
    assert without_id.returncode == 2


def test_direct_resume_preserves_run_id_and_reuses_valid_predictions(config_factory):
    config_path, _ = config_factory()
    config = load_config(config_path)
    first = run_evaluation(config, "smoke", 2, None)
    manifest_path = first.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "inference_running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prediction_bytes = {
        dataset: (first.run_dir / "predictions" / f"{dataset}.jsonl").read_bytes()
        for dataset in DATASETS
    }

    resumed = run_evaluation(config, "smoke", 2, first.run_id)

    assert resumed.run_id == first.run_id
    assert resumed.status == "complete"
    assert prediction_bytes == {
        dataset: (first.run_dir / "predictions" / f"{dataset}.jsonl").read_bytes()
        for dataset in DATASETS
    }


def test_resume_reruns_only_missing_sample_and_preserves_valid_rows(config_factory):
    config_path, _ = config_factory()
    config = load_config(config_path)
    first = run_evaluation(config, "smoke", 2, None)
    manifest_path = first.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "inference_running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictions = first.run_dir / "predictions" / "levir_cc.jsonl"
    rows = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text(rows[0] + "\n", encoding="utf-8")

    resumed = run_evaluation(config, "smoke", 2, first.run_id)

    assert resumed.status == "complete"
    assert json.loads(predictions.read_text(encoding="utf-8").splitlines()[0]) == json.loads(rows[0])
    subset = first.run_dir / "evidence" / "resume" / "levir_cc.requests.jsonl"
    assert len(read_jsonl(subset, RequestRecord, unique_key="sample_id")) == 1


def test_resume_quarantines_malformed_extraneous_evidence_without_rerun(config_factory):
    config_path, _ = config_factory()
    config = load_config(config_path)
    first = run_evaluation(config, "smoke", 2, None)
    manifest_path = first.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "inference_running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictions = first.run_dir / "predictions" / "levir_cc.jsonl"
    valid = predictions.read_text(encoding="utf-8")
    predictions.write_text(valid + "not-json\n", encoding="utf-8")

    resumed = run_evaluation(config, "smoke", 2, first.run_id)

    assert resumed.status == "complete"
    assert [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()] == [
        json.loads(line) for line in valid.splitlines()
    ]
    quarantined = (
        first.run_dir / "evidence" / "resume" / "levir_cc.previous_predictions.jsonl"
    )
    assert "not-json" in quarantined.read_text(encoding="utf-8")
    assert not (first.run_dir / "evidence" / "resume" / "levir_cc.requests.jsonl").exists()


def test_unexpected_resume_failure_is_persisted_as_failed(config_factory, monkeypatch):
    config_path, _ = config_factory()
    config = load_config(config_path)
    first = run_evaluation(config, "smoke", 2, None)
    manifest_path = first.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "inference_running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictions = first.run_dir / "predictions" / "levir_cc.jsonl"
    predictions.write_text(
        predictions.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )

    def explode(*_args, **_kwargs):
        raise RuntimeError("unexpected resume execution failure")

    monkeypatch.setattr("m3rs_eval.orchestrator.run_system", explode)
    resumed = run_evaluation(config, "smoke", 2, first.run_id)

    assert resumed.exit_code == 1
    assert resumed.status == "failed"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize(
    "field",
    ["request_manifest_hash", "protocol_hash", "config_hash", "command_hash", "system_version_hash"],
)
def test_resume_rejects_every_identity_mismatch_before_reuse(config_factory, field):
    config_path, _ = config_factory()
    config = load_config(config_path)
    first = run_evaluation(config, "smoke", 2, None)
    manifest_path = first.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "inference_running"
    manifest[field] = "mismatch"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prediction = first.run_dir / "predictions" / "levir_cc.jsonl"
    before = prediction.read_bytes()

    with pytest.raises(ResumeMismatch, match=field):
        run_evaluation(config, "smoke", 2, first.run_id)

    assert prediction.read_bytes() == before


def test_prepare_report_latest_compatible_selects_newest_eligible(
    cli_runner, tmp_path, run_factory, metric_registry_path
):
    import shutil

    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    shutil.copy2(metric_registry_path, registry_dir / "metrics.yaml")
    run_factory(
        "r1",
        created_at="2025-01-15T10:00:00+08:00",
        metrics={"mme_rs.avg": 0.70},
    )
    run_factory(
        "r2",
        created_at="2025-02-15T10:00:00+08:00",
        metrics={"mme_rs.avg": 0.80},
    )

    result = cli_runner(
        "prepare-report", "--latest-compatible", "--project-root", tmp_path
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "r2"
    assert payload["compatible_runs"] == 1
    assert (tmp_path / "reports" / "report_context_r2.json").is_file()
    assert (tmp_path / "reports" / "report_context_r2.md").is_file()
