"""One-click orchestration for immutable M3-RS evaluation run packages."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from m3rs_eval.command_adapter import CommandResult, run_system
from m3rs_eval.config import EvaluationConfig, serialize_resolved_config
from m3rs_eval.contracts import (
    MetricRecord,
    PredictionRecord,
    RequestRecord,
    RunManifest,
    read_jsonl,
    write_jsonl,
)
from m3rs_eval.datasets import DatasetMaterialization, create_adapters
from m3rs_eval.evaluation import (
    MetricContext,
    align_predictions,
    read_prediction_evidence,
)
from m3rs_eval.fixture_input import prepare_fixture_command_input
from m3rs_eval.metadata import collect_environment_metadata, configuration_hash
from m3rs_eval.preflight import DoctorReport, run_doctor
from m3rs_eval.registry import MetricRegistry
from m3rs_eval.resources import ResourceSampler, summarize_latencies
from m3rs_eval.state import InvalidTransition, RunStateStore


class ResumeMismatch(ValueError):
    """Raised before reuse when a run identity no longer matches its inputs."""


@dataclass(frozen=True)
class RunOutcome:
    run_id: str | None
    run_dir: Path | None
    status: str
    eligible_for_history: bool
    exit_code: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir) if self.run_dir is not None else None,
            "status": self.status,
            "eligible_for_history": self.eligible_for_history,
            "exit_code": self.exit_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class _Identity:
    config_hash: str
    protocol_hash: str
    command_hash: str
    system_version_hash: str


def run_evaluation(
    config: EvaluationConfig,
    mode: str,
    limit: int | None,
    resume_run_id: str | None,
) -> RunOutcome:
    """Execute or safely resume one standardized evaluation run."""
    _validate_mode_limit(mode, limit)
    doctor = run_doctor(config)
    if not doctor.passed:
        return RunOutcome(None, None, "preflight_failed", False, 2, "doctor checks failed")

    protocol = _load_protocol(config.protocol_path)
    registry = MetricRegistry.load(config.metric_registry_path)
    adapters = create_adapters(config, protocol, registry)
    identity = _identity(config)
    if resume_run_id is not None:
        return _resume_run(
            config, mode, limit, resume_run_id, doctor, protocol, registry, adapters, identity
        )
    return _new_run(config, mode, limit, doctor, protocol, registry, adapters, identity)


def _new_run(
    config: EvaluationConfig,
    mode: str,
    limit: int | None,
    doctor: DoctorReport,
    protocol: Mapping[str, Any],
    registry: MetricRegistry,
    adapters: list[Any],
    identity: _Identity,
) -> RunOutcome:
    run_id = _make_run_id(config.system_version, identity.config_hash)
    run_dir = config.output_root / run_id
    manifest = RunManifest(
        run_id=run_id,
        status="created",
        mode=mode,
        protocol_id=str(protocol["protocol_id"]),
        created_at=datetime.now().astimezone().isoformat(),
        config_hash=identity.config_hash,
        request_manifest_hash="pending",
        eligible_for_history=False,
        protocol_hash=identity.protocol_hash,
        command_hash=identity.command_hash,
        system_version_hash=identity.system_version_hash,
        metadata={
            "system_version": config.system_version,
            "limit": limit,
            "doctor": doctor.to_dict(),
        },
    )
    store = RunStateStore.create(run_dir, manifest)
    try:
        _create_package_directories(run_dir)
        _write_yaml(run_dir / "resolved_config.redacted.yaml", serialize_resolved_config(config))
        environment = collect_environment_metadata(config)
        store.update(metadata={**store.manifest.metadata, "environment": environment})
        materializations = _materialize_into_run(adapters, mode, limit, run_dir)
        request_hash = _request_manifest_hash(materializations, mode, limit)
        store.update(
            request_manifest_hash=request_hash,
            metadata={
                **store.manifest.metadata,
                "dataset_manifest_hashes": {
                    dataset: item.manifest_hash
                    for dataset, item in sorted(materializations.items())
                },
            },
        )
        store.transition("preflight_passed")
        return _execute_run(
            config, protocol, registry, adapters, materializations, store, doctor, resume=False
        )
    except Exception as error:
        return _failed_outcome(store, doctor, error)


def _resume_run(
    config: EvaluationConfig,
    mode: str,
    limit: int | None,
    run_id: str,
    doctor: DoctorReport,
    protocol: Mapping[str, Any],
    registry: MetricRegistry,
    adapters: list[Any],
    identity: _Identity,
) -> RunOutcome:
    run_dir = config.output_root / run_id
    store = RunStateStore.load(run_dir)
    if store.manifest.status in {"complete", "incomplete", "failed"}:
        raise ResumeMismatch(f"terminal run is immutable: {store.manifest.status}")
    if store.manifest.mode != mode:
        raise ResumeMismatch("mode mismatch")
    if store.manifest.metadata.get("limit") != limit:
        raise ResumeMismatch("limit mismatch")

    with tempfile.TemporaryDirectory(prefix="m3rs-resume-") as temporary:
        materializations = _materialize(adapters, mode, limit, Path(temporary))
        expected = {
            "request_manifest_hash": _request_manifest_hash(materializations, mode, limit),
            "protocol_hash": identity.protocol_hash,
            "config_hash": identity.config_hash,
            "command_hash": identity.command_hash,
            "system_version_hash": identity.system_version_hash,
        }
        for field, value in expected.items():
            if getattr(store.manifest, field) != value:
                raise ResumeMismatch(f"{field} mismatch")

        _bind_run_evidence(materializations, run_dir)
        if store.manifest.status == "created":
            _copy_materialized_evidence(materializations, run_dir)
            store.transition("preflight_passed")
        try:
            return _execute_run(
                config,
                protocol,
                registry,
                adapters,
                materializations,
                store,
                doctor,
                resume=True,
            )
        except Exception as error:
            return _failed_outcome(store, doctor, error)


def _execute_run(
    config: EvaluationConfig,
    protocol: Mapping[str, Any],
    registry: MetricRegistry,
    adapters: list[Any],
    materializations: dict[str, DatasetMaterialization],
    store: RunStateStore,
    doctor: DoctorReport,
    *,
    resume: bool,
) -> RunOutcome:
    run_dir = store.run_dir
    if store.manifest.status == "preflight_passed":
        store.transition("inference_running")

    command_results: dict[str, CommandResult] = {}
    command_failure: dict[str, Any] | None = None
    if store.manifest.status == "inference_running":
        for adapter in adapters:
            materialization = materializations[adapter.dataset]
            predictions_path = run_dir / "predictions" / f"{adapter.dataset}.jsonl"
            if resume and _normalize_reusable_predictions(
                materialization, predictions_path, run_dir
            ):
                continue
            result = _run_dataset_command(
                config,
                adapter,
                materialization,
                predictions_path,
                run_dir,
                resume=resume,
            )
            command_results[adapter.dataset] = result
            _write_command_result(run_dir, adapter.dataset, result)
            if result.returncode != 0 or result.timed_out:
                command_failure = {
                    "dataset": adapter.dataset,
                    "reason": "system_timeout" if result.timed_out else "system_nonzero_exit",
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                }
                break
        store.transition("evaluating")

    if command_failure is not None:
        _write_json(run_dir / "coverage.json", {"status": "failed", "datasets": {}})
        _write_raw_jsonl(run_dir / "failures.jsonl", [command_failure])
        write_jsonl(run_dir / "metrics.jsonl", [])
        _write_resource_metrics(run_dir, command_results, (), ())
        _write_qc(run_dir, doctor, "failed", False, [command_failure])
        _write_report_context(run_dir, store.manifest.run_id, "failed", False)
        store.transition("failed")
        return RunOutcome(store.manifest.run_id, run_dir, "failed", False, 1, command_failure["reason"])

    try:
        recorded_at = datetime.now().astimezone().isoformat()
        metric_records: list[MetricRecord] = []
        failures: list[dict[str, Any]] = []
        coverage: dict[str, Any] = {}
        latencies: list[float] = []
        expected_failures = 0
        for adapter in adapters:
            dataset = adapter.dataset
            predictions_path = run_dir / "predictions" / f"{dataset}.jsonl"
            result = adapter.evaluate(
                materializations[dataset],
                predictions_path,
                registry,
                context=MetricContext(
                    run_id=store.manifest.run_id,
                    recorded_at=recorded_at,
                    protocol_id=store.manifest.protocol_id,
                    benchmark_version=materializations[dataset].manifest_hash,
                    source_log_path=f"logs/{dataset}/system.stdout.log",
                ),
                log_dir=run_dir / "logs",
            )
            metric_records.extend(result.metric_records)
            failures.extend({"dataset": dataset, **failure.to_dict()} for failure in result.failures)
            coverage[dataset] = result.coverage
            expected_failures += result.alignment.expected_failures
            latencies.extend(
                evidence.prediction.latency_ms
                for evidence in read_prediction_evidence(predictions_path)
                if evidence.prediction is not None and evidence.prediction.latency_ms is not None
            )

        complete = all(row["status"] == "complete" for row in coverage.values())
        eligible = _eligible_for_history(config, store.manifest.mode, complete, doctor)
        status = "complete" if complete else "incomplete"
        write_jsonl(run_dir / "metrics.jsonl", metric_records)
        _write_raw_jsonl(run_dir / "failures.jsonl", failures)
        _write_json(run_dir / "coverage.json", {"status": status, "datasets": coverage})
        _write_resource_metrics(
            run_dir,
            command_results,
            latencies,
            failures,
            expected_failures=expected_failures,
        )
        _write_qc(run_dir, doctor, status, eligible, failures)
        _write_report_context(run_dir, store.manifest.run_id, status, eligible)
        store.update(eligible_for_history=eligible)
        store.transition(status)
        return RunOutcome(
            store.manifest.run_id,
            run_dir,
            status,
            eligible,
            0 if status == "complete" else 3,
            "evaluation complete" if complete else "evaluation incomplete",
        )
    except Exception as error:
        return _failed_outcome(store, doctor, error)


def _materialize_into_run(
    adapters: list[Any], mode: str, limit: int | None, run_dir: Path
) -> dict[str, DatasetMaterialization]:
    materializations = _materialize(adapters, mode, limit, run_dir / "evidence" / "materialized")
    _copy_materialized_evidence(materializations, run_dir)
    return materializations


def _materialize(
    adapters: list[Any], mode: str, limit: int | None, destination: Path
) -> dict[str, DatasetMaterialization]:
    return {
        adapter.dataset: adapter.materialize(mode, limit, destination / adapter.dataset)
        for adapter in adapters
    }


def _copy_materialized_evidence(
    materializations: dict[str, DatasetMaterialization], run_dir: Path
) -> None:
    for dataset, materialization in materializations.items():
        requests = read_jsonl(materialization.requests_path, RequestRecord, unique_key="sample_id")
        canonical = run_dir / "requests" / f"{dataset}.jsonl"
        write_jsonl(canonical, requests)
        reference_target = run_dir / "evidence" / "references" / f"{dataset}.jsonl"
        reference_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(materialization.references_path, reference_target)
        materializations[dataset] = replace(
            materialization,
            requests_path=canonical,
            references_path=reference_target,
        )


def _bind_run_evidence(
    materializations: dict[str, DatasetMaterialization], run_dir: Path
) -> None:
    for dataset, materialization in materializations.items():
        request_path = run_dir / "requests" / f"{dataset}.jsonl"
        reference_path = run_dir / "evidence" / "references" / f"{dataset}.jsonl"
        if not request_path.is_file() or not reference_path.is_file():
            raise ResumeMismatch(f"request evidence missing for {dataset}")
        materializations[dataset] = replace(
            materialization,
            requests_path=request_path,
            references_path=reference_path,
        )


def _run_dataset_command(
    config: EvaluationConfig,
    adapter: Any,
    materialization: DatasetMaterialization,
    predictions_path: Path,
    run_dir: Path,
    *,
    resume: bool,
) -> CommandResult:
    input_path = materialization.requests_path
    if resume and predictions_path.is_file():
        input_path = _prepare_resume_request_subset(materialization, predictions_path, run_dir)
        if input_path is None:
            raise RuntimeError("resume requested execution without invalid predictions")
    if adapter.config.profile == "fixture":
        references_path = materialization.references_path
        if input_path != materialization.requests_path:
            references_path = _subset_references(
                input_path,
                materialization.references_path,
                run_dir / "evidence" / "resume" / f"{adapter.dataset}.references.jsonl",
            )
        fixture_input = prepare_fixture_command_input(
            input_path,
            references_path,
            run_dir / "evidence" / "command_inputs" / f"{adapter.dataset}.jsonl",
            profile=adapter.config.profile,
            formal_execution=False,
        )
        input_path = fixture_input.path

    output_path = predictions_path
    merge_existing: list[PredictionRecord] = []
    if resume and predictions_path.is_file():
        alignment = align_predictions(
            read_jsonl(materialization.requests_path, RequestRecord, unique_key="sample_id"),
            read_prediction_evidence(predictions_path),
        )
        merge_existing = [
            row.prediction for row in alignment.rows if row.failure is None and row.prediction is not None
        ]
        previous = run_dir / "evidence" / "resume" / f"{adapter.dataset}.previous_predictions.jsonl"
        previous.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(predictions_path, previous)
        output_path = run_dir / "evidence" / "resume" / f"{adapter.dataset}.new_predictions.jsonl"

    result = run_system(
        config.system,
        input_path,
        output_path,
        run_dir / "logs" / adapter.dataset,
        ResourceSampler(sample_interval_seconds=0.05),
    )
    if merge_existing and result.returncode == 0 and not result.timed_out:
        new = read_jsonl(output_path, PredictionRecord)
        by_id = {record.sample_id: record for record in [*merge_existing, *new]}
        ordered_requests = read_jsonl(
            materialization.requests_path, RequestRecord, unique_key="sample_id"
        )
        write_jsonl(predictions_path, [by_id[request.sample_id] for request in ordered_requests])
    return result


def _prepare_resume_request_subset(
    materialization: DatasetMaterialization, predictions_path: Path, run_dir: Path
) -> Path | None:
    requests = read_jsonl(materialization.requests_path, RequestRecord, unique_key="sample_id")
    alignment = align_predictions(requests, read_prediction_evidence(predictions_path))
    rerun = [row.request for row in alignment.rows if row.failure is not None]
    if not rerun:
        return None
    target = run_dir / "evidence" / "resume" / f"{materialization.dataset}.requests.jsonl"
    write_jsonl(target, rerun)
    return target


def _subset_references(requests_path: Path, references_path: Path, destination: Path) -> Path:
    request_ids = {
        request.sample_id
        for request in read_jsonl(requests_path, RequestRecord, unique_key="sample_id")
    }
    rows = [
        json.loads(line)
        for line in references_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _write_raw_jsonl(destination, [row for row in rows if row.get("sample_id") in request_ids])
    return destination


def _normalize_reusable_predictions(
    materialization: DatasetMaterialization, predictions_path: Path, run_dir: Path
) -> bool:
    if not predictions_path.is_file():
        return False
    requests = read_jsonl(materialization.requests_path, RequestRecord, unique_key="sample_id")
    alignment = align_predictions(requests, read_prediction_evidence(predictions_path))
    if any(row.failure is not None or row.prediction is None for row in alignment.rows):
        return False
    if not alignment.complete:
        previous = (
            run_dir
            / "evidence"
            / "resume"
            / f"{materialization.dataset}.previous_predictions.jsonl"
        )
        previous.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(predictions_path, previous)
        write_jsonl(
            predictions_path,
            [row.prediction for row in alignment.rows if row.prediction is not None],
        )
    return True


def _identity(config: EvaluationConfig) -> _Identity:
    serialized = serialize_resolved_config(config)
    return _Identity(
        config_hash=configuration_hash(config),
        protocol_hash=_sha256_file(config.protocol_path),
        command_hash=_hash_json(serialized["system"]),
        system_version_hash=_sha256_text(config.system_version),
    )


def _request_manifest_hash(
    materializations: Mapping[str, DatasetMaterialization], mode: str, limit: int | None
) -> str:
    return _hash_json(
        {
            "mode": mode,
            "limit": limit,
            "datasets": {
                dataset: materialization.manifest_hash
                for dataset, materialization in sorted(materializations.items())
            },
        }
    )


def _eligible_for_history(
    config: EvaluationConfig, mode: str, complete: bool, doctor: DoctorReport
) -> bool:
    return bool(
        mode == "full"
        and complete
        and doctor.passed
        and all(dataset.profile == "official" for dataset in config.datasets.values())
        and all(dataset.official_scorer_output is None for dataset in config.datasets.values())
    )


def _failed_outcome(
    store: RunStateStore, doctor: DoctorReport, error: Exception
) -> RunOutcome:
    failure = {"reason": "orchestration_error", "detail": f"{type(error).__name__}: {error}"}
    try:
        _write_raw_jsonl(store.run_dir / "failures.jsonl", [failure])
        _write_json(store.run_dir / "coverage.json", {"status": "failed", "datasets": {}})
        write_jsonl(store.run_dir / "metrics.jsonl", [])
        _write_json(store.run_dir / "resource_metrics.json", {"datasets": {}})
        _write_qc(store.run_dir, doctor, "failed", False, [failure])
        _write_report_context(store.run_dir, store.manifest.run_id, "failed", False)
        while store.manifest.status != "evaluating":
            next_status = {
                "created": "preflight_passed",
                "preflight_passed": "inference_running",
                "inference_running": "evaluating",
            }[store.manifest.status]
            store.transition(next_status)
        store.transition("failed")
    except (OSError, ValueError, InvalidTransition):
        pass
    return RunOutcome(store.manifest.run_id, store.run_dir, "failed", False, 1, failure["detail"])


def _write_command_result(run_dir: Path, dataset: str, result: CommandResult) -> None:
    payload = {
        "argv": result.argv,
        "returncode": result.returncode,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_seconds": result.duration_seconds,
        "stdout_path": _relative_path(result.stdout_path, run_dir),
        "stderr_path": _relative_path(result.stderr_path, run_dir),
        "timed_out": result.timed_out,
        "environment_overrides": result.environment_overrides,
        "resource_summary": asdict(result.resource_summary) if result.resource_summary else None,
        "resource_sampler_error": result.resource_sampler_error,
    }
    _write_json(run_dir / "logs" / dataset / "command_result.json", payload)


def _write_resource_metrics(
    run_dir: Path,
    command_results: Mapping[str, CommandResult],
    latencies: Iterable[float],
    failures: Iterable[Any],
    *,
    expected_failures: int = 0,
) -> None:
    latencies = list(latencies)
    failures = list(failures)
    summary = summarize_latencies(latencies, max(expected_failures, len(failures)))
    _write_json(
        run_dir / "resource_metrics.json",
        {
            "datasets": {
                dataset: {
                    "duration_seconds": result.duration_seconds,
                    "resource_summary": asdict(result.resource_summary) if result.resource_summary else None,
                    "resource_sampler_error": result.resource_sampler_error,
                }
                for dataset, result in sorted(command_results.items())
            },
            "latency": asdict(summary),
            "total_command_duration_seconds": sum(
                result.duration_seconds for result in command_results.values()
            ),
        },
    )


def _write_qc(
    run_dir: Path,
    doctor: DoctorReport,
    status: str,
    eligible: bool,
    failures: Iterable[Any],
) -> None:
    _write_json(
        run_dir / "qc_summary.json",
        {
            "status": status,
            "eligible_for_history": eligible,
            "doctor": doctor.to_dict(),
            "failure_count": len(list(failures)),
        },
    )


def _write_report_context(run_dir: Path, run_id: str, status: str, eligible: bool) -> None:
    _write_json(
        run_dir / "report_context.json",
        {
            "run_id": run_id,
            "status": status,
            "eligible_for_history": eligible,
            "availability": "deferred_to_task_8",
        },
    )


def _create_package_directories(run_dir: Path) -> None:
    for name in ("requests", "predictions", "logs", "evidence"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _write_raw_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _load_protocol(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("protocol_id"), str):
        raise ValueError("protocol requires protocol_id")
    return raw


def _validate_mode_limit(mode: str, limit: int | None) -> None:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if mode == "full" and limit is not None:
        raise ValueError("full mode does not accept limit")


def _make_run_id(system_version: str, config_hash: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", system_version.lower()).strip("-") or "system"
    return f"{datetime.now().astimezone():%Y%m%dT%H%M%S%z}__{slug}__{config_hash[:8]}"


def _relative_path(path: Path, run_dir: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"evidence path escapes run directory: {path}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
