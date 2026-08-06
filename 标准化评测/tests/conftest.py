from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import json
import os
import subprocess
import sys

import pytest
import yaml


SOURCE_REGISTRY_SHA256 = "d14b983ff823e9fe78e294f50ca50700db73511a572f8120dcff4264033306cf"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def metric_registry_path(project_root: Path) -> Path:
    return project_root / "registry" / "metrics.yaml"


@pytest.fixture(scope="session")
def registry(metric_registry_path: Path):
    from m3rs_eval.registry import MetricRegistry

    return MetricRegistry.load(metric_registry_path)


@pytest.fixture(scope="session")
def fixture_config(project_root: Path):
    from m3rs_eval.config import load_config

    return load_config(project_root / "configs" / "fixture.yaml")


@pytest.fixture(scope="session")
def protocol(fixture_config):
    import yaml

    return yaml.safe_load(fixture_config.protocol_path.read_text(encoding="utf-8"))


@pytest.fixture
def levir_adapter(fixture_config, protocol, registry):
    from m3rs_eval.datasets import create_adapters

    return _adapter_by_name(create_adapters(fixture_config, protocol, registry), "levir_cc")


@pytest.fixture
def vrsbench_adapter(fixture_config, protocol, registry):
    from m3rs_eval.datasets import create_adapters

    return _adapter_by_name(create_adapters(fixture_config, protocol, registry), "vrsbench")


@pytest.fixture
def xlrs_bench_adapter(fixture_config, protocol, registry):
    from m3rs_eval.datasets import create_adapters

    return _adapter_by_name(create_adapters(fixture_config, protocol, registry), "xlrs_bench")


@pytest.fixture
def mme_rs_adapter(fixture_config, protocol, registry):
    from m3rs_eval.datasets import create_adapters

    return _adapter_by_name(create_adapters(fixture_config, protocol, registry), "mme_rs")


def _adapter_by_name(adapters, dataset: str):
    return next(adapter for adapter in adapters if adapter.dataset == dataset)


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str
    output_root: Path


@pytest.fixture
def config_factory(project_root: Path, fixture_config, tmp_path):
    def create(behavior: str = "ok", *, secret: str | None = None):
        output_root = tmp_path / f"runs-{behavior}-{len(list(tmp_path.glob('config-*.yaml')))}"
        protocol = yaml.safe_load(fixture_config.protocol_path.read_text(encoding="utf-8"))
        protocol["metric_namespace"] = str(fixture_config.metric_registry_path)
        protocol["datasets"]["xlrs_bench"]["variants"]["full"]["required_metric_ids"] = [
            "xlrs.caption.en.bleu_4",
            "xlrs.caption.zh.bleu_4",
            "xlrs.grounding.en.all.acc_0_5",
            "xlrs.grounding.zh.all.acc_0_5",
        ]
        protocol["datasets"]["xlrs_bench"]["variants"]["lite"]["required_metric_ids"] = [
            "xlrs.vqa.en.lite.micro_acc"
        ]
        protocol_path = tmp_path / f"fixture-protocol-{len(list(tmp_path.glob('config-*.yaml')))}.yaml"
        protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
        command = [
            sys.executable,
            str(project_root / "tools" / "fake_system.py"),
            "--input",
            "{input_jsonl}",
            "--output",
            "{output_jsonl}",
            "--behavior",
            behavior,
        ]
        payload = {
            "project_root": str(project_root.parent),
            "protocol_path": str(protocol_path),
            "metric_registry_path": str(fixture_config.metric_registry_path),
            "output_root": str(output_root),
            "system_version": fixture_config.system_version,
            "model_name": fixture_config.model_name,
            "model_weights": str(project_root / "test_fixtures" / "images" / "fixture.png"),
            "training_data_version": fixture_config.training_data_version,
            "operator": fixture_config.operator,
            "system": {
                "command": command,
                "working_directory": str(project_root),
                "timeout_seconds": 1 if behavior == "timeout" else 5,
                "environment": {"API_TOKEN": secret} if secret is not None else {},
            },
            "datasets": {},
        }
        for name, dataset in fixture_config.datasets.items():
            row = {
                "root": str(dataset.root),
                "asset_root": str(dataset.asset_root),
                "profile": dataset.profile,
            }
            if dataset.official_scorer_output is not None:
                row["official_scorer_output"] = str(dataset.official_scorer_output)
            if dataset.official_scorer_expected_version is not None:
                row["official_scorer_expected_version"] = dataset.official_scorer_expected_version
            payload["datasets"][name] = row
        path = tmp_path / f"config-{behavior}-{len(list(tmp_path.glob('config-*.yaml')))}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path, output_root

    return create


@pytest.fixture
def cli_runner(project_root: Path):
    def run(*args: object, output_root: Path | None = None) -> CliResult:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project_root / "src")
        completed = subprocess.run(
            [sys.executable, "-m", "m3rs_eval", *(str(arg) for arg in args)],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=60,
        )
        return CliResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
            output_root or Path(),
        )

    return run


def load_only_run_manifest(output_root: Path) -> tuple[Path, dict]:
    manifests = list(output_root.glob("*/run_manifest.json"))
    assert len(manifests) == 1
    return manifests[0].parent, json.loads(manifests[0].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Task 7 fixtures: run_factory, history_fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def run_factory(tmp_path):
    """Return a factory that creates synthetic run packages under tmp_path/runs/.

    Creates *run_manifest.json*, *metrics.jsonl*, *coverage.json*,
    *resource_metrics.json*, and *failures.jsonl*.

    Usage::

        run_factory("r1", mode="full", status="complete", eligible=True,
                    metrics={"mme_rs.avg": 0.85}, created_at="2025-01-01T00:00:00+08:00")
    """

    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    def create(
        run_id: str,
        *,
        mode: str = "full",
        status: str = "complete",
        eligible: bool = True,
        hardware: str | None = None,
        xlrs_protocol: str | None = None,
        created_at: str | None = None,
        metrics: dict[str, float] | None = None,
        protocol_id: str = "test-protocol-v1",
    ) -> Path:
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        gpu_name = hardware or "NVIDIA-A100"
        metadata: dict[str, Any] = {
            "system_version": f"test-system-{run_id}",
            "limit": None,
            "doctor": {"passed": True},
            "environment": {
                "gpus": [gpu_name],
                "platform": "linux",
                "cpu": "Intel-Xeon",
            },
            "dataset_manifest_hashes": {"test_ds": "hash_ds_123"},
        }
        if xlrs_protocol is not None:
            metadata["xlrs_protocol"] = xlrs_protocol

        manifest = {
            "run_id": run_id,
            "status": status,
            "mode": mode,
            "protocol_id": protocol_id,
            "created_at": created_at or "2025-06-01T00:00:00+08:00",
            "config_hash": "abc12345",
            "request_manifest_hash": "reqhash001",
            "eligible_for_history": eligible,
            "protocol_hash": "prothash001",
            "command_hash": "cmdhash001",
            "system_version_hash": f"syshash_{run_id}",
            "metadata": metadata,
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

        # Write metrics.jsonl
        if metrics:
            records = []
            for metric_id_key, value in metrics.items():
                records.append({
                    "record_schema_version": 2,
                    "run_id": run_id,
                    "metric_id": metric_id_key,
                    "availability": "available",
                    "provenance": "official",
                    "value_canonical": value,
                    "n_samples": 100,
                    "n_failures": 0,
                    "recorded_at": created_at or "2025-06-01T00:00:00+08:00",
                    "protocol_id": protocol_id,
                    "benchmark_version": "bench_v1",
                })
            with (run_dir / "metrics.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
                    fh.write("\n")
        else:
            # Empty metrics.jsonl
            (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")

        # Write coverage.json
        coverage = {
            "status": status,
            "datasets": {
                "test_ds": {
                    "status": "complete" if status == "complete" else "incomplete",
                    "expected": 100,
                    "requested": 100,
                    "predicted": 100,
                    "failed": 0,
                    "coverage": 1.0 if status == "complete" else 0.0,
                }
            },
        }
        (run_dir / "coverage.json").write_text(
            json.dumps(coverage, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

        # Write resource_metrics.json (minimal)
        resource = {
            "datasets": {
                "test_ds": {"duration_seconds": 60.0}
            },
            "latency": {},
            "total_command_duration_seconds": 60.0,
        }
        (run_dir / "resource_metrics.json").write_text(
            json.dumps(resource, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

        # Write empty failures.jsonl
        (run_dir / "failures.jsonl").write_text("", encoding="utf-8")

        return run_dir

    create.runs_root = runs_root
    return create


@pytest.fixture
def history_fixture(run_factory, tmp_path, registry):
    """Build a synthetic 3-run compatible history plus an incompatible run.

    r1 (oldest) -> r2 -> r3 (newest, baseline).
    mme_rs.avg: r2=0.85 (best), r3=0.80, r1=0.70
    system.latency.e2e_p50_ms: r1=200, r2=150, r3=100 (best)
    Also creates a candidate run with mme_rs.avg=0.78, latency=90.
    """
    from m3rs_eval.history import rebuild_history

    # r1 - oldest, lower mme_rs
    run_factory(
        "r1",
        mode="full",
        status="complete",
        eligible=True,
        created_at="2025-01-15T10:00:00+08:00",
        metrics={
            "mme_rs.avg": 0.70,
            "system.latency.e2e_p50_ms": 200.0,
        },
    )

    # r2 - middle, best mme_rs.avg
    run_factory(
        "r2",
        mode="full",
        status="complete",
        eligible=True,
        created_at="2025-02-15T10:00:00+08:00",
        metrics={
            "mme_rs.avg": 0.85,
            "system.latency.e2e_p50_ms": 150.0,
        },
    )

    # r3 - newest, baseline, best latency
    run_factory(
        "r3",
        mode="full",
        status="complete",
        eligible=True,
        created_at="2025-03-15T10:00:00+08:00",
        metrics={
            "mme_rs.avg": 0.80,
            "system.latency.e2e_p50_ms": 100.0,
        },
    )

    # incompatible run - different xlrs_protocol
    run_factory(
        "r4-incompat",
        mode="full",
        status="complete",
        eligible=True,
        xlrs_protocol="lite",
        created_at="2025-03-20T10:00:00+08:00",
        metrics={
            "mme_rs.avg": 0.75,
        },
    )

    # candidate run
    run_factory(
        "candidate",
        mode="full",
        status="complete",
        eligible=True,
        created_at="2025-04-01T10:00:00+08:00",
        metrics={
            "mme_rs.avg": 0.78,
            "system.latency.e2e_p50_ms": 90.0,
        },
    )

    return rebuild_history(run_factory.runs_root, tmp_path / "history")
