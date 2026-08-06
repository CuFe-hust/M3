from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from m3rs_eval.metadata import collect_environment_metadata, collect_git_metadata
from m3rs_eval.resources import NvidiaSmiProbe, ResourceSampler, parse_gpu_process_rows, summarize_latencies


def test_metadata_marks_unavailable_git_without_a_repository(tmp_path: Path):
    result = collect_git_metadata(tmp_path)

    assert result["git_commit"] == "unavailable"
    assert result["git_dirty"] is None
    assert "git metadata unavailable" in result["warnings"]


def test_environment_metadata_records_real_provenance_and_redacts_overrides(
    fixture_config, tmp_path: Path
):
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"weights")
    config = replace(
        fixture_config,
        model_weights=weights,
        system=replace(fixture_config.system, environment={"DB_CREDENTIAL": "secret-value", "BATCH": "2"}),
    )

    metadata = collect_environment_metadata(config)

    assert metadata["hostname"]
    assert metadata["cpu"]["logical_count"] is not None
    assert metadata["ram_bytes"] is not None
    assert metadata["python"]["version"]
    assert metadata["packages"]["psutil"]
    assert metadata["model_bytes"] == 7
    assert len(metadata["config_hash"]) == 64
    assert metadata["environment_overrides"] == {"BATCH": "2"}
    assert "secret-value" not in repr(metadata)
    assert "DB_CREDENTIAL" not in repr(metadata)
    assert "environment" not in metadata


def test_environment_metadata_sums_a_model_directory_deterministically(fixture_config, tmp_path: Path):
    weights = tmp_path / "weights"
    (weights / "nested").mkdir(parents=True)
    (weights / "a.bin").write_bytes(b"abc")
    (weights / "nested" / "b.bin").write_bytes(b"defgh")

    metadata = collect_environment_metadata(replace(fixture_config, model_weights=weights))

    assert metadata["model_bytes"] == 8


def test_latency_summary_uses_successes_for_quantiles_and_failures_for_rate():
    summary = summarize_latencies([10.0, 20.0, 30.0], failures=1)

    assert summary.successes == 3
    assert summary.failures == 1
    assert summary.p50_ms == 20.0
    assert summary.p95_ms == pytest.approx(29.0)
    assert summary.failure_rate == pytest.approx(0.25)


def test_latency_summary_has_stable_null_quantiles_without_successes():
    summary = summarize_latencies([], failures=2)

    assert summary.p50_ms is None
    assert summary.p95_ms is None
    assert summary.failure_rate == 1.0


def test_resource_sampler_is_idempotent_and_observes_a_process_tree():
    process = subprocess.Popen(["python", "-c", "import time; time.sleep(0.25)"], shell=False)
    sampler = ResourceSampler(sample_interval_seconds=0.02)
    try:
        sampler.start(process.pid)
        sampler.start(process.pid)
        process.wait(timeout=2)
        time.sleep(0.05)
        first = sampler.stop()
        second = sampler.stop()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)

    assert first == second
    assert first.sample_count >= 1
    assert first.duration_seconds >= 0
    assert hasattr(first, "peak_rss_bytes")
    assert hasattr(first, "peak_gpu_memory_bytes")


def test_resource_sampler_reports_positive_cpu_for_a_cpu_bound_child():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; end=time.monotonic()+0.6; value=0\nwhile time.monotonic()<end: value += 1",
        ],
        shell=False,
    )
    sampler = ResourceSampler(sample_interval_seconds=0.03)
    try:
        sampler.start(process.pid)
        process.wait(timeout=2)
        summary = sampler.stop()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)

    assert summary.peak_cpu_percent is not None
    assert summary.peak_cpu_percent > 0


def test_resource_sampler_concurrent_stop_callers_share_one_summary():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"], shell=False)
    sampler = ResourceSampler(sample_interval_seconds=0.01)
    sampler.start(process.pid)
    barrier = threading.Barrier(5)
    results = []

    def stop() -> None:
        barrier.wait()
        results.append(sampler.stop())

    threads = [threading.Thread(target=stop) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    if process.poll() is None:
        process.kill()
        process.wait(timeout=2)

    assert len(results) == 5
    assert all(result is results[0] for result in results)


def test_gpu_process_parser_sums_only_requested_pids_and_reports_bad_rows():
    memory, warnings = parse_gpu_process_rows("10, 100\n11, 25\nbad-row\n12, nope\n", {10, 12})

    assert memory == 100 * 1024 * 1024
    assert warnings == {"nvidia-smi returned an unparseable process memory value", "nvidia-smi returned an unparseable process row"}


def test_gpu_probe_caches_nvidia_smi_unavailability(monkeypatch):
    calls = []

    def unavailable(*args, **kwargs):
        calls.append((args, kwargs))
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("m3rs_eval.resources.subprocess.run", unavailable)
    probe = NvidiaSmiProbe()

    assert probe.memory_for_pids({1}) is None
    assert probe.memory_for_pids({1}) is None
    assert len(calls) == 1
    assert probe.warnings == {"nvidia-smi unavailable"}


def test_gpu_probe_caches_timeout_failure(monkeypatch):
    calls = []

    def timed_out(*args, **kwargs):
        calls.append((args, kwargs))
        raise subprocess.TimeoutExpired("nvidia-smi", 1)

    monkeypatch.setattr("m3rs_eval.resources.subprocess.run", timed_out)
    probe = NvidiaSmiProbe()

    assert probe.memory_for_pids({1}) is None
    assert probe.memory_for_pids({1}) is None
    assert len(calls) == 1
    assert probe.warnings == {"nvidia-smi unavailable"}
