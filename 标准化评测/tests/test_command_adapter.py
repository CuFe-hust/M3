from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import psutil
import pytest

import m3rs_eval.command_adapter as command_adapter
from m3rs_eval.command_adapter import (
    CommandConfigError,
    CommandExecutionError,
    render_argv,
    run_system,
)
from m3rs_eval.config import SystemCommandConfig
from m3rs_eval.contracts import RequestRecord, read_jsonl
from m3rs_eval.fixture_input import FixtureInputError, prepare_fixture_command_input
from m3rs_eval.resources import ResourceSampler


@pytest.fixture
def request_path(tmp_path: Path) -> Path:
    path = tmp_path / "requests.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "fixture:001",
                "dataset": "fixture",
                "fixture_prediction": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_command_config(project_root: Path) -> SystemCommandConfig:
    return SystemCommandConfig(
        command=(
            sys.executable,
            str(project_root / "tools" / "fake_system.py"),
            "--input",
            "{input_jsonl}",
            "--output",
            "{output_jsonl}",
            "--behavior",
            "ok",
        ),
        working_directory=project_root,
        timeout_seconds=2,
        environment={"DEMO_TOKEN": "must-not-appear", "FIXTURE_FLAG": "enabled"},
    )


def test_runner_substitutes_only_exact_known_placeholders(
    fake_command_config: SystemCommandConfig, request_path: Path, tmp_path: Path
):
    output_path = tmp_path / "predictions.jsonl"

    result = run_system(fake_command_config, request_path, output_path, tmp_path / "logs")

    assert result.returncode == 0
    assert result.argv[2:6] == ["--input", str(request_path), "--output", str(output_path)]
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "sample_id": "fixture:001",
        "status": "ok",
        "prediction": "A",
        "raw_output": "fixture_prediction",
        "latency_ms": 1.0,
    }
    assert result.stdout_path.read_text(encoding="utf-8") == ""
    assert result.stderr_path.read_text(encoding="utf-8") == ""
    assert result.environment_overrides == {"FIXTURE_FLAG": "enabled"}


@pytest.mark.parametrize("argument", ["{shell_payload}", "prefix-{input_jsonl}", "{input}"])
def test_runner_rejects_unknown_or_nonexact_placeholders(
    fake_command_config: SystemCommandConfig, argument: str
):
    bad = replace(fake_command_config, command=(sys.executable, "x.py", argument))

    with pytest.raises(CommandConfigError, match="placeholder"):
        render_argv(bad, Path("in.jsonl"), Path("out.jsonl"))


def test_runner_reports_the_placeholder_migration_for_legacy_alias(fake_command_config: SystemCommandConfig):
    legacy = replace(fake_command_config, command=(sys.executable, "x.py", "{input}"))

    with pytest.raises(CommandConfigError, match="input_jsonl"):
        render_argv(legacy, Path("in.jsonl"), Path("out.jsonl"))


def test_runner_redacts_emitted_environment_and_argv_secrets_before_persistence(
    fake_command_config: SystemCommandConfig, tmp_path: Path
):
    script = tmp_path / "emit_secret.py"
    script.write_text(
        "import os, sys\n"
        "print('DB_CREDENTIAL=' + os.environ['DB_CREDENTIAL'])\n"
        "print('DB_CREDENTIAL=' + os.environ['DB_CREDENTIAL'], file=sys.stderr)\n",
        encoding="utf-8",
    )
    config = replace(
        fake_command_config,
        command=(sys.executable, str(script), "--api-key", "argv-secret"),
        environment={"DB_CREDENTIAL": "environment-secret", "BATCH": "2"},
    )

    result = run_system(config, tmp_path / "in.jsonl", tmp_path / "out.jsonl", tmp_path / "logs")

    persisted = "\n".join(
        [
            repr(result),
            result.stdout_path.read_text(encoding="utf-8"),
            result.stderr_path.read_text(encoding="utf-8"),
        ]
    )
    assert "environment-secret" not in persisted
    assert "argv-secret" not in persisted
    assert "DB_CREDENTIAL" not in persisted
    assert "--api-key" not in persisted
    assert result.environment_overrides == {"BATCH": "2"}


def test_runner_redacts_emitted_secret_names_case_insensitively(
    fake_command_config: SystemCommandConfig, tmp_path: Path
):
    script = tmp_path / "emit_lowercase_secret.py"
    script.write_text(
        "import os\nprint('db_credential=' + os.environ['DB_CREDENTIAL'])\n",
        encoding="utf-8",
    )
    config = replace(
        fake_command_config,
        command=(sys.executable, str(script)),
        environment={"DB_CREDENTIAL": "environment-secret"},
    )

    result = run_system(config, tmp_path / "in.jsonl", tmp_path / "out.jsonl", tmp_path / "logs")

    assert "db_credential" not in result.stdout_path.read_text(encoding="utf-8").casefold()


def test_runner_redacts_inline_option_secret_from_partial_nonzero_output(
    fake_command_config: SystemCommandConfig, tmp_path: Path
):
    script = tmp_path / "emit_partial_secret.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write('inline-'); sys.stdout.flush()\n"
        "sys.stdout.write('secret'); sys.stdout.flush()\n"
        "sys.stderr.write('TOKEN=inline-secret'); sys.stderr.flush()\n"
        "raise SystemExit(9)\n",
        encoding="utf-8",
    )
    config = replace(
        fake_command_config,
        command=(sys.executable, str(script), "--token=inline-secret"),
    )

    result = run_system(config, tmp_path / "in", tmp_path / "out", tmp_path / "logs")

    persisted = result.stdout_path.read_text(encoding="utf-8") + result.stderr_path.read_text(encoding="utf-8")
    assert result.returncode == 9
    assert "inline-secret" not in persisted
    assert "TOKEN" not in persisted


def test_runner_captures_sampler_start_error_without_losing_command_evidence(
    fake_command_config: SystemCommandConfig, request_path: Path, tmp_path: Path
):
    class FailingSampler:
        def start(self, pid: int) -> None:
            del pid
            raise RuntimeError("sampler unavailable")

        def stop(self):
            raise AssertionError("stop must not be called after start failure")

    result = run_system(
        fake_command_config,
        request_path,
        tmp_path / "out.jsonl",
        tmp_path / "logs",
        resource_sampler=FailingSampler(),
    )

    assert result.returncode == 0
    assert result.resource_sampler_error == "sampler start failed: sampler unavailable"
    assert result.stdout_path.is_file()


def test_runner_reports_log_pump_failure_after_command_completion(
    fake_command_config: SystemCommandConfig, request_path: Path, tmp_path: Path, monkeypatch
):
    def fail_pump(self) -> None:
        self._error = OSError("simulated log write failure")

    monkeypatch.setattr(command_adapter._RedactedLogPump, "_run", fail_pump)

    with pytest.raises(CommandExecutionError, match="log pump failed"):
        run_system(
            fake_command_config,
            request_path,
            tmp_path / "out.jsonl",
            tmp_path / "logs",
        )


def test_runner_records_sampler_stop_error_after_a_successful_start(
    fake_command_config: SystemCommandConfig, request_path: Path, tmp_path: Path
):
    class StopFailingSampler:
        def start(self, pid: int) -> None:
            del pid

        def stop(self):
            raise RuntimeError("sampler shutdown failed")

    result = run_system(
        fake_command_config,
        request_path,
        tmp_path / "out.jsonl",
        tmp_path / "logs",
        resource_sampler=StopFailingSampler(),
    )

    assert result.returncode == 0
    assert result.resource_sampler_error == "sampler stop failed: sampler shutdown failed"


def test_fixture_input_transform_is_ephemeral_and_drives_fake_system(
    vrsbench_adapter, fake_command_config: SystemCommandConfig, tmp_path: Path
):
    materialization = vrsbench_adapter.materialize("smoke", 1, tmp_path / "materialized")
    original_requests = materialization.requests_path.read_text(encoding="utf-8")
    original_references = materialization.references_path.read_text(encoding="utf-8")

    artifact = prepare_fixture_command_input(
        materialization.requests_path,
        materialization.references_path,
        tmp_path / "ephemeral" / "fake_input.jsonl",
        profile="fixture",
        formal_execution=False,
    )
    result = run_system(
        fake_command_config,
        artifact.path,
        tmp_path / "predictions.jsonl",
        tmp_path / "logs",
    )

    assert artifact.ephemeral and not artifact.eligible_for_history
    assert "fixture_prediction" not in original_requests
    assert all(
        "fixture_prediction" not in request.to_dict()
        for request in read_jsonl(materialization.requests_path, RequestRecord)
    )
    assert materialization.requests_path.read_text(encoding="utf-8") == original_requests
    assert materialization.references_path.read_text(encoding="utf-8") == original_references
    assert result.returncode == 0
    assert len((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    with pytest.raises(FixtureInputError):
        prepare_fixture_command_input(
            materialization.requests_path,
            materialization.references_path,
            tmp_path / "forbidden.jsonl",
            profile="official",
            formal_execution=False,
        )


def test_runner_attaches_resource_summary_without_exposing_a_process_handle(
    fake_command_config: SystemCommandConfig, request_path: Path, tmp_path: Path
):
    sampler = ResourceSampler(sample_interval_seconds=0.01)

    result = run_system(
        fake_command_config,
        request_path,
        tmp_path / "out.jsonl",
        tmp_path / "logs",
        resource_sampler=sampler,
    )

    assert result.resource_summary is not None
    assert result.resource_summary.sample_count >= 1


@pytest.mark.skipif(os.name == "nt", reason="escaped-session descendant fixture is POSIX-specific")
def test_timeout_cleans_a_descendant_that_escapes_the_parent_process_group(tmp_path: Path):
    script = tmp_path / "spawn_descendant.py"
    child_pid_path = tmp_path / "child.pid"
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)\n"
        "open(sys.argv[1], 'w', encoding='utf-8').write(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    config = SystemCommandConfig(
        command=(sys.executable, str(script), str(child_pid_path)),
        working_directory=tmp_path,
        timeout_seconds=1,
        environment={},
    )
    child_pid: int | None = None
    try:
        result = run_system(config, tmp_path / "in", tmp_path / "out", tmp_path / "logs")
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)

        assert result.timed_out
        assert not psutil.pid_exists(child_pid)
    finally:
        if child_pid is not None and psutil.pid_exists(child_pid):
            psutil.Process(child_pid).kill()


def test_runner_keeps_nonzero_exit_as_evidence(
    fake_command_config: SystemCommandConfig, request_path: Path, tmp_path: Path
):
    config = replace(fake_command_config, command=(*fake_command_config.command, "--behavior", "nonzero"))

    result = run_system(config, request_path, tmp_path / "out.jsonl", tmp_path / "logs")

    assert result.returncode == 17
    assert not result.timed_out
    assert result.duration_seconds >= 0


def test_runner_times_out_and_returns_timeout_evidence(
    fake_command_config: SystemCommandConfig, request_path: Path, tmp_path: Path
):
    config = replace(
        fake_command_config,
        command=(*fake_command_config.command, "--behavior", "timeout", "--sleep-seconds", "2"),
        timeout_seconds=1,
    )

    result = run_system(config, request_path, tmp_path / "out.jsonl", tmp_path / "logs")

    assert result.timed_out
    assert result.returncode is not None
    assert result.duration_seconds < 2


@pytest.mark.parametrize(
    ("behavior", "expected_lines", "expected_status"),
    [
        ("missing", 0, None),
        ("duplicate", 2, "ok"),
        ("malformed", 1, None),
        ("error", 1, "error"),
    ],
)
def test_fake_system_behaviors_are_deterministic(
    fake_command_config: SystemCommandConfig,
    request_path: Path,
    tmp_path: Path,
    behavior: str,
    expected_lines: int,
    expected_status: str | None,
):
    output_path = tmp_path / f"{behavior}.jsonl"
    config = replace(fake_command_config, command=(*fake_command_config.command, "--behavior", behavior))

    result = run_system(config, request_path, output_path, tmp_path / "logs")

    assert result.returncode == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == expected_lines
    if expected_status is not None:
        assert all(json.loads(line)["status"] == expected_status for line in lines)
    if behavior == "malformed":
        assert lines == ["not-json"]
