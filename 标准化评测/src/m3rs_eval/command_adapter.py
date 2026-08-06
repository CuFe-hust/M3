"""Run an external evaluation system without invoking a shell."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import psutil

from m3rs_eval.config import SystemCommandConfig
from m3rs_eval.redaction import TextRedactor, redact_argv, redact_mapping
from m3rs_eval.windows_job import WindowsJobController, WindowsJobError

if TYPE_CHECKING:
    from m3rs_eval.resources import ResourceSampler, ResourceSummary


class CommandConfigError(ValueError):
    """Raised when a configured command cannot be rendered safely."""


class CommandExecutionError(RuntimeError):
    """Raised when execution evidence cannot be safely collected."""


_PLACEHOLDERS = {"{input_jsonl}", "{output_jsonl}"}
_TERMINATE_GRACE_SECONDS = 0.5


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int | None
    started_at: str
    finished_at: str
    duration_seconds: float
    stdout_path: Path
    stderr_path: Path
    timed_out: bool
    environment_overrides: dict[str, str]
    resource_summary: "ResourceSummary | None"
    resource_sampler_error: str | None


def render_argv(
    command_config: SystemCommandConfig, input_path: Path, output_path: Path
) -> list[str]:
    """Render only whole-argument input/output placeholders into an argv array."""
    replacements = {
        "{input_jsonl}": str(Path(input_path)),
        "{output_jsonl}": str(Path(output_path)),
    }
    argv: list[str] = []
    for argument in command_config.command:
        if argument in replacements:
            argv.append(replacements[argument])
        elif "{" in argument or "}" in argument:
            if argument in {"{input}", "{output}"}:
                raise CommandConfigError(
                    "placeholder migration: use {input_jsonl} and {output_jsonl}"
                )
            raise CommandConfigError(f"unknown or nonexact placeholder in command argument: {argument}")
        else:
            argv.append(argument)
    if not argv:
        raise CommandConfigError("command must contain at least one argv element")
    return argv


def run_system(
    command_config: SystemCommandConfig,
    input_path: Path,
    output_path: Path,
    log_dir: Path,
    resource_sampler: "ResourceSampler | None" = None,
) -> CommandResult:
    """Execute one configured command and retain independent stdout/stderr evidence."""
    execution_argv = render_argv(command_config, input_path, output_path)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "system.stdout.log"
    stderr_path = log_dir / "system.stderr.log"
    environment = _resolved_environment(command_config.environment)
    started_at = _utc_now()
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None

    popen_kwargs: dict[str, object] = {
        "cwd": str(command_config.working_directory),
        "env": environment,
        "shell": False,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    redactor = TextRedactor.from_execution(
        command_config.environment, execution_argv, command_config.sensitive_argument_positions
    )
    resource_summary: ResourceSummary | None = None
    resource_sampler_error: str | None = None
    sampler_started = False
    with subprocess.Popen(
        execution_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **popen_kwargs,
    ) as process:
        job: WindowsJobController | None = None
        if os.name == "nt":
            try:
                job = WindowsJobController.create()
                job.assign(process)
            except WindowsJobError as error:
                if job is not None:
                    try:
                        job.close()
                    except WindowsJobError:
                        pass
                _terminate_process_tree(process)
                raise CommandExecutionError(f"could not establish Windows Job Object ownership: {error}") from error
        stdout_pump = _RedactedLogPump(process.stdout, stdout_path, redactor, "stdout")
        stderr_pump = _RedactedLogPump(process.stderr, stderr_path, redactor, "stderr")
        stdout_pump.start()
        stderr_pump.start()
        try:
            if resource_sampler is not None:
                try:
                    resource_sampler.start(process.pid)
                    sampler_started = True
                except Exception as error:
                    resource_sampler_error = f"sampler start failed: {error}"
            returncode = process.wait(timeout=command_config.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if job is not None:
                try:
                    job.close()
                except WindowsJobError as error:
                    resource_sampler_error = _append_error(
                        resource_sampler_error, f"Windows Job Object close failed: {error}"
                    )
            returncode = _terminate_process_tree(process)
        finally:
            if resource_sampler is not None and sampler_started:
                try:
                    resource_summary = resource_sampler.stop()
                except Exception as error:
                    resource_sampler_error = _append_error(
                        resource_sampler_error, f"sampler stop failed: {error}"
                    )
            if job is not None:
                try:
                    job.close()
                except WindowsJobError as error:
                    resource_sampler_error = _append_error(
                        resource_sampler_error, f"Windows Job Object close failed: {error}"
                    )
            stdout_pump.join()
            stderr_pump.join()

    finished_at = _utc_now()
    return CommandResult(
        argv=redact_argv(execution_argv, command_config.sensitive_argument_positions),
        returncode=returncode,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.monotonic() - started,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timed_out=timed_out,
        environment_overrides=_redacted_environment(command_config.environment),
        resource_summary=resource_summary,
        resource_sampler_error=resource_sampler_error,
    )


def _resolved_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(overrides)
    return environment


def _redacted_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    return {str(key): str(value) for key, value in redact_mapping(overrides).items()}


class _RedactedLogPump:
    """Write only redacted complete records; retain incomplete chunks in memory."""

    def __init__(self, stream: object, path: Path, redactor: TextRedactor, label: str) -> None:
        self._stream = stream
        self._path = path
        self._redactor = redactor
        self._label = label
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name=f"m3rs-{label}-pump", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join()
        if self._error is not None:
            raise CommandExecutionError(f"{self._label} log pump failed: {self._error}") from self._error

    def _run(self) -> None:
        try:
            pending = ""
            with self._path.open("w", encoding="utf-8", newline="\n") as handle:
                while True:
                    chunk = self._stream.read(4096)  # type: ignore[union-attr]
                    if not chunk:
                        break
                    pending += chunk
                    complete, pending = _complete_lines(pending)
                    if complete:
                        handle.write(self._redactor.redact(complete))
                        handle.flush()
                if pending:
                    handle.write(self._redactor.redact(pending))
                    handle.flush()
        except Exception as error:
            self._error = error


def _complete_lines(value: str) -> tuple[str, str]:
    index = max(value.rfind("\n"), value.rfind("\r"))
    return (value[: index + 1], value[index + 1 :]) if index >= 0 else ("", value)


def _append_error(existing: str | None, message: str) -> str:
    return message if existing is None else f"{existing}; {message}"


def _terminate_process_tree(process: subprocess.Popen[object]) -> int | None:
    """Terminate a timed-out child and descendants with a bounded hard-kill escalation."""
    snapshot = _process_tree(process.pid)
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        _taskkill_tree(process.pid)
    _wait_for_processes(snapshot, _TERMINATE_GRACE_SECONDS)
    _terminate_processes(_live_processes(snapshot + _process_tree(process.pid)))
    _wait_for_processes(snapshot + _process_tree(process.pid), _TERMINATE_GRACE_SECONDS)
    survivors = _live_processes(snapshot + _process_tree(process.pid))
    for item in reversed(survivors):
        _ignore_process_error(item.kill)
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _wait_for_processes(survivors, _TERMINATE_GRACE_SECONDS)
    try:
        return process.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        return process.poll()


def _taskkill_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
            shell=False,
            timeout=_TERMINATE_GRACE_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _terminate_processes(processes: list[psutil.Process]) -> None:
    for item in reversed(processes):
        _ignore_process_error(item.terminate)


def _wait_for_processes(processes: list[psutil.Process], timeout: float) -> None:
    if processes:
        psutil.wait_procs(processes, timeout=timeout)


def _live_processes(processes: list[psutil.Process]) -> list[psutil.Process]:
    live: list[psutil.Process] = []
    seen: set[tuple[int, float]] = set()
    for item in processes:
        try:
            identity = (item.pid, item.create_time())
            if identity not in seen and item.is_running() and item.status() != psutil.STATUS_ZOMBIE:
                live.append(item)
                seen.add(identity)
        except (psutil.Error, OSError):
            continue
    return live


def _process_tree(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        return []


def _ignore_process_error(action: object) -> None:
    try:
        action()  # type: ignore[operator]
    except (psutil.Error, OSError):
        pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
