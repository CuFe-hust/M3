"""Collect auditable host and configuration metadata without persisting secrets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from m3rs_eval.config import EvaluationConfig
from m3rs_eval.redaction import redact_argv, redact_mapping


_PACKAGE_NAMES = ("m3rs-eval", "jsonschema", "PyYAML", "psutil")
_TOOL_TIMEOUT_SECONDS = 2


def collect_environment_metadata(config: EvaluationConfig) -> dict[str, Any]:
    """Return only observed environment/configuration provenance and warnings."""
    warnings: list[str] = []
    git_metadata = collect_git_metadata(config.project_root)
    warnings.extend(git_metadata.pop("warnings"))
    model_bytes = _model_size(config.model_weights, warnings)
    packages = _package_versions(warnings)
    gpus = _collect_nvidia_gpus(warnings)
    cuda_version = _collect_cuda_version(warnings) if gpus is not None else None
    virtual_memory = psutil.virtual_memory()
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "platform": {
            "system": platform.system() or None,
            "release": platform.release() or None,
            "version": platform.version() or None,
            "machine": platform.machine() or None,
        },
        "hostname": socket.gethostname() or None,
        "cpu": {
            "logical_count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "model": platform.processor() or None,
        },
        "ram_bytes": virtual_memory.total,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": packages,
        "model_path": str(config.model_weights),
        "model_bytes": model_bytes,
        "config_hash": configuration_hash(config),
        "command": redact_argv(
            config.system.command, config.system.sensitive_argument_positions
        ),
        "working_directory": str(config.system.working_directory),
        "environment_overrides": _redact_environment(config.system.environment),
        "gpus": gpus,
        "cuda_version": cuda_version,
        **git_metadata,
        "warnings": sorted(set(warnings)),
    }


def collect_git_metadata(root: Path) -> dict[str, Any]:
    """Query Git without assuming the configured directory is a repository."""
    root = Path(root)
    try:
        commit = _run_text(["git", "rev-parse", "HEAD"], root)
        dirty_output = _run_text(["git", "status", "--porcelain"], root)
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return {
            "git_available": False,
            "git_commit": "unavailable",
            "git_dirty": None,
            "warnings": ["git metadata unavailable"],
        }
    return {
        "git_available": True,
        "git_commit": commit.strip() or "unavailable",
        "git_dirty": bool(dirty_output.strip()),
        "warnings": [],
    }


def configuration_hash(config: EvaluationConfig) -> str:
    """Hash full resolved configuration values while persisting only the digest."""
    payload = asdict(config)
    normalized = _normalize(payload)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_text(argv: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        check=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        timeout=_TOOL_TIMEOUT_SECONDS,
    )
    return completed.stdout


def _collect_nvidia_gpus(warnings: list[str]) -> list[dict[str, Any]] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=_TOOL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        warnings.append("nvidia-smi unavailable")
        return None
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            warnings.append("nvidia-smi returned an unparseable GPU row")
            continue
        try:
            memory_total_mib = int(parts[2])
        except ValueError:
            memory_total_mib = None
            warnings.append("nvidia-smi returned an unparseable GPU memory value")
        rows.append(
            {
                "name": parts[0] or None,
                "driver_version": parts[1] or None,
                "memory_total_mib": memory_total_mib,
            }
        )
    return rows


def _collect_cuda_version(warnings: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=_TOOL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        warnings.append("nvidia-smi CUDA version unavailable")
        return None
    match = re.search(r"CUDA Version:\s*([^\s|]+)", completed.stdout)
    if match is None:
        warnings.append("nvidia-smi CUDA version unavailable")
        return None
    return match.group(1)


def _model_size(path: Path, warnings: list[str]) -> int | None:
    try:
        resolved = Path(path)
        if resolved.is_file():
            return resolved.stat().st_size
        if not resolved.is_dir():
            raise OSError("model path is neither a file nor directory")
        total = 0
        for entry in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                warnings.append("model directory entry size unavailable")
        return total
    except OSError:
        warnings.append("model file size unavailable")
        return None


def _package_versions(warnings: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            warnings.append(f"package version unavailable: {name}")
    return versions


def _redact_environment(environment: dict[str, str]) -> dict[str, str]:
    return {str(key): str(value) for key, value in redact_mapping(environment).items()}


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(child) for child in value]
    return value
