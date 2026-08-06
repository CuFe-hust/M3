"""Preflight diagnostics for configuration, protocol, data, and host readiness."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from m3rs_eval.config import EvaluationConfig
from m3rs_eval.datasets import create_adapters
from m3rs_eval.metadata import collect_git_metadata
from m3rs_eval.registry import MetricRegistry


MINIMUM_FREE_DISK_BYTES = 1024**3


@dataclass(frozen=True)
class DoctorCheck:
    code: str
    level: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def passed(self) -> bool:
        return not any(check.level == "error" and not check.passed for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "checks": [check.to_dict() for check in self.checks],
        }


def run_doctor(config: EvaluationConfig) -> DoctorReport:
    """Run deterministic blocking checks and optional warning diagnostics."""
    checks: list[DoctorCheck] = []
    _check_required_paths(config, checks)
    _check_command(config, checks)
    output_ready = _check_output(config.output_root, checks)
    if output_ready:
        _check_disk(config.output_root, checks)

    protocol = _load_yaml(config.protocol_path, "PROTOCOL_INVALID", checks)
    registry = _load_registry(config.metric_registry_path, checks)
    if protocol is not None:
        _check_locked_protocol(protocol, checks)
    if protocol is not None and registry is not None:
        _check_datasets(config, protocol, registry, checks)
    _check_official_scorers(config, checks)
    _check_optional_host_tools(config, checks)
    return DoctorReport(tuple(checks))


def _check_required_paths(config: EvaluationConfig, checks: list[DoctorCheck]) -> None:
    paths = {
        "project_root": (config.project_root, "directory"),
        "protocol_path": (config.protocol_path, "file"),
        "metric_registry_path": (config.metric_registry_path, "file"),
        "system.working_directory": (config.system.working_directory, "directory"),
    }
    for name, dataset in config.datasets.items():
        paths[f"datasets.{name}.root"] = (dataset.root, "directory")
        if dataset.asset_root is not None:
            paths[f"datasets.{name}.asset_root"] = (dataset.asset_root, "directory")
    official = all(dataset.profile == "official" for dataset in config.datasets.values())
    if official:
        paths["model_weights"] = (config.model_weights, "path")
    elif not config.model_weights.exists():
        checks.append(
            DoctorCheck(
                "FIXTURE_MODEL_UNAVAILABLE",
                "warning",
                False,
                f"fixture model path is unavailable and was not required: {config.model_weights}",
            )
        )

    for name, (path, kind) in paths.items():
        exists = path.is_dir() if kind == "directory" else path.exists()
        if kind == "file":
            exists = path.is_file()
        checks.append(
            DoctorCheck(
                "PATH_OK" if exists else "REQUIRED_PATH_MISSING",
                "info" if exists else "error",
                exists,
                f"{name}: {path}",
            )
        )


def _check_command(config: EvaluationConfig, checks: list[DoctorCheck]) -> None:
    executable = config.system.command[0]
    candidate = Path(executable)
    if candidate.is_absolute() or candidate.parent != Path("."):
        if not candidate.is_absolute():
            candidate = config.system.working_directory / candidate
        found = candidate.is_file()
        detail = str(candidate)
    else:
        search_path = config.system.environment.get("PATH", os.environ.get("PATH"))
        resolved = shutil.which(executable, path=search_path)
        found = resolved is not None
        detail = resolved or executable
    checks.append(
        DoctorCheck(
            "COMMAND_EXECUTABLE" if found else "COMMAND_NOT_EXECUTABLE",
            "info" if found else "error",
            found,
            detail,
        )
    )


def _check_output(output_root: Path, checks: list[DoctorCheck]) -> bool:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_root, prefix=".doctor-", suffix=".tmp", delete=False
        ) as handle:
            handle.write("m3rs doctor probe\n")
            handle.flush()
            os.fsync(handle.fileno())
            probe = Path(handle.name)
        probe.unlink()
    except OSError as error:
        checks.append(DoctorCheck("OUTPUT_NOT_WRITABLE", "error", False, str(error)))
        return False
    checks.append(DoctorCheck("OUTPUT_WRITABLE", "info", True, str(output_root)))
    return True


def _check_disk(output_root: Path, checks: list[DoctorCheck]) -> None:
    try:
        free = shutil.disk_usage(output_root).free
    except OSError as error:
        checks.append(DoctorCheck("DISK_SPACE_UNKNOWN", "error", False, str(error)))
        return
    enough = free >= MINIMUM_FREE_DISK_BYTES
    checks.append(
        DoctorCheck(
            "DISK_SPACE_OK" if enough else "DISK_SPACE_LOW",
            "info" if enough else "error",
            enough,
            f"free_bytes={free}; required_bytes={MINIMUM_FREE_DISK_BYTES}",
        )
    )


def _load_yaml(path: Path, code: str, checks: list[DoctorCheck]) -> Mapping[str, Any] | None:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("top level must be a mapping")
    except (OSError, yaml.YAMLError, ValueError) as error:
        checks.append(DoctorCheck(code, "error", False, str(error)))
        return None
    checks.append(DoctorCheck("PROTOCOL_VALID", "info", True, str(path)))
    return raw


def _load_registry(path: Path, checks: list[DoctorCheck]) -> MetricRegistry | None:
    try:
        registry = MetricRegistry.load(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        checks.append(DoctorCheck("METRIC_REGISTRY_INVALID", "error", False, str(error)))
        return None
    checks.append(DoctorCheck("METRIC_REGISTRY_VALID", "info", True, str(path)))
    return registry


def _check_locked_protocol(protocol: Mapping[str, Any], checks: list[DoctorCheck]) -> None:
    datasets = protocol.get("datasets")
    expected = {
        "levir_cc": ("formal_split", "test"),
        "vrsbench": ("formal_split", "test"),
        "xlrs_bench": ("formal_split", "public_test"),
        "mme_rs": ("formal_domain", "Remote_Sensing"),
    }
    for dataset, (field, value) in expected.items():
        actual = datasets.get(dataset, {}).get(field) if isinstance(datasets, Mapping) else None
        passed = actual == value
        checks.append(
            DoctorCheck(
                "PROTOCOL_SCOPE_LOCKED" if passed else "DATASET_SPLIT_LEAKAGE",
                "info" if passed else "error",
                passed,
                f"{dataset}.{field}={actual!r}; required={value!r}",
            )
        )


def _check_datasets(
    config: EvaluationConfig,
    protocol: Mapping[str, Any],
    registry: MetricRegistry,
    checks: list[DoctorCheck],
) -> None:
    try:
        adapters = create_adapters(config, protocol, registry)
    except (OSError, ValueError) as error:
        checks.append(DoctorCheck("DATASET_CONFIGURATION_INVALID", "error", False, str(error)))
        return
    for adapter in adapters:
        for result in adapter.preflight():
            checks.append(
                DoctorCheck(
                    "DATASET_FORMAT_VALID" if result.passed else "DATASET_FORMAT_INVALID",
                    "info" if result.passed else "error",
                    result.passed,
                    f"{adapter.dataset}: {result.detail}",
                )
            )


def _check_official_scorers(config: EvaluationConfig, checks: list[DoctorCheck]) -> None:
    for name, dataset in config.datasets.items():
        if dataset.official_scorer_output is not None:
            passed = dataset.official_scorer_output.is_file()
            checks.append(
                DoctorCheck(
                    "OFFICIAL_SCORER_EVIDENCE_VALID" if passed else "OFFICIAL_SCORER_PATH_MISSING",
                    "info" if passed else "error",
                    passed,
                    f"{name}: {dataset.official_scorer_output}",
                )
            )
        elif dataset.official_scorer_command is not None:
            working_directory = dataset.official_scorer_working_directory
            executable = dataset.official_scorer_command[0]
            resolved_executable = shutil.which(
                executable,
                path=dataset.official_scorer_environment.get("PATH", os.environ.get("PATH")),
            )
            if Path(executable).is_absolute() or Path(executable).parent != Path("."):
                executable_path = Path(executable)
                if not executable_path.is_absolute() and working_directory is not None:
                    executable_path = working_directory / executable_path
                executable_ok = executable_path.is_file()
            else:
                executable_ok = resolved_executable is not None
            script_paths = []
            for argument in dataset.official_scorer_command[1:]:
                candidate = Path(argument)
                if "{" in argument or argument.startswith("-"):
                    continue
                if candidate.is_absolute() or candidate.parent != Path("."):
                    if not candidate.is_absolute() and working_directory is not None:
                        candidate = working_directory / candidate
                    script_paths.append(candidate)
            passed = bool(
                working_directory is not None
                and working_directory.is_dir()
                and executable_ok
                and all(path.is_file() for path in script_paths)
            )
            checks.append(
                DoctorCheck(
                    "OFFICIAL_SCORER_COMMAND_VALID" if passed else "OFFICIAL_SCORER_COMMAND_INVALID",
                    "info" if passed else "error",
                    passed,
                    f"{name}: executable={executable!r}; working_directory={working_directory}; "
                    f"script_paths={[str(path) for path in script_paths]}",
                )
            )


def _check_optional_host_tools(config: EvaluationConfig, checks: list[DoctorCheck]) -> None:
    git = collect_git_metadata(config.project_root)
    git_available = bool(git["git_available"])
    checks.append(
        DoctorCheck(
            "GIT_AVAILABLE" if git_available else "GIT_UNAVAILABLE",
            "info" if git_available else "warning",
            git_available,
            str(git.get("git_commit", "unavailable")),
        )
    )
    nvidia_smi = shutil.which("nvidia-smi")
    checks.append(
        DoctorCheck(
            "GPU_TOOLING_AVAILABLE" if nvidia_smi else "GPU_TOOLING_UNAVAILABLE",
            "info" if nvidia_smi else "warning",
            nvidia_smi is not None,
            nvidia_smi or "nvidia-smi is optional and was not found",
        )
    )
