"""Configuration loading and secret redaction for evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from m3rs_eval.redaction import redact_argv, redact_mapping


class ConfigError(ValueError):
    """Raised when an evaluation configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    asset_root: Path | None = None
    profile: str = "official"
    official_scorer_command: tuple[str, ...] | None = None
    official_scorer_output: Path | None = None
    official_scorer_expected_version: str | None = None
    official_scorer_expected_commit: str | None = None
    official_scorer_working_directory: Path | None = None
    official_scorer_timeout_seconds: int = 300
    official_scorer_environment: dict[str, str] = field(default_factory=dict)
    scorer_sensitive_argument_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class SystemCommandConfig:
    command: tuple[str, ...]
    working_directory: Path
    timeout_seconds: int
    environment: dict[str, str]
    sensitive_argument_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class EvaluationConfig:
    project_root: Path
    protocol_path: Path
    metric_registry_path: Path
    output_root: Path
    system_version: str
    model_name: str
    model_weights: Path
    training_data_version: str
    operator: str
    system: SystemCommandConfig
    datasets: dict[str, DatasetConfig]


_REQUIRED_DATASETS = ("levir_cc", "vrsbench", "xlrs_bench", "mme_rs")
_EXAMPLE_ROOT = "/srv/m3rs"


def load_config(path: Path) -> EvaluationConfig:
    """Load a run configuration and resolve all paths from its directory."""
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"could not read configuration: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in configuration: {config_path}") from error

    config = _mapping(raw, "configuration")
    base_directory = config_path.parent
    datasets = _load_datasets(config.get("datasets"), base_directory)

    evaluation_config = EvaluationConfig(
        project_root=_path(config.get("project_root", ".."), "project_root", base_directory),
        protocol_path=_path(config.get("protocol_path"), "protocol_path", base_directory),
        metric_registry_path=_path(
            config.get("metric_registry_path"), "metric_registry_path", base_directory
        ),
        output_root=_path(config.get("output_root"), "output_root", base_directory),
        system_version=_text(config.get("system_version"), "system_version"),
        model_name=_text(config.get("model_name"), "model_name"),
        model_weights=_path(config.get("model_weights"), "model_weights", base_directory),
        training_data_version=_text(
            config.get("training_data_version"), "training_data_version"
        ),
        operator=_text(config.get("operator"), "operator"),
        system=_load_system(config.get("system"), base_directory),
        datasets=datasets,
    )
    _validate_metric_contract(evaluation_config.protocol_path, evaluation_config.metric_registry_path)
    _validate_scorer_profiles(evaluation_config.datasets)
    return evaluation_config


def redact_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a configuration with secret-bearing values masked."""
    return redact_mapping(raw)


def serialize_resolved_config(config: EvaluationConfig) -> dict[str, Any]:
    """Return the only command-aware resolved-config form safe for run-package persistence."""
    payload = _normalize_paths(asdict(config))
    system = payload["system"]
    system["command"] = redact_argv(
        config.system.command, config.system.sensitive_argument_positions
    )
    system["environment"] = redact_mapping(config.system.environment)
    for name, dataset in config.datasets.items():
        if dataset.official_scorer_command is not None:
            payload["datasets"][name]["official_scorer_command"] = redact_argv(
                dataset.official_scorer_command, dataset.scorer_sensitive_argument_positions
            )
        payload["datasets"][name]["official_scorer_environment"] = redact_mapping(
            dataset.official_scorer_environment
        )
    return payload


def _load_datasets(raw: Any, base_directory: Path) -> dict[str, DatasetConfig]:
    datasets = _mapping(raw, "datasets")
    loaded: dict[str, DatasetConfig] = {}
    for name in _REQUIRED_DATASETS:
        if name not in datasets:
            continue
        data = _mapping(datasets[name], f"datasets.{name}")
        root = _path(data.get("root"), f"datasets.{name}.root", base_directory)
        command = _optional_command(
            data.get("official_scorer_command"), f"datasets.{name}.official_scorer_command"
        )
        if command is not None:
            _validate_official_scorer_placeholders(command, f"datasets.{name}.official_scorer_command")
        output = _optional_path(
            data.get("official_scorer_output"),
            f"datasets.{name}.official_scorer_output",
            base_directory,
        )
        if command is not None and output is not None:
            raise ConfigError(
                f"datasets.{name} cannot configure both official_scorer_command and official_scorer_output"
            )
        profile = _optional_text(
            data.get("profile"), f"datasets.{name}.profile", default="official"
        )
        expected_version = _optional_nullable_text(
            data.get("official_scorer_expected_version"),
            f"datasets.{name}.official_scorer_expected_version",
        )
        expected_commit = _optional_nullable_text(
            data.get("official_scorer_expected_commit"),
            f"datasets.{name}.official_scorer_expected_commit",
        )
        if (command is not None or output is not None) and expected_version is None:
            raise ConfigError(
                f"datasets.{name} configured scorer requires official_scorer_expected_version"
            )
        environment = _mapping(
            data.get("official_scorer_environment", {}),
            f"datasets.{name}.official_scorer_environment",
        )
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
            raise ConfigError(
                f"datasets.{name}.official_scorer_environment must map strings to strings"
            )
        loaded[name] = DatasetConfig(
            root=root,
            asset_root=_optional_path(
                data.get("asset_root"), f"datasets.{name}.asset_root", base_directory
            ),
            profile=profile,
            official_scorer_command=command,
            official_scorer_output=output,
            official_scorer_expected_version=expected_version,
            official_scorer_expected_commit=expected_commit,
            official_scorer_working_directory=_optional_path(
                data.get("official_scorer_working_directory"),
                f"datasets.{name}.official_scorer_working_directory",
                base_directory,
            ),
            official_scorer_timeout_seconds=_optional_positive_int(
                data.get("official_scorer_timeout_seconds"),
                f"datasets.{name}.official_scorer_timeout_seconds",
                300,
            ),
            official_scorer_environment=dict(environment),
            scorer_sensitive_argument_positions=_sensitive_positions(
                data.get("scorer_sensitive_argument_positions"),
                f"datasets.{name}.scorer_sensitive_argument_positions",
                command,
            ),
        )
    missing = [name for name in _REQUIRED_DATASETS if name not in datasets]
    if missing:
        raise ConfigError(f"datasets missing required entries: {', '.join(missing)}")
    return loaded


def _validate_scorer_profiles(datasets: Mapping[str, DatasetConfig]) -> None:
    for name, dataset in datasets.items():
        if dataset.official_scorer_output is not None and dataset.profile != "fixture":
            raise ConfigError(
                f"datasets.{name}.official_scorer_output is fixture-only and forbidden for profile=official"
            )
        if dataset.profile != "official" or name not in {"levir_cc", "vrsbench", "xlrs_bench"}:
            continue
        if dataset.official_scorer_command is None:
            raise ConfigError(f"datasets.{name} profile=official requires official_scorer_command")
        if dataset.official_scorer_expected_version is None:
            raise ConfigError(
                f"datasets.{name} profile=official requires official_scorer_expected_version"
            )
        if dataset.official_scorer_expected_commit is None:
            raise ConfigError(
                f"datasets.{name} profile=official requires official_scorer_expected_commit"
            )


def _load_system(raw: Any, base_directory: Path) -> SystemCommandConfig:
    system = _mapping(raw, "system")
    command = system.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part.strip() for part in command
    ):
        raise ConfigError("system.command must be a nonempty list of strings")
    _validate_command_placeholders(command)

    timeout = system.get("timeout_seconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise ConfigError("system.timeout_seconds must be a positive integer")

    environment = _mapping(system.get("environment", {}), "system.environment")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
        raise ConfigError("system.environment must map strings to strings")

    sensitive_positions = system.get("sensitive_argument_positions", [])
    if not isinstance(sensitive_positions, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(command)
        for index in sensitive_positions
    ):
        raise ConfigError("system.sensitive_argument_positions must contain valid command indexes")

    return SystemCommandConfig(
        command=tuple(command),
        working_directory=_path(
            system.get("working_directory"), "system.working_directory", base_directory
        ),
        timeout_seconds=timeout,
        environment=dict(environment),
        sensitive_argument_positions=tuple(sensitive_positions),
    )


def _validate_command_placeholders(command: list[str]) -> None:
    allowed = {"{input_jsonl}", "{output_jsonl}"}
    for argument in command:
        if "{" not in argument and "}" not in argument:
            continue
        if argument in allowed:
            continue
        if argument in {"{input}", "{output}"}:
            raise ConfigError(
                "system.command placeholder migration: use {input_jsonl} and {output_jsonl}"
            )
        raise ConfigError("system.command contains an unknown or nonexact placeholder")


def _validate_official_scorer_placeholders(command: tuple[str, ...], field: str) -> None:
    allowed = {
        "{requests_jsonl}",
        "{references_jsonl}",
        "{predictions_jsonl}",
        "{output_json}",
    }
    for argument in command:
        if "{" not in argument and "}" not in argument:
            continue
        if argument not in allowed:
            raise ConfigError(f"{field} contains an unknown or nonexact placeholder")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a nonempty string")
    return value.strip()


def _optional_text(value: Any, field: str, default: str) -> str:
    if value is None:
        return default
    return _text(value, field)


def _optional_nullable_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _optional_command(value: Any, field: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(
        isinstance(part, str) and part.strip() for part in value
    ):
        raise ConfigError(f"{field} must be a nonempty list of strings when provided")
    return tuple(value)


def _sensitive_positions(
    value: Any, field: str, command: tuple[str, ...] | None
) -> tuple[int, ...]:
    if value is None:
        return ()
    if command is None:
        raise ConfigError(f"{field} requires official_scorer_command")
    if not isinstance(value, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(command)
        for index in value
    ):
        raise ConfigError(f"{field} must contain valid official_scorer_command indexes")
    return tuple(value)


def _optional_positive_int(value: Any, field: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _path(value: Any, field: str, base_directory: Path) -> Path:
    text = _text(value, field)
    normalized = text.replace("\\", "/").rstrip("/")
    if normalized == _EXAMPLE_ROOT or normalized.startswith(f"{_EXAMPLE_ROOT}/"):
        raise ConfigError(f"{field} uses the unchanged example path: {text}")
    candidate = Path(text)
    return candidate if candidate.is_absolute() else (base_directory / candidate).resolve()


def _optional_path(value: Any, field: str, base_directory: Path) -> Path | None:
    if value is None:
        return None
    return _path(value, field, base_directory)


def _validate_metric_contract(protocol_path: Path, registry_path: Path) -> None:
    protocol = _load_yaml_mapping(protocol_path, "protocol")
    declared_registry = _path(
        protocol.get("metric_namespace"), "protocol.metric_namespace", protocol_path.parent
    )
    if declared_registry != registry_path:
        raise ConfigError(
            "protocol.metric_namespace must resolve to the configured metric_registry_path"
        )
    registry = _load_yaml_mapping(registry_path, "metric registry")
    metric_ids = _metric_ids(registry)
    for metric_id in _required_metric_ids(protocol):
        if metric_id not in metric_ids:
            raise ConfigError(f"protocol requires unknown metric ID: {metric_id}")


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"could not read {label}: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {label}: {path}") from error
    return _mapping(raw, label)


def _metric_ids(registry: Mapping[str, Any]) -> set[str]:
    metrics = registry.get("metrics")
    if not isinstance(metrics, list):
        raise ConfigError("metric registry metrics must be a list")
    metric_ids = set()
    for index, raw_metric in enumerate(metrics):
        metric = _mapping(raw_metric, f"metric registry metrics[{index}]")
        metric_id = _text(metric.get("metric_id"), f"metric registry metrics[{index}].metric_id")
        if metric_id in metric_ids:
            raise ConfigError(f"metric registry contains duplicate metric ID: {metric_id}")
        metric_ids.add(metric_id)
    if not metric_ids:
        raise ConfigError("metric registry contains no metric IDs")
    return metric_ids


def _required_metric_ids(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, child in value.items():
            if key == "required_metric_ids":
                if not isinstance(child, list):
                    raise ConfigError("protocol required_metric_ids must be a list")
                result.extend(_text(metric_id, "protocol required_metric_ids item") for metric_id in child)
            else:
                result.extend(_required_metric_ids(child))
        return result
    if isinstance(value, list):
        return [metric_id for child in value for metric_id in _required_metric_ids(child)]
    return []


def _normalize_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_paths(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_paths(child) for child in value]
    return value
