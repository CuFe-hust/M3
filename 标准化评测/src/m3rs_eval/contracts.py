"""Typed JSONL contracts for evaluation requests, predictions, and run evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, TypeVar

from jsonschema import Draft202012Validator, ValidationError, validators


class ContractError(ValueError):
    """Raised when a record violates a persisted evaluation contract."""


METRIC_RECORD_SCHEMA_VERSION = 2
PREDICTION_ERROR_CODES = frozenset(
    {"timeout", "inference_error", "parse_error", "cancelled", "unknown"}
)


class RecordModel(Protocol):
    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RecordModel": ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RequestRecord:
    sample_id: str
    dataset: str
    benchmark_version: str
    split: str
    task: str
    images: tuple[str, ...]
    prompt: str
    expected_output: str
    request_hash: str
    choices: tuple[str, ...] | None = None
    language: str | None = None
    variant: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RequestRecord":
        data = _object(raw, "request")
        _reject_unknown_fields(data, _REQUEST_FIELDS, "request")
        choices = data.get("choices")
        if choices is not None:
            choices = _text_list(choices, "choices", allow_empty=False)
        return cls(
            sample_id=_text(data.get("sample_id"), "sample_id"),
            dataset=_text(data.get("dataset"), "dataset"),
            benchmark_version=_text(data.get("benchmark_version"), "benchmark_version"),
            split=_text(data.get("split"), "split"),
            task=_text(data.get("task"), "task"),
            images=_text_list(data.get("images"), "images", allow_empty=False),
            prompt=_text(data.get("prompt"), "prompt"),
            expected_output=_text(data.get("expected_output"), "expected_output"),
            request_hash=_text(data.get("request_hash"), "request_hash"),
            choices=choices,
            language=_optional_text(data.get("language"), "language"),
            variant=_optional_text(data.get("variant"), "variant"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "benchmark_version": self.benchmark_version,
            "split": self.split,
            "task": self.task,
            "images": list(self.images),
            "prompt": self.prompt,
            "expected_output": self.expected_output,
            "request_hash": self.request_hash,
        }
        if self.choices is not None:
            result["choices"] = list(self.choices)
        if self.language is not None:
            result["language"] = self.language
        if self.variant is not None:
            result["variant"] = self.variant
        return result


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    status: str
    prediction: str | None = None
    boxes: tuple[tuple[float, float, float, float], ...] | None = None
    raw_output: str | None = None
    latency_ms: float | None = None
    error_code: str | None = None
    error: str | None = None
    trace: Any = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PredictionRecord":
        data = _object(raw, "prediction")
        _reject_unknown_fields(data, _PREDICTION_FIELDS, "prediction")
        status = _text(data.get("status"), "status")
        if status not in {"ok", "error"}:
            raise ContractError("status must be 'ok' or 'error'")

        prediction = _optional_text(data.get("prediction"), "prediction")
        boxes = _boxes(data.get("boxes"))
        error_code = _optional_text(data.get("error_code"), "error_code")
        error = _optional_text(data.get("error"), "error")
        if status == "ok" and prediction is None and not boxes:
            raise ContractError("status=ok requires prediction or boxes")
        if status == "error":
            if error_code not in PREDICTION_ERROR_CODES:
                raise ContractError(
                    "status=error requires error_code in: "
                    + ", ".join(sorted(PREDICTION_ERROR_CODES))
                )
            if not error:
                raise ContractError("status=error requires a nonempty error")
            if prediction is not None or boxes is not None:
                raise ContractError("status=error forbids prediction and boxes")
        elif error_code is not None:
            raise ContractError("status=ok forbids error_code")

        latency = data.get("latency_ms")
        if latency is not None:
            if not _number(latency) or latency < 0:
                raise ContractError("latency_ms must be a nonnegative finite number")
            latency = float(latency)

        return cls(
            sample_id=_text(data.get("sample_id"), "sample_id"),
            status=status,
            prediction=prediction,
            boxes=boxes,
            raw_output=_optional_text(data.get("raw_output"), "raw_output"),
            latency_ms=latency,
            error_code=error_code,
            error=error,
            trace=data.get("trace"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"sample_id": self.sample_id, "status": self.status}
        if self.prediction is not None:
            result["prediction"] = self.prediction
        if self.boxes is not None:
            result["boxes"] = [list(box) for box in self.boxes]
        if self.raw_output is not None:
            result["raw_output"] = self.raw_output
        if self.latency_ms is not None:
            result["latency_ms"] = self.latency_ms
        if self.error_code is not None:
            result["error_code"] = self.error_code
        if self.error is not None:
            result["error"] = self.error
        if self.trace is not None:
            result["trace"] = self.trace
        return result


@dataclass(frozen=True)
class MetricRecord:
    record_schema_version: int
    run_id: str
    metric_id: str
    availability: str
    provenance: str
    value_canonical: float | None
    n_samples: int
    n_failures: int
    recorded_at: str
    protocol_id: str
    benchmark_version: str
    dataset: str | None = None
    task: str | None = None
    slice: str | None = None
    language: str | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None
    source_log_path: str | None = None
    notes: str | None = None
    baseline_run_id: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MetricRecord":
        data = _object(raw, "metric record")
        if "record_schema_version" not in data:
            raise ContractError(
                "legacy MetricRecord has no record_schema_version; implicit migration is unsafe, "
                "regenerate it from source evaluation evidence"
            )
        _reject_unknown_fields(data, _METRIC_FIELDS, "metric record")
        version = data.get("record_schema_version")
        if version != METRIC_RECORD_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported MetricRecord record_schema_version={version!r}; "
                f"expected {METRIC_RECORD_SCHEMA_VERSION}, regenerate the record"
            )
        low_present = "ci95_low" in data
        high_present = "ci95_high" in data
        if low_present != high_present:
            raise ContractError("ci95_low and ci95_high must be provided together")
        availability = _text(data.get("availability"), "availability")
        if availability not in {"available", "missing", "not_applicable", "failed"}:
            raise ContractError("invalid metric availability")
        provenance = _text(data.get("provenance"), "provenance")
        if provenance not in {"official", "supplemental"}:
            raise ContractError("invalid metric provenance")
        value = _optional_number(data.get("value_canonical"), "value_canonical")
        if availability == "available" and value is None:
            raise ContractError("available metric requires value_canonical")
        if availability != "available" and value is not None:
            raise ContractError("unavailable metric requires null value_canonical")
        if availability != "available" and (low_present or high_present):
            raise ContractError("unavailable metric forbids a confidence interval")
        if low_present and data["ci95_low"] is None:
            raise ContractError("ci95_low must be a finite number")
        if high_present and data["ci95_high"] is None:
            raise ContractError("ci95_high must be a finite number")
        low = _optional_number(data.get("ci95_low"), "ci95_low")
        high = _optional_number(data.get("ci95_high"), "ci95_high")
        if low is not None and low > high:
            raise ContractError("ci95_low must not exceed ci95_high")
        n_samples = _nonnegative_int(data.get("n_samples"), "n_samples")
        n_failures = _nonnegative_int(data.get("n_failures"), "n_failures")
        if availability == "available" and n_samples == 0:
            raise ContractError("available metric requires n_samples > 0")
        if n_failures > n_samples:
            raise ContractError("n_failures must not exceed n_samples")
        source_log_path = _optional_text(data.get("source_log_path"), "source_log_path")
        if source_log_path is not None and not _is_run_relative_path(source_log_path):
            raise ContractError("source_log_path must be a run-relative path without parent traversal")
        return cls(
            record_schema_version=version,
            run_id=_text(data.get("run_id"), "run_id"),
            metric_id=_text(data.get("metric_id"), "metric_id"),
            availability=availability,
            provenance=provenance,
            value_canonical=value,
            n_samples=n_samples,
            n_failures=n_failures,
            recorded_at=_text(data.get("recorded_at"), "recorded_at"),
            protocol_id=_text(data.get("protocol_id"), "protocol_id"),
            benchmark_version=_text(data.get("benchmark_version"), "benchmark_version"),
            dataset=_optional_text(data.get("dataset"), "dataset"),
            task=_optional_text(data.get("task"), "task"),
            slice=_optional_text(data.get("slice"), "slice"),
            language=_optional_text(data.get("language"), "language"),
            ci95_low=low,
            ci95_high=high,
            source_log_path=source_log_path,
            notes=_optional_text(data.get("notes"), "notes"),
            baseline_run_id=_optional_text(data.get("baseline_run_id"), "baseline_run_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "record_schema_version": self.record_schema_version,
            "run_id": self.run_id,
            "metric_id": self.metric_id,
            "availability": self.availability,
            "provenance": self.provenance,
            "value_canonical": self.value_canonical,
            "n_samples": self.n_samples,
            "n_failures": self.n_failures,
            "recorded_at": self.recorded_at,
            "protocol_id": self.protocol_id,
            "benchmark_version": self.benchmark_version,
        }
        for key in (
            "dataset", "task", "slice", "language", "ci95_low", "ci95_high",
            "source_log_path", "notes", "baseline_run_id",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    status: str
    mode: str
    protocol_id: str
    created_at: str
    config_hash: str
    request_manifest_hash: str
    eligible_for_history: bool
    protocol_hash: str
    command_hash: str
    system_version_hash: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunManifest":
        data = _object(raw, "run manifest")
        _reject_unknown_fields(data, _MANIFEST_FIELDS, "run manifest")
        status = _text(data.get("status"), "status")
        if status not in {"created", "preflight_passed", "inference_running", "evaluating", "complete", "incomplete", "failed"}:
            raise ContractError("invalid run manifest status")
        mode = _text(data.get("mode"), "mode")
        if mode not in {"smoke", "full"}:
            raise ContractError("mode must be 'smoke' or 'full'")
        metadata = _object(data.get("metadata"), "metadata")
        return cls(
            run_id=_text(data.get("run_id"), "run_id"),
            status=status,
            mode=mode,
            protocol_id=_text(data.get("protocol_id"), "protocol_id"),
            created_at=_text(data.get("created_at"), "created_at"),
            config_hash=_text(data.get("config_hash"), "config_hash"),
            request_manifest_hash=_text(data.get("request_manifest_hash"), "request_manifest_hash"),
            eligible_for_history=_boolean(
                data.get("eligible_for_history"), "eligible_for_history"
            ),
            protocol_hash=_text(data.get("protocol_hash"), "protocol_hash"),
            command_hash=_text(data.get("command_hash"), "command_hash"),
            system_version_hash=_text(
                data.get("system_version_hash"), "system_version_hash"
            ),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "mode": self.mode,
            "protocol_id": self.protocol_id,
            "created_at": self.created_at,
            "config_hash": self.config_hash,
            "request_manifest_hash": self.request_manifest_hash,
            "eligible_for_history": self.eligible_for_history,
            "protocol_hash": self.protocol_hash,
            "command_hash": self.command_hash,
            "system_version_hash": self.system_version_hash,
            "metadata": self.metadata,
        }


T = TypeVar("T", bound=RecordModel)


def _validate_m3rs_relations(
    validator: Any,
    relations: list[Mapping[str, Any]],
    instance: Any,
    schema: Mapping[str, Any],
) -> Iterator[ValidationError]:
    del validator, schema
    if not isinstance(instance, Mapping):
        return
    for relation in relations:
        rule = relation.get("rule")
        if rule == "all_or_none":
            fields = relation.get("fields", [])
            present = [field for field in fields if field in instance]
            if present and len(present) != len(fields):
                yield ValidationError(f"{', '.join(fields)} must be provided together")
        elif rule == "less_than_or_equal":
            left = relation.get("left")
            right = relation.get("right")
            if left not in instance or right not in instance:
                continue
            try:
                invalid = instance[left] > instance[right]
            except TypeError:
                continue
            if invalid:
                yield ValidationError(f"{left} must not exceed {right}")
        elif rule == "run_relative_path":
            field = relation.get("field")
            value = instance.get(field)
            if value is not None and isinstance(value, str) and not _is_run_relative_path(value):
                yield ValidationError(
                    f"{field} must be a run-relative path without parent traversal"
                )


M3RSPersistedRecordValidator = validators.extend(
    Draft202012Validator,
    {"x-m3rs-relations": _validate_m3rs_relations},
    version="m3rs-draft2020-12",
)


def validate_persisted_record(raw: Mapping[str, Any], model_type: type[RecordModel]) -> None:
    """Validate a serialized record with the registered M3-RS schema validator."""
    schema = _schema_for(model_type)
    errors = sorted(
        M3RSPersistedRecordValidator(schema).iter_errors(raw),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    if errors:
        raise ContractError(f"{model_type.__name__} schema validation failed: {errors[0].message}")


@lru_cache(maxsize=None)
def _schema_for(model_type: type[RecordModel]) -> Mapping[str, Any]:
    try:
        schema_name = _SCHEMA_NAMES[model_type]
    except KeyError as error:
        raise ContractError(f"no persisted schema registered for {model_type.__name__}") from error
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / schema_name
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"could not load persisted schema: {schema_path}") from error
    try:
        M3RSPersistedRecordValidator.check_schema(schema)
    except Exception as error:
        raise ContractError(f"invalid persisted schema: {schema_path}") from error
    return schema


def read_jsonl(path: Path, model_type: type[T], unique_key: str | None = None) -> list[T]:
    """Read validated JSONL records, retaining line context in raised errors."""
    records: list[T] = []
    key_name = "sample_id" if model_type is PredictionRecord else unique_key
    seen: set[Any] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ContractError(f"could not read JSONL: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ContractError(f"line {line_number}: blank JSONL records are not allowed")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"line {line_number}: invalid JSON: {error.msg}") from error
        try:
            record_data = _object(raw, "record")
            validate_persisted_record(record_data, model_type)
            record = model_type.from_dict(record_data)
        except ContractError as error:
            raise ContractError(f"line {line_number}: {error}") from error
        if key_name is not None:
            try:
                key = getattr(record, key_name)
            except AttributeError as error:
                raise ContractError(f"unknown uniqueness key: {key_name}") from error
            if key in seen:
                raise ContractError(f"line {line_number}: duplicate {key_name}: {key}")
            seen.add(key)
        records.append(record)
    return records


def write_jsonl(path: Path, records: list[RecordModel]) -> None:
    """Write validated record objects as UTF-8 JSONL with a final newline."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            try:
                payload = record.to_dict()
                validate_persisted_record(payload, record.__class__)
                record.__class__.from_dict(payload)
            except (AttributeError, ContractError) as error:
                raise ContractError(f"write_jsonl received an invalid record: {error}") from error
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an object")
    return value


def _reject_unknown_fields(data: Mapping[str, Any], allowed: frozenset[str], record: str) -> None:
    unknown = sorted(str(key) for key in data.keys() if key not in allowed)
    if unknown:
        raise ContractError(f"{record} contains unknown field: {', '.join(unknown)}")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    return value.strip()


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{field} must be a boolean")
    return value


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _text_list(value: Any, field: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ContractError(f"{field} must be a {'possibly empty ' if allow_empty else 'nonempty '}list of strings")
    return tuple(_text(item, f"{field} item") for item in value)


def _boxes(value: Any) -> tuple[tuple[float, float, float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ContractError("boxes must be a nonempty list of [x1, y1, x2, y2]")
    result = []
    for index, box in enumerate(value):
        if not isinstance(box, list) or len(box) != 4 or not all(_number(item) for item in box):
            raise ContractError(f"boxes[{index}] must contain four finite numbers")
        result.append(tuple(float(item) for item in box))
    return tuple(result)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _optional_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if not _number(value):
        raise ContractError(f"{field} must be a finite number or null")
    return float(value)


def _is_run_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or normalized.startswith("/"):
        return False
    if len(normalized) >= 2 and normalized[1] == ":":
        return False
    return ".." not in path.parts


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{field} must be a nonnegative integer")
    return value


_REQUEST_FIELDS = frozenset(
    {
        "sample_id", "dataset", "benchmark_version", "split", "task", "images", "prompt",
        "expected_output", "request_hash", "choices", "language", "variant",
    }
)
_PREDICTION_FIELDS = frozenset(
    {
        "sample_id", "status", "prediction", "boxes", "raw_output", "latency_ms",
        "error_code", "error", "trace",
    }
)
_METRIC_FIELDS = frozenset(
    {
        "record_schema_version", "run_id", "metric_id", "availability", "provenance", "value_canonical", "n_samples", "n_failures", "recorded_at",
        "protocol_id", "benchmark_version", "dataset", "task", "slice", "language", "ci95_low",
        "ci95_high", "source_log_path", "notes", "baseline_run_id",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "run_id", "status", "mode", "protocol_id", "created_at", "config_hash",
        "request_manifest_hash", "eligible_for_history", "protocol_hash", "command_hash",
        "system_version_hash", "metadata",
    }
)
_SCHEMA_NAMES = {
    RequestRecord: "request.schema.json",
    PredictionRecord: "prediction.schema.json",
    MetricRecord: "metric_record.schema.json",
    RunManifest: "run_manifest.schema.json",
}
