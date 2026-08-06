"""Prediction evidence alignment and benchmark evaluation primitives."""

from __future__ import annotations

import json
import hashlib
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

from m3rs_eval.command_adapter import run_system
from m3rs_eval.config import DatasetConfig, SystemCommandConfig
from m3rs_eval.contracts import (
    ContractError,
    METRIC_RECORD_SCHEMA_VERSION,
    MetricRecord,
    PredictionRecord,
    RequestRecord,
    read_jsonl,
)
from m3rs_eval.registry import MetricRegistry, RegistryError, normalize_value
from m3rs_eval.statistics import bootstrap_interval, wilson_interval


class EvaluationError(ValueError):
    """Raised when aligned evidence cannot produce a valid metric record."""


@dataclass(frozen=True)
class MetricContext:
    """Stable run provenance copied to every emitted metric record."""

    run_id: str
    recorded_at: str
    protocol_id: str
    benchmark_version: str
    source_log_path: str


@dataclass(frozen=True)
class OfficialMetricScore:
    """Strict aggregate score parsed from an external official scorer artifact."""

    metric_id: str
    value_canonical: float | None
    n_samples: int
    n_failures: int
    scorer_version: str
    availability: str = "available"


@dataclass(frozen=True)
class PredictionEvidence:
    """One physical prediction row, including rows that violate the contract."""

    prediction_index: int
    prediction: PredictionRecord | None
    sample_id: str | None
    error: str | None = None
    raw: Any = None


@dataclass(frozen=True)
class FailureRow:
    """Serializable evidence for one alignment failure event."""

    sample_id: str | None
    reason: str
    detail: str
    expected: bool
    prediction_indexes: tuple[int, ...] = ()
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_id": self.sample_id,
            "reason": self.reason,
            "detail": self.detail,
            "expected": self.expected,
            "prediction_indexes": list(self.prediction_indexes),
        }
        if self.raw is not None:
            result["raw"] = _json_safe(self.raw)
        return result


@dataclass(frozen=True)
class AlignedPrediction:
    """Exactly one alignment outcome for an expected request."""

    request: RequestRecord
    prediction: PredictionRecord | None
    failure: FailureRow | None


@dataclass(frozen=True)
class AlignmentResult:
    rows: tuple[AlignedPrediction, ...]
    failures: tuple[FailureRow, ...]
    expected: int
    valid_expected: int
    expected_failures: int
    extraneous_predictions: int
    duplicate_predictions: int

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def valid(self) -> int:
        """Brief-compatible alias for valid expected predictions."""
        return self.valid_expected

    @property
    def complete(self) -> bool:
        return self.expected_failures == 0 and self.failure_count == 0


@dataclass(frozen=True)
class OfficialScorerResult:
    scores: tuple[OfficialMetricScore, ...]
    raw_output_path: Path
    version_path: Path
    command_path: Path
    source_log_path: str
    scope_manifest_path: Path
    scope_manifest_hash: str


@dataclass(frozen=True)
class MetricScopeRule:
    dataset: str
    task: str | None
    provenance: str
    language: str | None = None
    variant: str | None = None
    slice_name: str = "all"
    dimension: str | None = None
    reference_dimension: str | None = None


@dataclass(frozen=True)
class OfficialMetricBinding:
    metric_id: str
    n_samples: int
    minimum_n_failures: int
    rule: MetricScopeRule


@dataclass(frozen=True)
class ScorerInputBundle:
    requests_path: Path
    references_path: Path
    predictions_path: Path
    manifest_path: Path
    manifest_hash: str
    bindings: Mapping[str, OfficialMetricBinding]


@dataclass(frozen=True)
class EvaluationResult:
    metric_records: tuple[MetricRecord, ...]
    failures: tuple[FailureRow, ...]
    coverage: dict[str, Any]
    alignment: AlignmentResult

    @property
    def status(self) -> str:
        """The sole persisted completeness authority exposed to Task 6."""
        return str(self.coverage["status"])

    @property
    def complete(self) -> bool:
        return self.status == "complete"


_XLRS_L2 = (
    "counting",
    "scene_classification",
    "object_spatial_relationship",
    "object_properties",
    "complex_reasoning",
    "planning",
    "spatiotemporal_reasoning",
    "anomaly_reasoning",
)
_XLRS_L3 = (
    "object_classification",
    "object_color",
    "object_motion_state",
    "object_spatial_relationship",
    "overall_counting",
    "regional_counting",
    "counting_with_complex_reasoning",
    "regional_counting_with_change_detection",
    "overall_land_use_classification",
    "regional_land_use_classification",
    "route_planning",
    "environmental_condition_reasoning",
    "anomaly_detection_and_interpretation",
)


def _build_metric_scope_rules() -> dict[str, MetricScopeRule]:
    """Declare every Task 5 metric scope and provenance without name heuristics."""
    rules: dict[str, MetricScopeRule] = {}

    for scope in ("all", "change", "no_change"):
        for metric in ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l", "cider_d"):
            rules[f"levir.caption.{scope}.{metric}"] = MetricScopeRule(
                "levir_cc", "caption", "official", slice_name=scope
            )
        rules[f"levir.discrimination.{scope}.accuracy"] = MetricScopeRule(
            "levir_cc", "caption", "supplemental", slice_name=scope
        )

    for metric in ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l", "cider", "clair", "avg_l"):
        rules[f"vrs.caption.{metric}"] = MetricScopeRule("vrsbench", "caption", "official")
    for scope in ("all", "unique", "non_unique"):
        for metric in ("acc_0_5", "acc_0_7", "mean_iou"):
            rules[f"vrs.grounding.hbb.{scope}.{metric}"] = MetricScopeRule(
                "vrsbench", "grounding", "supplemental", slice_name=scope
            )
    for category in (
        "all", "category", "presence", "quantity", "color", "shape", "size",
        "position", "direction", "scene", "reasoning",
    ):
        rules[f"vrs.vqa.acc.{category}"] = MetricScopeRule(
            "vrsbench", "vqa", "supplemental", slice_name=category
        )

    for language in ("en", "zh"):
        for metric in ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l"):
            rules[f"xlrs.caption.{language}.{metric}"] = MetricScopeRule(
                "xlrs_bench", "caption", "official", language, "full"
            )
        for scope in ("all", "fine_grained", "condition_based"):
            metrics = ("acc_0_5", "acc_0_7", "mean_iou") if scope == "all" else ("acc_0_5", "acc_0_7")
            for metric in metrics:
                rules[f"xlrs.grounding.{language}.{scope}.{metric}"] = MetricScopeRule(
                    "xlrs_bench", "grounding", "official", language, "full", scope
                )
        for dimension in _XLRS_L2:
            rules[f"xlrs.vqa.{language}.l2.{dimension}.acc"] = MetricScopeRule(
                "xlrs_bench", "vqa", "supplemental", language, "full",
                dimension=dimension, reference_dimension="l2"
            )
        for dimension in _XLRS_L3:
            rules[f"xlrs.vqa.{language}.l3.{dimension}.acc"] = MetricScopeRule(
                "xlrs_bench", "vqa", "supplemental", language, "full",
                dimension=dimension, reference_dimension="l3"
            )
        rules[f"xlrs.vqa.{language}.paper_avg_l2"] = MetricScopeRule(
            "xlrs_bench", "vqa", "supplemental", language, "full"
        )
    for metric in ("micro_acc", "macro_l3_acc"):
        rules[f"xlrs.vqa.en.lite.{metric}"] = MetricScopeRule(
            "xlrs_bench", "vqa", "supplemental", "en", "lite"
        )

    for task in ("color", "count", "position"):
        rules[f"mme_rs.acc.{task}"] = MetricScopeRule(
            "mme_rs", task, "supplemental", slice_name=task
        )
    for metric in ("avg", "avg_c", "invalid_parse_rate", "choice_e_rate"):
        rules[f"mme_rs.{metric}"] = MetricScopeRule("mme_rs", None, "supplemental")
    return rules


METRIC_SCOPE_RULES = _build_metric_scope_rules()


def make_metric_record(
    registry: MetricRegistry,
    context: MetricContext,
    metric_id: str,
    value: float | None,
    *,
    n_samples: int,
    n_failures: int,
    provenance: str,
    availability: str = "available",
    binomial_successes: int | None = None,
    bootstrap_observations: Iterable[float] | None = None,
    bootstrap_statistic: Callable[[list[float]], float] = fmean,
    notes: str | None = None,
) -> MetricRecord:
    """Create one registry-backed metric with an evidence-appropriate interval."""
    try:
        metric = registry.require(metric_id)
    except RegistryError as error:
        raise EvaluationError(str(error)) from error
    if provenance not in {"official", "supplemental"}:
        raise EvaluationError("provenance must be 'official' or 'supplemental'")
    if availability not in {"available", "missing", "not_applicable", "failed"}:
        raise EvaluationError("invalid metric availability")
    registry_availability = str(metric.metadata.get("availability", "available"))
    if registry_availability.startswith("not_applicable") and (
        availability != "not_applicable" or value is not None
    ):
        raise EvaluationError(f"{metric_id} is not applicable according to the registry")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 0:
        raise EvaluationError("n_samples must be a nonnegative integer")
    if not isinstance(n_failures, int) or isinstance(n_failures, bool) or not 0 <= n_failures <= n_samples:
        raise EvaluationError("n_failures must be between zero and n_samples")
    _validate_context(context)

    if availability == "available":
        if n_samples == 0:
            raise EvaluationError("available metric requires n_samples > 0")
        try:
            canonical_value = normalize_value(metric, value)
        except (TypeError, ValueError) as error:
            raise EvaluationError(str(error)) from error
        if not _in_canonical_range(canonical_value, str(metric.metadata.get("canonical_range", ""))):
            raise EvaluationError(f"{metric_id} value is outside its canonical range")
    else:
        if value is not None:
            raise EvaluationError("unavailable metrics require value=None")
        canonical_value = None

    if binomial_successes is not None and bootstrap_observations is not None:
        raise EvaluationError("choose either Wilson or bootstrap interval evidence")
    if availability != "available" and (
        binomial_successes is not None or bootstrap_observations is not None
    ):
        raise EvaluationError("unavailable metrics cannot have interval evidence")

    low: float | None = None
    high: float | None = None
    if binomial_successes is not None:
        if n_samples == 0:
            raise EvaluationError("Wilson interval requires at least one sample")
        try:
            low, high = wilson_interval(binomial_successes, n_samples)
        except ValueError as error:
            raise EvaluationError(str(error)) from error
        if not math.isclose(canonical_value, binomial_successes / n_samples, abs_tol=1e-12):
            raise EvaluationError("binomial successes do not reproduce value_canonical")
    elif bootstrap_observations is not None:
        observations = list(bootstrap_observations)
        if len(observations) != n_samples:
            raise EvaluationError("bootstrap observations must match n_samples")
        try:
            low, high = bootstrap_interval(observations, bootstrap_statistic)
        except ValueError as error:
            raise EvaluationError(str(error)) from error
        observed = float(bootstrap_statistic(observations))
        if not math.isclose(canonical_value, observed, abs_tol=1e-12):
            raise EvaluationError("bootstrap observations do not reproduce value_canonical")

    metadata = metric.metadata
    return MetricRecord(
        record_schema_version=METRIC_RECORD_SCHEMA_VERSION,
        run_id=context.run_id,
        metric_id=metric_id,
        availability=availability,
        provenance=provenance,
        value_canonical=canonical_value,
        n_samples=n_samples,
        n_failures=n_failures,
        recorded_at=context.recorded_at,
        protocol_id=context.protocol_id,
        benchmark_version=context.benchmark_version,
        dataset=_metadata_text(metadata, "dataset"),
        task=_metadata_text(metadata, "task"),
        slice=_metadata_text(metadata, "slice"),
        language=_metadata_text(metadata, "language"),
        ci95_low=low,
        ci95_high=high,
        source_log_path=context.source_log_path,
        notes=notes,
    )


def ingest_official_scorer_output(
    path: Path, registry: MetricRegistry
) -> list[OfficialMetricScore]:
    """Parse the narrow scorer JSON contract and fail closed on any ambiguity."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"malformed official scorer output: {error}") from error
    try:
        if not isinstance(raw, Mapping) or set(raw) != {"scorer_version", "metrics"}:
            raise ValueError("top-level fields must be scorer_version and metrics")
        version = raw["scorer_version"]
        rows = raw["metrics"]
        if not isinstance(version, str) or not version.strip():
            raise ValueError("scorer_version must be a nonempty string")
        if not isinstance(rows, list):
            raise ValueError("metrics must be a list")
        scores: list[OfficialMetricScore] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"metrics[{index}] must be an object")
            allowed = {"metric_id", "value_canonical", "n_samples", "n_failures", "availability"}
            if set(row) - allowed:
                raise ValueError(f"metrics[{index}] has unknown fields")
            metric_id = row.get("metric_id")
            if not isinstance(metric_id, str) or not metric_id.strip():
                raise ValueError(f"metrics[{index}].metric_id is required")
            metric = registry.require(metric_id)
            if metric_id in seen:
                raise ValueError(f"duplicate metric_id: {metric_id}")
            seen.add(metric_id)
            availability = row.get("availability", "available")
            if availability not in {"available", "missing", "not_applicable", "failed"}:
                raise ValueError(f"metrics[{index}].availability is invalid")
            value = row.get("value_canonical")
            if availability == "available":
                canonical = normalize_value(metric, value)
                if not _in_canonical_range(
                    canonical, str(metric.metadata.get("canonical_range", ""))
                ):
                    raise ValueError(f"{metric_id} is outside its canonical range")
                if str(metric.metadata.get("availability", "available")).startswith("not_applicable"):
                    raise ValueError(f"{metric_id} is not applicable in the registry")
            else:
                if value is not None:
                    raise ValueError(f"metrics[{index}] unavailable value must be null")
                canonical = None
            n_samples = row.get("n_samples")
            n_failures = row.get("n_failures")
            if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 0:
                raise ValueError(f"metrics[{index}].n_samples is invalid")
            if (
                not isinstance(n_failures, int)
                or isinstance(n_failures, bool)
                or not 0 <= n_failures <= n_samples
            ):
                raise ValueError(f"metrics[{index}].n_failures is invalid")
            scores.append(
                OfficialMetricScore(
                    metric_id,
                    canonical,
                    n_samples,
                    n_failures,
                    version.strip(),
                    availability,
                )
            )
    except (KeyError, RegistryError, TypeError, ValueError) as error:
        raise EvaluationError(f"malformed official scorer output: {error}") from error
    return scores


def official_metric_record(
    registry: MetricRegistry, context: MetricContext, score: OfficialMetricScore
) -> MetricRecord:
    """Convert one external aggregate without inventing sample-level intervals."""
    return make_metric_record(
        registry,
        context,
        score.metric_id,
        score.value_canonical,
        n_samples=score.n_samples,
        n_failures=score.n_failures,
        provenance="official",
        availability=score.availability,
        notes=f"external official scorer version: {score.scorer_version}; aggregate-only, CI unavailable",
    )


def merge_metric_records(
    official: Iterable[MetricRecord], supplemental: Iterable[MetricRecord]
) -> list[MetricRecord]:
    """Merge provenance lanes while rejecting any supplemental overwrite."""
    result: list[MetricRecord] = []
    seen: dict[str, str] = {}
    for record in official:
        if record.metric_id in seen:
            raise EvaluationError(f"duplicate official metric record: {record.metric_id}")
        if record.provenance != "official":
            raise EvaluationError("official metric lane contains a non-official record")
        seen[record.metric_id] = "official"
        result.append(record)
    for record in supplemental:
        if record.metric_id in seen:
            if seen[record.metric_id] == "official":
                raise EvaluationError(
                    f"supplemental metric cannot overwrite official metric: {record.metric_id}"
                )
            raise EvaluationError(f"duplicate supplemental metric record: {record.metric_id}")
        if record.provenance != "supplemental":
            raise EvaluationError("supplemental metric lane contains an official record")
        seen[record.metric_id] = "supplemental"
        result.append(record)
    return result


def render_official_scorer_argv(
    command: Iterable[str],
    materialization: Any,
    predictions_path: Path,
    output_path: Path,
) -> list[str]:
    """Render only exact whole-argument scorer placeholders."""
    replacements = {
        "{requests_jsonl}": str(Path(materialization.requests_path)),
        "{references_jsonl}": str(Path(materialization.references_path)),
        "{predictions_jsonl}": str(Path(predictions_path)),
        "{output_json}": str(Path(output_path)),
    }
    argv = []
    for argument in command:
        if argument in replacements:
            argv.append(replacements[argument])
        elif "{" in argument or "}" in argument:
            raise EvaluationError(
                f"official scorer command contains unknown or nonexact placeholder: {argument}"
            )
        else:
            argv.append(argument)
    if not argv:
        raise EvaluationError("official scorer command must not be empty")
    return argv


def _scope_rows(
    rule: MetricScopeRule,
    alignment: AlignmentResult,
    references: Mapping[str, Mapping[str, Any]],
) -> list[AlignedPrediction]:
    selected: list[AlignedPrediction] = []
    for row in alignment.rows:
        request = row.request
        if request.dataset != rule.dataset:
            continue
        if rule.task is not None and request.task != rule.task:
            continue
        if rule.language is not None and request.language != rule.language:
            continue
        if rule.variant is not None and request.variant != rule.variant:
            continue
        reference = references.get(request.sample_id, {})
        if rule.dataset == "levir_cc" and rule.slice_name != "all":
            expected_slice = "no_change" if reference.get("change") == "no-change" else "change"
            if expected_slice != rule.slice_name:
                continue
        if rule.dataset == "vrsbench" and request.task == "grounding" and rule.slice_name != "all":
            if reference.get("grounding_slice") != rule.slice_name:
                continue
        if rule.dataset == "vrsbench" and request.task == "vqa" and rule.slice_name != "all":
            if reference.get("vqa_category") != rule.slice_name:
                continue
        if rule.dataset == "xlrs_bench" and request.task == "grounding" and rule.slice_name != "all":
            if reference.get("grounding_slice") != rule.slice_name:
                continue
        if rule.dimension is not None:
            if reference.get(rule.reference_dimension or "") != rule.dimension:
                continue
        selected.append(row)
    return selected


def _official_input_row(dataset: str, row: AlignedPrediction) -> bool:
    if dataset == "levir_cc":
        return row.request.task == "caption"
    if dataset == "vrsbench":
        return row.request.task == "caption"
    if dataset == "xlrs_bench":
        return row.request.variant == "full" and row.request.task in {"caption", "grounding"}
    return False


def _scorer_prediction_payload(row: AlignedPrediction) -> dict[str, Any]:
    if row.prediction is not None and row.failure is None:
        return row.prediction.to_dict()
    failure = row.failure
    error_code = "timeout" if failure and failure.reason == "timed_out_prediction" else "parse_error"
    return {
        "sample_id": row.request.sample_id,
        "status": "error",
        "error_code": error_code,
        "error": failure.detail if failure is not None else "prediction unavailable",
    }


def _write_raw_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_scorer_input_bundle(
    dataset: str,
    requests: Iterable[RequestRecord],
    references: Mapping[str, Mapping[str, Any]],
    alignment: AlignmentResult,
    scorer_dir: Path,
    *,
    expected_version: str | None,
    expected_commit: str | None,
) -> ScorerInputBundle:
    """Persist deterministic scorer inputs and materialization-derived metric bindings."""
    scorer_dir = Path(scorer_dir)
    input_dir = scorer_dir / "scorer_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    selected = [row for row in alignment.rows if _official_input_row(dataset, row)]
    selected_ids = {row.request.sample_id for row in selected}
    request_by_id = {request.sample_id: request for request in requests}

    combined_requests = input_dir / "combined.requests.jsonl"
    combined_references = input_dir / "combined.references.jsonl"
    combined_predictions = input_dir / "combined.predictions.jsonl"
    _write_raw_rows(combined_requests, (request_by_id[row.request.sample_id].to_dict() for row in selected))
    _write_raw_rows(combined_references, (references[row.request.sample_id] for row in selected))
    _write_raw_rows(combined_predictions, (_scorer_prediction_payload(row) for row in selected))

    bindings: dict[str, OfficialMetricBinding] = {}
    for metric_id, rule in METRIC_SCOPE_RULES.items():
        if rule.dataset != dataset or rule.provenance != "official":
            continue
        if metric_id == "levir.caption.no_change.cider_d":
            continue
        rows = _scope_rows(rule, alignment, references)
        if not rows:
            continue
        bindings[metric_id] = OfficialMetricBinding(
            metric_id,
            len(rows),
            sum(row.failure is not None for row in rows),
            rule,
        )

    scope_groups: dict[tuple[str, str, str], list[AlignedPrediction]] = {}
    for row in selected:
        request = row.request
        key = (request.variant or "all", request.language or "all", request.task)
        scope_groups.setdefault(key, []).append(row)
    scopes = []
    for (variant, language, task), rows in sorted(scope_groups.items()):
        stem = f"{variant}.{language}.{task}"
        request_path = input_dir / f"{stem}.requests.jsonl"
        reference_path = input_dir / f"{stem}.references.jsonl"
        prediction_path = input_dir / f"{stem}.predictions.jsonl"
        _write_raw_rows(request_path, (row.request.to_dict() for row in rows))
        _write_raw_rows(reference_path, (references[row.request.sample_id] for row in rows))
        _write_raw_rows(prediction_path, (_scorer_prediction_payload(row) for row in rows))
        metric_ids = sorted(
            metric_id
            for metric_id, binding in bindings.items()
            if binding.rule.task == task
            and (binding.rule.language or "all") == language
            and (binding.rule.variant or "all") == variant
        )
        scopes.append(
            {
                "variant": variant,
                "language": language,
                "task": task,
                "n_samples": len(rows),
                "minimum_aligned_failures": sum(row.failure is not None for row in rows),
                "sample_ids": [row.request.sample_id for row in rows],
                "allowed_metric_ids": metric_ids,
                "requests_path": request_path.relative_to(scorer_dir).as_posix(),
                "references_path": reference_path.relative_to(scorer_dir).as_posix(),
                "predictions_path": prediction_path.relative_to(scorer_dir).as_posix(),
                "requests_sha256": _sha256_file(request_path),
                "references_sha256": _sha256_file(reference_path),
                "predictions_sha256": _sha256_file(prediction_path),
            }
        )
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "expected_scorer_version": expected_version,
        "expected_scorer_commit": expected_commit,
        "selected_sample_ids": [row.request.sample_id for row in selected],
        "excluded_sample_ids": sorted(set(request_by_id) - selected_ids),
        "combined_inputs": {
            "requests_path": combined_requests.relative_to(scorer_dir).as_posix(),
            "references_path": combined_references.relative_to(scorer_dir).as_posix(),
            "predictions_path": combined_predictions.relative_to(scorer_dir).as_posix(),
            "requests_sha256": _sha256_file(combined_requests),
            "references_sha256": _sha256_file(combined_references),
            "predictions_sha256": _sha256_file(combined_predictions),
        },
        "metric_bindings": {
            metric_id: {
                "n_samples": binding.n_samples,
                "minimum_n_failures": binding.minimum_n_failures,
                "task": binding.rule.task,
                "language": binding.rule.language,
                "variant": binding.rule.variant,
                "slice": binding.rule.slice_name,
            }
            for metric_id, binding in sorted(bindings.items())
        },
        "scopes": scopes,
    }
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    manifest["manifest_hash"] = manifest_hash
    manifest_path = scorer_dir / "scorer_scope_manifest.json"
    _write_json(manifest_path, manifest)
    return ScorerInputBundle(
        combined_requests,
        combined_references,
        combined_predictions,
        manifest_path,
        manifest_hash,
        bindings,
    )


def _validate_bound_official_scores(
    scores: Iterable[OfficialMetricScore],
    bindings: Mapping[str, OfficialMetricBinding],
    expected_version: str | None,
) -> None:
    for score in scores:
        if expected_version is None or score.scorer_version != expected_version:
            raise EvaluationError(
                f"official scorer version mismatch: expected {expected_version!r}, got {score.scorer_version!r}"
            )
        binding = bindings.get(score.metric_id)
        if binding is None:
            raise EvaluationError(
                f"official scorer metric_id not allowed for materialized dataset/scope: {score.metric_id}"
            )
        if score.n_samples != binding.n_samples:
            raise EvaluationError(
                f"official scorer {score.metric_id} n_samples={score.n_samples} does not match scoped n_samples={binding.n_samples}"
            )
        if score.n_failures < binding.minimum_n_failures:
            raise EvaluationError(
                f"official scorer {score.metric_id} n_failures={score.n_failures} is below aligned scope minimum={binding.minimum_n_failures}"
            )


def run_official_scorer(
    config: DatasetConfig,
    materialization: Any,
    predictions_path: Path,
    registry: MetricRegistry,
    log_dir: Path,
    dataset: str,
    *,
    alignment: AlignmentResult | None = None,
    references: Mapping[str, Mapping[str, Any]] | None = None,
) -> OfficialScorerResult:
    """Run or ingest a configured official scorer while retaining raw evidence."""
    if config.profile == "official":
        if config.official_scorer_output is not None:
            raise EvaluationError(
                "official_scorer_output is fixture-only and forbidden for profile=official"
            )
        if config.official_scorer_command is None:
            raise EvaluationError("profile=official requires an official scorer command")
        if not config.official_scorer_expected_version or not config.official_scorer_expected_commit:
            raise EvaluationError(
                "profile=official requires pinned scorer version and commit evidence"
            )
    scorer_dir = Path(log_dir) / dataset
    scorer_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = scorer_dir / "official_scores.raw.json"
    version_path = scorer_dir / "official_scorer.version.txt"
    command_path = scorer_dir / "official_scorer.command.json"
    requests = read_jsonl(
        Path(materialization.requests_path), RequestRecord, unique_key="sample_id"
    )
    reference_map = references or _read_references(Path(materialization.references_path))
    aligned = alignment or align_predictions(
        requests, read_prediction_evidence(Path(predictions_path))
    )
    bundle = build_scorer_input_bundle(
        dataset,
        requests,
        reference_map,
        aligned,
        scorer_dir,
        expected_version=config.official_scorer_expected_version,
        expected_commit=config.official_scorer_expected_commit,
    )

    if config.official_scorer_output is not None:
        if config.profile != "fixture":
            raise EvaluationError("official_scorer_output is fixture-only and forbidden for profile=official")
        try:
            shutil.copyfile(config.official_scorer_output, raw_output_path)
        except OSError as error:
            raise EvaluationError(f"could not copy fixture official scorer output: {error}") from error
        command_evidence = {
            "mode": "fixture_output_ingestion",
            "source": str(config.official_scorer_output),
            "shell": False,
            "timed_out": False,
        }
    elif config.official_scorer_command is not None:
        argv = render_official_scorer_argv(
            config.official_scorer_command,
            bundle,
            bundle.predictions_path,
            raw_output_path,
        )
        command_config = SystemCommandConfig(
            command=tuple(argv),
            working_directory=config.official_scorer_working_directory or config.root,
            timeout_seconds=config.official_scorer_timeout_seconds,
            environment=dict(config.official_scorer_environment),
            sensitive_argument_positions=config.scorer_sensitive_argument_positions,
        )
        result = run_system(
            command_config,
            bundle.requests_path,
            raw_output_path,
            scorer_dir,
        )
        command_evidence = {
            "argv": result.argv,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "duration_seconds": result.duration_seconds,
            "stdout_path": result.stdout_path.name,
            "stderr_path": result.stderr_path.name,
            "environment_overrides": result.environment_overrides,
            "shell": False,
        }
        if result.timed_out:
            _write_json(command_path, command_evidence)
            raise EvaluationError("official scorer timed out")
        if result.returncode != 0:
            _write_json(command_path, command_evidence)
            raise EvaluationError(f"official scorer exited with code {result.returncode}")
        if not raw_output_path.is_file():
            _write_json(command_path, command_evidence)
            raise EvaluationError("official scorer did not create output JSON")
    else:
        raise EvaluationError("official scorer is not configured")

    command_evidence.update(
        {
            "scope_manifest_path": bundle.manifest_path.name,
            "scope_manifest_hash": bundle.manifest_hash,
            "expected_scorer_version": config.official_scorer_expected_version,
            "expected_scorer_commit": config.official_scorer_expected_commit,
        }
    )
    _write_json(command_path, command_evidence)
    scores = ingest_official_scorer_output(raw_output_path, registry)
    version = _scorer_version(raw_output_path)
    version_path.write_text(version + "\n", encoding="utf-8", newline="\n")
    if version != config.official_scorer_expected_version:
        raise EvaluationError(
            f"official scorer version mismatch: expected {config.official_scorer_expected_version!r}, got {version!r}"
        )
    _validate_bound_official_scores(
        scores, bundle.bindings, config.official_scorer_expected_version
    )
    source_log_path = (Path(Path(log_dir).name) / dataset / raw_output_path.name).as_posix()
    return OfficialScorerResult(
        tuple(scores),
        raw_output_path,
        version_path,
        command_path,
        source_log_path,
        bundle.manifest_path,
        bundle.manifest_hash,
    )


def evaluate_materialization(
    adapter: Any,
    materialization: Any,
    predictions_path: Path,
    registry: MetricRegistry,
    *,
    context: MetricContext,
    log_dir: Path,
) -> EvaluationResult:
    """Read, align once, route by adapter, and return Task 6-ready evidence."""
    _validate_context(context)
    requests = read_jsonl(Path(materialization.requests_path), RequestRecord, unique_key="sample_id")
    references = _read_references(Path(materialization.references_path))
    request_ids = [request.sample_id for request in requests]
    if len(requests) != materialization.expected_samples:
        raise EvaluationError("materialization expected_samples does not match requests JSONL")
    if set(request_ids) != set(references) or len(request_ids) != len(references):
        raise EvaluationError("requests and references are not one-to-one")

    alignment = align_predictions(requests, read_prediction_evidence(Path(predictions_path)))
    failures = list(alignment.failures)
    scores: tuple[OfficialMetricScore, ...] = ()
    metric_context = context
    scorer_scope_manifest_hash: str | None = None
    scorer_required = _scorer_required(adapter.dataset, requests)
    scorer_status = "not_required"
    if adapter.config.official_scorer_command is not None or adapter.config.official_scorer_output is not None:
        try:
            scorer_result = run_official_scorer(
                adapter.config,
                materialization,
                predictions_path,
                registry,
                log_dir,
                adapter.dataset,
                alignment=alignment,
                references=references,
            )
            scores = scorer_result.scores
            metric_context = dataclass_replace(
                context, source_log_path=scorer_result.source_log_path
            )
            scorer_status = "succeeded"
            scorer_scope_manifest_hash = scorer_result.scope_manifest_hash
        except EvaluationError as error:
            scorer_status = "failed"
            failures.append(
                FailureRow(
                    None,
                    "official_scorer_failed",
                    str(error),
                    False,
                )
            )
    elif scorer_required:
        scorer_status = "missing"
        failures.append(
            FailureRow(
                None,
                "official_scorer_missing",
                f"{adapter.dataset} materialization requires configured official scorer evidence",
                False,
            )
        )

    records = _route_benchmark_metrics(
        adapter.dataset, alignment, references, registry, metric_context, scores
    )
    records = _add_required_unavailable_records(
        adapter,
        records,
        registry,
        metric_context,
        alignment,
        references,
        scorer_status,
    )
    failures.extend(_required_metric_failures(adapter, records))
    status = "complete" if not failures else "incomplete"
    coverage = dict(materialization.coverage)
    coverage.update(
        {
            "expected_requests": alignment.expected,
            "valid_expected_predictions": alignment.valid_expected,
            "expected_prediction_failures": alignment.expected_failures,
            "extraneous_predictions": alignment.extraneous_predictions,
            "duplicate_predictions": alignment.duplicate_predictions,
            "failure_count": len(failures),
            "official_scorer_status": scorer_status,
            "status": status,
        }
    )
    if scorer_scope_manifest_hash is not None:
        coverage["scorer_scope_manifest_hash"] = scorer_scope_manifest_hash
    if adapter.dataset == "xlrs_bench":
        coverage["metric_scope_notes"] = [
            "Lite Chinese rows have no protocol-supported Lite aggregate/dimension lane; no Lite-zh metric IDs are emitted."
        ]
    return EvaluationResult(
        tuple(records),
        tuple(failures),
        coverage,
        alignment,
    )


def read_prediction_evidence(path: Path) -> list[PredictionEvidence]:
    """Read every physical JSONL row without discarding malformed evidence."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ContractError(f"could not read prediction evidence: {path}") from error

    evidence: list[PredictionEvidence] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            evidence.append(PredictionEvidence(index, None, None, "blank JSONL record", line))
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            evidence.append(
                PredictionEvidence(index, None, None, f"invalid JSON: {error.msg}", line)
            )
            continue
        evidence.append(_prediction_evidence(raw, index))
    return evidence


def align_predictions(
    requests: Iterable[RequestRecord],
    predictions: Iterable[PredictionRecord | PredictionEvidence | Mapping[str, Any]],
) -> AlignmentResult:
    """Align prediction evidence while preserving every expected request and every failure."""
    expected_requests = tuple(requests)
    expected_by_id: dict[str, RequestRecord] = {}
    for request in expected_requests:
        if request.sample_id in expected_by_id:
            raise ContractError(f"duplicate expected request sample_id: {request.sample_id}")
        expected_by_id[request.sample_id] = request

    evidence = tuple(_coerce_evidence(item, index) for index, item in enumerate(predictions, 1))
    by_id: dict[str, list[PredictionEvidence]] = {}
    unkeyed: list[PredictionEvidence] = []
    for item in evidence:
        if item.sample_id is None:
            unkeyed.append(item)
        else:
            by_id.setdefault(item.sample_id, []).append(item)

    rows: list[AlignedPrediction] = []
    failures: list[FailureRow] = []
    duplicate_predictions = 0
    valid_expected = 0
    for request in expected_requests:
        matches = by_id.pop(request.sample_id, [])
        if not matches:
            failure = FailureRow(
                request.sample_id,
                "missing_prediction",
                "no prediction row matched the expected sample_id",
                True,
            )
        elif len(matches) > 1:
            duplicate_predictions += len(matches)
            failure = FailureRow(
                request.sample_id,
                "duplicate_prediction",
                f"{len(matches)} prediction rows matched one expected sample_id",
                True,
                tuple(item.prediction_index for item in matches),
                [item.raw for item in matches],
            )
        else:
            failure = _expected_prediction_failure(request, matches[0])
            if failure is None:
                rows.append(AlignedPrediction(request, matches[0].prediction, None))
                valid_expected += 1
                continue
        rows.append(AlignedPrediction(request, None, failure))
        failures.append(failure)

    extraneous = [*unkeyed, *(item for matches in by_id.values() for item in matches)]
    for item in sorted(extraneous, key=lambda row: row.prediction_index):
        malformed = item.prediction is None
        failures.append(
            FailureRow(
                item.sample_id,
                "malformed_prediction" if malformed else "unknown_sample_id",
                item.error or "prediction sample_id is not present in expected requests",
                False,
                (item.prediction_index,),
                item.raw,
            )
        )

    return AlignmentResult(
        rows=tuple(rows),
        failures=tuple(failures),
        expected=len(expected_requests),
        valid_expected=valid_expected,
        expected_failures=len(expected_requests) - valid_expected,
        extraneous_predictions=len(extraneous),
        duplicate_predictions=duplicate_predictions,
    )


def _coerce_evidence(
    item: PredictionRecord | PredictionEvidence | Mapping[str, Any], index: int
) -> PredictionEvidence:
    if isinstance(item, PredictionEvidence):
        return item
    if isinstance(item, PredictionRecord):
        return PredictionEvidence(index, item, item.sample_id, raw=item.to_dict())
    return _prediction_evidence(item, index)


def _prediction_evidence(raw: Any, index: int) -> PredictionEvidence:
    sample_id = None
    if isinstance(raw, Mapping):
        candidate = raw.get("sample_id")
        if isinstance(candidate, str) and candidate.strip():
            sample_id = candidate.strip()
    try:
        prediction = PredictionRecord.from_dict(raw)
    except (ContractError, TypeError) as error:
        return PredictionEvidence(index, None, sample_id, str(error), raw)
    return PredictionEvidence(index, prediction, prediction.sample_id, raw=raw)


def _expected_prediction_failure(
    request: RequestRecord, evidence: PredictionEvidence
) -> FailureRow | None:
    if evidence.prediction is None:
        return FailureRow(
            request.sample_id,
            "malformed_prediction",
            evidence.error or "prediction row violates the persisted contract",
            True,
            (evidence.prediction_index,),
            evidence.raw,
        )
    prediction = evidence.prediction
    if prediction.status == "error":
        return FailureRow(
            request.sample_id,
            "timed_out_prediction"
            if prediction.error_code == "timeout"
            else "explicit_error_prediction",
            prediction.error or "prediction status is error",
            True,
            (evidence.prediction_index,),
            evidence.raw,
        )
    expects_boxes = request.expected_output == "boxes"
    shape_matches = (
        prediction.boxes is not None and prediction.prediction is None
        if expects_boxes
        else prediction.prediction is not None and prediction.boxes is None
    )
    if not shape_matches:
        return FailureRow(
            request.sample_id,
            "output_shape_mismatch",
            f"expected_output={request.expected_output} does not match prediction fields",
            True,
            (evidence.prediction_index,),
            evidence.raw,
        )
    return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _validate_context(context: MetricContext) -> None:
    for field in ("run_id", "recorded_at", "protocol_id", "benchmark_version", "source_log_path"):
        value = getattr(context, field)
        if not isinstance(value, str) or not value.strip():
            raise EvaluationError(f"metric context {field} must be a nonempty string")
    source_path = Path(context.source_log_path)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise EvaluationError("source_log_path must be a run-relative path")


def _metadata_text(metadata: Mapping[str, Any], field: str) -> str | None:
    value = metadata.get(field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _in_canonical_range(value: float, expression: str) -> bool:
    if expression == "[0,1]":
        return 0.0 <= value <= 1.0
    if expression == "[0,+inf)":
        return value >= 0.0
    if expression in {"(-inf,+inf)", ""}:
        return True
    raise EvaluationError(f"unsupported canonical range expression: {expression}")


def _read_references(path: Path) -> dict[str, Mapping[str, Any]]:
    references: dict[str, Mapping[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationError(f"could not read references: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"references line {line_number}: invalid JSON") from error
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"references line {line_number}: expected an object")
        sample_id = raw.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise EvaluationError(f"references line {line_number}: sample_id is required")
        if sample_id in references:
            raise EvaluationError(f"references line {line_number}: duplicate sample_id: {sample_id}")
        references[sample_id] = raw
    return references


def _scorer_required(dataset: str, requests: Iterable[RequestRecord]) -> bool:
    tasks = {request.task for request in requests}
    if dataset in {"levir_cc", "vrsbench"}:
        return "caption" in tasks
    if dataset == "xlrs_bench":
        return bool(tasks & {"caption", "grounding"})
    return False


def _route_benchmark_metrics(
    dataset: str,
    alignment: AlignmentResult,
    references: Mapping[str, Mapping[str, Any]],
    registry: MetricRegistry,
    context: MetricContext,
    scores: Iterable[OfficialMetricScore],
) -> list[MetricRecord]:
    if dataset == "mme_rs":
        from m3rs_eval.datasets.mme_rs import evaluate_mme_alignment

        return evaluate_mme_alignment(alignment, references, registry, context)
    if dataset == "vrsbench":
        from m3rs_eval.datasets.vrsbench import evaluate_vrs_alignment

        return evaluate_vrs_alignment(alignment, references, registry, context, scores)
    if dataset == "xlrs_bench":
        from m3rs_eval.datasets.xlrs_bench import evaluate_xlrs_alignment

        return evaluate_xlrs_alignment(alignment, references, registry, context, scores)
    if dataset == "levir_cc":
        from m3rs_eval.datasets.levir_cc import evaluate_levir_alignment

        return evaluate_levir_alignment(alignment, references, registry, context, scores)
    raise EvaluationError(f"unsupported dataset adapter: {dataset}")


def _add_required_unavailable_records(
    adapter: Any,
    records: list[MetricRecord],
    registry: MetricRegistry,
    context: MetricContext,
    alignment: AlignmentResult,
    references: Mapping[str, Mapping[str, Any]],
    scorer_status: str,
) -> list[MetricRecord]:
    existing = {record.metric_id for record in records}
    required = _required_metric_ids_for_adapter(adapter)
    result = list(records)
    for metric_id in required:
        if metric_id in existing:
            continue
        rule = METRIC_SCOPE_RULES.get(metric_id)
        if rule is None:
            raise EvaluationError(f"required metric has no explicit Task 5 scope rule: {metric_id}")
        scoped_rows = _scope_rows(rule, alignment, references)
        n_samples = len(scoped_rows)
        n_failures = sum(row.failure is not None for row in scoped_rows)
        availability = (
            "failed"
            if rule.provenance == "official" and scorer_status == "failed"
            else "missing"
        )
        result.append(
            make_metric_record(
                registry,
                context,
                metric_id,
                None,
                n_samples=n_samples,
                n_failures=n_failures,
                provenance=rule.provenance,
                availability=availability,
                notes=f"required protocol metric unavailable; official_scorer_status={scorer_status}",
            )
        )
    return result


def _required_metric_failures(adapter: Any, records: Iterable[MetricRecord]) -> list[FailureRow]:
    required = _required_metric_ids_for_adapter(adapter)
    allowed_na = set(_collect_protocol_list(adapter.protocol, "allowed_not_applicable_metric_ids"))
    by_id: dict[str, list[MetricRecord]] = {}
    for record in records:
        by_id.setdefault(record.metric_id, []).append(record)
    failures: list[FailureRow] = []
    for metric_id in required:
        matches = by_id.get(metric_id, [])
        if len(matches) != 1:
            failures.append(
                FailureRow(
                    None,
                    "required_metric_duplicate" if matches else "required_metric_missing",
                    f"required metric {metric_id} produced {len(matches)} records",
                    False,
                    raw={"metric_id": metric_id, "record_count": len(matches)},
                )
            )
            continue
        record = matches[0]
        accepted_na = record.availability == "not_applicable" and metric_id in allowed_na
        if record.availability != "available" and not accepted_na:
            failures.append(
                FailureRow(
                    None,
                    "required_metric_unavailable",
                    f"required metric {metric_id} availability={record.availability}",
                    False,
                    raw={
                        "metric_id": metric_id,
                        "availability": record.availability,
                        "allowlisted_not_applicable": False,
                    },
                )
            )
    return failures


def _collect_protocol_list(value: Any, key_name: str) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, child in value.items():
            if key == key_name and isinstance(child, list):
                result.extend(str(item) for item in child)
            else:
                result.extend(_collect_protocol_list(child, key_name))
        return result
    if isinstance(value, list):
        return [item for child in value for item in _collect_protocol_list(child, key_name)]
    return []


def _required_metric_ids_for_adapter(adapter: Any) -> list[str]:
    protocol = adapter.protocol
    if not isinstance(protocol, Mapping):
        return []
    datasets = protocol.get("datasets")
    if not isinstance(datasets, Mapping):
        return []
    dataset = datasets.get(adapter.dataset)
    return _collect_required_metric_ids(dataset)


def _collect_required_metric_ids(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, child in value.items():
            if key == "required_metric_ids" and isinstance(child, list):
                result.extend(str(metric_id) for metric_id in child)
            else:
                result.extend(_collect_required_metric_ids(child))
        return result
    if isinstance(value, list):
        return [metric_id for child in value for metric_id in _collect_required_metric_ids(child)]
    return []


def _scorer_version(path: Path) -> str:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"malformed official scorer output: {error}") from error
    version = raw.get("scorer_version") if isinstance(raw, Mapping) else None
    if not isinstance(version, str) or not version.strip():
        raise EvaluationError("malformed official scorer output: scorer_version is required")
    return version.strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
