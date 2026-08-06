"""Metric registry loading and canonical-value normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import yaml


class RegistryError(ValueError):
    """Raised when a metric registry is invalid or a metric is unavailable."""


@dataclass(frozen=True)
class MetricDefinition:
    """Validated metric fields needed by evaluation and presentation layers."""

    metric_id: str
    authority: str
    direction: str
    canonical_unit: str
    display_multiplier: float
    source_refs: tuple[str, ...]
    metadata: Mapping[str, Any]


class MetricRegistry:
    """A validated, immutable lookup table for registered metrics."""

    def __init__(self, metrics: tuple[MetricDefinition, ...]):
        self._metrics = metrics
        self._by_id = {metric.metric_id: metric for metric in metrics}

    @classmethod
    def load(cls, path: Path) -> "MetricRegistry":
        """Load and validate a YAML registry snapshot."""
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise RegistryError(f"could not load metric registry: {path}") from error
        if not isinstance(raw, Mapping):
            raise RegistryError("metric registry must be an object")

        authorities = raw.get("authority_tiers")
        if not isinstance(authorities, Mapping) or not authorities:
            raise RegistryError("metric registry requires authority_tiers")
        source_ids = _source_ids(raw.get("sources"))

        raw_metrics = raw.get("metrics")
        if not isinstance(raw_metrics, list):
            raise RegistryError("metric registry requires a metrics list")

        metrics: list[MetricDefinition] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(raw_metrics):
            metric = _metric_definition(item, index, set(authorities), source_ids)
            if metric.metric_id in seen_ids:
                raise RegistryError(f"duplicate metric_id: {metric.metric_id}")
            seen_ids.add(metric.metric_id)
            metrics.append(metric)
        return cls(tuple(metrics))

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(metric.metric_id for metric in self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def require(self, metric_id: str) -> MetricDefinition:
        """Return a registered metric or raise a contextual validation error."""
        try:
            return self._by_id[metric_id]
        except KeyError as error:
            raise RegistryError(f"unknown metric_id: {metric_id}") from error


def normalize_value(metric: MetricDefinition, raw_value: object) -> float:
    """Validate a canonical metric value without applying display formatting."""
    if not isinstance(metric, MetricDefinition):
        raise TypeError("metric must be a MetricDefinition")
    if not _finite_number(raw_value):
        raise ValueError(f"{metric.metric_id} value must be a finite number")
    return float(raw_value)


def _source_ids(raw_sources: object) -> set[str]:
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RegistryError("metric registry requires source references")
    source_ids: set[str] = set()
    for index, source in enumerate(raw_sources):
        if not isinstance(source, Mapping):
            raise RegistryError(f"source {index} must be an object")
        source_id = _required_text(source, "id", f"source {index}")
        if source_id in source_ids:
            raise RegistryError(f"duplicate source id: {source_id}")
        source_ids.add(source_id)
    return source_ids


def _metric_definition(
    raw_metric: object, index: int, authorities: set[object], source_ids: set[str]
) -> MetricDefinition:
    if not isinstance(raw_metric, Mapping):
        raise RegistryError(f"metric {index} must be an object")
    label = f"metric {index}"
    metric_id = _required_text(raw_metric, "metric_id", label)
    authority = _required_text(raw_metric, "authority", metric_id)
    if authority not in authorities:
        raise RegistryError(f"{metric_id} has unknown authority: {authority}")
    direction = _required_text(raw_metric, "direction", metric_id)
    canonical_unit = _required_text(raw_metric, "canonical_unit", metric_id)
    multiplier = raw_metric.get("display_multiplier")
    if not _finite_number(multiplier):
        raise RegistryError(f"{metric_id} display_multiplier must be a finite number")

    raw_source_refs = raw_metric.get("source_refs")
    if not isinstance(raw_source_refs, list) or not raw_source_refs:
        raise RegistryError(f"{metric_id} source_refs must be a nonempty list")
    source_refs = tuple(_required_text({"source_ref": ref}, "source_ref", metric_id) for ref in raw_source_refs)
    unknown_sources = sorted(set(source_refs) - source_ids)
    if unknown_sources:
        raise RegistryError(f"{metric_id} has unknown source reference: {unknown_sources[0]}")

    return MetricDefinition(
        metric_id=metric_id,
        authority=authority,
        direction=direction,
        canonical_unit=canonical_unit,
        display_multiplier=float(multiplier),
        source_refs=source_refs,
        metadata=dict(raw_metric),
    )


def _required_text(raw: Mapping[str, Any], field: str, label: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{label} requires {field}")
    return value.strip()


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
