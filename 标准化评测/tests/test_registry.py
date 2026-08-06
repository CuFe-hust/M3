from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conftest import SOURCE_REGISTRY_SHA256
from m3rs_eval.registry import MetricRegistry, RegistryError, normalize_value


def test_registry_has_194_unique_ids(metric_registry_path: Path):
    registry = MetricRegistry.load(metric_registry_path)

    assert len(registry) == 194
    assert len(set(registry.ids)) == 194


def test_vendored_registry_matches_recorded_source_provenance(
    project_root: Path, metric_registry_path: Path
):
    source_path = (
        project_root.parent
        / "指标字典"
        / "outputs"
        / "019fc7cf-aa6b-7143-b248-647f0db1037d"
        / "metrics.yaml"
    )

    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == SOURCE_REGISTRY_SHA256
    assert hashlib.sha256(metric_registry_path.read_bytes()).hexdigest() == SOURCE_REGISTRY_SHA256


def test_ratio_display_does_not_change_canonical_value(registry):
    metric = registry.require("mme_rs.avg")

    assert metric.canonical_unit == "ratio"
    assert metric.display_multiplier == 100
    assert normalize_value(metric, 0.42) == pytest.approx(0.42)


def test_require_rejects_unknown_metric_id(registry):
    with pytest.raises(RegistryError, match="unknown metric_id: missing.metric"):
        registry.require("missing.metric")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("direction", None, "direction"),
        ("canonical_unit", None, "canonical_unit"),
        ("display_multiplier", None, "display_multiplier"),
        ("source_refs", [], "source_refs"),
    ],
)
def test_loader_rejects_missing_required_metric_metadata(
    metric_registry_path: Path, tmp_path: Path, field: str, replacement: object, message: str
):
    payload = _read_registry(metric_registry_path)
    if replacement is None:
        del payload["metrics"][0][field]
    else:
        payload["metrics"][0][field] = replacement

    with pytest.raises(RegistryError, match=message):
        MetricRegistry.load(_write_registry(tmp_path, payload))


def test_loader_rejects_duplicate_metric_id(metric_registry_path: Path, tmp_path: Path):
    payload = _read_registry(metric_registry_path)
    payload["metrics"][1]["metric_id"] = payload["metrics"][0]["metric_id"]

    with pytest.raises(RegistryError, match="duplicate metric_id"):
        MetricRegistry.load(_write_registry(tmp_path, payload))


def test_loader_rejects_unknown_authority(metric_registry_path: Path, tmp_path: Path):
    payload = _read_registry(metric_registry_path)
    payload["metrics"][0]["authority"] = "Z"

    with pytest.raises(RegistryError, match="unknown authority"):
        MetricRegistry.load(_write_registry(tmp_path, payload))


def test_loader_rejects_unknown_source_reference(metric_registry_path: Path, tmp_path: Path):
    payload = _read_registry(metric_registry_path)
    payload["metrics"][0]["source_refs"] = ["not-a-source"]

    with pytest.raises(RegistryError, match="unknown source reference"):
        MetricRegistry.load(_write_registry(tmp_path, payload))


@pytest.mark.parametrize("replacement", ["100", True, float("nan")])
def test_loader_rejects_nonnumeric_multiplier(
    metric_registry_path: Path, tmp_path: Path, replacement: object
):
    payload = _read_registry(metric_registry_path)
    payload["metrics"][0]["display_multiplier"] = replacement

    with pytest.raises(RegistryError, match="display_multiplier"):
        MetricRegistry.load(_write_registry(tmp_path, payload))


def _read_registry(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "metrics.yaml"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path
