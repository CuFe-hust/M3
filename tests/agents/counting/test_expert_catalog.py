"""Contract tests for the data-driven counting expert catalog."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from agents.counting.expert_catalog import ExpertCatalog, ExpertCatalogError
from agents.counting.schema import CountTargetSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "agents" / "counting" / "expert_catalog.json"


def _target(label: str) -> CountTargetSpec:
    return CountTargetSpec(
        canonical_label=label,
        inclusion_rule="count each visible instance",
        exclusion_rule="exclude ambiguous fragments",
    )


def _payload() -> dict[str, object]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _load_payload(tmp_path: Path, payload: dict[str, object]) -> ExpertCatalog:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ExpertCatalog.load(path)


def test_valid_catalog_loads_and_exposes_capability_specs() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    expert = catalog.expert("segmenter_isaid_001")

    assert expert.kind == "semantic_segmentation"
    assert expert.priority == 100
    assert expert.supports["small-vehicle"].model_labels == ("Small_Vehicle",)
    assert expert.supports["small-vehicle"].counting_mode == "connected_components"


def test_yolo_capabilities_use_declared_detector_identity_and_labels() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "yolo.example.yaml").read_text(encoding="utf-8")
    )
    detector = config["backend"]["yolo"]["detectors"][0]
    expert = ExpertCatalog.load(CATALOG_PATH).expert("yolov5m_obb_csl_dotav20")
    capability_labels = {
        label
        for support in expert.supports.values()
        for label in support.model_labels
    }

    assert expert.backend_name == detector["name"]
    assert expert.logical_model_id == detector["model_id"]
    assert expert.asset.sha256 == detector["sha256"]
    assert capability_labels <= set(detector["classes"])


def test_isaid_capabilities_use_only_verified_class_map_labels() -> None:
    class_map = json.loads(
        (
            REPO_ROOT / "models" / "segformer_mitb2_isaid" / "classes.json"
        ).read_text(encoding="utf-8")
    )
    expert = ExpertCatalog.load(CATALOG_PATH).expert("segmenter_isaid_001")
    capability_labels = {
        label
        for support in expert.supports.values()
        for label in support.model_labels
    }

    assert capability_labels <= set(class_map["id2name"].values())
    assert not any(label.startswith("LABEL_") for label in capability_labels)


def test_duplicate_backend_name_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["experts"].append(copy.deepcopy(payload["experts"][0]))

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_unknown_kind_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["experts"][0]["kind"] = "mystery_model"

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_counting_mode_must_match_expert_kind(tmp_path: Path) -> None:
    payload = _payload()
    payload["experts"][0]["supports"]["plane"]["counting_mode"] = "connected_components"
    payload["experts"][0]["supports"]["plane"]["policy"] = {
        "min_component_area_px": 16,
        "min_mean_confidence": 0.45,
    }

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_disabled_expert_is_excluded_by_default(tmp_path: Path) -> None:
    payload = _payload()
    disabled = copy.deepcopy(payload["experts"][0])
    disabled["backend_name"] = "disabled_detector"
    disabled["enabled"] = False
    disabled["priority"] = 500
    payload["experts"].append(disabled)
    catalog = _load_payload(tmp_path, payload)

    default_names = tuple(expert.backend_name for expert in catalog.candidates(_target("plane")))
    all_names = tuple(
        expert.backend_name
        for expert in catalog.candidates(_target("plane"), enabled_only=False)
    )

    assert "disabled_detector" not in default_names
    assert "disabled_detector" in all_names


def test_explicit_alias_resolves_to_canonical_target() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    candidates = catalog.candidates(_target("passenger car"))
    hints = catalog.target_hints(_target("car"))

    assert tuple(expert.backend_name for expert in candidates) == (
        "yolov5m_obb_csl_dotav20",
        "segmenter_isaid_001",
    )
    assert hints["canonical_label"] == "small-vehicle"


def test_candidates_have_deterministic_kind_priority_name_order(tmp_path: Path) -> None:
    payload = _payload()
    low = copy.deepcopy(payload["experts"][0])
    low["backend_name"] = "detector_z"
    low["priority"] = 10
    high_a = copy.deepcopy(payload["experts"][0])
    high_a["backend_name"] = "detector_a"
    high_a["priority"] = 200
    high_b = copy.deepcopy(payload["experts"][0])
    high_b["backend_name"] = "detector_b"
    high_b["priority"] = 200
    payload["experts"] = [payload["experts"][1], low, high_b, payload["experts"][2], high_a]
    catalog = _load_payload(tmp_path, payload)

    assert tuple(expert.backend_name for expert in catalog.candidates(_target("plane"))) == (
        "detector_a",
        "detector_b",
        "detector_z",
        "segmenter_isaid_001",
    )


def test_enabled_semantic_expert_requires_verified_class_map(tmp_path: Path) -> None:
    payload = _payload()
    payload["experts"][1]["verification"]["class_map"] = "unverified"

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_blocked_oem_expert_loads_but_is_not_a_candidate() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    oem = catalog.expert("segmenter_oem_001")

    assert oem.enabled is False
    assert oem.status == "blocked_unverified_class_map"
    assert oem.supports == {}
    assert oem not in catalog.candidates(_target("small vehicle"), enabled_only=False)


def test_separator_and_case_normalization_maps_isaid_label() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    candidates = catalog.candidates(_target("Small_Vehicle"))

    assert tuple(expert.backend_name for expert in candidates) == (
        "yolov5m_obb_csl_dotav20",
        "segmenter_isaid_001",
    )


def test_unsupported_target_has_no_candidates_or_hints() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)
    isaid = catalog.expert("segmenter_isaid_001")

    assert catalog.candidates(_target("water")) == ()
    assert catalog.target_hints(_target("water")) == {}
    assert catalog.candidates(_target("bridge")) == ()
    assert isaid.supports["background"].counting_mode == "unsupported"
    assert isaid.supports["bridge"].counting_mode == "unsupported"
    assert isaid.supports["harbor"].counting_mode == "unsupported"


def test_parse_error_does_not_leak_absolute_path(tmp_path: Path) -> None:
    secret_parent = tmp_path / "machine-specific-secret-directory"
    secret_parent.mkdir()
    path = secret_parent / "catalog.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ExpertCatalogError) as error:
        ExpertCatalog.load(path)

    assert str(path) not in str(error.value)
    assert str(secret_parent) not in str(error.value)
    assert str(error.value) == "expert catalog is not valid JSON"


def test_unknown_support_target_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["experts"][0]["supports"]["invented-target"] = {
        "model_labels": ["plane"],
        "counting_mode": "native_detection",
    }

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_alias_conflict_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["targets"]["ship"]["aliases"].append("car")

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_placeholder_label_cannot_become_a_mapping(tmp_path: Path) -> None:
    payload = _payload()
    payload["targets"]["label-0"] = {
        "aliases": [],
        "countable": True,
        "hints": [],
    }

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_kind_filter_is_explicit_and_validated() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    semantic = catalog.candidates(
        _target("plane"), kinds=frozenset({"semantic_segmentation"})
    )

    assert tuple(expert.backend_name for expert in semantic) == ("segmenter_isaid_001",)
    with pytest.raises(ValueError, match="unknown expert kind filter"):
        catalog.candidates(_target("plane"), kinds=frozenset({"unknown"}))
