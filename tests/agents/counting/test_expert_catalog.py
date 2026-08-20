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
    return ExpertCatalog.load(path, asset_root=REPO_ROOT)


def _expert_payload(payload: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in payload["experts"] if item["backend_name"] == name)


def test_semantic_label_absent_from_verified_class_map_fails_closed(
    tmp_path: Path,
) -> None:
    payload = _payload()
    _expert_payload(payload, "segmenter_mitb2_001")["supports"]["small-vehicle"]["model_labels"] = [
        "not_a_verified_label"
    ]

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_valid_catalog_loads_and_exposes_capability_specs() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    expert = catalog.expert("segmenter_mitb2_001")

    assert expert.kind == "semantic_segmentation"
    assert expert.priority == 100
    assert expert.supports["small-vehicle"].model_labels == ("Small_Vehicle",)
    assert expert.supports["small-vehicle"].counting_mode == "connected_components"


def test_change_semantic_roles_are_explicit_and_verified() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    expert = catalog.expert("segmenter_mitb2_001")

    assert expert.change_semantics is not None
    assert expert.change_semantics.enabled is True
    assert expert.change_semantics.role == "object_semantic"
    assert "background" in expert.change_semantics.neutral_model_labels
    assert "plane" in expert.change_semantics.transient_model_labels
    assert "storage_tank" in expert.change_semantics.persistent_model_labels


def test_change_semantic_label_absent_from_verified_class_map_fails_closed(
    tmp_path: Path,
) -> None:
    payload = _payload()
    _expert_payload(payload, "segmenter_mitb2_001")["change_semantics"][
        "persistent_model_labels"
    ] = ["not_a_verified_label"]

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_expert_supports_are_physical_canonical_leaves_only() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)
    for expert in catalog.experts(enabled_only=False):
        assert "vehicle" not in expert.supports
        assert "aircraft" not in expert.supports
        assert "background" not in expert.supports
    assert catalog.candidates(_target("vehicle")) == ()
    assert catalog.candidates(_target("aircraft")) == ()


def test_duplicate_or_placeholder_leaf_model_labels_fail_closed(
    tmp_path: Path,
) -> None:
    for labels in (
        ["Small_Vehicle", "small_vehicle"],
        ["Small_Vehicle", "LABEL_7"],
    ):
        payload = _payload()
        _expert_payload(payload, "segmenter_mitb2_001")["supports"]["small-vehicle"]["model_labels"] = labels
        with pytest.raises(ExpertCatalogError, match="validation failed"):
            _load_payload(tmp_path, payload)


def test_yolo_capabilities_use_declared_detector_identity_and_labels() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "yolo.example.yaml").read_text(encoding="utf-8")
    )
    detector = config["backend"]["yolo"]["detectors"][0]
    expert = ExpertCatalog.load(CATALOG_PATH).expert("detector_obb_csl_001")
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
    expert = ExpertCatalog.load(CATALOG_PATH).expert("segmenter_mitb2_001")
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
        "detector_yolo_detect_001",
        "detector_obb_csl_001",
        "segmenter_mitb2_001",
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
    payload["experts"] = [
        _expert_payload(payload, "segmenter_mitb2_001"), low, high_b,
        _expert_payload(payload, "segmenter_oem_001"), high_a,
    ]
    catalog = _load_payload(tmp_path, payload)

    assert tuple(expert.backend_name for expert in catalog.candidates(_target("plane"))) == (
        "detector_a",
        "detector_b",
        "detector_z",
        "segmenter_mitb2_001",
    )


def test_enabled_semantic_expert_requires_verified_class_map(tmp_path: Path) -> None:
    payload = _payload()
    _expert_payload(payload, "segmenter_mitb2_001")["verification"]["class_map"] = "unverified"

    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


def test_unverified_oem_expert_is_blocked_until_a_channel_map_is_verified() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    oem = catalog.expert("segmenter_oem_001")

    assert oem.enabled is False
    assert oem.status == "blocked_unverified_class_map"
    assert oem.verification.class_map == "unverified"
    assert oem.asset.class_map is None
    assert oem.supports == {}
    assert oem not in catalog.candidates(_target("small vehicle"), enabled_only=False)


def test_separator_and_case_normalization_maps_isaid_label() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    candidates = catalog.candidates(_target("Small_Vehicle"))

    assert tuple(expert.backend_name for expert in candidates) == (
        "detector_yolo_detect_001",
        "detector_obb_csl_001",
        "segmenter_mitb2_001",
    )


def test_unsupported_target_has_no_candidates_or_hints() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)
    isaid = catalog.expert("segmenter_mitb2_001")

    assert catalog.candidates(_target("water")) == ()
    assert catalog.target_hints(_target("water")) == {}
    assert tuple(
        expert.backend_name for expert in catalog.candidates(_target("bridge"))
    ) == ("detector_yolo_detect_001", "detector_obb_csl_001")
    assert "background" not in isaid.supports
    assert isaid.supports["bridge"].counting_mode == "unsupported"
    assert isaid.supports["harbor"].counting_mode == "unsupported"


def test_parent_target_cannot_be_added_to_physical_support(tmp_path: Path) -> None:
    payload = _payload()
    payload["experts"][0]["supports"]["vehicle"] = {
        "model_labels": ["small vehicle"],
        "counting_mode": "native_detection",
    }
    with pytest.raises(ExpertCatalogError, match="validation failed"):
        _load_payload(tmp_path, payload)


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

    assert tuple(expert.backend_name for expert in semantic) == ("segmenter_mitb2_001",)
    with pytest.raises(ValueError, match="unknown expert kind filter"):
        catalog.candidates(_target("plane"), kinds=frozenset({"unknown"}))


def test_public_expert_enumeration_is_immutable_stable_and_filtered() -> None:
    catalog = ExpertCatalog.load(CATALOG_PATH)

    enabled = catalog.experts()
    semantic = catalog.experts(kinds=frozenset({"semantic_segmentation"}))
    all_semantic = catalog.experts(
        kinds=frozenset({"semantic_segmentation"}), enabled_only=False
    )

    assert isinstance(enabled, tuple)
    assert tuple(expert.backend_name for expert in enabled) == (
        "detector_yolo_detect_001",
        "detector_obb_csl_001",
        "segmenter_mitb2_001",
    )
    assert tuple(expert.backend_name for expert in semantic) == (
        "segmenter_mitb2_001",
    )
    assert tuple(expert.backend_name for expert in all_semantic) == (
        "segmenter_mitb2_001",
        "segmenter_oem_001",
    )
    with pytest.raises(ValueError, match="unknown expert kind filter"):
        catalog.experts(kinds=frozenset({"unknown"}))
