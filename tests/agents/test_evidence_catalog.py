"""Contract tests for canonical leaves, aliases, parents, and capability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.counting.expert_catalog import ExpertCatalog
from agents.evidence_catalog import CatalogCategoryError, EvidenceCatalog
from application.settings import load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_CATALOG = REPO_ROOT / "agents" / "evidence_catalog.json"
_COUNTING_CATALOG = REPO_ROOT / "agents" / "counting" / "expert_catalog.json"
_TASKS = ("counting", "fine_grained_counting", "general_vqa", "grounding")


def _data() -> dict[str, object]:
    leaves = {
        "small-vehicle": {
            "yolo_labels": ["small vehicle"],
            "segformer_labels": ["Small_Vehicle"],
            "segformer_binding": "segmenter_mitb2_001",
            "yolo_enabled": True,
            "segformer_enabled": True,
        },
        "large-vehicle": {
            "yolo_labels": ["large vehicle"],
            "segformer_labels": None,
            "yolo_enabled": True,
            "segformer_enabled": False,
        },
        "plane": {
            "yolo_labels": ["plane"],
            "segformer_labels": None,
            "yolo_enabled": True,
            "segformer_enabled": False,
        },
        "helicopter": {
            "yolo_labels": ["helicopter"],
            "segformer_labels": None,
            "yolo_enabled": True,
            "segformer_enabled": False,
        },
    }
    return {
        "catalog_version": "catalog-test-v3",
        "aliases": {"airplane": "plane", "aeroplane": "plane"},
        "parents": {
            "vehicle": ["small-vehicle", "large-vehicle"],
            "aircraft": ["plane", "helicopter"],
        },
        "leaves": leaves,
        "task_capabilities": {task: list(leaves) for task in _TASKS},
    }


def _catalog(**overrides: object) -> EvidenceCatalog:
    data = _data()
    data.update(overrides)
    return EvidenceCatalog(data)


def test_catalog_exposes_stable_canonical_shape() -> None:
    catalog = _catalog()
    assert catalog.catalog_version == "catalog-test-v3"
    assert catalog.leaf_categories == (
        "small-vehicle", "large-vehicle", "plane", "helicopter"
    )
    assert catalog.parent_categories == ("vehicle", "aircraft")
    assert catalog.is_leaf("small-vehicle")
    assert catalog.is_parent("vehicle")
    assert not catalog.is_leaf("small_vehicle")
    assert dict(catalog.parent_expansions)["vehicle"] == (
        "small-vehicle", "large-vehicle"
    )


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("small_vehicle", "small-vehicle"),
        ("small vehicle", "small-vehicle"),
        ("small-vehicle", "small-vehicle"),
        ("airplane", "plane"),
        ("aeroplane", "plane"),
    ],
)
def test_alias_canonicalization_is_deterministic(raw: str, canonical: str) -> None:
    assert _catalog().canonicalize_alias(raw) == canonical


def test_parent_and_alias_expansion_is_central_and_ordered() -> None:
    catalog = _catalog()
    assert catalog.expand_target("vehicle") == ("small-vehicle", "large-vehicle")
    assert catalog.expand_target("aircraft") == ("plane", "helicopter")
    assert catalog.expand_target("airplane") == ("plane",)
    assert catalog.expand_target("unknown-object") == ()


def test_plan_validation_accepts_only_canonical_executable_leaves() -> None:
    catalog = _catalog()
    assert catalog.validate_plan_leaves(
        ["small-vehicle", "large-vehicle"], task="counting"
    ) == ("small-vehicle", "large-vehicle")
    for invalid in (
        "vehicle", "airplane", "small_vehicle", "small vehicle", "LABEL_0",
        "background", "unknown-vehical",
    ):
        with pytest.raises(CatalogCategoryError, match="PLAN_CATEGORY_NOT_CANONICAL_LEAF"):
            catalog.validate_plan_leaves([invalid], task="counting")
    with pytest.raises(CatalogCategoryError, match="PLAN_CATEGORY_DUPLICATED"):
        catalog.validate_plan_leaves(["plane", "plane"], task="grounding")


def test_per_task_capability_is_explicit() -> None:
    data = _data()
    data["task_capabilities"]["grounding"] = ["plane"]
    catalog = EvidenceCatalog(data)
    assert catalog.executable_leaves_for_task("grounding") == ("plane",)
    assert catalog.executable_leaves_for_target("aircraft", task="grounding") == ()
    assert catalog.executable_leaves_for_target("airplane", task="grounding") == (
        "plane",
    )
    with pytest.raises(CatalogCategoryError, match="NOT_EXECUTABLE"):
        catalog.validate_plan_leaves(["helicopter"], task="grounding")


@pytest.mark.parametrize("invalid", ["../plane", "/plane", "plane\\x", "bad\nname", ""])
def test_category_inputs_reject_paths_controls_and_empty(invalid: str) -> None:
    with pytest.raises(CatalogCategoryError, match="INVALID_CATEGORY_NAME"):
        _catalog().canonicalize_alias(invalid)


def test_catalog_rejects_invalid_structure_and_alias_conflicts() -> None:
    data = _data()
    data["unexpected"] = True
    with pytest.raises(CatalogCategoryError, match="INVALID_TOP_LEVEL_KEYS"):
        EvidenceCatalog(data)

    data = _data()
    data["parents"] = {"plane": ["plane"]}
    with pytest.raises(CatalogCategoryError, match="COLLIDES_WITH_LEAF"):
        EvidenceCatalog(data)

    data = _data()
    data["aliases"] = {"flying-machine": "missing"}
    with pytest.raises(CatalogCategoryError, match="ALIAS_TARGET_UNKNOWN"):
        EvidenceCatalog(data)

    data = _data()
    data["parents"] = {"vehicle": ["small-vehicle", "small-vehicle"]}
    with pytest.raises(CatalogCategoryError, match="PARENT_TARGETS_DUPLICATED"):
        EvidenceCatalog(data)


def test_enabled_capability_requires_a_verified_raw_label() -> None:
    data = _data()
    data["leaves"]["plane"]["yolo_labels"] = []
    with pytest.raises(CatalogCategoryError, match="INVALID_LEAF_CAPABILITIES"):
        EvidenceCatalog(data)


@pytest.mark.parametrize(
    "capability",
    [
        {"segformer_labels": ["Small_Vehicle"]},
        {"segformer_labels": None, "segformer_binding": "segmenter_mitb2_001"},
    ],
)
def test_segformer_labels_and_binding_are_all_or_none(capability: dict[str, object]) -> None:
    data = _data()
    data["leaves"]["plane"].update(capability)
    with pytest.raises(CatalogCategoryError, match="INVALID_LEAF_CAPABILITIES"):
        EvidenceCatalog(data)


@pytest.mark.parametrize("binding", ["segmenter mitb2", "Segmenter_MiTB2", "LABEL_7"])
def test_segformer_binding_must_be_stable_lowercase_identifier(binding: str) -> None:
    data = _data()
    data["leaves"]["plane"].update(
        {
            "segformer_labels": ["plane"],
            "segformer_binding": binding,
            "yolo_enabled": False,
            "segformer_enabled": True,
        }
    )
    with pytest.raises(CatalogCategoryError, match="INVALID_LEAF_CAPABILITIES"):
        EvidenceCatalog(data)


def test_unknown_leaf_capability_queries_fail_closed() -> None:
    catalog = _catalog()
    with pytest.raises(CatalogCategoryError):
        catalog.leaf_yolo_labels("unknown-object")
    with pytest.raises(CatalogCategoryError):
        catalog.leaf_segformer_binding("unknown-object")
    with pytest.raises(CatalogCategoryError):
        catalog.capability_enabled("unknown-object", "yolo")


def test_catalog_rejects_background_placeholder_and_bad_model_labels() -> None:
    for leaf in ("background", "label-0"):
        data = _data()
        data["leaves"] = {leaf: {"yolo_labels": [leaf], "yolo_enabled": True}}
        data["parents"] = {}
        data["task_capabilities"] = {task: [leaf] for task in _TASKS}
        with pytest.raises(CatalogCategoryError, match="INVALID_LEAF"):
            EvidenceCatalog(data)

    data = _data()
    data["leaves"]["plane"]["yolo_labels"] = ["LABEL_7"]
    with pytest.raises(CatalogCategoryError, match="INVALID_LEAF_CAPABILITIES"):
        EvidenceCatalog(data)


def test_capability_labels_and_identity_are_versioned() -> None:
    catalog = _catalog()
    assert catalog.leaf_yolo_labels("small-vehicle") == ("small vehicle",)
    assert catalog.leaf_segformer_labels("small-vehicle") == ("Small_Vehicle",)
    assert catalog.leaf_segformer_binding("small-vehicle") == "segmenter_mitb2_001"
    assert catalog.leaf_segformer_binding("large-vehicle") is None
    assert catalog.capability_enabled("small-vehicle", "segformer")
    assert not catalog.capability_enabled("large-vehicle", "segformer")
    assert catalog.capability_identity("small-vehicle", "yolo") == (
        "catalog-test-v3:small-vehicle:yolo"
    )


def test_production_catalog_publishes_only_verified_current_leaves() -> None:
    catalog = EvidenceCatalog.from_file(_PRODUCTION_CATALOG)
    expected_leaves = (
        # 15 iSAID-bound leaves (YOLO + SegFormer capability).
        "plane", "baseball-diamond", "bridge", "ground-track-field",
        "small-vehicle", "large-vehicle", "ship", "tennis-court",
        "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
        "harbor", "swimming-pool", "helicopter", "container-crane", "airport",
        "helipad",
        # 8 OEM-bound leaves (SegFormer-only capability).
        "bareland", "rangeland", "developed-space", "road", "tree", "water",
        "agriculture-land", "building",
    )
    expected_isaid = frozenset(expected_leaves[:15])
    expected_oem = frozenset(expected_leaves[18:])
    expected_yolo = frozenset(expected_leaves[:18])
    assert catalog.catalog_version == "visual-evidence-catalog-v4"
    assert catalog.leaf_categories == expected_leaves
    assert catalog.executable_leaves_for_task("counting") == expected_leaves[:18]
    assert set(catalog.executable_leaves_for_task("general_vqa")) == set(expected_leaves)
    assert {
        leaf for leaf in catalog.leaf_categories
        if catalog.capability_enabled(leaf, "segformer")
    } == expected_isaid | expected_oem
    for leaf in expected_isaid:
        assert catalog.leaf_segformer_binding(leaf) == "segmenter_mitb2_001"
    for leaf in expected_oem:
        assert catalog.leaf_segformer_binding(leaf) == "segmenter_oem_001"
        assert catalog.leaf_yolo_labels(leaf) == ()
        assert not catalog.capability_enabled(leaf, "yolo")
    assert "background" not in catalog.leaf_categories
    assert not any(leaf.startswith("label-") for leaf in catalog.leaf_categories)


def test_production_raw_labels_are_backed_by_current_model_maps() -> None:
    evidence = EvidenceCatalog.from_file(_PRODUCTION_CATALOG)
    experts = ExpertCatalog.load(_COUNTING_CATALOG, asset_root=REPO_ROOT)
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    (yolo,) = settings.backend.yolo.detectors
    semantic = experts.expert("segmenter_mitb2_001")
    semantic_labels = {
        label for support in semantic.supports.values() for label in support.model_labels
    }
    oem = experts.expert("segmenter_oem_001")
    oem_class_map = json.loads(
        (REPO_ROOT / oem.asset.class_map).read_text(encoding="utf-8")
    )
    oem_labels = set(oem_class_map["id2name"].values())
    assert oem.status == "active"
    assert oem.verification.class_map == "verified"
    for leaf in evidence.leaf_categories:
        if evidence.capability_enabled(leaf, "yolo"):
            assert set(evidence.leaf_yolo_labels(leaf)) <= set(yolo.classes)
        if evidence.capability_enabled(leaf, "segformer"):
            binding = evidence.leaf_segformer_binding(leaf)
            expected = (
                oem_labels if binding == "segmenter_oem_001" else semantic_labels
            )
            assert set(evidence.leaf_segformer_labels(leaf) or ()) <= expected


def test_catalog_asset_has_no_physical_paths_or_unverified_classes() -> None:
    text = _PRODUCTION_CATALOG.read_text(encoding="utf-8")
    for token in ("/Users", "C:\\", "checkpoints", ".onnx", "LABEL_", "background"):
        assert token not in text


def test_from_file_missing_asset_fails_stable(tmp_path: Path) -> None:
    with pytest.raises(CatalogCategoryError, match="CATALOG_LOAD_FAILED"):
        EvidenceCatalog.from_file(tmp_path / "missing.json")
