"""Contract tests for the versioned evidence catalog.

版本化证据目录契约测试：版本固定、组合展开顺序与去重、能力判断、逻辑身份、
安全校验、严格失败与生产资产一致性。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.counting.expert_catalog import ExpertCatalog
from agents.evidence_catalog import (
    CatalogCategoryError,
    EvidenceCatalog,
)
from application.settings import load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_CATALOG = REPO_ROOT / "agents" / "evidence_catalog.json"
_COUNTING_EXPERT_CATALOG = (
    REPO_ROOT / "agents" / "counting" / "expert_catalog.json"
)


def _catalog(**overrides) -> EvidenceCatalog:
    data = {
        "catalog_version": "catalog-test-v1",
        "composites": {
            "vehicle": ["small_vehicle", "large_vehicle"],
            "aircraft": ["plane"],
        },
        "leaves": {
            "small_vehicle": {
                "yolo_labels": ["small vehicle"],
                "segformer_labels": ["SmallVehicle"],
                "yolo_enabled": True,
                "segformer_enabled": True,
            },
            "large_vehicle": {
                "yolo_labels": ["large vehicle"],
                "segformer_labels": None,
                "yolo_enabled": True,
                "segformer_enabled": False,
            },
            "plane": {
                "yolo_labels": ["airplane"],
                "segformer_labels": None,
                "yolo_enabled": True,
                "segformer_enabled": False,
            },
        },
    }
    data.update(overrides)
    return EvidenceCatalog(data)


# ── 版本与结构 / version and structure ───────────────────────────────────


def test_catalog_exposes_version_and_stable_order() -> None:
    catalog = _catalog()
    assert catalog.catalog_version == "catalog-test-v1"
    assert catalog.composite_categories == ("vehicle", "aircraft")
    assert catalog.leaf_categories == ("small_vehicle", "large_vehicle", "plane")
    assert catalog.is_composite("vehicle") is True
    assert catalog.is_leaf("plane") is True
    assert catalog.is_composite("plane") is False


def test_catalog_rejects_bad_versions() -> None:
    for version in ("", "  ", "has space", "UPPER", "has/control\x00"):
        with pytest.raises(CatalogCategoryError, match="INVALID_CATALOG_VERSION"):
            _catalog(catalog_version=version)


def test_catalog_rejects_extra_top_level_keys() -> None:
    with pytest.raises(CatalogCategoryError, match="INVALID_TOP_LEVEL_KEYS"):
        EvidenceCatalog(
            {
                "catalog_version": "v1",
                "composites": {},
                "leaves": {},
                "unexpected": True,
            }
        )


def test_catalog_rejects_invalid_category_names() -> None:
    with pytest.raises(CatalogCategoryError, match="INVALID_CATEGORY_NAME"):
        _catalog(leaves={"Bad Name": {"yolo_labels": ["x"]}})
    with pytest.raises(CatalogCategoryError, match="INVALID_CATEGORY_NAME"):
        _catalog(leaves={"has-dash": {"yolo_labels": ["x"]}})


def test_catalog_rejects_composite_targeting_unknown_leaf() -> None:
    with pytest.raises(CatalogCategoryError, match="COMPOSITE_TARGETS_UNKNOWN_LEAF"):
        _catalog(composites={"vehicle": ["missing_leaf"]})


def test_catalog_rejects_composite_name_colliding_with_leaf() -> None:
    with pytest.raises(CatalogCategoryError, match="COMPOSITE_NAME_COLLIDES_WITH_LEAF"):
        _catalog(composites={"plane": ["small_vehicle"]})


def test_catalog_rejects_duplicated_composite_targets() -> None:
    with pytest.raises(CatalogCategoryError, match="COMPOSITE_TARGETS_DUPLICATED"):
        _catalog(composites={"vehicle": ["small_vehicle", "small_vehicle"]})


# ── 展开 / expansion ─────────────────────────────────────────────────────


def test_expand_composites_preserves_order_and_dedupes() -> None:
    catalog = _catalog()
    assert catalog.composite_leaves("vehicle") == ("small_vehicle", "large_vehicle")
    assert catalog.expand_composites(["vehicle", "aircraft"]) == (
        "small_vehicle",
        "large_vehicle",
        "plane",
    )
    assert catalog.expand_composites(["vehicle", "vehicle"]) == (
        "small_vehicle",
        "large_vehicle",
    )
    assert catalog.expand_composites([]) == ()


def test_expand_unknown_or_leaf_category_fails_strictly() -> None:
    catalog = _catalog()
    with pytest.raises(CatalogCategoryError, match="UNKNOWN_COMPOSITE"):
        catalog.expand_composites(["tank"])
    with pytest.raises(CatalogCategoryError, match="UNKNOWN_COMPOSITE"):
        catalog.expand_composites(["plane"])  # a leaf is not a composite / 叶子不是组合
    with pytest.raises(CatalogCategoryError, match="UNKNOWN_COMPOSITE"):
        catalog.validate_plan_categories(["vehicle", "tank"])


def test_validate_plan_categories_accepts_known_composites() -> None:
    catalog = _catalog()
    catalog.validate_plan_categories(["vehicle"])
    catalog.validate_plan_categories(["aircraft", "vehicle"])


# ── 能力与身份 / capabilities and identity ──────────────────────────────


def test_capabilities_expose_labels_and_enabled_flags() -> None:
    catalog = _catalog()
    assert catalog.leaf_yolo_labels("small_vehicle") == ("small vehicle",)
    assert catalog.leaf_segformer_labels("small_vehicle") == ("SmallVehicle",)
    assert catalog.leaf_segformer_labels("large_vehicle") is None
    assert catalog.capability_enabled("small_vehicle", "yolo") is True
    assert catalog.capability_enabled("large_vehicle", "segformer") is False
    assert catalog.capability_enabled("plane", "yolo") is True


def test_capability_identity_is_logical_and_versioned() -> None:
    catalog = _catalog()
    assert catalog.capability_identity("small_vehicle", "yolo") == (
        "catalog-test-v1:small_vehicle:yolo"
    )
    assert catalog.capability_identity("small_vehicle", "segformer") == (
        "catalog-test-v1:small_vehicle:segformer"
    )


def test_unknown_leaf_capability_fails_stable() -> None:
    catalog = _catalog()
    for call in (
        lambda: catalog.leaf_yolo_labels("tank"),
        lambda: catalog.capability_enabled("tank", "yolo"),
        lambda: catalog.capability_identity("tank", "segformer"),
    ):
        with pytest.raises(CatalogCategoryError, match="UNKNOWN_LEAF"):
            call()


def test_enabled_capability_requires_verified_labels() -> None:
    with pytest.raises(CatalogCategoryError, match="INVALID_LEAF_CAPABILITIES"):
        _catalog(
            leaves={
                "tank": {"yolo_labels": [], "yolo_enabled": True},
            }
        )
    with pytest.raises(CatalogCategoryError, match="INVALID_LEAF_CAPABILITIES"):
        _catalog(
            leaves={
                "tank": {
                    "yolo_labels": ["tank"],
                    "segformer_labels": [],
                    "segformer_enabled": True,
                },
            }
        )


def test_uncalibrated_capability_stays_disabled() -> None:
    catalog = EvidenceCatalog(
        {
            "catalog_version": "seed-v1",
            "composites": {"vehicle": ["small_vehicle"]},
            "leaves": {
                "small_vehicle": {
                    "yolo_labels": [],
                    "segformer_labels": None,
                    "yolo_enabled": False,
                    "segformer_enabled": False,
                }
            },
        }
    )
    assert catalog.capability_enabled("small_vehicle", "yolo") is False
    assert catalog.capability_enabled("small_vehicle", "segformer") is False
    assert catalog.leaf_yolo_labels("small_vehicle") == ()


# ── 安全 / safety ────────────────────────────────────────────────────────


def test_catalog_rejects_sensitive_keys() -> None:
    with pytest.raises(CatalogCategoryError, match="SENSITIVE_KEY_IN_CATALOG"):
        EvidenceCatalog(
            {
                "catalog_version": "v1",
                "composites": {},
                "leaves": {"x": {"yolo_labels": [], "api_key": "sk-123"}},
            }
        )


def test_catalog_rejects_path_like_labels() -> None:
    with pytest.raises(CatalogCategoryError, match="INVALID_LEAF_CAPABILITIES"):
        _catalog(
            leaves={
                "tank": {"yolo_labels": ["/home/user/models/class.txt"]},
            }
        )
    with pytest.raises(CatalogCategoryError, match="INVALID_LEAF_CAPABILITIES"):
        _catalog(
            leaves={
                "tank": {"yolo_labels": ["C:\\models\\class"]},
            }
        )
    with pytest.raises(CatalogCategoryError, match="INVALID_LEAF_CAPABILITIES"):
        _catalog(
            leaves={
                "tank": {"yolo_labels": ["bad\x00label"]},
            }
        )


# ── 生产资产 / production asset ──────────────────────────────────────────


def test_production_catalog_loads_and_is_consistent() -> None:
    catalog = EvidenceCatalog.from_file(_PRODUCTION_CATALOG)
    assert catalog.catalog_version == "first-qwen-evidence-catalog-v2"
    assert catalog.composite_categories == (
        "vehicle",
        "aircraft",
        "watercraft",
        "sports_facility",
        "transport_infrastructure",
        "industrial_facility",
        "aviation_infrastructure",
    )
    assert catalog.composite_leaves("vehicle") == ("small_vehicle", "large_vehicle")
    assert catalog.composite_leaves("aircraft") == ("plane", "helicopter")
    assert catalog.composite_leaves("watercraft") == ("ship",)
    assert catalog.leaf_yolo_labels("small_vehicle") == ("small vehicle",)
    assert catalog.leaf_segformer_labels("small_vehicle") == ("Small_Vehicle",)
    assert catalog.capability_enabled("small_vehicle", "yolo") is True
    assert catalog.capability_enabled("small_vehicle", "segformer") is True
    assert catalog.capability_enabled("container_crane", "yolo") is True
    assert catalog.capability_enabled("container_crane", "segformer") is False


def test_production_catalog_labels_are_backed_by_current_model_maps() -> None:
    evidence = EvidenceCatalog.from_file(_PRODUCTION_CATALOG)
    experts = ExpertCatalog.load(_COUNTING_EXPERT_CATALOG, asset_root=REPO_ROOT)
    settings = load_settings(REPO_ROOT / "configs" / "local.yaml", environ={})
    (yolo,) = settings.backend.yolo.detectors
    segformer = experts.expert("segmenter_mitb2_001")
    yolo_labels = set(yolo.classes)
    segformer_labels = {
        label
        for support in segformer.supports.values()
        for label in support.model_labels
    }

    for leaf in evidence.leaf_categories:
        if evidence.capability_enabled(leaf, "yolo"):
            assert set(evidence.leaf_yolo_labels(leaf)) <= yolo_labels
        if evidence.capability_enabled(leaf, "segformer"):
            assert set(evidence.leaf_segformer_labels(leaf) or ()) <= segformer_labels


def test_catalog_asset_has_no_physical_paths() -> None:
    text = _PRODUCTION_CATALOG.read_text(encoding="utf-8")
    for token in ("/Users", "C:\\", "checkpoints", ".pt", ".onnx", "sk-", "data:image"):
        assert token not in text


def test_from_file_missing_asset_fails_stable(tmp_path: Path) -> None:
    with pytest.raises(CatalogCategoryError, match="CATALOG_LOAD_FAILED"):
        EvidenceCatalog.from_file(tmp_path / "missing.json")
