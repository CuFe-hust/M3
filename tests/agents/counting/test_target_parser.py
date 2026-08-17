"""Contract tests for deterministic count-target reconciliation."""

import re
from pathlib import Path

import pytest

from agents.counting.target_parser import CountTargetResolutionError, CountTargetResolver
from agents.evidence_catalog import EvidenceCatalog

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def resolver() -> CountTargetResolver:
    catalog = EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")
    return CountTargetResolver(evidence_catalog=catalog)


def _resolve(resolver: CountTargetResolver, **overrides):
    values = {
        "question": "How many vehicles are visible?",
        "planner_target": "vehicle",
        "planner_object_categories": ("small-vehicle", "large-vehicle"),
        "count_target_hint": "vehicle",
        "legacy_metadata": None,
    }
    values.update(overrides)
    return resolver.resolve(**values)


def test_scenario_a_narrows_approved_vehicle_scope(resolver) -> None:
    result = _resolve(
        resolver, question="How many small vehicals are visible?",
        count_target_hint="vehicle",
    )
    assert result.target.canonical_label == "small-vehicle"
    assert result.executable_leaf_categories == ("small-vehicle",)
    assert result.validation_status == "planner_scope_broadened_corrected"


def test_specificity_guard_handles_only_approved_vehicle_variants(resolver) -> None:
    large = _resolve(
        resolver, question="How many large vehicals are visible?",
        count_target_hint="vehicle",
    )
    assert large.target.canonical_label == "large-vehicle"
    assert large.executable_leaf_categories == ("large-vehicle",)

    unknown_typo = _resolve(
        resolver, question="How many tiny vehicals are visible?",
        count_target_hint="vehicle",
    )
    assert unknown_typo.target.canonical_label == "vehicle"
    assert unknown_typo.executable_leaf_categories == (
        "small-vehicle", "large-vehicle"
    )


def test_specificity_guard_never_widens_a_leaf_verifier(resolver) -> None:
    result = _resolve(
        resolver, question="How many vehicles are visible?",
        planner_target="small-vehicle",
        planner_object_categories=("small-vehicle",),
        count_target_hint="small-vehicle",
    )
    assert result.target.canonical_label == "small-vehicle"
    assert result.executable_leaf_categories == ("small-vehicle",)
    assert result.validation_status == "matched"


def test_scenario_b_repairs_incomplete_parent_expansion(resolver) -> None:
    result = _resolve(resolver, planner_object_categories=("small-vehicle",))
    assert result.executable_leaf_categories == ("small-vehicle", "large-vehicle")
    assert result.validation_status == "incomplete_parent_expansion_corrected"


def test_exact_leaf_and_parent_matches_are_distinct(resolver) -> None:
    leaf = _resolve(
        resolver, planner_target="small-vehicle",
        planner_object_categories=("small-vehicle",),
        count_target_hint="small_vehicle",
    )
    assert leaf.validation_status == "matched"
    assert _resolve(resolver).validation_status == "matched_parent_expansion"


def test_planner_narrower_than_verifier_expands_scope(resolver) -> None:
    result = _resolve(
        resolver, planner_target="small-vehicle",
        planner_object_categories=("small-vehicle",), count_target_hint="vehicle",
    )
    assert result.target.canonical_label == "vehicle"
    assert result.executable_leaf_categories == ("small-vehicle", "large-vehicle")
    assert result.validation_status == "planner_scope_narrowed_corrected"


def test_scenario_c_unrelated_targets_fail_closed(resolver) -> None:
    with pytest.raises(CountTargetResolutionError) as caught:
        _resolve(
            resolver, planner_target="ship", planner_object_categories=("ship",),
            count_target_hint="small-vehicle",
        )
    assert caught.value.code == str(caught.value) == "COUNT_TARGET_CONFLICT"


def test_scenario_d_direct_structured_hint_preserves_reviewed_fields(resolver) -> None:
    result = _resolve(
        resolver, question="", planner_target=None, planner_object_categories=(),
        count_target_hint={
            "canonical_label": "airplane", "aliases": ["aeroplane"],
            "required_attributes": ["visible fuselage"],
            "excluded_attributes": ["shadow"],
            "spatial_constraints": ["inside image"],
            "inclusion_rule": "Count visible airplanes.",
            "exclusion_rule": "Exclude shadows.", "ambiguity": ["partial wing"],
        },
    )
    assert result.target_source == "explicit_hint"
    assert result.target.canonical_label == "plane"
    assert result.target.required_attributes == ["visible fuselage"]
    assert result.executable_leaf_categories == ("plane",)


def test_scenario_e_requires_a_source(resolver) -> None:
    with pytest.raises(CountTargetResolutionError) as caught:
        _resolve(
            resolver, planner_target=None, planner_object_categories=(),
            count_target_hint=None,
        )
    assert caught.value.code == "COUNT_TARGET_SOURCE_REQUIRED"


def test_invalid_normalization_hint_never_falls_back(resolver) -> None:
    with pytest.raises(CountTargetResolutionError) as caught:
        _resolve(
            resolver, count_target_hint={"canonical_label": 42},
            legacy_metadata={"count_target_hint": {
                "canonical_label": "vehicle", "inclusion_rule": "Count vehicles.",
                "exclusion_rule": "Exclude shadows.",
            }},
        )
    assert caught.value.code == "COUNT_TARGET_VERIFIER_INVALID"


def test_legacy_hint_is_used_only_when_normalization_hint_is_absent(resolver) -> None:
    result = _resolve(
        resolver, planner_target="ship", planner_object_categories=("ship",),
        count_target_hint=None, legacy_metadata={"count_target_hint": "ships"},
    )
    assert result.target.canonical_label == "ship"
    assert result.verifier_source == "legacy_metadata.count_target_hint"


def test_unknown_planner_target_without_verifier_is_preserved(resolver) -> None:
    result = _resolve(
        resolver, planner_target="building", planner_object_categories=(),
        count_target_hint=None,
    )
    assert result.target.canonical_label == "building"
    assert result.executable_leaf_categories == ()
    assert result.validation_status == "planner_only_no_visual_expert"


def test_noncanonical_or_unrelated_planner_leaves_fail(resolver) -> None:
    for leaves in (("small_vehicle",), ("ship",)):
        with pytest.raises(CountTargetResolutionError) as caught:
            _resolve(resolver, planner_object_categories=leaves)
        assert caught.value.code == "COUNT_TARGET_PLANNER_LEAVES_INVALID"


def test_resolver_has_zero_model_or_async_dependencies() -> None:
    source = (REPO_ROOT / "agents" / "counting" / "target_parser.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "VisionLanguageClient", "CallBudget", "RequestMeta", "build_request_hash",
        "require_model_cache_identity", "_parse_via_qwen", "target_parse",
        "async def", "TargetParser =",
    )
    assert all(token not in source for token in forbidden)


def test_final_small_vehicle_typo_acceptance_payload(resolver) -> None:
    result = _resolve(
        resolver,
        question="How many small vehicals are visible?",
        planner_target="vehicle",
        planner_object_categories=("small-vehicle", "large-vehicle"),
        count_target_hint="vehicle",
    )
    acceptance = {
        "target": result.target.canonical_label,
        "planner_target": result.planner_target,
        "planner_object_categories": list(result.planner_object_categories),
        "executable_leaf_categories": list(result.executable_leaf_categories),
        "target_validation": result.validation_status,
        "target_qwen_calls": 0,
    }
    assert acceptance == {
        "target": "small-vehicle",
        "planner_target": "vehicle",
        "planner_object_categories": ["small-vehicle", "large-vehicle"],
        "executable_leaf_categories": ["small-vehicle"],
        "target_validation": "planner_scope_broadened_corrected",
        "target_qwen_calls": 0,
    }


def test_active_production_has_no_target_qwen_wiring() -> None:
    forbidden_patterns = (
        "target_parse_v1", "_parse_via_qwen", "target_prompt",
        "target_prompt_version", "artifact_dir.+target_parse",
        "request_id=.*:target",
    )
    production_roots = (
        REPO_ROOT / "agents", REPO_ROOT / "application",
        REPO_ROOT / "workflows", REPO_ROOT / "evaluation",
    )
    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in production_roots
        for path in root.rglob("*.py")
    )
    assert all(
        re.search(pattern, production_text) is None
        for pattern in forbidden_patterns
    )
