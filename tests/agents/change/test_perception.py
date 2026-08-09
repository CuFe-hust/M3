from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import agents.change.perception as perception_module
from agents.change.perception import (
    PERCEPTION_VERSION,
    ChangePerceptionError,
    ChangePerceptionPipeline,
)
from agents.change.preprocess import ChangePreparedPair
from agents.change.schema import HarmonizationDecision, PairValidationReport
from agents.change.settings import (
    AgentChangeSettings,
    ChangeHarmonizationSettings,
    ChangeProposalSettings,
    ChangeSemanticSettings,
)
from agents.errors import OptionalDependencyMissingError
from models.base import DenseSemanticOutput, ModelAssetMissingError, ModelCacheIdentity


def _prepared() -> ChangePreparedPair:
    first = np.full((64, 64, 3), 80, dtype=np.uint8)
    second = first.copy()
    pif = np.ones((64, 64), dtype=np.uint8)
    pif[16:32, 16:32] = 0
    return ChangePreparedPair(
        raw_t1=first.copy(),
        raw_t2=second.copy(),
        comparison_t1=first,
        comparison_t2=second,
        pif_mask=pif,
        validation=PairValidationReport(
            valid=True,
            temporal_roles_valid=True,
            same_size=True,
            alignment_status="assumed_dataset_aligned",
        ),
        decision=HarmonizationDecision(
            version="pif_lab_midpoint_v1",
            status="applied",
            reason_codes=["PIF_MATCHED"],
            metrics=None,
            used_for_proposal=True,
        ),
        transform_summary={},
    )


def _settings(*, policy: str = "fallback_legacy", enabled: bool = True) -> AgentChangeSettings:
    return AgentChangeSettings(
        harmonization=ChangeHarmonizationSettings(min_pif_pixels=32),
        semantic=ChangeSemanticSettings(
            enabled=enabled,
            local_match_radius=0,
            min_pif_feature_cells=16,
            failure_policy=policy,
        ),
        proposals=ChangeProposalSettings(
            min_component_area_ratio=0.001,
            max_component_area_ratio=0.50,
            mask_close_kernel=1,
        ),
    )


def _outputs() -> tuple[DenseSemanticOutput, DenseSemanticOutput]:
    rng = np.random.default_rng(73)
    first_features = rng.normal(size=(8, 16, 16)).astype(np.float32)
    second_features = first_features.copy()
    second_features[:, 4:8, 4:8] *= np.float32(-1.0)
    first_probabilities = np.empty((2, 16, 16), dtype=np.float32)
    first_probabilities[0] = 0.95
    first_probabilities[1] = 0.05
    second_probabilities = first_probabilities.copy()
    second_probabilities[0, 4:8, 4:8] = 0.05
    second_probabilities[1, 4:8, 4:8] = 0.95
    common = {
        "semantic_stride": (4.0, 4.0),
        "feature_stride": (4.0, 4.0),
        "original_size": (64, 64),
        "class_names": ("unchanged", "changed"),
        "diagnostics": {},
    }
    return (
        DenseSemanticOutput(
            probabilities=first_probabilities,
            features=first_features,
            **common,
        ),
        DenseSemanticOutput(
            probabilities=second_probabilities,
            features=second_features,
            **common,
        ),
    )


class _DenseClient:
    def __init__(self, outputs: tuple[DenseSemanticOutput, DenseSemanticOutput] | None = None) -> None:
        self.outputs = outputs or _outputs()
        self.calls: list[dict[str, Any]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="segformer-logical-test",
            generation={"backend": "fake"},
            client_version="test-v1",
        )

    def infer(self, image: Any, **kwargs: Any) -> DenseSemanticOutput:
        self.calls.append({"image": image, **kwargs})
        return self.outputs[len(self.calls) - 1]


class _RaisingDenseClient(_DenseClient):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def infer(self, image: Any, **kwargs: Any) -> DenseSemanticOutput:
        self.calls.append({"image": image, **kwargs})
        raise self.error


class SegFormerInferenceError(RuntimeError):
    """Concrete-backend-shaped fake without importing the concrete backend."""


def test_semantic_disabled_uses_legacy_without_dense_calls() -> None:
    client = _DenseClient()

    result = ChangePerceptionPipeline(client, _settings(enabled=False)).run(_prepared())

    assert client.calls == []
    assert result.diagnostics["semantic_status"] == "disabled"
    assert result.diagnostics["proposal_source"] == "difference_map_v1"
    assert result.component_maps is None


def test_enabled_pipeline_calls_two_frames_and_returns_v2_proposals() -> None:
    client = _DenseClient()

    result = ChangePerceptionPipeline(client, _settings()).run(_prepared())

    assert len(client.calls) == 2
    assert all(call["tile_size"] == 768 for call in client.calls)
    assert result.proposals
    assert all(proposal.source == "fused_change_v2" for proposal in result.proposals)
    assert result.diagnostics["semantic_status"] == "success"
    assert result.diagnostics["segformer_model"] == "segformer-logical-test"
    assert result.diagnostics["perception_version"] == PERCEPTION_VERSION
    assert set(result.component_maps or {}) == {
        "low_level_difference_map",
        "feature_residual_map",
        "semantic_difference_map",
        "binary_change_mask",
    }
    assert result.diagnostics["perception_mode"] == "fused_v2"
    assert result.diagnostics["semantic_model"] == "segformer-logical-test"
    assert result.diagnostics["semantic_client_version"] == "test-v1"
    assert set(result.diagnostics["score_maps"]) == {
        "low_level",
        "feature",
        "semantic",
        "fused",
    }
    assert result.component_masks


def test_client_missing_falls_back_to_legacy_with_stable_reason() -> None:
    result = ChangePerceptionPipeline(None, _settings()).run(_prepared())

    assert result.diagnostics["semantic_status"] == "fallback"
    assert result.diagnostics["semantic_reason_code"] == "SEGFORMER_CLIENT_MISSING"
    assert result.diagnostics["proposal_source"] == "difference_map_v1"


def test_client_missing_fail_policy_raises() -> None:
    with pytest.raises(ChangePerceptionError, match="SEGFORMER_CLIENT_MISSING"):
        ChangePerceptionPipeline(None, _settings(policy="fail")).run(_prepared())


def test_pair_grid_mismatch_falls_back_without_hiding_reason() -> None:
    first, second = _outputs()
    mismatched = DenseSemanticOutput(
        probabilities=second.probabilities,
        features=second.features,
        semantic_stride=second.semantic_stride,
        feature_stride=second.feature_stride,
        original_size=(63, 64),
        class_names=second.class_names,
        diagnostics={},
    )
    client = _DenseClient((first, mismatched))

    result = ChangePerceptionPipeline(client, _settings()).run(_prepared())

    assert len(client.calls) == 2
    assert result.diagnostics["semantic_status"] == "fallback"
    assert result.diagnostics["semantic_reason_code"] == "SEGFORMER_PAIR_GRID_MISMATCH"


def test_pair_grid_mismatch_fail_policy_raises() -> None:
    first, second = _outputs()
    mismatched = DenseSemanticOutput(
        probabilities=second.probabilities[:, :, :-1],
        features=second.features,
        semantic_stride=(64 / 15, 4.0),
        feature_stride=second.feature_stride,
        original_size=second.original_size,
        class_names=second.class_names,
        diagnostics={},
    )

    with pytest.raises(ChangePerceptionError, match="SEGFORMER_PAIR_GRID_MISMATCH"):
        ChangePerceptionPipeline(
            _DenseClient((first, mismatched)),
            _settings(policy="fail"),
        ).run(_prepared())


def test_model_identity_drift_is_rejected() -> None:
    class _DriftingIdentityClient(_DenseClient):
        def __init__(self) -> None:
            super().__init__()
            self.identity_reads = 0

        @property
        def cache_identity(self) -> ModelCacheIdentity:
            self.identity_reads += 1
            return ModelCacheIdentity(
                model=f"segformer-logical-{self.identity_reads}",
                generation={"backend": "fake"},
                client_version="test-v1",
            )

    result = ChangePerceptionPipeline(_DriftingIdentityClient(), _settings()).run(
        _prepared()
    )

    assert result.diagnostics["semantic_status"] == "fallback"
    assert (
        result.diagnostics["semantic_reason_code"]
        == "SEGFORMER_MODEL_IDENTITY_MISMATCH"
    )


def test_invalid_prepared_pair_fails_before_dense_call() -> None:
    prepared = _prepared()
    invalid = ChangePreparedPair(
        **{
            **prepared.__dict__,
            "raw_t1": None,
            "validation": prepared.validation.model_copy(update={"valid": False}),
        }
    )
    client = _DenseClient()

    with pytest.raises(ChangePerceptionError, match="INVALID_CHANGE_PAIR"):
        ChangePerceptionPipeline(client, _settings()).run(invalid)
    assert client.calls == []


def test_unlisted_programming_error_is_not_swallowed() -> None:
    class _BuggyClient(_DenseClient):
        def infer(self, image: Any, **kwargs: Any) -> DenseSemanticOutput:
            raise RuntimeError("unexpected implementation bug")

    with pytest.raises(RuntimeError, match="unexpected implementation bug"):
        ChangePerceptionPipeline(_BuggyClient(), _settings()).run(_prepared())


def _failure_matrix_case(
    scenario: str,
) -> tuple[_DenseClient, ChangePreparedPair]:
    prepared = _prepared()
    if scenario == "torch_missing":
        return (
            _RaisingDenseClient(
                OptionalDependencyMissingError("change-semantic", dependency="torch")
            ),
            prepared,
        )
    if scenario == "checkpoint_missing":
        return (
            _RaisingDenseClient(ModelAssetMissingError("checkpoint missing")),
            prepared,
        )
    if scenario == "hidden_state_invalid":
        return (
            _RaisingDenseClient(
                SegFormerInferenceError("SEGFORMER_FEATURE_GRID_UNRESOLVED")
            ),
            prepared,
        )
    if scenario == "insufficient_pif":
        return _DenseClient(), replace(
            prepared,
            pif_mask=np.zeros_like(prepared.pif_mask),
        )
    first, second = _outputs()
    if scenario == "nan_feature":
        features = np.asarray(second.features).copy()
        features[0, 0, 0] = np.nan
        invalid = replace(second, features=features)
        return _DenseClient((first, invalid)), prepared
    if scenario == "grid_mismatch":
        invalid = replace(second, original_size=(63, 64))
        return _DenseClient((first, invalid)), prepared
    raise AssertionError(scenario)


@pytest.mark.parametrize(
    ("scenario", "reason_code"),
    [
        ("torch_missing", "SEGFORMER_DEPENDENCY_MISSING"),
        ("checkpoint_missing", "SEGFORMER_CHECKPOINT_MISSING"),
        ("hidden_state_invalid", "SEGFORMER_FEATURE_GRID_UNRESOLVED"),
        ("insufficient_pif", "FEATURE_RESIDUAL_INSUFFICIENT_PIF"),
        ("nan_feature", "FEATURE_RESIDUAL_NONFINITE"),
        ("grid_mismatch", "SEGFORMER_PAIR_GRID_MISMATCH"),
    ],
)
@pytest.mark.parametrize("policy", ["fallback_legacy", "fail"])
def test_semantic_failure_matrix_has_explicit_policy_behavior(
    scenario: str,
    reason_code: str,
    policy: str,
) -> None:
    client, prepared = _failure_matrix_case(scenario)
    pipeline = ChangePerceptionPipeline(client, _settings(policy=policy))

    if policy == "fail":
        with pytest.raises(ChangePerceptionError, match=reason_code):
            pipeline.run(prepared)
        return

    result = pipeline.run(prepared)
    assert result.diagnostics["perception_mode"] == "fallback_legacy"
    assert result.diagnostics["semantic_reason_code"] == reason_code
    assert result.diagnostics["proposal_source"] == "difference_map_v1"
    assert result.component_maps is None


def test_perception_imports_only_abstract_model_contract() -> None:
    source_path = Path(perception_module.__file__ or "")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    model_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("models")
    }

    assert model_imports == {"models.base"}
    assert "segformer_transformers" not in source
