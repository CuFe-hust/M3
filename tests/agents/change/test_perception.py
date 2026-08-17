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
    ChangeLearnedChangeSettings,
    ChangeSemanticSettings,
)
from agents.errors import OptionalDependencyMissingError
from models.base import (
    DenseSemanticOutput,
    DenseSemanticPyramidOutput,
    LearnedChangeOutput,
    ModelAssetMissingError,
    ModelCacheIdentity,
)


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
        pif_valid=True,
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
        "weights_sha256": "a" * 64,
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


class _PyramidClient(_DenseClient):
    def infer_pyramid(self, image: Any, **kwargs: Any) -> DenseSemanticPyramidOutput:
        self.calls.append({"image": image, **kwargs})
        output = self.outputs[len(self.calls) - 1]
        return DenseSemanticPyramidOutput(
            probabilities=output.probabilities,
            features_by_stage={1: output.features, 2: output.features.copy()},
            semantic_stride=output.semantic_stride,
            feature_strides_by_stage={1: output.feature_stride, 2: output.feature_stride},
            original_size=output.original_size,
            class_names=output.class_names,
            diagnostics={"pyramid": True},
            weights_sha256=output.weights_sha256,
        )


class _LearnedClient:
    def __init__(self, score_map: np.ndarray | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.score_map = score_map if score_map is not None else np.full(
            (16, 16), 0.2, dtype=np.float32
        )

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="learned-change-head-test",
            generation={"backend": "fake"},
            client_version="head-v1",
            revision="adapter-0",
        )

    def infer(self, *, first, second, valid_mask) -> LearnedChangeOutput:
        self.calls.append(
            {"first": first, "second": second, "valid_mask": valid_mask}
        )
        return LearnedChangeOutput(
            score_map=self.score_map,
            diagnostics={"head": "frozen-feature-test"},
        )


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
    assert result.diagnostics["semantic_weights_sha256"] == "a" * 64
    assert result.diagnostics["pif_valid"] is True
    assert result.diagnostics["pif_used_for_feature_alignment"] is True
    assert result.diagnostics["pif_used_for_threshold"] is True
    assert set(result.diagnostics["score_maps"]) == {
        "low_level",
        "feature",
        "semantic",
        "fused",
    }


def test_multiscale_pipeline_uses_pyramid_contract_and_records_stages() -> None:
    settings = _settings()
    settings.semantic.feature_stages = (1, 2)
    settings.semantic.feature_stage_weights = {1: 0.6, 2: 0.4}
    client = _PyramidClient()

    result = ChangePerceptionPipeline(client, settings).run(_prepared())

    assert len(client.calls) == 2
    assert all(call["feature_stages"] == (1, 2) for call in client.calls)
    assert result.diagnostics["multiscale_enabled"] is True
    assert result.diagnostics["feature_stages"] == [1, 2]
    assert result.diagnostics["feature_residual"]["effective_stages"] == [1, 2]
    assert result.component_masks


def test_client_missing_falls_back_to_legacy_with_stable_reason() -> None:
    result = ChangePerceptionPipeline(None, _settings()).run(_prepared())

    assert result.diagnostics["semantic_status"] == "fallback"
    assert result.diagnostics["semantic_reason_code"] == "SEGFORMER_CLIENT_MISSING"
    assert result.diagnostics["proposal_source"] == "difference_map_v1"
    assert result.diagnostics["pif_used_for_feature_alignment"] is False
    assert result.diagnostics["pif_used_for_threshold"] is False


def test_learned_change_is_disabled_without_a_client() -> None:
    client = _DenseClient()

    result = ChangePerceptionPipeline(client, _settings()).run(_prepared())

    assert result.diagnostics["learned_change"]["status"] == "disabled"
    assert result.diagnostics["learned_change"]["available"] is False
    assert "learned" not in result.diagnostics.get("fusion", {}).get(
        "available_components", []
    )


def test_enabled_learned_change_without_client_falls_back_to_rule_branch() -> None:
    settings = _settings()
    settings.learned_change = ChangeLearnedChangeSettings(enabled=True, fusion_weight=0.2)

    result = ChangePerceptionPipeline(_DenseClient(), settings).run(_prepared())

    assert result.diagnostics["semantic_status"] == "success"
    assert result.diagnostics["learned_change"] == {
        "enabled": True,
        "status": "fallback",
        "available": False,
        "reason_codes": ["LEARNED_CHANGE_CLIENT_MISSING"],
        "fusion_weight": 0.2,
    }
    assert "learned" in result.diagnostics["fusion"]["missing_components"]


def test_enabled_learned_change_client_is_injected_without_training_runtime() -> None:
    settings = _settings()
    settings.learned_change = ChangeLearnedChangeSettings(enabled=True, fusion_weight=0.2)
    learned = _LearnedClient()

    result = ChangePerceptionPipeline(
        _DenseClient(), settings, learned_change_client=learned
    ).run(_prepared())

    assert len(learned.calls) == 1
    assert isinstance(learned.calls[0]["first"], DenseSemanticPyramidOutput)
    assert learned.calls[0]["first"].features_by_stage[1].shape == (8, 16, 16)
    assert result.diagnostics["learned_change"]["status"] == "available"
    assert result.diagnostics["learned_change"]["model"] == "learned-change-head-test"
    assert "learned" in result.diagnostics["fusion"]["available_components"]


def test_enabled_learned_change_fail_policy_is_strict() -> None:
    settings = _settings()
    settings.learned_change = ChangeLearnedChangeSettings(
        enabled=True,
        fusion_weight=0.2,
        failure_policy="fail",
    )

    with pytest.raises(ChangePerceptionError, match="LEARNED_CHANGE_CLIENT_MISSING"):
        ChangePerceptionPipeline(_DenseClient(), settings).run(_prepared())


def test_invalid_pif_falls_back_before_dense_inference() -> None:
    client = _DenseClient()
    prepared = replace(_prepared(), pif_valid=False)

    result = ChangePerceptionPipeline(client, _settings()).run(prepared)

    assert client.calls == []
    assert result.diagnostics["semantic_status"] == "fallback"
    assert (
        result.diagnostics["semantic_reason_code"]
        == "FEATURE_RESIDUAL_INSUFFICIENT_PIF"
    )
    assert result.diagnostics["proposal_source"] == "difference_map_v1"
    assert result.diagnostics["pif_valid"] is False


def test_invalid_pif_fail_policy_raises_before_dense_inference() -> None:
    client = _DenseClient()
    prepared = replace(_prepared(), pif_valid=False)

    with pytest.raises(
        ChangePerceptionError,
        match="FEATURE_RESIDUAL_INSUFFICIENT_PIF",
    ):
        ChangePerceptionPipeline(client, _settings(policy="fail")).run(prepared)
    assert client.calls == []


def test_rejected_transform_with_valid_pif_still_runs_v2() -> None:
    prepared = replace(
        _prepared(),
        comparison_t1=_prepared().raw_t1.copy(),
        comparison_t2=_prepared().raw_t2.copy(),
        decision=HarmonizationDecision(
            version="pif_lab_midpoint_v1",
            status="rejected",
            reason_codes=["REJECTED_UNSTABLE_TRANSFORM", "RAW_FALLBACK_USED"],
            metrics=None,
            used_for_proposal=False,
        ),
        pif_valid=True,
    )
    client = _DenseClient()

    result = ChangePerceptionPipeline(client, _settings()).run(prepared)

    assert len(client.calls) == 2
    assert result.diagnostics["semantic_status"] == "success"
    assert result.diagnostics["proposal_source"] == "fused_change_v2"


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


def test_pair_weight_digest_mismatch_is_rejected() -> None:
    first, second = _outputs()
    mismatched = replace(second, weights_sha256="b" * 64)

    result = ChangePerceptionPipeline(
        _DenseClient((first, mismatched)),
        _settings(),
    ).run(_prepared())

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
