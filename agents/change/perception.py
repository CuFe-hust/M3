"""Abstract-model orchestration for Change V2 auxiliary perception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from agents.change.difference_proposal import propose_changes
from agents.change.feature_residual import (
    FEATURE_RESIDUAL_VERSION,
    compute_feature_residual,
)
from agents.change.preprocess import ChangePreparedPair
from agents.change.proposal_fusion import (
    PROPOSAL_FUSION_VERSION,
    fuse_change_proposals,
)
from agents.change.schema import ChangeProposal
from agents.change.semantic_difference import (
    SEMANTIC_DIFFERENCE_VERSION,
    compute_semantic_difference,
)
from agents.change.settings import AgentChangeSettings
from agents.errors import OptionalDependencyMissingError
from models.base import (
    DenseSemanticClient,
    DenseSemanticOutput,
    MissingModelCacheIdentityError,
    ModelAssetError,
    ModelAssetHashMismatchError,
    ModelAssetMissingError,
    ModelAssetPointerError,
    ModelCacheIdentity,
    require_model_cache_identity,
)


PERCEPTION_VERSION = "change_auxiliary_perception_v1"


class ChangePerceptionError(RuntimeError):
    """Stable fatal or fallback-eligible Change V2 perception failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ChangePerceptionResult:
    """In-memory proposal result consumed by the artifact publisher."""

    score_map: Any
    proposals: list[ChangeProposal]
    diagnostics: dict[str, object]
    component_maps: dict[str, Any] | None = None
    component_masks: dict[str, Any] | None = None


class ChangePerceptionPipeline:
    """Orchestrate Change V1 or V2 through an abstract dense client."""

    def __init__(
        self,
        semantic_client: DenseSemanticClient | None,
        settings: AgentChangeSettings,
    ) -> None:
        self._semantic_client = semantic_client
        self._settings = settings

    def run(self, prepared: ChangePreparedPair) -> ChangePerceptionResult:
        """Run deterministic proposal perception without publishing artifacts."""

        np = _require_numpy()
        if (
            not prepared.validation.valid
            or prepared.raw_t1 is None
            or prepared.raw_t2 is None
            or prepared.comparison_t1 is None
            or prepared.comparison_t2 is None
        ):
            raise ChangePerceptionError("INVALID_CHANGE_PAIR")

        if self._settings.proposals.enabled:
            low_level_map, legacy_proposals = propose_changes(
                prepared.comparison_t1,
                prepared.comparison_t2,
                self._settings.proposals,
            )
        else:
            low_level_map = np.zeros(prepared.raw_t1.shape[:2], dtype=np.float32)
            legacy_proposals = []

        if not self._settings.semantic.enabled or not self._settings.proposals.enabled:
            reason = (
                "SEMANTIC_DISABLED"
                if not self._settings.semantic.enabled
                else "CHANGE_PROPOSALS_DISABLED"
            )
            return self._legacy_result(
                low_level_map,
                legacy_proposals,
                semantic_status="disabled",
                semantic_reason_code=reason,
                identity=None,
                pif_valid=prepared.pif_valid,
            )

        if not prepared.pif_valid:
            return self._handle_failure(
                ChangePerceptionError("FEATURE_RESIDUAL_INSUFFICIENT_PIF"),
                low_level_map=low_level_map,
                legacy_proposals=legacy_proposals,
                identity=None,
                pif_valid=False,
            )

        if self._semantic_client is None:
            return self._handle_failure(
                ChangePerceptionError("SEGFORMER_CLIENT_MISSING"),
                low_level_map=low_level_map,
                legacy_proposals=legacy_proposals,
                identity=None,
                pif_valid=True,
            )

        identity: ModelCacheIdentity | None = None
        try:
            identity = require_model_cache_identity(
                self._semantic_client,
                component="change semantic client",
            )
            semantic_settings = self._settings.semantic
            first_output = self._semantic_client.infer(
                Image.fromarray(prepared.comparison_t1),
                tile_size=semantic_settings.tile_size,
                tile_overlap=semantic_settings.tile_overlap,
                feature_stage=semantic_settings.feature_stage,
            )
            second_output = self._semantic_client.infer(
                Image.fromarray(prepared.comparison_t2),
                tile_size=semantic_settings.tile_size,
                tile_overlap=semantic_settings.tile_overlap,
                feature_stage=semantic_settings.feature_stage,
            )
            if require_model_cache_identity(
                self._semantic_client,
                component="change semantic client",
            ) != identity:
                raise ChangePerceptionError("SEGFORMER_MODEL_IDENTITY_MISMATCH")
            _validate_pair_grids(
                first_output,
                second_output,
                expected_size=(prepared.raw_t1.shape[1], prepared.raw_t1.shape[0]),
                np=np,
            )
            weights_sha256 = _validate_pair_weight_identity(
                first_output,
                second_output,
            )
            feature_result = compute_feature_residual(
                first_output.features,
                second_output.features,
                prepared.pif_mask,
                local_match_radius=semantic_settings.local_match_radius,
                min_pif_feature_cells=semantic_settings.min_pif_feature_cells,
                feature_scale_epsilon=semantic_settings.feature_scale_epsilon,
            )
            if feature_result.diagnostics["alignment_status"] != "aligned":
                raise ChangePerceptionError("FEATURE_RESIDUAL_INSUFFICIENT_PIF")
            semantic_result = compute_semantic_difference(
                first_output.probabilities,
                second_output.probabilities,
                confidence_floor=semantic_settings.semantic_confidence_floor,
                epsilon=semantic_settings.js_epsilon,
            )
            fusion_result = fuse_change_proposals(
                low_level_map,
                feature_result.score_map,
                semantic_result.score_map,
                prepared.pif_mask,
                self._settings.proposals,
                min_pif_pixels=self._settings.harmonization.min_pif_pixels,
            )
        except (
            ChangePerceptionError,
            MissingModelCacheIdentityError,
            ModelAssetError,
            OptionalDependencyMissingError,
            RuntimeError,
            ValueError,
        ) as error:
            reason_code = _fallback_reason_code(error)
            if reason_code is None:
                raise
            return self._handle_failure(
                ChangePerceptionError(reason_code),
                low_level_map=low_level_map,
                legacy_proposals=legacy_proposals,
                identity=identity,
                pif_valid=prepared.pif_valid,
            )

        diagnostics = self._base_diagnostics(
            semantic_status="success",
            semantic_reason_code=None,
            proposal_source="fused_change_v2",
            identity=identity,
            weights_sha256=weights_sha256,
            pif_valid=prepared.pif_valid,
            pif_used_for_feature_alignment=True,
            pif_used_for_threshold=(
                fusion_result.diagnostics.get("threshold_mode") == "pif_robust"
            ),
        )
        diagnostics.update(
            {
                "feature_residual": feature_result.diagnostics,
                "semantic_difference": semantic_result.diagnostics,
                "fusion": fusion_result.diagnostics,
                "score_maps": {
                    "low_level": _score_statistics(low_level_map, np=np),
                    "feature": _score_statistics(feature_result.score_map, np=np),
                    "semantic": _score_statistics(semantic_result.score_map, np=np),
                    "fused": _score_statistics(fusion_result.fused_score_map, np=np),
                },
            }
        )
        return ChangePerceptionResult(
            score_map=fusion_result.fused_score_map,
            proposals=fusion_result.proposals,
            diagnostics=diagnostics,
            component_maps={
                "low_level_difference_map": low_level_map,
                "feature_residual_map": feature_result.score_map,
                "semantic_difference_map": semantic_result.score_map,
                "binary_change_mask": fusion_result.binary_change_mask,
            },
            component_masks=fusion_result.component_masks,
        )

    def _handle_failure(
        self,
        error: ChangePerceptionError,
        *,
        low_level_map: Any,
        legacy_proposals: list[ChangeProposal],
        identity: ModelCacheIdentity | None,
        pif_valid: bool,
    ) -> ChangePerceptionResult:
        if self._settings.semantic.failure_policy == "fail":
            raise error
        return self._legacy_result(
            low_level_map,
            legacy_proposals,
            semantic_status="fallback",
            semantic_reason_code=error.reason_code,
            identity=identity,
            pif_valid=pif_valid,
        )

    def _legacy_result(
        self,
        low_level_map: Any,
        legacy_proposals: list[ChangeProposal],
        *,
        semantic_status: str,
        semantic_reason_code: str | None,
        identity: ModelCacheIdentity | None,
        pif_valid: bool,
    ) -> ChangePerceptionResult:
        np = _require_numpy()
        return ChangePerceptionResult(
            score_map=low_level_map,
            proposals=legacy_proposals,
            diagnostics={
                **self._base_diagnostics(
                    semantic_status=semantic_status,
                    semantic_reason_code=semantic_reason_code,
                    proposal_source="difference_map_v1",
                    identity=identity,
                    weights_sha256=None,
                    pif_valid=pif_valid,
                    pif_used_for_feature_alignment=False,
                    pif_used_for_threshold=False,
                ),
                "score_maps": {
                    "low_level": _score_statistics(low_level_map, np=np),
                },
            },
        )

    def _base_diagnostics(
        self,
        *,
        semantic_status: str,
        semantic_reason_code: str | None,
        proposal_source: str,
        identity: ModelCacheIdentity | None,
        weights_sha256: str | None,
        pif_valid: bool,
        pif_used_for_feature_alignment: bool,
        pif_used_for_threshold: bool,
    ) -> dict[str, object]:
        semantic = self._settings.semantic
        perception_mode = (
            "fused_v2"
            if semantic_status == "success"
            else "fallback_legacy"
            if semantic_status == "fallback"
            else "legacy"
        )
        return {
            "perception_mode": perception_mode,
            "perception_version": PERCEPTION_VERSION,
            "semantic_enabled": semantic.enabled,
            "semantic_status": semantic_status,
            "semantic_reason_code": semantic_reason_code,
            "semantic_reason_codes": (
                [semantic_reason_code] if semantic_reason_code is not None else []
            ),
            "proposal_source": proposal_source,
            "semantic_model": identity.model if identity is not None else None,
            "semantic_client_version": (
                identity.client_version if identity is not None else None
            ),
            "semantic_model_revision": identity.revision if identity is not None else None,
            "semantic_weights_sha256": weights_sha256,
            # Backward-compatible logical-id alias. Never a physical checkpoint path.
            "segformer_model": identity.model if identity is not None else None,
            "feature_stage": semantic.feature_stage,
            "tile_size": semantic.tile_size,
            "tile_overlap": semantic.tile_overlap,
            "local_match_radius": semantic.local_match_radius,
            "semantic_confidence_floor": semantic.semantic_confidence_floor,
            "js_epsilon": semantic.js_epsilon,
            "min_pif_feature_cells": semantic.min_pif_feature_cells,
            "feature_scale_epsilon": semantic.feature_scale_epsilon,
            "pif_threshold_k": self._settings.proposals.pif_threshold_k,
            "pif_fallback_quantile": self._settings.proposals.pif_fallback_quantile,
            "pif_valid": pif_valid,
            "pif_used_for_feature_alignment": pif_used_for_feature_alignment,
            "pif_used_for_threshold": pif_used_for_threshold,
            "threshold_floor": self._settings.proposals.threshold_floor,
            "feature_residual_version": FEATURE_RESIDUAL_VERSION,
            "semantic_difference_version": SEMANTIC_DIFFERENCE_VERSION,
            "fusion_version": PROPOSAL_FUSION_VERSION,
        }


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise OptionalDependencyMissingError("change", dependency="numpy") from error
    return np


def _validate_pair_grids(
    first: DenseSemanticOutput,
    second: DenseSemanticOutput,
    *,
    expected_size: tuple[int, int],
    np: Any,
) -> None:
    try:
        probability_1 = np.asarray(first.probabilities)
        probability_2 = np.asarray(second.probabilities)
        feature_1 = np.asarray(first.features)
        feature_2 = np.asarray(second.features)
        output_shapes_match = (
            probability_1.ndim == 3
            and probability_2.shape == probability_1.shape
            and feature_1.ndim == 3
            and feature_2.shape == feature_1.shape
        )
        size_match = (
            tuple(first.original_size) == expected_size
            and tuple(second.original_size) == expected_size
        )
        stride_match = _compatible_stride(
            first.semantic_stride, second.semantic_stride, np=np
        ) and _compatible_stride(first.feature_stride, second.feature_stride, np=np)
        stride_matches_shapes = (
            _stride_matches_grid(
                first.semantic_stride,
                expected_size,
                probability_1.shape,
                np=np,
            )
            and _stride_matches_grid(
                second.semantic_stride,
                expected_size,
                probability_2.shape,
                np=np,
            )
            and _stride_matches_grid(
                first.feature_stride,
                expected_size,
                feature_1.shape,
                np=np,
            )
            and _stride_matches_grid(
                second.feature_stride,
                expected_size,
                feature_2.shape,
                np=np,
            )
        )
        classes_match = tuple(first.class_names) == tuple(second.class_names)
    except (AttributeError, TypeError, ValueError, IndexError):
        raise ChangePerceptionError("SEGFORMER_PAIR_GRID_MISMATCH") from None
    if not (
        output_shapes_match
        and size_match
        and stride_match
        and stride_matches_shapes
        and classes_match
    ):
        raise ChangePerceptionError("SEGFORMER_PAIR_GRID_MISMATCH")


def _compatible_stride(first: Any, second: Any, *, np: Any) -> bool:
    if len(first) != 2 or len(second) != 2:
        return False
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    return bool(
        np.all(np.isfinite(first_values))
        and np.all(np.isfinite(second_values))
        and np.all(first_values > 0.0)
        and np.all(second_values > 0.0)
        and np.allclose(first_values, second_values, rtol=1e-6, atol=1e-9)
    )


def _validate_pair_weight_identity(
    first: DenseSemanticOutput,
    second: DenseSemanticOutput,
) -> str:
    """Require the same verified checkpoint digest for both frame outputs.
    要求两帧输出携带相同的已验证 checkpoint digest。"""

    first_digest = first.weights_sha256
    second_digest = second.weights_sha256
    if (
        not isinstance(first_digest, str)
        or len(first_digest) != 64
        or any(character not in "0123456789abcdef" for character in first_digest)
        or first_digest != second_digest
    ):
        raise ChangePerceptionError("SEGFORMER_MODEL_IDENTITY_MISMATCH")
    return first_digest


def _stride_matches_grid(
    stride: Any,
    original_size: tuple[int, int],
    shape: tuple[int, ...],
    *,
    np: Any,
) -> bool:
    if len(shape) != 3 or shape[1] <= 0 or shape[2] <= 0 or len(stride) != 2:
        return False
    expected = np.asarray(
        [original_size[0] / shape[2], original_size[1] / shape[1]],
        dtype=np.float64,
    )
    return bool(
        np.allclose(np.asarray(stride, dtype=np.float64), expected, rtol=1e-6, atol=1e-9)
    )


def _fallback_reason_code(error: BaseException) -> str | None:
    if isinstance(error, ChangePerceptionError):
        return error.reason_code
    if isinstance(error, ModelAssetMissingError):
        return "SEGFORMER_CHECKPOINT_MISSING"
    if isinstance(error, ModelAssetPointerError):
        return "SEGFORMER_CHECKPOINT_LFS_POINTER"
    if isinstance(error, ModelAssetHashMismatchError):
        return "SEGFORMER_CHECKPOINT_HASH_MISMATCH"
    if isinstance(error, ModelAssetError):
        return "SEGFORMER_MODEL_ASSET_INVALID"
    if isinstance(error, OptionalDependencyMissingError):
        return "SEGFORMER_DEPENDENCY_MISSING"
    if isinstance(error, MissingModelCacheIdentityError):
        return "SEGFORMER_MODEL_IDENTITY_INVALID"
    concrete_error_codes = {
        "SegFormerDependencyError": "SEGFORMER_DEPENDENCY_MISSING",
        "SegFormerMetadataError": "SEGFORMER_METADATA_INVALID",
        "SegFormerLoadError": "SEGFORMER_LOAD_FAILED",
        "SegFormerDeviceError": "SEGFORMER_DEVICE_UNAVAILABLE",
    }
    if type(error).__name__ in concrete_error_codes:
        return concrete_error_codes[type(error).__name__]
    message = str(error).strip()
    if type(error).__name__ == "SegFormerInferenceError":
        return message if message.startswith("SEGFORMER_") else "SEGFORMER_INFERENCE_FAILED"
    allowed_prefixes = (
        "FEATURE_RESIDUAL_",
        "SEMANTIC_DIFFERENCE_",
        "PROPOSAL_FUSION_",
        "SEGFORMER_PAIR_GRID_MISMATCH",
    )
    if isinstance(error, ValueError) and message.startswith(allowed_prefixes):
        return message.split(":", 1)[0]
    return None


def _score_statistics(value: Any, *, np: Any) -> dict[str, float]:
    """Return compact numeric audit data for a score map."""

    score_map = np.asarray(value, dtype=np.float64)
    if score_map.ndim != 2 or not bool(np.all(np.isfinite(score_map))):
        raise ValueError("CHANGE_SCORE_MAP_INVALID")
    return {
        "min": float(np.min(score_map)),
        "median": float(np.median(score_map)),
        "p95": float(np.quantile(score_map, 0.95)),
        "max": float(np.max(score_map)),
    }


__all__ = [
    "PERCEPTION_VERSION",
    "ChangePerceptionError",
    "ChangePerceptionPipeline",
    "ChangePerceptionResult",
]
