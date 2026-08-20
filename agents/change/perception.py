"""Abstract-model orchestration for Change V2 auxiliary perception."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math
from typing import Any

from PIL import Image

from agents.change.difference_proposal import propose_changes
from agents.change.feature_residual import (
    FEATURE_RESIDUAL_VERSION,
    compute_feature_residual,
    compute_multiscale_feature_residual,
)
from agents.change.preprocess import ChangePreparedPair
from agents.change.proposal_fusion import (
    PROPOSAL_FUSION_VERSION,
    compute_reliabilities,
    fuse_feature_evidence,
    fuse_semantic_evidence,
    fuse_change_proposals,
)
from agents.change.schema import ChangeProposal
from agents.change.semantic_difference import (
    SEMANTIC_DIFFERENCE_VERSION,
    compute_semantic_difference,
)
from agents.change.semantic_transition import infer_semantic_transition
from agents.change.settings import AgentChangeSettings
from agents.errors import OptionalDependencyMissingError
from models.base import (
    DenseSemanticClient,
    DenseSemanticOutput,
    DenseSemanticPyramidOutput,
    LearnedChangeClient,
    LearnedChangeOutput,
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


class LearnedChangeFailure(ChangePerceptionError):
    """Failure that must honor the learned hook's own strict policy."""


@dataclass(frozen=True)
class ChangePerceptionResult:
    """In-memory proposal result consumed by the artifact publisher."""

    score_map: Any
    proposals: list[ChangeProposal]
    diagnostics: dict[str, object]
    component_maps: dict[str, Any] | None = None
    component_masks: dict[str, Any] | None = None


@dataclass(frozen=True)
class SemanticExpertBinding:
    """Runtime-only verified semantic expert binding."""

    expert_id: str
    logical_model_id: str
    priority: int
    role: str
    neutral_labels: frozenset[str]
    transient_labels: frozenset[str]
    persistent_labels: frozenset[str]
    client: DenseSemanticClient


@dataclass(frozen=True)
class SemanticExpertRun:
    binding: SemanticExpertBinding
    identity: ModelCacheIdentity
    first_output: DenseSemanticOutput | DenseSemanticPyramidOutput
    second_output: DenseSemanticOutput | DenseSemanticPyramidOutput
    weights_sha256: str | None


@dataclass(frozen=True)
class SemanticExpertEvidence:
    """Taxonomy-local evidence produced by one successful expert."""

    run: SemanticExpertRun
    semantic_result: Any
    feature_result: Any | None
    feature_diagnostics: dict[str, object]
    reliability: dict[str, float]
    reliability_diagnostics: dict[str, object]


class ChangePerceptionPipeline:
    """Orchestrate Change V1 or V2 through an abstract dense client."""

    def __init__(
        self,
        semantic_client: DenseSemanticClient | None,
        settings: AgentChangeSettings,
        learned_change_client: LearnedChangeClient | None = None,
        semantic_experts: tuple[SemanticExpertBinding, ...] = (),
    ) -> None:
        self._semantic_client = semantic_client
        compatibility = (
            SemanticExpertBinding(
                expert_id="injected_semantic",
                logical_model_id="injected_semantic",
                priority=0,
                role="generic",
                neutral_labels=frozenset(),
                transient_labels=frozenset(),
                persistent_labels=frozenset(),
                client=semantic_client,
            )
            if semantic_client is not None
            else None
        )
        self._semantic_experts = (
            tuple(semantic_experts)
            if semantic_experts
            else tuple(
                item
                for item in ((compatibility,) if compatibility is not None else ())
            )
        )
        self._learned_change_client = learned_change_client
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
                valid_mask=getattr(prepared, "registration_valid_mask", None),
            )
        else:
            low_level_map = np.zeros(prepared.raw_t1.shape[:2], dtype=np.float32)
            legacy_proposals = []

        if (
            self._settings.learned_change.enabled
            and self._learned_change_client is None
            and self._settings.learned_change.failure_policy == "fail"
        ):
            raise LearnedChangeFailure("LEARNED_CHANGE_CLIENT_MISSING")

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

        if (
            not prepared.pif_valid
            and self._settings.semantic.failure_policy == "fail"
        ):
            raise ChangePerceptionError("FEATURE_RESIDUAL_INSUFFICIENT_PIF")

        if not self._semantic_experts:
            return self._handle_failure(
                ChangePerceptionError("SEGFORMER_CLIENT_MISSING"),
                low_level_map=low_level_map,
                legacy_proposals=legacy_proposals,
                identity=None,
                pif_valid=True,
            )

        identity: ModelCacheIdentity | None = None
        try:
            semantic_settings = self._settings.semantic
            requested_stages = tuple(semantic_settings.feature_stages)
            expected_size = (prepared.raw_t1.shape[1], prepared.raw_t1.shape[0])
            semantic_runs: list[SemanticExpertRun] = []
            expert_failures: list[dict[str, str]] = []
            expert_errors: list[BaseException] = []
            for binding in self._semantic_experts[: semantic_settings.max_experts if semantic_settings.multi_expert_enabled else 1]:
                try:
                    semantic_runs.append(
                        _infer_semantic_expert_pair(
                            binding,
                            prepared,
                            semantic_settings,
                            expected_size=expected_size,
                            requested_stages=requested_stages,
                            np=np,
                        )
                    )
                except Exception as error:
                    expert_errors.append(error)
                    expert_failures.append(
                        {"expert_id": binding.expert_id, "error_type": type(error).__name__}
                    )
            if len(semantic_runs) < semantic_settings.min_successful_experts:
                if len(self._semantic_experts) == 1 and expert_errors:
                    raise expert_errors[0]
                raise ChangePerceptionError("SEGFORMER_ALL_EXPERTS_FAILED")
            expert_evidence: list[SemanticExpertEvidence] = []
            for run in semantic_runs:
                try:
                    semantic_result = compute_semantic_difference(
                        run.first_output.probabilities,
                        run.second_output.probabilities,
                        confidence_floor=semantic_settings.semantic_confidence_floor,
                        epsilon=semantic_settings.js_epsilon,
                        valid_mask=getattr(prepared, "registration_valid_mask", None),
                    )
                except Exception as error:
                    if len(self._semantic_experts) == 1:
                        raise
                    expert_failures.append(
                        {"expert_id": run.binding.expert_id, "error_type": type(error).__name__}
                    )
                    continue
                try:
                    feature_result, feature_diagnostics = _compute_feature_evidence(
                        run.first_output,
                        run.second_output,
                        prepared,
                        semantic_settings,
                        requested_stages=requested_stages,
                        expected_size=expected_size,
                        np=np,
                    )
                    if (
                        feature_result is None
                        and prepared.pif_valid
                        and semantic_settings.failure_policy == "fail"
                        and len(self._semantic_experts) == 1
                    ):
                        raise ChangePerceptionError("FEATURE_RESIDUAL_INSUFFICIENT_PIF")
                except Exception as error:
                    if len(self._semantic_experts) == 1:
                        raise
                    feature_result = None
                    feature_diagnostics = {
                        "alignment_status": "failed",
                        "reason_code": _fallback_reason_code(error)
                        or "FEATURE_RESIDUAL_INFERENCE_FAILED",
                        "valid_feature_fraction": 0.0,
                        "effective_stages": [],
                        "missing_stages": list(requested_stages),
                    }
                    expert_failures.append(
                        {
                            "expert_id": run.binding.expert_id,
                            "error_type": type(error).__name__,
                            "branch": "feature",
                        }
                    )
                reliability, reliability_diagnostics = compute_reliabilities(
                    registration_report=getattr(prepared, "registration_report", None),
                    feature_diagnostics=feature_diagnostics,
                    semantic_diagnostics=semantic_result.diagnostics,
                    harmonization_decision=prepared.decision,
                    settings=self._settings.reliability,
                )
                expert_evidence.append(
                    SemanticExpertEvidence(
                        run=run,
                        semantic_result=semantic_result,
                        feature_result=feature_result,
                        feature_diagnostics=feature_diagnostics,
                        reliability=reliability,
                        reliability_diagnostics=reliability_diagnostics,
                    )
                )
            if len(expert_evidence) < semantic_settings.min_successful_experts:
                raise ChangePerceptionError("SEGFORMER_ALL_EXPERTS_FAILED")
            selected_evidence = expert_evidence[0]
            selected_run = selected_evidence.run
            identity = selected_run.identity
            first_output = selected_run.first_output
            second_output = selected_run.second_output
            weights_sha256 = selected_run.weights_sha256
            semantic_map, semantic_fusion_diagnostics = fuse_semantic_evidence(
                [item.semantic_result.score_map for item in expert_evidence],
                [item.reliability["semantic"] for item in expert_evidence],
                consensus_weight=semantic_settings.semantic_consensus_weight,
                union_weight=semantic_settings.semantic_union_weight,
            )
            feature_results = [
                item for item in expert_evidence if item.feature_result is not None
            ]
            feature_map, feature_fusion_diagnostics = fuse_feature_evidence(
                [item.feature_result.score_map for item in feature_results],
                [item.reliability["feature"] for item in feature_results],
            )
            feature_diagnostics = _aggregate_branch_diagnostics(
                [item.feature_diagnostics for item in expert_evidence],
                branch="feature",
            )
            semantic_diagnostics = _aggregate_branch_diagnostics(
                [item.semantic_result.diagnostics for item in expert_evidence],
                branch="semantic",
            )
            reliability, reliability_diagnostics = _aggregate_reliability(
                expert_evidence
            )
            learned_map, learned_diagnostics = self._run_learned_change_hook(
                first_output=first_output,
                second_output=second_output,
                feature_stage=(
                    requested_stages[0] if requested_stages else semantic_settings.feature_stage
                ),
                valid_mask=getattr(prepared, "registration_valid_mask", None),
            )
            fusion_result = fuse_change_proposals(
                low_level_map,
                feature_map,
                semantic_map,
                prepared.pif_mask,
                self._settings.proposals,
                min_pif_pixels=self._settings.harmonization.min_pif_pixels,
                fallback_reason=(
                    "FEATURE_RESIDUAL_INSUFFICIENT_PIF"
                    if feature_map is None
                    else None
                ),
                reliability=reliability,
                valid_overlap_mask=getattr(prepared, "registration_valid_mask", None),
                registration_confidence=reliability["registration"],
                learned_map=learned_map,
                learned_weight=self._settings.learned_change.fusion_weight,
                learned_requested=self._settings.learned_change.enabled,
            )
        except LearnedChangeFailure:
            raise
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

        proposals = _attach_semantic_transitions(
            fusion_result.proposals,
            component_masks=fusion_result.component_masks,
            valid_mask=getattr(prepared, "registration_valid_mask", None),
            confidence_floor=self._settings.semantic.semantic_confidence_floor,
            expert_evidence=expert_evidence,
        )
        diagnostics = self._base_diagnostics(
            semantic_status="success",
            semantic_reason_code=None,
            proposal_source="fused_change_v2",
            identity=identity,
            weights_sha256=weights_sha256,
            pif_valid=prepared.pif_valid,
            pif_used_for_feature_alignment=feature_map is not None,
            pif_used_for_threshold=(
                fusion_result.diagnostics.get("threshold_mode") == "pif_robust"
            ),
        )
        diagnostics.update(
            {
                "feature_residual": feature_diagnostics,
                "semantic_difference": semantic_diagnostics,
                "reliability": reliability_diagnostics,
                "learned_change": learned_diagnostics,
                "semantic_transition_note": (
                    "auxiliary model evidence; not ground truth; raw evidence must be reviewed"
                ),
                "semantic_experts": [
                    {
                        "expert_id": item.run.binding.expert_id,
                        "logical_model_id": item.run.identity.model,
                        "priority": item.run.binding.priority,
                        "role": item.run.binding.role,
                        "status": "success",
                        "weights_sha256": item.run.weights_sha256,
                        "semantic": dict(item.semantic_result.diagnostics),
                        "feature": dict(item.feature_diagnostics),
                        "reliability": dict(item.reliability),
                    }
                    for item in expert_evidence
                ],
                "semantic_expert_failures": expert_failures,
                "semantic_fusion": semantic_fusion_diagnostics,
                "feature_fusion": feature_fusion_diagnostics,
                "fusion": fusion_result.diagnostics,
                "score_maps": {
                    "low_level": _score_statistics(low_level_map, np=np),
                    **(
                        {
                            "feature": _score_statistics(
                                feature_map, np=np
                            )
                        }
                        if feature_map is not None
                        else {}
                    ),
                    "semantic": _score_statistics(semantic_map, np=np),
                    "fused": _score_statistics(fusion_result.fused_score_map, np=np),
                    **(
                        {"learned": _score_statistics(learned_map, np=np)}
                        if learned_map is not None
                        else {}
                    ),
                },
            }
        )
        return ChangePerceptionResult(
            score_map=fusion_result.fused_score_map,
            proposals=proposals,
            diagnostics=diagnostics,
            component_maps={
                "low_level_difference_map": low_level_map,
                **(
                    {"feature_residual_map": feature_map}
                    if feature_map is not None
                    else {}
                ),
                "semantic_difference_map": semantic_map,
                "binary_change_mask": fusion_result.binary_change_mask,
                **(
                    {"learned_change_map": learned_map}
                    if learned_map is not None
                    else {}
                ),
            },
            component_masks=fusion_result.component_masks,
        )

    def _run_learned_change_hook(
        self,
        *,
        first_output: DenseSemanticOutput | DenseSemanticPyramidOutput,
        second_output: DenseSemanticOutput | DenseSemanticPyramidOutput,
        feature_stage: int,
        valid_mask: Any,
    ) -> tuple[Any | None, dict[str, object]]:
        """Run the optional learned-head seam without fabricating a map.

        The head is an inference-only dependency.  Its concrete runtime and
        checkpoint identity stay outside ``agents/``; when unavailable, the
        deterministic rule branches remain the complete fallback path.
        """

        settings = self._settings.learned_change
        if not settings.enabled:
            return None, {
                "enabled": False,
                "status": "disabled",
                "available": False,
                "reason_codes": ["LEARNED_CHANGE_DISABLED"],
                "fusion_weight": settings.fusion_weight,
            }
        if self._learned_change_client is None:
            return self._learned_change_failure("LEARNED_CHANGE_CLIENT_MISSING")
        try:
            identity = require_model_cache_identity(
                self._learned_change_client,
                component="learned change client",
            )
            output = self._learned_change_client.infer(
                first=_as_pyramid_output(first_output, feature_stage=feature_stage),
                second=_as_pyramid_output(second_output, feature_stage=feature_stage),
                valid_mask=valid_mask,
            )
            if not isinstance(output, LearnedChangeOutput):
                raise ChangePerceptionError("LEARNED_CHANGE_OUTPUT_INVALID")
            np = _require_numpy()
            score_map = np.asarray(output.score_map)
            if score_map.ndim != 2 or not bool(np.isfinite(score_map).all()):
                raise ChangePerceptionError("LEARNED_CHANGE_OUTPUT_INVALID")
            return score_map, {
                "enabled": True,
                "status": "available",
                "available": True,
                "reason_codes": [],
                "fusion_weight": settings.fusion_weight,
                "model": identity.model,
                "revision": identity.revision,
                "client_version": identity.client_version,
                "diagnostics": _json_safe_diagnostics(output.diagnostics),
            }
        except (
            ChangePerceptionError,
            MissingModelCacheIdentityError,
            ModelAssetError,
            OptionalDependencyMissingError,
            RuntimeError,
            ValueError,
        ) as error:
            reason_code = (
                error.reason_code
                if isinstance(error, ChangePerceptionError)
                else _fallback_reason_code(error) or "LEARNED_CHANGE_INFERENCE_FAILED"
            )
            return self._learned_change_failure(reason_code)

    def _learned_change_failure(
        self,
        reason_code: str,
    ) -> tuple[Any | None, dict[str, object]]:
        settings = self._settings.learned_change
        if settings.failure_policy == "fail":
            raise LearnedChangeFailure(reason_code)
        return None, {
            "enabled": True,
            "status": "fallback",
            "available": False,
            "reason_codes": [reason_code],
            "fusion_weight": settings.fusion_weight,
        }

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
            "feature_stages": list(semantic.feature_stages),
            "feature_stage_weights": {
                str(stage): float(weight)
                for stage, weight in semantic.feature_stage_weights.items()
            },
            "multiscale_enabled": len(semantic.feature_stages) > 1,
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
            "learned_change": {
                "enabled": self._settings.learned_change.enabled,
                "status": (
                    "unavailable"
                    if self._settings.learned_change.enabled
                    else "disabled"
                ),
                "available": False,
                "reason_codes": (
                    ["LEARNED_CHANGE_NOT_RUN"]
                    if self._settings.learned_change.enabled
                    else ["LEARNED_CHANGE_DISABLED"]
                ),
                "fusion_weight": self._settings.learned_change.fusion_weight,
            },
        }


def _compute_feature_evidence(
    first_output: DenseSemanticOutput | DenseSemanticPyramidOutput,
    second_output: DenseSemanticOutput | DenseSemanticPyramidOutput,
    prepared: ChangePreparedPair,
    settings: Any,
    *,
    requested_stages: tuple[int, ...],
    expected_size: tuple[int, int],
    np: Any,
) -> tuple[Any | None, dict[str, object]]:
    if not prepared.pif_valid:
        return None, {
            "alignment_status": "insufficient_pif",
            "reason_code": "FEATURE_RESIDUAL_INSUFFICIENT_PIF",
            "pif_feature_cells": 0,
            "valid_feature_fraction": 0.0,
            "effective_stages": [],
            "missing_stages": list(requested_stages),
        }
    if len(requested_stages) > 1:
        if not isinstance(first_output, DenseSemanticPyramidOutput) or not isinstance(
            second_output, DenseSemanticPyramidOutput
        ):
            raise ChangePerceptionError("SEGFORMER_PYRAMID_GRID_MISMATCH")
        result = compute_multiscale_feature_residual(
            first_output.features_by_stage,
            second_output.features_by_stage,
            prepared.pif_mask,
            feature_stages=requested_stages,
            feature_stage_weights=settings.feature_stage_weights,
            feature_strides_by_stage=first_output.feature_strides_by_stage,
            image_size=expected_size,
            valid_mask=getattr(prepared, "registration_valid_mask", None),
            local_match_radius=settings.local_match_radius,
            min_pif_feature_cells=settings.min_pif_feature_cells,
            feature_scale_epsilon=settings.feature_scale_epsilon,
        )
    else:
        if not isinstance(first_output, DenseSemanticOutput) or not isinstance(
            second_output, DenseSemanticOutput
        ):
            raise ChangePerceptionError("SEGFORMER_PAIR_GRID_MISMATCH")
        result = compute_feature_residual(
            first_output.features,
            second_output.features,
            prepared.pif_mask,
            local_match_radius=settings.local_match_radius,
            min_pif_feature_cells=settings.min_pif_feature_cells,
            feature_scale_epsilon=settings.feature_scale_epsilon,
        )
    diagnostics = dict(result.diagnostics)
    diagnostics["valid_feature_fraction"] = float(
        np.mean(np.asarray(result.valid_mask, dtype=bool))
    )
    if result.diagnostics.get("alignment_status") != "aligned":
        diagnostics["alignment_status"] = "insufficient_pif"
        diagnostics["reason_code"] = "FEATURE_RESIDUAL_INSUFFICIENT_PIF"
        result = None
    return result, diagnostics


def _aggregate_branch_diagnostics(
    diagnostics: list[dict[str, object]],
    *,
    branch: str,
) -> dict[str, object]:
    if not diagnostics:
        return {"expert_count": 0}
    if len(diagnostics) == 1:
        result = dict(diagnostics[0])
        result["expert_count"] = 1
        return result
    result: dict[str, object] = {"expert_count": len(diagnostics)}
    numeric_keys = {
        key
        for item in diagnostics
        for key, value in item.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for key in sorted(numeric_keys):
        values = [
            float(item[key])
            for item in diagnostics
            if isinstance(item.get(key), (int, float))
            and not isinstance(item.get(key), bool)
        ]
        if values:
            result[key] = sum(values) / len(values)
    for key in ("version", "effective_stages", "missing_stages"):
        if key in diagnostics[0]:
            result[key] = diagnostics[0][key]
    statuses = [str(item.get("alignment_status", "")) for item in diagnostics]
    if branch == "feature":
        result["alignment_status"] = (
            "aligned" if all(value == "aligned" for value in statuses) else "partial"
        )
    return result


def _aggregate_reliability(
    evidence: list[SemanticExpertEvidence],
) -> tuple[dict[str, float], dict[str, object]]:
    names = ("low_level", "feature", "semantic", "registration")
    reliability = {
        name: sum(item.reliability[name] for item in evidence) / len(evidence)
        for name in names
    }
    return reliability, {
        "enabled": True,
        "reliability": reliability,
        "expert_count": len(evidence),
        "per_expert": [
            {
                "expert_id": item.run.binding.expert_id,
                "reliability": dict(item.reliability),
            }
            for item in evidence
        ],
    }


def _infer_semantic_expert_pair(
    binding: SemanticExpertBinding,
    prepared: ChangePreparedPair,
    settings: Any,
    *,
    expected_size: tuple[int, int],
    requested_stages: tuple[int, ...],
    np: Any,
) -> SemanticExpertRun:
    """Run and validate one semantic expert independently for both frames."""

    identity = require_model_cache_identity(
        binding.client,
        component=f"change semantic expert {binding.expert_id}",
    )
    infer_pyramid = getattr(binding.client, "infer_pyramid", None)
    if len(requested_stages) > 1:
        if not callable(infer_pyramid):
            raise ChangePerceptionError("SEGFORMER_PYRAMID_UNSUPPORTED")
        first_output = infer_pyramid(
            Image.fromarray(prepared.comparison_t1),
            tile_size=settings.tile_size,
            tile_overlap=settings.tile_overlap,
            feature_stages=requested_stages,
        )
        second_output = infer_pyramid(
            Image.fromarray(prepared.comparison_t2),
            tile_size=settings.tile_size,
            tile_overlap=settings.tile_overlap,
            feature_stages=requested_stages,
        )
        if not isinstance(first_output, DenseSemanticPyramidOutput) or not isinstance(
            second_output, DenseSemanticPyramidOutput
        ):
            raise ChangePerceptionError("SEGFORMER_PYRAMID_GRID_MISMATCH")
        _validate_pyramid_pair_grids(
            first_output,
            second_output,
            expected_size=expected_size,
            expected_stages=requested_stages,
            np=np,
        )
    else:
        stage = requested_stages[0]
        first_output = binding.client.infer(
            Image.fromarray(prepared.comparison_t1),
            tile_size=settings.tile_size,
            tile_overlap=settings.tile_overlap,
            feature_stage=stage,
        )
        second_output = binding.client.infer(
            Image.fromarray(prepared.comparison_t2),
            tile_size=settings.tile_size,
            tile_overlap=settings.tile_overlap,
            feature_stage=stage,
        )
        if not isinstance(first_output, DenseSemanticOutput) or not isinstance(
            second_output, DenseSemanticOutput
        ):
            raise ChangePerceptionError("SEGFORMER_PAIR_GRID_MISMATCH")
        _validate_pair_grids(
            first_output,
            second_output,
            expected_size=expected_size,
            np=np,
        )
    if require_model_cache_identity(
        binding.client,
        component=f"change semantic expert {binding.expert_id}",
    ) != identity:
        raise ChangePerceptionError("SEGFORMER_MODEL_IDENTITY_MISMATCH")
    weights_sha256 = _validate_pair_weight_identity(first_output, second_output)
    return SemanticExpertRun(
        binding=binding,
        identity=identity,
        first_output=first_output,
        second_output=second_output,
        weights_sha256=weights_sha256,
    )


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise OptionalDependencyMissingError("change", dependency="numpy") from error
    return np


def _as_pyramid_output(
    output: DenseSemanticOutput | DenseSemanticPyramidOutput,
    *,
    feature_stage: int,
) -> DenseSemanticPyramidOutput:
    """Adapt the legacy single-stage output to the future head contract."""

    if isinstance(output, DenseSemanticPyramidOutput):
        return output
    return DenseSemanticPyramidOutput(
        probabilities=output.probabilities,
        features_by_stage={feature_stage: output.features},
        semantic_stride=output.semantic_stride,
        feature_strides_by_stage={feature_stage: output.feature_stride},
        original_size=output.original_size,
        class_names=output.class_names,
        diagnostics=output.diagnostics,
        weights_sha256=output.weights_sha256,
    )


def _json_safe_diagnostics(value: Mapping[str, Any]) -> dict[str, object]:
    """Keep optional-head diagnostics compact and artifact-safe."""

    def compact(item: Any) -> object | None:
        if item is None or isinstance(item, (str, bool)):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, Mapping):
            return {
                str(key): child
                for key, raw in item.items()
                if (child := compact(raw)) is not None
            }
        if isinstance(item, (list, tuple)):
            return [child for raw in item if (child := compact(raw)) is not None]
        return None

    compacted = compact(value)
    return compacted if isinstance(compacted, dict) else {}


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


def _validate_pyramid_pair_grids(
    first: DenseSemanticPyramidOutput,
    second: DenseSemanticPyramidOutput,
    *,
    expected_size: tuple[int, int],
    expected_stages: tuple[int, ...],
    np: Any,
) -> None:
    """Validate stage presence, native grids and class order for both frames."""

    try:
        probability_1 = np.asarray(first.probabilities)
        probability_2 = np.asarray(second.probabilities)
        output_shapes_match = (
            probability_1.ndim == 3
            and probability_2.shape == probability_1.shape
        )
        size_match = (
            tuple(first.original_size) == expected_size
            and tuple(second.original_size) == expected_size
        )
        semantic_stride_match = _compatible_stride(
            first.semantic_stride, second.semantic_stride, np=np
        ) and _stride_matches_grid(
            first.semantic_stride, expected_size, probability_1.shape, np=np
        ) and _stride_matches_grid(
            second.semantic_stride, expected_size, probability_2.shape, np=np
        )
        classes_match = tuple(first.class_names) == tuple(second.class_names)
        first_stages = tuple(first.features_by_stage)
        second_stages = tuple(second.features_by_stage)
        stages_match = all(
            stage in first.features_by_stage and stage in second.features_by_stage
            for stage in expected_stages
        )
        stage_keys_match = first_stages == second_stages
        feature_shapes_match = True
        feature_strides_match = True
        for stage in expected_stages:
            feature_1 = np.asarray(first.features_by_stage[stage])
            feature_2 = np.asarray(second.features_by_stage[stage])
            if (
                feature_1.ndim != 3
                or feature_2.shape != feature_1.shape
                or not _stride_matches_grid(
                    first.feature_strides_by_stage[stage],
                    expected_size,
                    feature_1.shape,
                    np=np,
                )
                or not _stride_matches_grid(
                    second.feature_strides_by_stage[stage],
                    expected_size,
                    feature_2.shape,
                    np=np,
                )
            ):
                feature_shapes_match = False
            if not _compatible_stride(
                first.feature_strides_by_stage[stage],
                second.feature_strides_by_stage[stage],
                np=np,
            ):
                feature_strides_match = False
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        raise ChangePerceptionError("SEGFORMER_PYRAMID_GRID_MISMATCH") from None
    if not (
        output_shapes_match
        and size_match
        and semantic_stride_match
        and classes_match
        and stages_match
        and stage_keys_match
        and feature_shapes_match
        and feature_strides_match
    ):
        raise ChangePerceptionError("SEGFORMER_PYRAMID_GRID_MISMATCH")


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
    if isinstance(error, NotImplementedError):
        return "SEGFORMER_PYRAMID_UNSUPPORTED"
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
        "SEMANTIC_TRANSITION_",
        "PROPOSAL_FUSION_",
        "SEGFORMER_PAIR_GRID_MISMATCH",
    )
    if isinstance(error, ValueError) and message.startswith(allowed_prefixes):
        return message.split(":", 1)[0]
    return None


def _attach_semantic_transitions(
    proposals: list[ChangeProposal],
    *,
    component_masks: dict[str, Any],
    valid_mask: Any | None,
    confidence_floor: float,
    expert_evidence: list[SemanticExpertEvidence],
) -> list[ChangeProposal]:
    """Attach concise, taxonomy-aware evidence while preserving raw authority."""

    np = _require_numpy()
    if not proposals:
        return []
    image_shape = np.asarray(valid_mask).shape if valid_mask is not None else None
    if image_shape is None or len(image_shape) != 2:
        image_shape = (
            max((int(item.pixel_box[3]) for item in proposals), default=0),
            max((int(item.pixel_box[2]) for item in proposals), default=0),
        )
        if image_shape == (0, 0):
            image_shape = tuple(
                np.asarray(expert_evidence[0].run.first_output.probabilities).shape[1:]
            )
    updated: list[ChangeProposal] = []
    for proposal in proposals:
        crop = component_masks.get(proposal.mask_filename or "")
        if crop is None:
            updated.append(proposal)
            continue
        full_mask = np.zeros(image_shape, dtype=bool)
        x1, y1, x2, y2 = proposal.pixel_box
        x1 = max(0, min(int(x1), image_shape[1]))
        y1 = max(0, min(int(y1), image_shape[0]))
        x2 = max(x1, min(int(x2), image_shape[1]))
        y2 = max(y1, min(int(y2), image_shape[0]))
        crop_array = np.asarray(crop) != 0
        target_height, target_width = y2 - y1, x2 - x1
        if crop_array.shape != (target_height, target_width):
            # A malformed component mask must not become semantic evidence.
            updated.append(proposal)
            continue
        full_mask[y1:y2, x1:x2] = crop_array
        evidence_items: list[dict[str, object]] = []
        transitions: list[tuple[str, SemanticExpertEvidence, Any, str]] = []
        for evidence in expert_evidence:
            transition = infer_semantic_transition(
                evidence.run.first_output.probabilities,
                evidence.run.second_output.probabilities,
                full_mask,
                evidence.run.first_output.class_names,
                confidence_floor=confidence_floor,
                valid_mask=valid_mask,
            )
            evidence_type = _transition_evidence_type(
                transition,
                evidence.run.binding,
                confidence_floor=confidence_floor,
            )
            confidence = float(transition.transition_confidence)
            item = {
                "expert_id": evidence.run.binding.expert_id,
                "expert_role": evidence.run.binding.role,
                "from_class": transition.from_class,
                "to_class": transition.to_class,
                "evidence_type": evidence_type,
                "confidence": confidence,
                "from_confidence": float(transition.from_confidence),
                "to_confidence": float(transition.to_confidence),
                "support_ratio": float(transition.support_ratio),
                "support_level": _support_level(confidence, confidence_floor),
            }
            evidence_items.append(item)
            if evidence_type in {"persistent", "transient"}:
                transitions.append(
                    (evidence_type, evidence, transition, evidence_type)
                )
        persistent = [item for item in transitions if item[0] == "persistent"]
        transient = [item for item in transitions if item[0] == "transient"]
        persistent_support = max(
            (float(item[2].transition_confidence) for item in persistent),
            default=0.0,
        )
        transient_support = max(
            (float(item[2].transition_confidence) for item in transient),
            default=0.0,
        )
        persistent_pairs = {
            (item[2].from_class, item[2].to_class) for item in persistent
        }
        consensus = {
            "persistent_support": persistent_support,
            "transient_support": transient_support,
            "disagreement": len(persistent_pairs) > 1,
            "informative_expert_count": len(transitions),
            "expert_count": len(expert_evidence),
        }
        chosen = max(persistent, key=lambda item: _transition_sort_key(item[2])) if persistent else (
            max(transient, key=lambda item: _transition_sort_key(item[2])) if transient else None
        )
        updated.append(
            proposal.model_copy(
                update={
                    "semantic_transition": chosen[2] if chosen is not None else None,
                    "semantic_transitions": evidence_items,
                    "semantic_consensus": consensus,
                }
            )
        )
    return updated


def _transition_evidence_type(
    transition: Any,
    binding: SemanticExpertBinding,
    *,
    confidence_floor: float,
) -> str:
    if (
        transition.from_class.casefold() == "unknown"
        or transition.to_class.casefold() == "unknown"
        or transition.from_class.casefold() == transition.to_class.casefold()
        or transition.from_confidence < confidence_floor
        or transition.to_confidence < confidence_floor
    ):
        return "neutral"
    labels = {
        transition.from_class,
        transition.to_class,
    }
    if labels & binding.persistent_labels or binding.role == "persistent_landcover":
        return "persistent"
    if labels & binding.transient_labels:
        return "transient"
    return "transient"


def _support_level(confidence: float, confidence_floor: float) -> str:
    if confidence >= max(0.75, confidence_floor):
        return "strong"
    if confidence >= confidence_floor:
        return "moderate"
    return "weak"


def _transition_sort_key(transition: Any) -> tuple[float, float, str, str]:
    return (
        float(transition.transition_confidence),
        float(transition.support_ratio),
        str(transition.from_class),
        str(transition.to_class),
    )


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
    "SemanticExpertBinding",
    "SemanticExpertEvidence",
    "SemanticExpertRun",
]
