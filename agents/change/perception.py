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
    compute_temporal_semantic_stability,
)
from agents.change.schema import ChangeProposal, StructuralRescueCandidate
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
    LearnedChangeExpertPair,
    LearnedChangeClient,
    LearnedChangeInputSpec,
    LearnedChangeRequest,
    LearnedChangeOutput,
    MissingModelCacheIdentityError,
    ModelAssetError,
    ModelAssetHashMismatchError,
    ModelAssetMissingError,
    ModelAssetPointerError,
    ModelCacheIdentity,
    hash_class_names,
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
    participation: str = "core"
    structural_labels: frozenset[str] = frozenset()
    landcover_candidate_labels: frozenset[str] = frozenset()
    rescue_model_labels: frozenset[str] = frozenset()
    rescue_strategy: str = "none"
    class_names: tuple[str, ...] = ()
    class_names_sha256: str | None = None
    weights_sha256: str | None = None


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


@dataclass(frozen=True)
class SemanticExpertExecutionPlan:
    """Explicit, order-independent plan for all semantic consumers."""

    bindings_by_id: Mapping[str, SemanticExpertBinding]
    requested_stages_by_id: Mapping[str, tuple[int, ...]]
    core_expert_ids: tuple[str, ...]
    rescue_expert_ids: tuple[str, ...]
    learned_required_expert_ids: tuple[str, ...]
    learned_optional_expert_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChangePerceptionExecution:
    """One complete perception execution, including reusable rescue output."""

    result: ChangePerceptionResult
    rescue_candidates: tuple[StructuralRescueCandidate, ...]
    rescue_diagnostics: Mapping[str, object]
    learned_request: LearnedChangeRequest | None = None


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
        self._last_runs_by_id: dict[str, SemanticExpertRun] = {}
        self._last_prepared: ChangePreparedPair | None = None
        self._include_rescue_in_pool = False
        self._last_learned_request: LearnedChangeRequest | None = None

    def _build_execution_plan(self) -> SemanticExpertExecutionPlan:
        bindings_by_id = {
            binding.expert_id: binding for binding in self._semantic_experts
        }
        semantic = self._settings.semantic
        core = sorted(
            (
                binding
                for binding in self._semantic_experts
                if binding.participation == "core"
            ),
            key=lambda item: (-item.priority, item.expert_id),
        )
        if semantic.multi_expert_enabled:
            core = core[: semantic.max_experts]
        else:
            core = core[:1]
        rescue = (
            sorted(
                (
                    binding
                    for binding in self._semantic_experts
                    if binding.participation == "rescue"
                ),
                key=lambda item: (-item.priority, item.expert_id),
            )
            if self._include_rescue_in_pool or self._settings.learned_change.enabled
            else []
        )
        required: list[str] = []
        optional: list[str] = []
        learned_spec = getattr(self._learned_change_client, "input_spec", None)
        if self._settings.learned_change.enabled and learned_spec is not None:
            for requirement in learned_spec.expert_requirements:
                (required if requirement.required else optional).append(
                    requirement.expert_id
                )
        requested: dict[str, tuple[int, ...]] = {}
        for binding in [*core, *rescue]:
            requested[binding.expert_id] = tuple(semantic.feature_stages)
        if learned_spec is not None:
            for requirement in learned_spec.expert_requirements:
                requested[requirement.expert_id] = tuple(
                    sorted(set(requested.get(requirement.expert_id, ()))
                           | set(requirement.feature_stages))
                )
        return SemanticExpertExecutionPlan(
            bindings_by_id=bindings_by_id,
            requested_stages_by_id=requested,
            core_expert_ids=tuple(binding.expert_id for binding in core),
            rescue_expert_ids=tuple(binding.expert_id for binding in rescue),
            learned_required_expert_ids=tuple(required),
            learned_optional_expert_ids=tuple(optional),
        )

    def _infer_semantic_expert_pool(
        self,
        *,
        plan: SemanticExpertExecutionPlan,
        prepared: ChangePreparedPair,
        np: Any,
    ) -> tuple[dict[str, SemanticExpertRun], list[dict[str, str]]]:
        expected_size = (prepared.raw_t1.shape[1], prepared.raw_t1.shape[0])
        runs_by_id: dict[str, SemanticExpertRun] = {}
        failures: list[dict[str, str]] = []
        planned_ids = tuple(dict.fromkeys(
            [*plan.core_expert_ids, *plan.rescue_expert_ids,
             *plan.learned_required_expert_ids, *plan.learned_optional_expert_ids]
        ))
        for expert_id in planned_ids:
            binding = plan.bindings_by_id.get(expert_id)
            if binding is None:
                continue
            stages = plan.requested_stages_by_id.get(
                expert_id, tuple(self._settings.semantic.feature_stages)
            )
            try:
                runs_by_id[expert_id] = _infer_semantic_expert_pair(
                    binding,
                    prepared,
                    self._settings.semantic,
                    expected_size=expected_size,
                    requested_stages=stages,
                    np=np,
                )
            except Exception as error:
                if (
                    expert_id in plan.core_expert_ids
                    and len(plan.core_expert_ids) == 1
                ):
                    raise
                failures.append({
                    "expert_id": expert_id,
                    "error_type": type(error).__name__,
                })
        return runs_by_id, failures

    def run_full(self, prepared: ChangePreparedPair) -> ChangePerceptionExecution:
        """Run perception and consume the same expert pool for rescue."""
        previous = self._include_rescue_in_pool
        self._include_rescue_in_pool = True
        try:
            result = self.run(prepared)
        finally:
            self._include_rescue_in_pool = previous
        rescue_candidates, rescue_diagnostics = self.run_rescue_candidates(prepared)
        return ChangePerceptionExecution(
            result=result,
            rescue_candidates=tuple(rescue_candidates),
            rescue_diagnostics=rescue_diagnostics,
            learned_request=self._last_learned_request,
        )

    def run_rescue_candidates(
        self, prepared: ChangePreparedPair
    ) -> tuple[list[StructuralRescueCandidate], dict[str, object]]:
        """Run building-only rescue inference outside the core fusion path."""

        np = _require_numpy()
        rescue_settings = self._settings.building_rescue
        if not rescue_settings.enabled:
            return [], {"status": "disabled", "shadow_only": rescue_settings.shadow_only}
        if (
            prepared.comparison_t1 is None
            or prepared.comparison_t2 is None
            or not prepared.validation.valid
        ):
            return [], {"status": "unavailable", "reason_code": "INVALID_CHANGE_PAIR"}
        rescue_bindings = tuple(
            binding
            for binding in self._semantic_experts
            if binding.participation == "rescue"
        )
        if not rescue_bindings:
            return [], {"status": "unavailable", "reason_code": "RESCUE_EXPERT_MISSING"}

        expected_size = (prepared.raw_t1.shape[1], prepared.raw_t1.shape[0])
        requested_stages = tuple(self._settings.semantic.feature_stages)
        candidates: list[StructuralRescueCandidate] = []
        failures: list[dict[str, str]] = []
        expert_diagnostics: list[dict[str, object]] = []
        component_diagnostics: list[dict[str, object]] = []
        cached_runs = (
            self._last_runs_by_id if self._last_prepared is prepared else {}
        )
        for binding in rescue_bindings:
            try:
                run = cached_runs.get(binding.expert_id)
                if run is None:
                    run = _infer_semantic_expert_pair(
                        binding,
                        prepared,
                        self._settings.semantic,
                        expected_size=expected_size,
                        requested_stages=requested_stages,
                        np=np,
                    )
                extracted = _extract_building_rescue_candidates(
                    run,
                    prepared,
                    rescue_settings,
                    np=np,
                    component_diagnostics=component_diagnostics,
                )
                label_names = [str(name).casefold() for name in run.first_output.class_names]
                rescue_labels = run.binding.rescue_model_labels or frozenset({"building"})
                class_index = next(
                    (
                        index
                        for index, name in enumerate(label_names)
                        if name in {label.casefold() for label in rescue_labels}
                    ),
                    None,
                )
                expert_diagnostics.append(
                    {
                        "expert_id": binding.expert_id,
                        "expert_model": run.identity.model,
                        "weights_sha256": run.weights_sha256,
                        "label": "building",
                        "class_index": class_index,
                        "status": "success",
                        "candidate_count": len(extracted),
                        "registration_tolerance_px": _rescue_tolerance_px(
                            prepared, rescue_settings
                        ),
                    }
                )
                candidates.extend(extracted)
            except Exception as error:
                failures.append(
                    {"expert_id": binding.expert_id, "error_type": type(error).__name__}
                )
        candidates = _rank_rescue_candidates(candidates, rescue_settings.max_candidates)
        return candidates, {
            "status": "success" if not failures else "partial",
            "shadow_only": rescue_settings.shadow_only,
            "expert_ids": [binding.expert_id for binding in rescue_bindings],
            "experts": expert_diagnostics,
            "component_diagnostics": component_diagnostics,
            "candidate_count": len(candidates),
            "edge_candidate_count": sum(bool(item.edge_flags) for item in candidates),
            "failures": failures,
        }

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
            expected_size = (prepared.raw_t1.shape[1], prepared.raw_t1.shape[0])
            plan = self._build_execution_plan()
            if not plan.core_expert_ids:
                raise ChangePerceptionError("SEGFORMER_CORE_EXPERT_MISSING")
            runs_by_id, expert_failures = self._infer_semantic_expert_pool(
                plan=plan, prepared=prepared, np=np
            )
            self._last_runs_by_id = runs_by_id
            self._last_prepared = prepared
            semantic_runs = [
                runs_by_id[expert_id]
                for expert_id in plan.core_expert_ids
                if expert_id in runs_by_id
            ]
            core_experts = tuple(
                plan.bindings_by_id[expert_id] for expert_id in plan.core_expert_ids
            )
            if len(semantic_runs) < semantic_settings.min_successful_experts:
                if len(core_experts) == 1 and expert_failures:
                    raise ChangePerceptionError(
                        expert_failures[0].get(
                            "reason_code", "SEGFORMER_ALL_EXPERTS_FAILED"
                        )
                    )
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
                    if len(core_experts) == 1:
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
                        requested_stages=plan.requested_stages_by_id[run.binding.expert_id],
                        expected_size=expected_size,
                        np=np,
                    )
                    if (
                        feature_result is None
                        and prepared.pif_valid
                        and semantic_settings.failure_policy == "fail"
                        and len(core_experts) == 1
                    ):
                        raise ChangePerceptionError("FEATURE_RESIDUAL_INSUFFICIENT_PIF")
                except Exception as error:
                    if len(core_experts) == 1:
                        raise
                    feature_result = None
                    feature_diagnostics = {
                        "alignment_status": "failed",
                        "reason_code": _fallback_reason_code(error)
                        or "FEATURE_RESIDUAL_INFERENCE_FAILED",
                        "valid_feature_fraction": 0.0,
                        "effective_stages": [],
                        "missing_stages": list(
                            plan.requested_stages_by_id[run.binding.expert_id]
                        ),
                    }
                    expert_failures.append(
                        {
                            "expert_id": run.binding.expert_id,
                            "error_type": type(error).__name__,
                            "branch": "feature",
                        }
                    )
                stability_diagnostics = compute_temporal_semantic_stability(
                    run.first_output.probabilities,
                    run.second_output.probabilities,
                    pif_mask=prepared.pif_mask,
                    registration_valid_mask=getattr(
                        prepared, "registration_valid_mask", None
                    ),
                    class_names=run.first_output.class_names,
                    neutral_labels=run.binding.neutral_labels,
                    score_map=semantic_result.score_map,
                    enabled=semantic_settings.temporal_stability_enabled,
                    soft_flip_rate=semantic_settings.temporal_stability_soft_flip_rate,
                    hard_flip_rate=semantic_settings.temporal_stability_hard_flip_rate,
                    floor=semantic_settings.temporal_stability_floor,
                )
                semantic_result.diagnostics.update(stability_diagnostics)
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
            selected_run = min(
                (item.run for item in expert_evidence),
                key=lambda item: (-item.binding.priority, item.binding.expert_id),
            )
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
            learned_output, learned_diagnostics = self._run_learned_change_hook(
                runs_by_id=runs_by_id,
                prepared=prepared,
            )
            learned_map = (
                learned_output.probability_map if learned_output is not None else None
            )
            fusion_reliability = dict(reliability)
            if learned_output is not None:
                fusion_reliability["learned"] = float(learned_output.reliability)
            learned_mode = self._settings.learned_change.mode
            if not isinstance(
                getattr(self._learned_change_client, "input_spec", None),
                LearnedChangeInputSpec,
            ):
                learned_mode = "assist"
            rescue_threshold = None
            rescue_area = None
            try:
                calibration_diagnostics = (
                    learned_output.diagnostics if learned_output is not None else {}
                )
                rescue_threshold = calibration_diagnostics.get("rescue_probability_threshold")
                threshold_override = self._settings.learned_change.rescue.probability_threshold_override
                if rescue_threshold is not None and threshold_override is not None:
                    if self._settings.learned_change.strict_contract and threshold_override < float(rescue_threshold):
                        raise ChangePerceptionError("LEARNED_CHANGE_CALIBRATION_INVALID")
                    rescue_threshold = max(float(rescue_threshold), float(threshold_override))
                rescue_area = calibration_diagnostics.get("rescue_min_component_area_ratio")
                area_override = self._settings.learned_change.rescue.min_component_area_ratio_override
                if rescue_area is not None and area_override is not None:
                    if self._settings.learned_change.strict_contract and area_override < float(rescue_area):
                        raise ChangePerceptionError("LEARNED_CHANGE_CALIBRATION_INVALID")
                    rescue_area = max(float(rescue_area), float(area_override))
            except (ChangePerceptionError, RuntimeError, ValueError) as error:
                reason_code = _fallback_reason_code(error) or "LEARNED_CHANGE_CALIBRATION_INVALID"
                if self._settings.learned_change.failure_policy == "fail":
                    raise LearnedChangeFailure(reason_code) from None
                learned_output = None
                learned_map = None
                fusion_reliability.pop("learned", None)
                learned_mode = "disabled"
                learned_diagnostics = {
                    "enabled": True,
                    "status": "fallback",
                    "available": False,
                    "reason_codes": [reason_code],
                    "fusion_weight": self._settings.learned_change.fusion_weight,
                }
            try:
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
                    reliability=fusion_reliability,
                    valid_overlap_mask=getattr(prepared, "registration_valid_mask", None),
                    registration_confidence=reliability["registration"],
                    learned_map=learned_map,
                    learned_weight=self._settings.learned_change.fusion_weight,
                    learned_requested=self._settings.learned_change.enabled,
                    learned_mode=learned_mode,
                    learned_rescue_threshold=(
                        float(rescue_threshold) if rescue_threshold is not None else None
                    ),
                    learned_rescue_min_reliability=self._settings.learned_change.rescue.min_reliability,
                    learned_rescue_min_component_area_ratio=(
                        float(rescue_area) if rescue_area is not None else None
                    ),
                    learned_rescue_max_proposals=self._settings.learned_change.rescue.max_rescue_proposals,
                )
            except (ChangePerceptionError, RuntimeError, ValueError) as error:
                reason_code = _fallback_reason_code(error)
                if learned_output is None or reason_code is None:
                    raise
                if self._settings.learned_change.failure_policy == "fail":
                    raise LearnedChangeFailure(reason_code) from None
                learned_output = None
                learned_map = None
                fusion_reliability.pop("learned", None)
                learned_mode = "disabled"
                learned_diagnostics = {
                    "enabled": True,
                    "status": "fallback",
                    "available": False,
                    "reason_codes": [reason_code],
                    "fusion_weight": self._settings.learned_change.fusion_weight,
                }
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
                    reliability=fusion_reliability,
                    valid_overlap_mask=getattr(prepared, "registration_valid_mask", None),
                    registration_confidence=reliability["registration"],
                    learned_map=None,
                    learned_weight=self._settings.learned_change.fusion_weight,
                    learned_requested=self._settings.learned_change.enabled,
                    learned_mode="disabled",
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
                        "participation": item.run.binding.participation,
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
                "semantic_ensemble": {
                    "selected_experts": [
                        item.binding.expert_id for item in semantic_runs[: semantic_settings.max_experts]
                    ],
                    "successful_experts": [
                        item.run.binding.expert_id for item in expert_evidence
                    ],
                    "failed_experts": list(expert_failures),
                    "per_expert": [
                        item.run.binding.expert_id for item in expert_evidence
                    ],
                    "semantic_merge": dict(semantic_fusion_diagnostics),
                    "feature_merge": dict(feature_fusion_diagnostics),
                },
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
        runs_by_id: Mapping[str, SemanticExpertRun],
        prepared: ChangePreparedPair,
    ) -> tuple[LearnedChangeOutput | None, dict[str, object]]:
        """Run the explicit learned-head seam without fabricating a map.

        The head is an inference-only dependency.  Its concrete runtime and
        checkpoint identity stay outside ``agents/``; when unavailable, the
        deterministic rule branches remain the complete fallback path.
        """

        settings = self._settings.learned_change
        self._last_learned_request = None
        if not settings.enabled:
            return None, {
                "enabled": False,
                "status": "disabled",
                "available": False,
                "reason_codes": ["LEARNED_CHANGE_DISABLED"],
                "fusion_weight": settings.fusion_weight,
                "mode": settings.mode,
            }
        if self._learned_change_client is None:
            return self._learned_change_failure("LEARNED_CHANGE_CLIENT_MISSING")
        try:
            identity = require_model_cache_identity(
                self._learned_change_client,
                component="learned change client",
            )
            input_spec = getattr(self._learned_change_client, "input_spec", None)
            missing_optional: list[str] = []
            if isinstance(input_spec, LearnedChangeInputSpec):
                missing_required = [
                    requirement.expert_id
                    for requirement in input_spec.expert_requirements
                    if requirement.required and requirement.expert_id not in runs_by_id
                ]
                if missing_required:
                    return self._learned_change_failure(
                        "LEARNED_CHANGE_REQUIRED_EXPERT_MISSING"
                    )
                for requirement in input_spec.expert_requirements:
                    if requirement.required or requirement.expert_id in runs_by_id:
                        continue
                    if (
                        requirement.missing_policy != "zero_with_presence_mask"
                        or not input_spec.optional_expert_dropout_supported
                    ):
                        return self._learned_change_failure(
                            "LEARNED_CHANGE_REQUIRED_EXPERT_MISSING"
                        )
                    missing_optional.append(requirement.expert_id)
                request = _build_learned_change_request(
                    input_spec=input_spec,
                    runs_by_id=runs_by_id,
                    prepared=prepared,
                )
                self._last_learned_request = request
                output = self._learned_change_client.infer(request)
            else:
                # Compatibility bridge for one release: old injected clients
                # are still deterministic test seams, while production clients
                # must publish input_spec and accept LearnedChangeRequest.
                primary = min(
                    runs_by_id.values(),
                    key=lambda item: (-item.binding.priority, item.binding.expert_id),
                )
                output = self._learned_change_client.infer(
                    first=_as_pyramid_output(
                        primary.first_output,
                        feature_stage=self._settings.semantic.feature_stage,
                    ),
                    second=_as_pyramid_output(
                        primary.second_output,
                        feature_stage=self._settings.semantic.feature_stage,
                    ),
                    valid_mask=getattr(prepared, "registration_valid_mask", None),
                )
            if not isinstance(output, LearnedChangeOutput):
                raise ChangePerceptionError("LEARNED_CHANGE_OUTPUT_INVALID")
            np = _require_numpy()
            probability_map = np.asarray(output.probability_map)
            if (
                probability_map.ndim != 2
                or not bool(np.isfinite(probability_map).all())
                or bool((probability_map < -1e-6).any())
                or bool((probability_map > 1.0 + 1e-6).any())
            ):
                raise ChangePerceptionError("LEARNED_CHANGE_OUTPUT_INVALID")
            reliability = float(output.reliability)
            if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
                raise ChangePerceptionError("LEARNED_CHANGE_OUTPUT_INVALID")
            if output.uncertainty_map is not None:
                uncertainty = np.asarray(output.uncertainty_map)
                if uncertainty.shape != probability_map.shape or not bool(
                    np.isfinite(uncertainty).all()
                ):
                    raise ChangePerceptionError("LEARNED_CHANGE_OUTPUT_INVALID")
            if bool((probability_map < 0.0).any()) or bool(
                (probability_map > 1.0).any()
            ):
                output = LearnedChangeOutput(
                    probability_map=np.clip(probability_map, 0.0, 1.0).astype(
                        np.float32, copy=False
                    ),
                    reliability=reliability,
                    diagnostics=output.diagnostics,
                    uncertainty_map=output.uncertainty_map,
                )
            return output, {
                "enabled": True,
                "status": "available",
                "available": True,
                "reason_codes": [],
                "fusion_weight": settings.fusion_weight,
                "mode": settings.mode,
                "model": identity.model,
                "revision": identity.revision,
                "client_version": identity.client_version,
                "required_experts": (
                    [item.expert_id for item in input_spec.expert_requirements if item.required]
                    if isinstance(input_spec, LearnedChangeInputSpec)
                    else []
                ),
                "optional_experts": (
                    [item.expert_id for item in input_spec.expert_requirements if not item.required]
                    if isinstance(input_spec, LearnedChangeInputSpec)
                    else []
                ),
                "available_experts": sorted(runs_by_id),
                "missing_optional_experts": missing_optional,
                "reliability": reliability,
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


def infer_semantic_expert_pair(
    binding: SemanticExpertBinding,
    prepared: ChangePreparedPair,
    settings: Any,
    *,
    requested_stages: tuple[int, ...],
) -> SemanticExpertRun:
    """Run one verified production expert pair for offline consumers.

    Training cache construction uses this narrow public adapter so it shares
    the exact T1/T2 validation, preprocessing, feature-stage, and client
    execution path used by online perception.
    """

    np = _require_numpy()
    if prepared.raw_t1 is None or prepared.raw_t2 is None:
        raise ChangePerceptionError("INVALID_CHANGE_PAIR")
    expected_size = (prepared.raw_t1.shape[1], prepared.raw_t1.shape[0])
    return _infer_semantic_expert_pair(
        binding,
        prepared,
        settings,
        expected_size=expected_size,
        requested_stages=requested_stages,
        np=np,
    )


def _build_learned_change_request(
    *,
    input_spec: LearnedChangeInputSpec,
    runs_by_id: Mapping[str, SemanticExpertRun],
    prepared: ChangePreparedPair,
) -> LearnedChangeRequest:
    """Map verified semantic runs to the explicit learned ABI."""

    pairs: dict[str, LearnedChangeExpertPair] = {}
    expected_size = (prepared.raw_t1.shape[1], prepared.raw_t1.shape[0])
    for requirement in input_spec.expert_requirements:
        run = runs_by_id.get(requirement.expert_id)
        if run is None:
            if requirement.required:
                raise ChangePerceptionError("LEARNED_CHANGE_REQUIRED_EXPERT_MISSING")
            continue
        if run.binding.expert_id != requirement.expert_id:
            raise ChangePerceptionError("LEARNED_CHANGE_EXPERT_ID_MISMATCH")
        if run.identity.model != requirement.logical_model_id:
            raise ChangePerceptionError("LEARNED_CHANGE_LOGICAL_MODEL_MISMATCH")
        if run.weights_sha256 != requirement.weights_sha256:
            raise ChangePerceptionError("LEARNED_CHANGE_BACKBONE_HASH_MISMATCH")
        first = _as_pyramid_output(
            run.first_output, feature_stage=requirement.feature_stages[0]
        )
        second = _as_pyramid_output(
            run.second_output, feature_stage=requirement.feature_stages[0]
        )
        if tuple(first.class_names) != tuple(second.class_names):
            raise ChangePerceptionError("LEARNED_CHANGE_CLASS_MAP_MISMATCH")
        if hash_class_names(first.class_names) != requirement.class_names_sha256:
            raise ChangePerceptionError("LEARNED_CHANGE_CLASS_MAP_MISMATCH")
        if tuple(first.original_size) != expected_size or tuple(second.original_size) != expected_size:
            raise ChangePerceptionError("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH")
        for stage in requirement.feature_stages:
            if stage not in first.features_by_stage or stage not in second.features_by_stage:
                raise ChangePerceptionError("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH")
            if first.feature_strides_by_stage[stage] != second.feature_strides_by_stage[stage]:
                raise ChangePerceptionError("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH")
        if first.weights_sha256 != second.weights_sha256:
            raise ChangePerceptionError("LEARNED_CHANGE_BACKBONE_HASH_MISMATCH")
        pairs[requirement.expert_id] = LearnedChangeExpertPair(
            expert_id=requirement.expert_id,
            logical_model_id=run.identity.model,
            weights_sha256=run.weights_sha256 or "",
            class_names_sha256=hash_class_names(first.class_names),
            first=first,
            second=second,
        )
    return LearnedChangeRequest(
        image_size=expected_size,
        experts=pairs,
        valid_mask=prepared.registration_valid_mask,
        pif_mask=prepared.pif_mask if input_spec.use_pif_mask else None,
        pif_valid=bool(prepared.pif_valid),
        comparison_t1=prepared.comparison_t1 if input_spec.use_rgb_pair else None,
        comparison_t2=prepared.comparison_t2 if input_spec.use_rgb_pair else None,
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


def _extract_building_rescue_candidates(
    run: SemanticExpertRun,
    prepared: ChangePreparedPair,
    settings: Any,
    *,
    np: Any,
    component_diagnostics: list[dict[str, object]] | None = None,
) -> list[StructuralRescueCandidate]:
    """Extract conservative added/removed building footprint candidates."""

    label_names = {str(name).casefold(): str(name) for name in run.first_output.class_names}
    requested_labels = run.binding.rescue_model_labels or frozenset({"building"})
    label = next(
        (label for label in requested_labels if label.casefold() in label_names),
        None,
    )
    if label is None:
        return []
    index = next(
        index
        for index, name in enumerate(run.first_output.class_names)
        if str(name).casefold() == label.casefold()
    )
    first_probability = _resize_probability_map(
        np.asarray(run.first_output.probabilities)[index],
        tuple(run.first_output.original_size),
        np=np,
    )
    second_probability = _resize_probability_map(
        np.asarray(run.second_output.probabilities)[index],
        tuple(run.second_output.original_size),
        np=np,
    )
    height, width = first_probability.shape
    tolerance = _rescue_tolerance_px(prepared, settings)
    first_mask = first_probability >= settings.building_probability_threshold
    second_mask = second_probability >= settings.building_probability_threshold
    first_tolerant = _binary_dilate(first_mask, tolerance, np=np)
    second_tolerant = _binary_dilate(second_mask, tolerance, np=np)
    valid_mask = getattr(prepared, "registration_valid_mask", None)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != first_mask.shape or not np.any(valid_mask):
            return []

    output: list[StructuralRescueCandidate] = []
    diagnostics = component_diagnostics if component_diagnostics is not None else []
    for direction, component_mask, target, source in (
        ("added", second_mask & ~first_tolerant, second_probability, first_probability),
        ("removed", first_mask & ~second_tolerant, first_probability, second_probability),
    ):
        for component_index, component in enumerate(_binary_components(component_mask, np=np)):
            ys, xs = np.where(component)
            if not len(xs):
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            area = int(component.sum())
            area_ratio = area / float(height * width)
            edge_flags = _rescue_edge_flags(
                x0, y0, x1, y1, width=width, height=height, margin=settings.edge_margin_ratio
            )
            min_area = (
                settings.min_component_area_ratio_edge
                if edge_flags
                else settings.min_component_area_ratio
            )
            target_values = target[component]
            source_values = source[component]
            target_p10 = float(np.percentile(target_values, 10))
            target_p50 = float(np.percentile(target_values, 50))
            source_mean = float(np.mean(source_values))
            source_p90 = float(np.percentile(source_values, 90))
            source_p95 = float(np.percentile(source_values, 95))
            source_max = float(np.max(source_values))
            target_mean = float(np.mean(target_values))
            registration_valid_ratio = (
                float(np.mean(component & valid_mask)) if valid_mask is not None else 1.0
            )
            rejection_reason: str | None = None
            if direction not in settings.allowed_directions:
                rejection_reason = "DIRECTION_DISABLED"
            elif settings.edge_only and not edge_flags:
                rejection_reason = "EDGE_ONLY_DISABLED"
            elif registration_valid_ratio <= 0.0:
                rejection_reason = "REGISTRATION_INVALID"
            elif area_ratio < min_area:
                rejection_reason = "AREA_TOO_SMALL"
            elif area_ratio > settings.max_component_area_ratio:
                rejection_reason = "AREA_TOO_LARGE"
            elif target_mean < settings.building_probability_threshold:
                rejection_reason = "TARGET_MEAN_TOO_LOW"
            elif target_p10 < settings.building_probability_threshold * 0.75:
                rejection_reason = "TARGET_P10_TOO_LOW"
            elif source_max > settings.source_absence_probability_max:
                rejection_reason = "SOURCE_ABSENCE_FAILED"
            record = {
                "direction": direction,
                "candidate_direction": direction,
                "component_id": f"{run.binding.expert_id}:{direction}:{component_index}",
                "area_px": area,
                "area_ratio": area_ratio,
                "target_mean_building_probability": target_mean,
                "target_p10_building_probability": target_p10,
                "target_p50_building_probability": target_p50,
                "source_mean_building_probability": source_mean,
                "source_p90_building_probability": source_p90,
                "source_p95_building_probability": source_p95,
                "source_max_building_probability": source_max,
                "registration_tolerance_px": tolerance,
                "touches_top": "top" in edge_flags,
                "touches_bottom": "bottom" in edge_flags,
                "touches_left": "left" in edge_flags,
                "touches_right": "right" in edge_flags,
                "touches_corner": "corner" in edge_flags,
                "registration_valid_ratio": registration_valid_ratio,
                "accepted": rejection_reason is None,
                "rejection_reason": rejection_reason,
            }
            diagnostics.append(record)
            if rejection_reason is not None:
                continue
            score = float(
                np.clip(
                    0.60 * target_mean
                    + 0.30 * (1.0 - float(np.mean(source_values)))
                    + 0.10 * min(1.0, area_ratio / max(min_area, 1e-9)),
                    0.0,
                    1.0,
                )
            )
            output.append(
                StructuralRescueCandidate(
                    candidate_id=(
                        f"{run.binding.expert_id}:{direction}:{component_index}:"
                        f"{x0}-{y0}-{x1}-{y1}"
                    ),
                    expert_id=run.binding.expert_id,
                    direction=direction,
                    box=(x0, y0, x1, y1),
                    normalized_box=(
                        round(999 * x0 / width),
                        round(999 * y0 / height),
                        round(999 * x1 / width),
                        round(999 * y1 / height),
                    ),
                    score=score,
                    target_mean_probability=target_mean,
                    target_p10_probability=target_p10,
                    target_p50_probability=target_p50,
                    source_mean_probability=source_mean,
                    source_p90_probability=source_p90,
                    source_p95_probability=source_p95,
                    source_max_probability=source_max,
                    area_px=area,
                    area_ratio=area_ratio,
                    edge_flags=tuple(edge_flags),
                    registration_tolerance_px=tolerance,
                )
            )
    return output


def _resize_probability_map(probability: Any, original_size: tuple[int, int], *, np: Any) -> Any:
    width, height = (int(original_size[0]), int(original_size[1]))
    array = np.asarray(probability, dtype=np.float32)
    if array.shape == (height, width):
        return array
    image = Image.fromarray(array, mode="F")
    return np.asarray(
        image.resize((width, height), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _binary_dilate(mask: Any, radius: int, *, np: Any) -> Any:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source, radius, mode="constant", constant_values=False)
    result = np.zeros_like(source, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + source.shape[0], dx : dx + source.shape[1]]
    return result


def _binary_components(mask: Any, *, np: Any) -> list[Any]:
    source = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(source, dtype=bool)
    height, width = source.shape
    components: list[Any] = []
    for y, x in np.argwhere(source):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and source[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        component = np.zeros_like(source, dtype=bool)
        ys, xs = zip(*pixels)
        component[list(ys), list(xs)] = True
        components.append(component)
    return components


def _rescue_edge_flags(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    width: int,
    height: int,
    margin: float,
) -> tuple[str, ...]:
    x_margin = width * margin
    y_margin = height * margin
    flags = []
    if x0 <= x_margin:
        flags.append("left")
    if x1 >= width - x_margin:
        flags.append("right")
    if y0 <= y_margin:
        flags.append("top")
    if y1 >= height - y_margin:
        flags.append("bottom")
    if len(flags) >= 2:
        flags.append("corner")
    return tuple(flags)


def _rescue_tolerance_px(prepared: ChangePreparedPair, settings: Any) -> int:
    reprojection_error = 0.0
    report = getattr(prepared, "registration_report", None)
    if report is not None and report.metrics is not None:
        reprojection_error = float(report.metrics.median_reprojection_error)
    return max(
        settings.registration_tolerance_min_px,
        min(
            settings.registration_tolerance_max_px,
            int(math.ceil(reprojection_error * settings.registration_tolerance_error_scale)),
        ),
    )


def _rank_rescue_candidates(
    candidates: list[StructuralRescueCandidate], max_candidates: int
) -> list[StructuralRescueCandidate]:
    ranked = sorted(candidates, key=lambda item: (-item.score, item.candidate_id))
    edge = next((item for item in ranked if item.edge_flags), None)
    selected = ranked[:max_candidates]
    if edge is not None and edge not in selected:
        selected = [*selected[:-1], edge] if selected else [edge]
    return sorted(selected, key=lambda item: (-item.score, item.candidate_id))


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
            primary_evidence = next(iter(expert_evidence), None)
            if primary_evidence is not None:
                image_shape = tuple(
                    np.asarray(primary_evidence.run.first_output.probabilities).shape[1:]
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
            if evidence_type in {
                "persistent",
                "structural_candidate",
                "landcover_candidate",
                "transient",
            }:
                transitions.append(
                    (evidence_type, evidence, transition, evidence_type)
                )
        structural = [
            item
            for item in transitions
            if item[0] in {"persistent", "structural_candidate"}
        ]
        landcover = [item for item in transitions if item[0] == "landcover_candidate"]
        transient = [item for item in transitions if item[0] == "transient"]
        structural_support = max(
            (float(item[2].transition_confidence) for item in structural),
            default=0.0,
        )
        landcover_support = max(
            (float(item[2].transition_confidence) for item in landcover),
            default=0.0,
        )
        transient_support = max(
            (float(item[2].transition_confidence) for item in transient),
            default=0.0,
        )
        structural_pairs = {
            (item[2].from_class, item[2].to_class) for item in structural
        }
        informative = [*structural, *landcover, *transient]
        informative_expert_ids = {
            item[1].run.binding.expert_id for item in informative
        }
        consensus = {
            "structural_support": structural_support,
            "landcover_support": landcover_support,
            "transient_support": transient_support,
            # Compatibility boundary: only structural candidates count as
            # legacy persistent support.  Land-cover-only flips remain
            # visible but cannot authorize a persistent conclusion.
            "persistent_support": structural_support,
            "neutral_expert_count": len(expert_evidence) - len(informative_expert_ids),
            "informative_expert_count": len(informative_expert_ids),
            "disagreement": len(structural_pairs) > 1,
            "expert_count": len(expert_evidence),
        }
        chosen = max(structural, key=lambda item: _transition_sort_key(item[2])) if structural else (
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
    if labels & binding.transient_labels:
        return "transient"
    if labels & binding.structural_labels:
        return "structural_candidate"
    if labels & binding.landcover_candidate_labels:
        return "landcover_candidate"
    # Catalogs predating typed evidence can still use the old persistent
    # label set, but the role itself is never a classification rule.
    if labels & binding.persistent_labels:
        return "persistent"
    return "neutral"


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
    "infer_semantic_expert_pair",
]
