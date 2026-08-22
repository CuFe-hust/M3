"""Torch ChangeHead runtime adapter with explicit compatibility checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.base import (
    LearnedChangeRequest,
    LearnedChangeOutput,
    ModelCacheIdentity,
    hash_class_names,
)
from models.change_head.calibration import ChangeHeadCalibration
from models.change_head.checkpoint import LoadedChangeHeadCheckpoint
from models.change_head.manifest import ChangeHeadManifest
from models.change_head.network import MultiExpertSiameseChangeHead


@dataclass(frozen=True)
class RuntimeCompatibilityReport:
    compatible: bool
    reason_codes: tuple[str, ...]
    missing_optional_experts: tuple[str, ...] = ()


class ChangeHeadRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {message}".strip())


def validate_change_head_runtime_compatibility(
    *,
    manifest: ChangeHeadManifest,
    semantic_experts: tuple[Any, ...],
    pipeline_fingerprint: str,
    strict: bool,
) -> RuntimeCompatibilityReport:
    by_id = {expert.expert_id: expert for expert in semantic_experts}
    reasons: list[str] = []
    missing_optional: list[str] = []
    if manifest.pipeline_fingerprint != pipeline_fingerprint:
        reasons.append("LEARNED_CHANGE_PIPELINE_FINGERPRINT_MISMATCH")
    for requirement in manifest.experts:
        binding = by_id.get(requirement.expert_id)
        if binding is None:
            if requirement.required:
                reasons.append("LEARNED_CHANGE_REQUIRED_EXPERT_MISSING")
            elif requirement.missing_policy == "zero_with_presence_mask" and manifest.architecture.optional_expert_dropout_supported:
                missing_optional.append(requirement.expert_id)
            else:
                reasons.append("LEARNED_CHANGE_REQUIRED_EXPERT_MISSING")
            continue
        if binding.logical_model_id != requirement.logical_model_id:
            reasons.append("LEARNED_CHANGE_LOGICAL_MODEL_MISMATCH")
        actual_weights = getattr(binding, "weights_sha256", None)
        if actual_weights is None or actual_weights != requirement.weights_sha256:
            reasons.append("LEARNED_CHANGE_BACKBONE_HASH_MISMATCH")
        actual_class_hash = getattr(binding, "class_names_sha256", None)
        if actual_class_hash is None and getattr(binding, "class_names", ()):
            actual_class_hash = hash_class_names(binding.class_names)
        if actual_class_hash is None or actual_class_hash != requirement.class_names_sha256:
            reasons.append("LEARNED_CHANGE_CLASS_MAP_MISMATCH")
        supported = getattr(binding.client, "supported_feature_stages", None)
        if supported is not None and not set(requirement.feature_stages).issubset(set(supported)):
            reasons.append("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH")
    report = RuntimeCompatibilityReport(
        compatible=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
        missing_optional_experts=tuple(missing_optional),
    )
    if strict and not report.compatible:
        raise ChangeHeadRuntimeError(report.reason_codes[0], "runtime compatibility mismatch")
    return report


def resolve_torch_device(device: str) -> str:
    try:
        import torch
    except ImportError as error:
        raise ChangeHeadRuntimeError("LEARNED_CHANGE_INFERENCE_FAILED", "torch dependency missing") from error
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return normalized
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ChangeHeadRuntimeError("LEARNED_CHANGE_INFERENCE_FAILED", "requested cuda is unavailable")
        return normalized
    raise ChangeHeadRuntimeError("LEARNED_CHANGE_INFERENCE_FAILED", "unsupported device")


class TorchLearnedChangeClient:
    """Frozen inference adapter; the shared ABI remains torch-free."""

    def __init__(
        self,
        checkpoint: LoadedChangeHeadCheckpoint,
        *,
        device: str = "auto",
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise ChangeHeadRuntimeError("LEARNED_CHANGE_INFERENCE_FAILED", "torch dependency missing") from error
        self._torch = torch
        self._checkpoint = checkpoint
        self._manifest = checkpoint.manifest
        self._calibration: ChangeHeadCalibration = checkpoint.calibration
        self._device = resolve_torch_device(device)
        self._network = MultiExpertSiameseChangeHead(self._manifest)
        try:
            self._network.load_state_dict(checkpoint.state_dict, strict=True)
            self._network.to(self._device)
            self._network.eval()
            for parameter in self._network.parameters():
                parameter.requires_grad_(False)
        except Exception as error:
            raise ChangeHeadRuntimeError("LEARNED_CHANGE_CONTRACT_MISMATCH", "state dict mismatch") from error
        self._identity = ModelCacheIdentity(
            model="M3/ChangeHead:multi_expert_siamese_change_head_v1",
            generation={
                "model_weights_sha256": self._manifest.model_weights_sha256,
                "pipeline_fingerprint": self._manifest.pipeline_fingerprint,
                "input_contract_version": self._manifest.input_contract_version,
                "experts": [
                    {
                        "expert_id": expert.expert_id,
                        "logical_model_id": expert.logical_model_id,
                        "weights_sha256": expert.weights_sha256,
                        "feature_stages": list(expert.feature_stages),
                    }
                    for expert in self._manifest.experts
                ],
            },
            client_version="learned-change-runtime-v2",
            revision=self._manifest.created_from_git_commit,
        )

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return self._identity

    @property
    def input_spec(self):
        return self._manifest.input_spec()

    @staticmethod
    def _tensor(value: Any, *, device: str, name: str) -> Any:
        import numpy as np

        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ChangeHeadRuntimeError("LEARNED_CHANGE_OUTPUT_INVALID", name)
        import torch
        tensor = torch.from_numpy(array.copy())
        return tensor.to(device)

    def infer(self, request: LearnedChangeRequest) -> LearnedChangeOutput:
        import numpy as np
        torch = self._torch
        requirements = self.input_spec.requirement_by_expert_id()
        for key, pair in request.experts.items():
            if key != pair.expert_id or key not in requirements:
                raise ChangeHeadRuntimeError("LEARNED_CHANGE_EXPERT_ID_MISMATCH")
        missing_optional: list[str] = []
        expert_features: dict[str, tuple[Any, Any]] = {}
        semantic_probabilities: dict[str, tuple[Any, Any]] = {}
        presence: dict[str, Any] = {}
        valid = self._tensor(request.valid_mask, device=self._device, name="valid_mask")
        if valid.ndim != 2:
            raise ChangeHeadRuntimeError("LEARNED_CHANGE_OUTPUT_INVALID", "valid mask")
        valid = valid > 0.5
        target_hw = tuple(valid.shape)
        for manifest_expert in self._manifest.experts:
            requirement = requirements[manifest_expert.expert_id]
            pair = request.experts.get(manifest_expert.expert_id)
            if pair is None:
                if requirement.required:
                    raise ChangeHeadRuntimeError("LEARNED_CHANGE_REQUIRED_EXPERT_MISSING")
                missing_optional.append(requirement.expert_id)
                presence[requirement.expert_id] = torch.zeros(1, device=self._device)
                continue
            if pair.logical_model_id != requirement.logical_model_id or pair.weights_sha256 != requirement.weights_sha256:
                raise ChangeHeadRuntimeError("LEARNED_CHANGE_BACKBONE_HASH_MISMATCH")
            if pair.class_names_sha256 != requirement.class_names_sha256:
                raise ChangeHeadRuntimeError("LEARNED_CHANGE_CLASS_MAP_MISMATCH")
            first_features: list[Any] = []
            second_features: list[Any] = []
            for stage in requirement.feature_stages:
                first = self._tensor(pair.first.features_by_stage[stage], device=self._device, name=f"{manifest_expert.expert_id}:{stage}")
                second = self._tensor(pair.second.features_by_stage[stage], device=self._device, name=f"{manifest_expert.expert_id}:{stage}")
                if first.ndim != 3 or second.shape != first.shape or first.shape[0] != manifest_expert.feature_channels_by_stage[stage]:
                    raise ChangeHeadRuntimeError("LEARNED_CHANGE_FEATURE_STAGE_MISMATCH")
                first_features.append(first.unsqueeze(0))
                second_features.append(second.unsqueeze(0))
            expert_features[requirement.expert_id] = (first_features, second_features)
            if requirement.use_semantic_probabilities:
                first_sem = self._tensor(pair.first.probabilities, device=self._device, name="semantic_probabilities")
                second_sem = self._tensor(pair.second.probabilities, device=self._device, name="semantic_probabilities")
                if first_sem.ndim != 3 or second_sem.shape != first_sem.shape:
                    raise ChangeHeadRuntimeError("LEARNED_CHANGE_OUTPUT_INVALID", "semantic probabilities")
                semantic_probabilities[requirement.expert_id] = (first_sem.unsqueeze(0), second_sem.unsqueeze(0))
            presence[requirement.expert_id] = torch.ones(1, device=self._device)
        pif = None
        if self._manifest.architecture.use_pif_mask:
            if request.pif_mask is None:
                raise ChangeHeadRuntimeError(
                    "LEARNED_CHANGE_CONTRACT_MISMATCH",
                    "manifest requires pif_mask",
                )
            pif = self._tensor(request.pif_mask, device=self._device, name="pif_mask")
            if pif.ndim != 2 or tuple(pif.shape) != target_hw:
                raise ChangeHeadRuntimeError(
                    "LEARNED_CHANGE_OUTPUT_INVALID",
                    "pif mask shape",
                )
        elif request.pif_mask is not None:
            raise ChangeHeadRuntimeError(
                "LEARNED_CHANGE_CONTRACT_MISMATCH",
                "pif_mask supplied to a manifest that disables it",
            )
        try:
            with torch.no_grad():
                logits = self._network(
                    expert_features=expert_features,
                    semantic_probabilities=semantic_probabilities,
                    expert_presence=presence,
                    valid_mask=valid.unsqueeze(0),
                    pif_mask=pif.unsqueeze(0) if pif is not None else None,
                )
                probability = torch.sigmoid(logits / self._calibration.temperature)
                probability = probability * valid.unsqueeze(0).unsqueeze(0).to(
                    dtype=probability.dtype
                )
                probability = probability.clamp(0.0, 1.0)[0, 0]
        except ChangeHeadRuntimeError:
            raise
        except Exception as error:
            raise ChangeHeadRuntimeError("LEARNED_CHANGE_INFERENCE_FAILED") from error
        probability_array = probability.detach().to("cpu").numpy().astype(np.float32)
        reliability = self._calibration.validation_reliability
        for _ in missing_optional:
            reliability *= self._calibration.optional_expert_missing_reliability_factor
        uncertainty = -(probability * torch.log(probability.clamp_min(1e-6)) + (1 - probability) * torch.log((1 - probability).clamp_min(1e-6)))
        uncertainty = uncertainty * valid.to(dtype=uncertainty.dtype)
        return LearnedChangeOutput(
            probability_map=probability_array,
            reliability=float(max(0.0, min(1.0, reliability))),
            uncertainty_map=uncertainty.detach().to("cpu").numpy().astype(np.float32),
            diagnostics={
                "missing_optional_experts": missing_optional,
                "device": self._device,
                "rescue_probability_threshold": self._calibration.rescue_probability_threshold,
                "rescue_min_component_area_ratio": self._calibration.rescue_min_component_area_ratio,
            },
        )
