"""Deterministic fusion of low-level, feature, and semantic change maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from agents.change.schema import ChangeProposal, HarmonizationDecision, RegistrationReport
from agents.change.settings import ChangeProposalSettings, ChangeReliabilitySettings
from agents.errors import OptionalDependencyMissingError


PROPOSAL_FUSION_VERSION = "weighted_pif_robust_fusion_v1"
_COMPONENT_NAMES = ("low_level", "feature", "semantic")


def compute_reliabilities(
    *,
    registration_report: RegistrationReport | None,
    feature_diagnostics: Mapping[str, object] | None,
    semantic_diagnostics: Mapping[str, object] | None,
    harmonization_decision: HarmonizationDecision | None,
    settings: ChangeReliabilitySettings,
) -> tuple[dict[str, float], dict[str, object]]:
    """Derive branch reliability from existing, serializable quality metrics."""

    if not settings.enabled:
        reliability = {
            name: 1.0
            for name in ("low_level", "feature", "semantic", "registration")
        }
        return reliability, {"enabled": False, "raw": reliability.copy()}

    raw = {
        "registration": _registration_reliability(
            registration_report, settings=settings
        ),
        "feature": _feature_reliability(
            feature_diagnostics,
            residual_scale=settings.feature_residual_scale,
        ),
        "semantic": _semantic_reliability(
            semantic_diagnostics,
            confidence_floor=settings.semantic_confidence_floor,
        ),
        "low_level": _harmonization_reliability(harmonization_decision),
    }
    reliability = {
        name: _apply_floor(value, settings.min_branch_reliability)
        for name, value in raw.items()
    }
    return reliability, {
        "enabled": True,
        "raw": raw,
        "reliability": reliability,
        "policy": {
            "registration_error_scale": settings.registration_error_scale,
            "min_branch_reliability": settings.min_branch_reliability,
            "semantic_confidence_floor": settings.semantic_confidence_floor,
            "feature_residual_scale": settings.feature_residual_scale,
        },
    }


def _registration_reliability(
    report: RegistrationReport | None,
    *,
    settings: ChangeReliabilitySettings,
) -> float:
    if report is None:
        return 1.0
    decision = report.decision
    if "REGISTRATION_DISABLED" in decision.reason_codes:
        return 1.0
    if decision.used_for_comparison and any(
        code in decision.reason_codes
        for code in (
            "REGISTRATION_NOT_NEEDED",
            "METADATA_ALIGNMENT_USED",
            "IDENTICAL_INPUTS",
        )
    ):
        return 1.0
    if decision.status == "skipped":
        return 0.35
    if decision.status != "applied" or report.metrics is None:
        return 0.20
    metrics = report.metrics
    reprojection = 1.0 / (
        1.0 + metrics.median_reprojection_error / settings.registration_error_scale
    )
    return _clamp(
        (metrics.inlier_ratio + metrics.overlap_ratio + reprojection) / 3.0
    )


def _feature_reliability(
    diagnostics: Mapping[str, object] | None,
    *,
    residual_scale: float,
) -> float:
    if not diagnostics:
        return 0.35
    values: list[float] = []
    cells = diagnostics.get("pif_feature_cells")
    if isinstance(cells, (int, float)) and math.isfinite(float(cells)):
        values.append(_clamp(float(cells) / 64.0))
    ratio = diagnostics.get("pif_feature_ratio")
    if isinstance(ratio, (int, float)) and math.isfinite(float(ratio)):
        values.append(_clamp(float(ratio) * 4.0))
    valid_fraction = diagnostics.get("valid_feature_fraction")
    if isinstance(valid_fraction, (int, float)) and math.isfinite(float(valid_fraction)):
        values.append(_clamp(float(valid_fraction)))
    residual = diagnostics.get("median_score_pif")
    if isinstance(residual, (int, float)) and math.isfinite(float(residual)):
        values.append(_clamp(1.0 - float(residual) / residual_scale))
    effective = diagnostics.get("effective_stages")
    missing = diagnostics.get("missing_stages")
    if isinstance(effective, list) and isinstance(missing, list):
        total = len(effective) + len(missing)
        values.append(float(len(effective) / total) if total else 0.0)
    if diagnostics.get("alignment_status") == "insufficient_pif":
        values.append(0.0)
    return sum(values) / len(values) if values else 0.35


def _semantic_reliability(
    diagnostics: Mapping[str, object] | None,
    *,
    confidence_floor: float,
) -> float:
    if not diagnostics:
        return 0.35
    confidence = diagnostics.get("mean_confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        return 0.35
    denominator = max(1e-6, 1.0 - confidence_floor)
    value = _clamp((float(confidence) - confidence_floor) / denominator)
    valid_fraction = diagnostics.get("valid_pixel_fraction")
    if isinstance(valid_fraction, (int, float)) and math.isfinite(float(valid_fraction)):
        value *= _clamp(float(valid_fraction))
    return value


def _harmonization_reliability(decision: HarmonizationDecision | None) -> float:
    if decision is None or decision.status == "skipped":
        return 0.75
    if decision.status in {"rejected", "failed"}:
        return 0.25
    metrics = decision.metrics
    if metrics is None or metrics.mad_pif_before <= 1e-6:
        return 0.75
    if metrics.mad_pif_after <= metrics.mad_pif_before:
        return 1.0
    return _clamp(metrics.mad_pif_before / metrics.mad_pif_after)


def _apply_floor(value: float, floor: float) -> float:
    return _clamp(floor + (1.0 - floor) * _clamp(value))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise OptionalDependencyMissingError("change", dependency="numpy") from error
    return np


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise OptionalDependencyMissingError(
            "change", dependency="opencv-python-headless"
        ) from error
    return cv2


@dataclass(frozen=True)
class ProposalFusionResult:
    """Fused grids, retained proposals, and crop-local component masks."""

    fused_score_map: Any
    binary_change_mask: Any
    proposals: list[ChangeProposal]
    diagnostics: dict[str, object]
    component_masks: dict[str, Any]


def fuse_semantic_evidence(
    score_maps: Sequence[Any],
    reliabilities: Sequence[float] | None = None,
    *,
    consensus_weight: float = 0.65,
    union_weight: float = 0.35,
) -> tuple[Any, dict[str, object]]:
    """Fuse already taxonomy-independent semantic change maps.

    This function intentionally accepts score maps only; it never receives or
    averages expert class probabilities.
    """

    return _fuse_expert_maps(
        score_maps,
        reliabilities,
        mode="semantic",
        consensus_weight=consensus_weight,
        union_weight=union_weight,
    )


def fuse_feature_evidence(
    score_maps: Sequence[Any],
    reliabilities: Sequence[float] | None = None,
) -> tuple[Any | None, dict[str, object]]:
    """Reliability-weight feature residual maps without requiring all experts."""

    if not score_maps:
        return None, {"method": "none", "expert_count": 0}
    fused, diagnostics = _fuse_expert_maps(
        score_maps,
        reliabilities,
        mode="feature",
        consensus_weight=1.0,
        union_weight=0.0,
    )
    diagnostics["method"] = "reliability_weighted_mean"
    return fused, diagnostics


def _fuse_expert_maps(
    score_maps: Sequence[Any],
    reliabilities: Sequence[float] | None,
    *,
    mode: str,
    consensus_weight: float,
    union_weight: float,
) -> tuple[Any, dict[str, object]]:
    np = _require_numpy()
    cv2 = _require_cv2()
    if not score_maps:
        raise ValueError("CHANGE_EXPERT_MAPS_EMPTY")
    if consensus_weight < 0.0 or union_weight < 0.0 or consensus_weight + union_weight <= 0.0:
        raise ValueError("CHANGE_EXPERT_FUSION_WEIGHTS_INVALID")
    weights = [1.0] * len(score_maps) if reliabilities is None else [
        _clamp(value) for value in reliabilities
    ]
    if len(weights) != len(score_maps) or not any(value > 0.0 for value in weights):
        raise ValueError("CHANGE_EXPERT_RELIABILITIES_INVALID")
    first = np.asarray(score_maps[0], dtype=np.float32)
    if first.ndim != 2 or not bool(np.isfinite(first).all()):
        raise ValueError("CHANGE_EXPERT_MAP_INVALID")
    height, width = first.shape
    maps = []
    for value in score_maps:
        current = np.asarray(value, dtype=np.float32)
        if current.ndim != 2 or not bool(np.isfinite(current).all()):
            raise ValueError("CHANGE_EXPERT_MAP_INVALID")
        if current.shape != (height, width):
            current = cv2.resize(current, (width, height), interpolation=cv2.INTER_LINEAR)
        maps.append(np.clip(current, 0.0, 1.0))
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    stack = np.stack(maps, axis=0)
    weighted_mean = np.sum(stack * normalized[:, None, None], axis=0)
    weighted_max = np.max(
        stack * np.asarray(weights, dtype=np.float32)[:, None, None],
        axis=0,
    )
    total = consensus_weight + union_weight
    merged = (
        consensus_weight * weighted_mean + union_weight * weighted_max
    ) / total
    return np.clip(merged, 0.0, 1.0).astype(np.float32, copy=False), {
        "method": (
            "semantic_consensus_union"
            if mode == "semantic"
            else "reliability_weighted_mean"
        ),
        "expert_count": len(maps),
        "weights": [float(value) for value in normalized],
        "consensus_weight": float(consensus_weight / total),
        "union_weight": float(union_weight / total),
    }


def fuse_change_proposals(
    low_level_map: Any,
    feature_map: Any | None,
    semantic_map: Any | None,
    pif_mask: Any,
    settings: ChangeProposalSettings,
    *,
    min_pif_pixels: int = 32,
    fallback_reason: str | None = None,
    reliability: Mapping[str, float] | None = None,
    valid_overlap_mask: Any | None = None,
    registration_confidence: float | None = None,
    learned_map: Any | None = None,
    learned_weight: float = 0.0,
    learned_requested: bool = False,
) -> ProposalFusionResult:
    """Fuse available score maps and extract deterministic V2 proposals."""

    np = _require_numpy()
    cv2 = _require_cv2()
    if isinstance(min_pif_pixels, bool) or not isinstance(min_pif_pixels, int):
        raise ValueError("PROPOSAL_FUSION_MIN_PIF_INVALID")
    if min_pif_pixels < 1:
        raise ValueError("PROPOSAL_FUSION_MIN_PIF_INVALID")
    if fallback_reason is not None and not isinstance(fallback_reason, str):
        raise ValueError("PROPOSAL_FUSION_FALLBACK_REASON_INVALID")
    if (
        isinstance(learned_weight, bool)
        or not isinstance(learned_weight, (int, float))
        or not math.isfinite(float(learned_weight))
        or float(learned_weight) < 0.0
    ):
        raise ValueError("PROPOSAL_FUSION_LEARNED_WEIGHT_INVALID")

    low_level = _validate_score_map(low_level_map, name="low_level", np=np)
    height, width = low_level.shape
    supplied_maps = {
        "low_level": low_level,
        "feature": feature_map,
        "semantic": semantic_map,
    }
    canonical_maps: dict[str, Any] = {"low_level": np.clip(low_level, 0.0, 1.0)}
    for name in ("feature", "semantic"):
        value = supplied_maps[name]
        if value is None:
            continue
        validated = _validate_score_map(value, name=name, np=np)
        canonical_maps[name] = np.clip(
            cv2.resize(validated, (width, height), interpolation=cv2.INTER_LINEAR),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

    requested_component_names = list(_COMPONENT_NAMES)
    if learned_requested or learned_map is not None:
        requested_component_names.append("learned")
    if learned_map is not None:
        validated_learned = _validate_score_map(learned_map, name="learned", np=np)
        canonical_maps["learned"] = np.clip(
            cv2.resize(
                validated_learned,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            ),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

    available_components = [
        name for name in requested_component_names if name in canonical_maps
    ]
    missing_components = [
        name for name in requested_component_names if name not in canonical_maps
    ]
    configured_weights = {
        "low_level": float(settings.fusion_low_level_weight),
        "feature": float(settings.fusion_feature_weight),
        "semantic": float(settings.fusion_semantic_weight),
    }
    if "learned" in requested_component_names:
        configured_weights["learned"] = float(learned_weight)
    resolved_reliability = {
        name: _reliability_value(
            None if reliability is None else reliability.get(name, 1.0),
            name=name,
        )
        for name in (*requested_component_names, "registration")
    }
    available_weight_sum = sum(
        configured_weights[name] * resolved_reliability[name]
        for name in available_components
    )
    if available_weight_sum <= 0.0:
        raise ValueError("PROPOSAL_FUSION_NO_AVAILABLE_WEIGHT")
    effective_weights = {
        name: configured_weights[name] * resolved_reliability[name] / available_weight_sum
        for name in available_components
    }
    resolved_registration_confidence = (
        resolved_reliability["registration"]
        if registration_confidence is None
        else _reliability_value(registration_confidence, name="registration")
    )
    fused_score = np.zeros((height, width), dtype=np.float32)
    for name in available_components:
        fused_score += np.float32(effective_weights[name]) * canonical_maps[name]
    # Registration quality is a global confidence gate.  It is intentionally
    # applied after branch normalization so it cannot be cancelled by weight
    # renormalization.
    fused_score = np.clip(
        fused_score * np.float32(resolved_registration_confidence), 0.0, 1.0
    ).astype(np.float32, copy=False)

    valid_overlap = _resize_mask(
        valid_overlap_mask,
        width=width,
        height=height,
        cv2=cv2,
        np=np,
    )
    if valid_overlap is not None:
        fused_score[~valid_overlap] = 0.0
    pif = _resize_pif(pif_mask, width=width, height=height, cv2=cv2, np=np)
    if valid_overlap is not None:
        pif &= valid_overlap
    pif_pixels = int(np.count_nonzero(pif))
    resolved_fallback_reason = fallback_reason
    if missing_components and resolved_fallback_reason is None:
        resolved_fallback_reason = "MISSING_COMPONENT_MAP"
    common_diagnostics: dict[str, object] = {
        "available_components": available_components,
        "missing_components": missing_components,
        "effective_weights": effective_weights,
        "base_weights": configured_weights,
        "reliability": resolved_reliability,
        "registration_confidence": resolved_registration_confidence,
        "fallback_reason": resolved_fallback_reason,
        "pif_pixels": pif_pixels,
        "pif_ratio": float(pif_pixels / (height * width)),
        "valid_overlap_pixels": (
            int(np.count_nonzero(valid_overlap)) if valid_overlap is not None else height * width
        ),
        "valid_overlap_ratio": (
            float(np.count_nonzero(valid_overlap) / (height * width))
            if valid_overlap is not None
            else 1.0
        ),
        "threshold_comparison": ">",
        "score_min": float(np.min(fused_score)),
        "score_median": float(np.median(fused_score)),
        "score_p95": float(np.quantile(fused_score, 0.95)),
        "score_max": float(np.max(fused_score)),
        "version": PROPOSAL_FUSION_VERSION,
    }

    if float(np.max(fused_score)) < settings.threshold_floor:
        return _empty_result(
            fused_score,
            diagnostics={
                **common_diagnostics,
                "threshold": float(settings.threshold_floor),
                "threshold_mode": "no_change_floor",
                "pif_threshold_fallback_used": False,
                "reason_code": "NO_SCORE_ABOVE_FLOOR",
                "component_count": 0,
                "proposal_count": 0,
                "components": [],
            },
            np=np,
        )

    if pif_pixels >= min_pif_pixels:
        pif_scores = fused_score[pif]
        median = float(np.median(pif_scores))
        mad = float(np.median(np.abs(pif_scores - median)))
        robust_sigma = 1.4826 * mad
        threshold = float(
            np.clip(
                max(settings.threshold_floor, median + settings.pif_threshold_k * robust_sigma),
                0.0,
                1.0,
            )
        )
        threshold_mode = "pif_robust"
        pif_threshold_fallback_used = False
    else:
        threshold = float(
            np.clip(
                max(
                    settings.threshold_floor,
                    float(np.quantile(fused_score, settings.pif_fallback_quantile)),
                ),
                0.0,
                1.0,
            )
        )
        threshold_mode = "quantile_fallback"
        pif_threshold_fallback_used = True

    binary = fused_score > threshold
    if valid_overlap is not None:
        binary &= valid_overlap
    close_kernel = np.ones(
        (settings.mask_close_kernel, settings.mask_close_kernel), dtype=np.uint8
    )
    binary = cv2.morphologyEx(
        binary.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        close_kernel,
    ).astype(bool)
    if valid_overlap is not None:
        # Closing can dilate pixels back across the comparison-canvas edge.
        # Re-apply the hard validity invariant after every morphology step.
        binary &= valid_overlap
    proposals, component_masks, component_diagnostics, component_count = _components(
        binary,
        fused_score=fused_score,
        component_maps=canonical_maps,
        settings=settings,
        cv2=cv2,
        np=np,
        effective_weights=effective_weights,
        reliability=resolved_reliability,
        registration_confidence=common_diagnostics["registration_confidence"],
    )
    return ProposalFusionResult(
        fused_score_map=fused_score,
        binary_change_mask=binary,
        proposals=proposals,
        component_masks=component_masks,
        diagnostics={
            **common_diagnostics,
            "threshold": threshold,
            "threshold_mode": threshold_mode,
            "pif_threshold_fallback_used": pif_threshold_fallback_used,
            "reason_code": None,
            "component_count": component_count,
            "proposal_count": len(proposals),
            "components": component_diagnostics,
        },
    )


def _validate_score_map(value: Any, *, name: str, np: Any) -> Any:
    score_map = np.asarray(value)
    if score_map.ndim != 2 or any(dimension <= 0 for dimension in score_map.shape):
        raise ValueError(f"PROPOSAL_FUSION_{name.upper()}_SHAPE_INVALID")
    if not np.issubdtype(score_map.dtype, np.number):
        raise ValueError(f"PROPOSAL_FUSION_{name.upper()}_DTYPE_INVALID")
    if not bool(np.all(np.isfinite(score_map))):
        raise ValueError(f"PROPOSAL_FUSION_{name.upper()}_NONFINITE")
    return score_map.astype(np.float32, copy=False)


def _reliability_value(value: float | None, *, name: str) -> float:
    if value is None:
        return 1.0
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"PROPOSAL_FUSION_{name.upper()}_RELIABILITY_INVALID") from None
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"PROPOSAL_FUSION_{name.upper()}_RELIABILITY_INVALID")
    return resolved


def _resize_mask(
    value: Any | None,
    *,
    width: int,
    height: int,
    cv2: Any,
    np: Any,
) -> Any | None:
    if value is None:
        return None
    mask = np.asarray(value)
    if mask.ndim != 2 or any(int(dimension) <= 0 for dimension in mask.shape):
        raise ValueError("PROPOSAL_FUSION_VALID_OVERLAP_SHAPE_INVALID")
    return cv2.resize(
        (mask != 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool, copy=False)


def _resize_pif(
    value: Any,
    *,
    width: int,
    height: int,
    cv2: Any,
    np: Any,
) -> Any:
    pif = np.asarray(value)
    if pif.ndim != 2 or any(dimension <= 0 for dimension in pif.shape):
        raise ValueError("PROPOSAL_FUSION_PIF_SHAPE_INVALID")
    if pif.dtype != np.bool_ and pif.dtype != np.uint8:
        raise ValueError("PROPOSAL_FUSION_PIF_DTYPE_INVALID")
    return cv2.resize(
        (pif != 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool, copy=False)


def _empty_result(
    fused_score: Any,
    *,
    diagnostics: dict[str, object],
    np: Any,
) -> ProposalFusionResult:
    return ProposalFusionResult(
        fused_score_map=fused_score,
        binary_change_mask=np.zeros(fused_score.shape, dtype=bool),
        proposals=[],
        diagnostics=diagnostics,
        component_masks={},
    )


def _components(
    binary: Any,
    *,
    fused_score: Any,
    component_maps: dict[str, Any],
    settings: ChangeProposalSettings,
    cv2: Any,
    np: Any,
    effective_weights: Mapping[str, float],
    reliability: Mapping[str, float],
    registration_confidence: float,
) -> tuple[list[ChangeProposal], dict[str, Any], list[dict[str, object]], int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    height, width = fused_score.shape
    image_area = float(height * width)
    candidates: list[dict[str, object]] = []
    for label_index in range(1, count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[label_index]
        )
        area_ratio = area / image_area
        if not (
            settings.min_component_area_ratio
            <= area_ratio
            <= settings.max_component_area_ratio
        ):
            continue
        component = labels == label_index
        component_scores = {
            name: float(np.mean(score_map[component]))
            for name, score_map in component_maps.items()
        }
        component_scores["fused"] = float(np.mean(fused_score[component]))
        candidates.append(
            {
                "label_index": label_index,
                "tight_pixel_box": [x, y, x + box_width, y + box_height],
                "area": area,
                "area_ratio": area_ratio,
                "max_fused_score": float(np.max(fused_score[component])),
                "component_scores": component_scores,
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item["component_scores"]["fused"]),
            -int(item["area"]),
            int(item["tight_pixel_box"][1]),
            int(item["tight_pixel_box"][0]),
            int(item["tight_pixel_box"][3]),
            int(item["tight_pixel_box"][2]),
        )
    )
    retained = candidates[: settings.max_proposals]
    proposals: list[ChangeProposal] = []
    component_masks: dict[str, Any] = {}
    component_diagnostics: list[dict[str, object]] = []
    for proposal_index, candidate in enumerate(retained):
        proposal_id = f"change_{proposal_index:03d}"
        tight_box = [int(value) for value in candidate["tight_pixel_box"]]
        padded_box = _padded_box(
            tight_box,
            width=width,
            height=height,
            padding_ratio=settings.proposal_padding_ratio,
            np=np,
        )
        normalized_box = _normalized_box(padded_box, width=width, height=height)
        component_scores = {
            str(name): float(score)
            for name, score in candidate["component_scores"].items()
        }
        mask_filename = f"{proposal_id}_mask.png"
        proposals.append(
            ChangeProposal(
                proposal_id=proposal_id,
                box=normalized_box,
                pixel_box=padded_box,
                score=component_scores["fused"],
                area_ratio=float(candidate["area_ratio"]),
                source="fused_change_v2",
                component_scores=component_scores,
                mask_filename=mask_filename,
                effective_weights={
                    str(name): float(weight)
                    for name, weight in effective_weights.items()
                },
                reliability={
                    str(name): float(value) for name, value in reliability.items()
                },
                registration_confidence=float(registration_confidence),
            )
        )
        x1, y1, x2, y2 = padded_box
        crop_mask = (
            labels[y1:y2, x1:x2] == int(candidate["label_index"])
        ).astype(np.uint8) * 255
        component_masks[mask_filename] = crop_mask
        component_diagnostics.append(
            {
                "proposal_id": proposal_id,
                "tight_pixel_box": tight_box,
                "padded_pixel_box": padded_box,
                "area_ratio": float(candidate["area_ratio"]),
                "mean_fused_score": component_scores["fused"],
                "max_fused_score": float(candidate["max_fused_score"]),
                "mean_low_level": component_scores.get("low_level"),
                "mean_feature": component_scores.get("feature"),
                "mean_semantic": component_scores.get("semantic"),
            }
        )
    return proposals, component_masks, component_diagnostics, len(candidates)


def _padded_box(
    tight_box: list[int],
    *,
    width: int,
    height: int,
    padding_ratio: float,
    np: Any,
) -> list[int]:
    x1, y1, x2, y2 = tight_box
    padding = int(np.ceil(padding_ratio * max(x2 - x1, y2 - y1)))
    return [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    ]


def _normalized_box(pixel_box: list[int], *, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = pixel_box
    box = [
        round(x1 * 999 / width),
        round(y1 * 999 / height),
        round(x2 * 999 / width),
        round(y2 * 999 / height),
    ]
    box[2] = min(999, max(box[0] + 1, box[2]))
    box[3] = min(999, max(box[1] + 1, box[3]))
    return box


__all__ = [
    "PROPOSAL_FUSION_VERSION",
    "ProposalFusionResult",
    "compute_reliabilities",
    "fuse_change_proposals",
]
