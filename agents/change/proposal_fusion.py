"""Deterministic fusion of low-level, feature, and semantic change maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.change.schema import ChangeProposal
from agents.change.settings import ChangeProposalSettings
from agents.errors import OptionalDependencyMissingError


PROPOSAL_FUSION_VERSION = "weighted_pif_robust_fusion_v1"
_COMPONENT_NAMES = ("low_level", "feature", "semantic")


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


def fuse_change_proposals(
    low_level_map: Any,
    feature_map: Any | None,
    semantic_map: Any | None,
    pif_mask: Any,
    settings: ChangeProposalSettings,
    *,
    min_pif_pixels: int = 32,
    fallback_reason: str | None = None,
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

    available_components = [
        name for name in _COMPONENT_NAMES if name in canonical_maps
    ]
    missing_components = [
        name for name in _COMPONENT_NAMES if name not in canonical_maps
    ]
    configured_weights = {
        "low_level": float(settings.fusion_low_level_weight),
        "feature": float(settings.fusion_feature_weight),
        "semantic": float(settings.fusion_semantic_weight),
    }
    available_weight_sum = sum(
        configured_weights[name] for name in available_components
    )
    if available_weight_sum <= 0.0:
        raise ValueError("PROPOSAL_FUSION_NO_AVAILABLE_WEIGHT")
    effective_weights = {
        name: configured_weights[name] / available_weight_sum
        for name in available_components
    }
    fused_score = np.zeros((height, width), dtype=np.float32)
    for name in available_components:
        fused_score += np.float32(effective_weights[name]) * canonical_maps[name]
    fused_score = np.clip(fused_score, 0.0, 1.0).astype(np.float32, copy=False)

    pif = _resize_pif(pif_mask, width=width, height=height, cv2=cv2, np=np)
    pif_pixels = int(np.count_nonzero(pif))
    resolved_fallback_reason = fallback_reason
    if missing_components and resolved_fallback_reason is None:
        resolved_fallback_reason = "MISSING_COMPONENT_MAP"
    common_diagnostics: dict[str, object] = {
        "available_components": available_components,
        "missing_components": missing_components,
        "effective_weights": effective_weights,
        "fallback_reason": resolved_fallback_reason,
        "pif_pixels": pif_pixels,
        "pif_ratio": float(pif_pixels / (height * width)),
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
    close_kernel = np.ones(
        (settings.mask_close_kernel, settings.mask_close_kernel), dtype=np.uint8
    )
    binary = cv2.morphologyEx(
        binary.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        close_kernel,
    ).astype(bool)
    proposals, component_masks, component_diagnostics, component_count = _components(
        binary,
        fused_score=fused_score,
        component_maps=canonical_maps,
        settings=settings,
        cv2=cv2,
        np=np,
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
    "fuse_change_proposals",
]
