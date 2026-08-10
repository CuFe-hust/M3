"""Semantic-mask counting through conservative connected components.

This backend approximates instances with semantic connected components and
emits their centroids into the shared point pipeline. It deliberately does not
select experts, split touching instances, or expose dense masks in traces.
本后端以语义连通域近似实例，仅向统一点流水线提交质心，不选择专家、不拆分相接对象，
也不在 trace 中暴露稠密 mask。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image

from agents.counting.backends.base import BackendKind, CountingBackendOutcome, CountingRequest
from agents.counting.expert_catalog import ExpertSpec, ExpertTargetSupportSpec
from agents.counting.geometry import (
    build_core_halo_tiles,
    convert_local_point_to_global,
    crop_for_tile,
    should_tile_image,
)
from agents.counting.point_pipeline import (
    apply_acceptance_policy,
    decide_seam_pairs,
    finalize_representatives,
    find_boundary_conflicts,
)
from agents.counting.schema import (
    CountTargetSpec,
    CountingResult,
    GlobalPointObservation,
    IssueRecord,
    LocalPointObservation,
    PointProvenance,
    TileSpec,
)
from agents.counting.settings import CountingSettings
from models.base import SemanticSegmentationClient, SemanticSegmentationOutput


class SemanticSegmentationBackendError(RuntimeError):
    """Stable failure raised when every semantic tile fails.
    所有语义切片均失败时抛出的稳定错误。"""

    code = "ALL_SEMANTIC_SEGMENTATION_TILES_FAILED"


@dataclass(frozen=True)
class _Component:
    model_label: str
    area_px: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    mean_confidence: float
    touches_tile_border: bool


@dataclass
class _Counters:
    raw_components: int = 0
    area_rejected: int = 0
    confidence_rejected: int = 0
    ownership_rejected: int = 0


class SemanticSegmentationCountingBackend:
    """Convert one catalog-declared semantic capability to point evidence.
    将 catalog 声明的语义能力转换为统一点证据。"""

    kind: BackendKind = "semantic_segmentation"

    def __init__(
        self,
        client: SemanticSegmentationClient,
        expert: ExpertSpec,
        counting: CountingSettings,
    ) -> None:
        self._client = client
        self._expert = expert
        self._counting = counting

    @property
    def name(self) -> str:
        return self._expert.backend_name

    @property
    def priority(self) -> int:
        return self._expert.priority

    def is_enabled(self) -> bool:
        return self._expert.enabled

    def is_available(self) -> bool:
        # Keep dependency, asset, and provider readiness lazy until predict.
        # 依赖、资产和 provider 就绪性保持惰性，直到 predict 才验证。
        return self._expert.enabled and self._expert.verification.class_map == "verified"

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        del hints
        return self._capability(target) is not None

    async def count(
        self,
        request: CountingRequest,
        context: object,
    ) -> CountingBackendOutcome:
        del context
        capability = self._capability(request.target)
        if capability is None:
            raise ValueError("semantic segmentation backend selected for unsupported target")

        image = request.image.convert("RGB")
        target = _normalize_target_label(request.target.canonical_label)
        tiles = self._tiles(image)
        points: list[GlobalPointObservation] = []
        warnings: list[IssueRecord] = []
        succeeded: list[str] = []
        failed: list[str] = []
        counters = _Counters()
        model_revision: str | None = None
        weights_sha256: str | None = None

        for tile in tiles:
            try:
                tile_points, revision, digest = self._count_tile(
                    crop_for_tile(image, tile),
                    tile,
                    request,
                    target,
                    capability,
                    counters,
                )
                points.extend(tile_points)
                model_revision = model_revision if model_revision is not None else revision
                weights_sha256 = weights_sha256 if weights_sha256 is not None else digest
                succeeded.append(tile.tile_id)
            except Exception as error:
                failed.append(tile.tile_id)
                warnings.append(
                    IssueRecord(
                        code="SEMANTIC_SEGMENTATION_TILE_FAILED",
                        message=(
                            f"Tile {tile.tile_id} semantic inference failed with "
                            f"{type(error).__name__}."
                        ),
                        tile_ids=[tile.tile_id],
                    )
                )

        if failed and not succeeded:
            raise SemanticSegmentationBackendError(
                SemanticSegmentationBackendError.code
            ) from None

        conflicts = find_boundary_conflicts(points, tiles)
        merged_pairs, unresolved_pairs = decide_seam_pairs(conflicts)
        points, merged_groups = finalize_representatives(points, merged_pairs)
        merged_duplicate_count = sum(len(group) - 1 for group in merged_groups)

        if merged_groups:
            warnings.append(
                IssueRecord(
                    code="SEMANTIC_DUPLICATE_MERGED",
                    message=(
                        f"Merged {merged_duplicate_count} semantic component "
                        "duplicate observations at tile seams."
                    ),
                    point_ids=[point_id for group in merged_groups for point_id in group],
                )
            )
        for first, second in unresolved_pairs:
            warnings.append(
                IssueRecord(
                    code="SEMANTIC_BOUNDARY_CONFLICT_UNRESOLVED",
                    message="Possible semantic boundary duplicate retained for review.",
                    point_ids=[first, second],
                )
            )

        unresolved_ids = [f"{first}|{second}" for first, second in unresolved_pairs]
        status = "partial" if failed else (
            "completed_with_warnings" if warnings else "completed"
        )
        final_count = sum(point.accepted for point in points)
        result = CountingResult(
            sample_id=request.sample.sample_id,
            target=target,
            question=request.sample.question,
            source_width=image.width,
            source_height=image.height,
            tile_count=len(tiles),
            initial_tile_count=len(tiles),
            leaf_tile_count=len(tiles),
            succeeded_tiles=succeeded,
            failed_tiles=failed,
            global_points=points,
            merged_groups=merged_groups,
            unresolved_conflicts=unresolved_ids,
            warnings=warnings,
            final_count=final_count,
            status=status,
        )
        trace: dict[str, object] = {
            "backend_kind": "semantic_segmentation",
            "backend_name": self.name,
            "logical_model_id": self._expert.logical_model_id,
            "model_revision": model_revision,
            "weights_sha256": weights_sha256,
            "counting_mode": "connected_components",
            "semantic_instance_approximation": True,
            "touching_objects_may_undercount": True,
            "target": target,
            "model_labels": list(capability.model_labels),
            "tile_count": len(tiles),
            "succeeded_tile_count": len(succeeded),
            "failed_tile_count": len(failed),
            "raw_component_count": counters.raw_components,
            "area_rejected_count": counters.area_rejected,
            "confidence_rejected_count": counters.confidence_rejected,
            "ownership_rejected_count": counters.ownership_rejected,
            "merged_duplicate_count": merged_duplicate_count,
            "accepted_count": result.final_count,
        }
        return CountingBackendOutcome(counting=result, trace=trace)

    def _capability(
        self,
        target: CountTargetSpec,
    ) -> ExpertTargetSupportSpec | None:
        if not self._expert.enabled or self._expert.kind != "semantic_segmentation":
            return None
        if self._expert.verification.class_map != "verified":
            return None
        capability = self._expert.supports.get(
            _normalize_target_label(target.canonical_label)
        )
        if capability is None or capability.counting_mode != "connected_components":
            return None
        if not capability.model_labels or any(
            not label.strip() for label in capability.model_labels
        ):
            return None
        policy = capability.policy
        if (
            policy.min_component_area_px is None
            or policy.min_component_area_px < 1
            or policy.max_component_area_ratio is None
            or not 0.0 < policy.max_component_area_ratio <= 1.0
            or policy.min_mean_confidence is None
            or not 0.0 <= policy.min_mean_confidence <= 1.0
            or any(
                kernel < 0 or kernel > 31 or (kernel != 0 and kernel % 2 == 0)
                for kernel in (
                    policy.morphology.open_kernel,
                    policy.morphology.close_kernel,
                )
            )
        ):
            return None
        return capability

    def _tiles(self, image: Image.Image) -> list[TileSpec]:
        tiled = should_tile_image(
            image.width,
            image.height,
            model_max_side=self._counting.model_max_side,
            max_pixels_without_tiling=self._counting.max_pixels_without_tiling,
        )
        return build_core_halo_tiles(
            image.width,
            image.height,
            core_size=self._counting.tile_core_size if tiled else max(image.size),
            halo_size=self._counting.halo_size if tiled else 0,
            model_max_side=self._counting.model_max_side,
        )

    def _count_tile(
        self,
        crop: Image.Image,
        tile: TileSpec,
        request: CountingRequest,
        target: str,
        capability: ExpertTargetSupportSpec,
        counters: _Counters,
    ) -> tuple[list[GlobalPointObservation], str | None, str]:
        output = self._client.predict(crop)
        mask, confidence = _validated_dense_maps(output, crop.size)
        label_ids = _resolve_label_ids(output.id_to_label, capability.model_labels)
        _validate_model_identity(output, self._expert)
        effective_confidence = max(
            self._counting.min_confidence,
            capability.policy.min_mean_confidence or 0.0,
        )
        local_points: list[LocalPointObservation] = []
        for model_label in capability.model_labels:
            class_id = label_ids[model_label]
            components = _components_for_label(
                mask,
                confidence,
                class_id=class_id,
                model_label=model_label,
                min_mean_confidence=effective_confidence,
                open_kernel=capability.policy.morphology.open_kernel,
                close_kernel=capability.policy.morphology.close_kernel,
                counters=counters,
            )
            for component in components:
                if (
                    component.area_px < (capability.policy.min_component_area_px or 1)
                    or component.area_px / (crop.width * crop.height)
                    > (capability.policy.max_component_area_ratio or 1.0)
                ):
                    counters.area_rejected += 1
                    continue
                local_points.append(
                    _component_point(component, len(local_points), crop.size)
                )

        global_points: list[GlobalPointObservation] = []
        for local_point in local_points:
            point = convert_local_point_to_global(
                local_point,
                tile,
                sample_id=request.sample.sample_id,
                target=target,
                boundary_band_px=self._counting.boundary_band_px,
            ).model_copy(
                update={
                    "provenance": PointProvenance(
                        source="semantic_component_centroid",
                        backend_name=self.name,
                        model_id=self._expert.logical_model_id,
                        source_class=_evidence_label(local_point.short_evidence),
                        weights_sha256=self._expert.asset.sha256,
                    )
                }
            )
            if not point.ownership_valid:
                counters.ownership_rejected += 1
            global_points.append(
                apply_acceptance_policy(point, min_confidence=effective_confidence)
            )
        return global_points, output.model_revision, output.weights_sha256


def _validated_dense_maps(
    output: SemanticSegmentationOutput,
    expected_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(output.mask)
    confidence = np.asarray(output.confidence_map, dtype=np.float32)
    expected_width, expected_height = expected_size
    if output.width != expected_width or output.height != expected_height:
        raise ValueError("semantic output dimensions differ from the tile")
    if mask.shape != (expected_height, expected_width):
        raise ValueError("semantic class map shape differs from the tile")
    if confidence.shape != mask.shape:
        raise ValueError("semantic confidence map shape differs from the class map")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("semantic class map must contain integer IDs")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("semantic confidence map must be finite probabilities")
    return mask, confidence


def _resolve_label_ids(
    id_to_label: Mapping[int, str],
    model_labels: tuple[str, ...],
) -> dict[str, int]:
    if not isinstance(id_to_label, Mapping) or not id_to_label:
        raise ValueError("semantic output is missing its verified class map")
    reverse: dict[str, int] = {}
    for class_id, label in id_to_label.items():
        if (
            not isinstance(class_id, int)
            or isinstance(class_id, bool)
            or not isinstance(label, str)
        ):
            raise ValueError("semantic output class map is invalid")
        if label in reverse:
            raise ValueError("semantic output class labels must be unique")
        reverse[label] = class_id
    missing = [label for label in model_labels if label not in reverse]
    if missing:
        raise ValueError("expert model label is absent from the verified class map")
    return {label: reverse[label] for label in model_labels}


def _validate_model_identity(output: SemanticSegmentationOutput, expert: ExpertSpec) -> None:
    if output.logical_model_id != expert.logical_model_id:
        raise ValueError("semantic client logical model ID differs from the expert")
    if expert.asset.sha256 is not None and output.weights_sha256 != expert.asset.sha256:
        raise ValueError("semantic client weight digest differs from the expert")


def _components_for_label(
    mask: np.ndarray,
    confidence: np.ndarray,
    *,
    class_id: int,
    model_label: str,
    min_mean_confidence: float,
    open_kernel: int,
    close_kernel: int,
    counters: _Counters,
) -> list[_Component]:
    cv2 = _load_cv2()
    raw_mask = (mask == class_id).astype(np.uint8)
    raw_count, raw_ids, _, _ = cv2.connectedComponentsWithStats(
        raw_mask, connectivity=8
    )
    counters.raw_components += raw_count - 1
    eligible = np.zeros_like(raw_mask)
    for component_id in range(1, raw_count):
        pixels = raw_ids == component_id
        mean_confidence = float(confidence[pixels].mean())
        if mean_confidence < min_mean_confidence:
            counters.confidence_rejected += 1
            continue
        eligible[pixels & (confidence >= min_mean_confidence)] = 1

    eligible = _apply_morphology(eligible, open_kernel, close_kernel)
    count, component_ids, stats, centroids = cv2.connectedComponentsWithStats(
        eligible, connectivity=8
    )
    components: list[_Component] = []
    height, width = eligible.shape
    for component_id in range(1, count):
        pixels = component_ids == component_id
        source_pixels = pixels & (raw_mask > 0)
        if not source_pixels.any():
            counters.confidence_rejected += 1
            continue
        mean_confidence = float(confidence[source_pixels].mean())
        if mean_confidence < min_mean_confidence:
            counters.confidence_rejected += 1
            continue
        left = int(stats[component_id, cv2.CC_STAT_LEFT])
        top = int(stats[component_id, cv2.CC_STAT_TOP])
        component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        components.append(
            _Component(
                model_label=model_label,
                area_px=area,
                bbox=(left, top, component_width, component_height),
                centroid=(
                    float(centroids[component_id, 0]),
                    float(centroids[component_id, 1]),
                ),
                mean_confidence=mean_confidence,
                touches_tile_border=(
                    left == 0
                    or top == 0
                    or left + component_width >= width
                    or top + component_height >= height
                ),
            )
        )
    return components


def _apply_morphology(mask: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
    cv2 = _load_cv2()
    result = mask
    if open_kernel:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    if close_kernel:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def _load_cv2() -> Any:
    """Import optional OpenCV only when semantic execution needs it.
    仅在语义执行需要时导入可选 OpenCV 依赖。"""

    import cv2

    return cv2


def _component_point(
    component: _Component,
    index: int,
    size: tuple[int, int],
) -> LocalPointObservation:
    width, height = size
    x, y = component.centroid
    radius_px = math.sqrt(component.area_px / math.pi)
    return LocalPointObservation(
        local_id=f"component-{index:05d}",
        x=0 if width == 1 else round(x / (width - 1) * 999),
        y=0 if height == 1 else round(y / (height - 1) * 999),
        confidence=component.mean_confidence,
        radius=min(250, round(radius_px / max(1, min(width, height)) * 999)),
        touches_crop_border=component.touches_tile_border,
        short_evidence=(
            f"semantic:{component.model_label} area={component.area_px} "
            f"mean={component.mean_confidence:.3f}"
        ),
    )


def _evidence_label(value: str) -> str | None:
    match = re.match(r"semantic:([^ ]+)", value)
    return match.group(1) if match else None


def _normalize_target_label(value: str) -> str:
    return re.sub(r"[-_\s]+", "-", value.strip().casefold()).strip("-")
