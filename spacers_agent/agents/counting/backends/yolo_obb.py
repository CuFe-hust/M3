"""Local YOLO OBB counting backend with point-derived final counts.
使用点导出最终计数的本地 YOLO OBB 计数后端。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from spacers_agent.agents.base import AgentContext
from spacers_agent.agents.counting.backends.base import CountingBackendOutcome, CountingRequest
from spacers_agent.agents.counting.backends.yolo_model_store import YoloModelStore
from spacers_agent.agents.counting.point_pipeline import apply_acceptance_policy, finalize_representatives
from spacers_agent.imaging import build_core_halo_tiles, convert_local_point_to_global, crop_for_tile
from spacers_agent.schemas import (
    CountTargetSpec,
    CountingResult,
    GlobalPointObservation,
    IssueRecord,
    LocalPointObservation,
    PointProvenance,
    TileSpec,
    YoloDetectorSettings,
)


class YoloOBBCountingBackend:
    """Count configured local DOTAv1 classes through verified OBB inference.
    通过经验证的 OBB 推理统计已配置的本地 DOTAv1 类别。
    """

    def __init__(self, detector: YoloDetectorSettings, *, model_store: YoloModelStore | None = None) -> None:
        self._detector = detector
        self._store = model_store or YoloModelStore()
        self._classes = {name.casefold(): name for name in detector.classes}
        self._aliases = {name.casefold(): target.casefold() for name, target in detector.aliases.items()}
        self._composites = {
            name.casefold(): tuple(target.casefold() for target in targets)
            for name, targets in detector.composite_targets.items()
        }

    @property
    def name(self) -> str:
        return self._detector.name

    @property
    def priority(self) -> int:
        return self._detector.priority

    def is_available(self) -> bool:
        """Report configured availability without importing or loading Ultralytics.
        在不导入或加载 Ultralytics 的条件下报告配置可用性。
        """
        return self._detector.enabled

    def trace_profile(self) -> dict[str, object]:
        """Return serializable detector identity without its absolute weight path.
        返回不含绝对权重路径的可序列化检测器身份信息。
        """
        return {
            "detector_name": self.name,
            "model_id": self._detector.model_id,
            "weights_file": self._detector.weights.name,
            "weights_sha256": self._detector.sha256,
            "task": self._detector.task,
            "source_dataset": self._detector.source_dataset,
            "confidence": self._detector.confidence,
            "iou": self._detector.iou,
            "image_size": self._detector.image_size,
            "max_detections": self._detector.max_detections,
        }

    def resolve_target_classes(self, target: CountTargetSpec) -> frozenset[str]:
        """Resolve one target to audited detector classes only.
        将一个目标仅解析为已审计的检测器类别。
        """
        values = [target.canonical_label, *target.aliases]
        resolved: set[str] = set()
        for value in values:
            normalized = value.strip().casefold().replace("_", " ").replace("-", " ")
            normalized = " ".join(normalized.split())
            if normalized.endswith("s") and normalized[:-1] in self._classes:
                normalized = normalized[:-1]
            if normalized in self._aliases:
                normalized = self._aliases[normalized]
            if normalized in self._composites:
                resolved.update(self._composites[normalized])
            elif normalized in self._classes:
                resolved.add(normalized)
        return frozenset(resolved)

    def supports(self, target: CountTargetSpec) -> bool:
        """Return whether the requested target resolves to at least one detector class.
        返回请求目标是否可解析为至少一个检测器类别。
        """
        return bool(self.resolve_target_classes(target))

    async def count(self, request: CountingRequest, context: AgentContext) -> CountingBackendOutcome:
        """Run sequential OBB tiles and preserve all visible failure evidence.
        顺序运行 OBB 切片并保留全部可见失败证据。
        """
        allowed = self.resolve_target_classes(request.target)
        if not allowed:
            raise ValueError("YOLO backend selected for an unsupported target")
        image = request.image.convert("RGB")
        settings = context.settings
        tiles = build_core_halo_tiles(
            *image.size,
            core_size=settings.counting.tile_core_size,
            halo_size=settings.counting.halo_size,
            model_max_side=settings.counting.model_max_side,
        )
        model = self._store.get(self._detector)
        points: list[GlobalPointObservation] = []
        warnings: list[IssueRecord] = []
        succeeded: list[str] = []
        failed: list[str] = []
        raw_count = unrelated_count = 0
        for tile in tiles:
            try:
                tile_points, tile_raw, tile_unrelated = self._detect_tile(
                    model, crop_for_tile(image, tile), tile, request, allowed, settings
                )
                points.extend(tile_points)
                raw_count += tile_raw
                unrelated_count += tile_unrelated
                succeeded.append(tile.tile_id)
            except Exception as exc:
                failed.append(tile.tile_id)
                warnings.append(IssueRecord(
                    code="YOLO_TILE_INFERENCE_FAILED",
                    message=f"{tile.tile_id}: {type(exc).__name__}: {exc}",
                    tile_ids=[tile.tile_id],
                ))
        effective_min_confidence = max(settings.counting.min_confidence, self._detector.confidence)
        points = [apply_acceptance_policy(point, min_confidence=effective_min_confidence) for point in points]
        border_fragment_ids = [
            point.global_id
            for point in points
            if point.accepted and _is_clipped_border_fragment(point, image.width, image.height)
        ]
        border_fragment_set = set(border_fragment_ids)
        points = [
            point.model_copy(update={"accepted": False, "rejection_reason": "IMAGE_BORDER_FRAGMENT"})
            if point.global_id in border_fragment_set else point
            for point in points
        ]
        if border_fragment_ids:
            warnings.append(IssueRecord(
                code="YOLO_IMAGE_BORDER_FRAGMENT_REJECTED",
                message=f"Rejected {len(border_fragment_ids)} detector observations clipped by the image border.",
                point_ids=border_fragment_ids,
            ))
        accepted_before_merge = sum(point.accepted for point in points)
        pairs, unresolved = _detector_duplicate_pairs(points, tiles, self._detector)
        points, merged_groups = finalize_representatives(points, pairs)
        if merged_groups:
            warnings.append(IssueRecord(
                code="YOLO_DUPLICATE_MERGED",
                message=f"Merged {len(merged_groups)} strongly matching detector duplicate groups.",
                point_ids=[point_id for group in merged_groups for point_id in group],
            ))
        for first, second in unresolved:
            warnings.append(IssueRecord(
                code="YOLO_BOUNDARY_CONFLICT_UNRESOLVED",
                message=f"Possible boundary duplicate retained for review: {first}|{second}",
                point_ids=[first, second],
            ))
        acceptance_rejected = len(points) - accepted_before_merge
        status = "failed" if failed and not succeeded else "partial" if failed else "completed_with_warnings" if warnings else "completed"
        counting = CountingResult(
            sample_id=request.sample.sample_id,
            target=request.target.canonical_label,
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
            unresolved_conflicts=[f"{first}|{second}" for first, second in unresolved],
            warnings=warnings,
            final_count=sum(point.accepted for point in points),
            status=status,
        )
        return CountingBackendOutcome(
            counting=counting,
            trace={
                "backend_kind": "yolo_obb",
                **self.trace_profile(),
                "resolved_target_classes": sorted(allowed),
                "confidence": self._detector.confidence,
                "iou": self._detector.iou,
                "image_size": self._detector.image_size,
                "max_detections": self._detector.max_detections,
                "tile_count": len(tiles),
                "raw_detection_count": raw_count,
                "unrelated_class_rejected_count": unrelated_count,
                "acceptance_rejected_count": acceptance_rejected,
                "border_fragment_rejected_count": len(border_fragment_ids),
                "merged_duplicate_count": sum(len(group) - 1 for group in merged_groups),
                "unresolved_conflict_count": len(unresolved),
                "accepted_count": counting.final_count,
                "effective_min_confidence": effective_min_confidence,
            },
        )

    def _detect_tile(
        self,
        model: Any,
        crop: Any,
        tile: TileSpec,
        request: CountingRequest,
        allowed: frozenset[str],
        settings: Any,
    ) -> tuple[list[GlobalPointObservation], int, int]:
        results = model.predict(
            source=crop,
            conf=self._detector.confidence,
            iou=self._detector.iou,
            imgsz=self._detector.image_size,
            device=self._detector.device,
            max_det=self._detector.max_detections,
            verbose=False,
        )
        if not results or getattr(results[0], "obb", None) is None:
            return [], 0, 0
        obb = results[0].obb
        polygons, classes, confidences = obb.xyxyxyxy, obb.cls, obb.conf
        names = _model_names(getattr(model, "names", {}))
        points: list[GlobalPointObservation] = []
        unrelated = 0
        raw_count = len(polygons)
        crop_w, crop_h = crop.size
        for index in range(raw_count):
            class_id = int(_scalar(classes[index]))
            if class_id not in names:
                raise ValueError(f"YOLO_CLASS_ID_UNKNOWN:{class_id}")
            class_name = names[class_id].casefold()
            if class_name not in allowed:
                unrelated += 1
                continue
            polygon = [[float(_scalar(corner[0])), float(_scalar(corner[1]))] for corner in polygons[index]]
            center_x = sum(point[0] for point in polygon) / len(polygon)
            center_y = sum(point[1] for point in polygon) / len(polygon)
            local_x = max(0, min(999, round(center_x / max(1, crop_w - 1) * 999)))
            local_y = max(0, min(999, round(center_y / max(1, crop_h - 1) * 999)))
            width = max(point[0] for point in polygon) - min(point[0] for point in polygon)
            height = max(point[1] for point in polygon) - min(point[1] for point in polygon)
            radius_px = max(1.0, min(width, height) / 2.0)
            local_radius = max(0, min(250, round(radius_px / max(1, max(crop_w, crop_h)) * 999)))
            confidence = float(_scalar(confidences[index]))
            local = LocalPointObservation(
                local_id=f"yolo_{tile.tile_id}_{index:04d}", x=local_x, y=local_y,
                confidence=confidence, radius=local_radius,
                short_evidence=f"YOLO OBB {names[class_id]} in {tile.tile_id}",
            )
            point = convert_local_point_to_global(
                local, tile, sample_id=request.sample.sample_id,
                target=request.target.canonical_label,
                boundary_band_px=settings.counting.boundary_band_px,
            )
            point = point.model_copy(update={"provenance": PointProvenance(
                source="yolo_obb_center", backend_name=self.name, model_id=self._detector.model_id,
                source_class=names[class_id], detector_confidence=confidence,
                obb_polygon_local_px=polygon,
                obb_polygon_global_px=[[x + tile.crop_global.left, y + tile.crop_global.top] for x, y in polygon],
                detector_task="obb", detector_source_dataset=self._detector.source_dataset,
                weights_sha256=self._detector.sha256,
            )})
            points.append(point)
        return points, raw_count, unrelated


def _scalar(value: Any) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def _model_names(value: object) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(index): str(name).strip() for index, name in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {index: str(name).strip() for index, name in enumerate(value)}
    return {}


def _detector_duplicate_pairs(
    points: Sequence[GlobalPointObservation],
    tiles: Sequence[TileSpec],
    detector: YoloDetectorSettings,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    tile_by_id = {tile.tile_id: tile for tile in tiles}
    merged: list[tuple[str, str]] = []
    unresolved: list[tuple[str, str]] = []
    candidates = [point for point in points if point.accepted]
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            if first.target != second.target or _source_class(first) != _source_class(second):
                continue
            same_tile = first.source_tile_id == second.source_tile_id
            neighbouring_boundary = (
                first.near_core_boundary
                and second.near_core_boundary
                and _neighbouring(tile_by_id.get(first.source_tile_id), tile_by_id.get(second.source_tile_id))
            )
            if not same_tile and not neighbouring_boundary:
                continue
            distance = ((first.global_x_px - second.global_x_px) ** 2 + (first.global_y_px - second.global_y_px) ** 2) ** 0.5
            iou = _polygon_envelope_iou(first, second)
            if iou >= detector.boundary_duplicate_iou or (
                neighbouring_boundary and distance <= detector.boundary_duplicate_center_px
            ):
                merged.append((first.global_id, second.global_id))
            elif neighbouring_boundary and distance <= detector.boundary_duplicate_center_px * 2:
                unresolved.append((first.global_id, second.global_id))
    return merged, unresolved


def _is_clipped_border_fragment(point: GlobalPointObservation, width: int, height: int) -> bool:
    """Reject detections whose centre and polygon are both clipped at an image edge.
    拒绝中心与多边形同时贴近图像边缘的截断检测。
    """
    polygon = point.provenance.obb_polygon_global_px if point.provenance else None
    if not polygon:
        return False
    xs = [item[0] for item in polygon]
    ys = [item[1] for item in polygon]
    return (
        (min(xs) <= 0 and point.global_x_norm < 25)
        or (min(ys) <= 0 and point.global_y_norm < 25)
        or (max(xs) >= width - 1 and point.global_x_norm > 974)
        or (max(ys) >= height - 1 and point.global_y_norm > 974)
    )


def _source_class(point: GlobalPointObservation) -> str | None:
    return point.provenance.source_class if point.provenance is not None else None


def _neighbouring(first: TileSpec | None, second: TileSpec | None) -> bool:
    if first is None or second is None:
        return False
    a, b = first.owner_core_global, second.owner_core_global
    return (a.right == b.left or b.right == a.left) or (a.bottom == b.top or b.bottom == a.top)


def _polygon_envelope_iou(first: GlobalPointObservation, second: GlobalPointObservation) -> float:
    polygons = [point.provenance.obb_polygon_global_px if point.provenance else None for point in (first, second)]
    if not all(polygons):
        return 0.0
    boxes = []
    for polygon in polygons:
        xs, ys = [item[0] for item in polygon], [item[1] for item in polygon]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    ax1, ay1, ax2, ay2 = boxes[0]
    bx1, by1, bx2, by2 = boxes[1]
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
    return overlap / union if union > 0 else 0.0
