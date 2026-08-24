"""Local YOLO detection counting backend with point-derived final counts.

使用点导出最终计数的本地 YOLO OBB 计数后端。保留多 detector、hash 校验、
类别映射（canonical leaf 到 raw label）与边界去重；不写入任何模型回退逻辑。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agents.counting.backends.base import (
    BackendKind,
    CountingBackendOutcome,
    CountingRequest,
)
from agents.counting.backends.yolo_model_store import YoloModelStore
from agents.errors import DetectorInferenceError
from agents.counting.geometry import (
    build_core_halo_tiles,
    convert_local_point_to_global,
    cores_are_neighbours,
    crop_for_tile,
)
from agents.counting.point_pipeline import (
    apply_acceptance_policy,
    finalize_representatives,
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
from agents.counting.settings import CountingSettings, YoloDetectorSettings
from models.base import ObjectDetectionClient, RuntimeObjectDetectionClient


class YoloOBBCountingBackend:
    """Count configured local classes through verified YOLO inference.
    通过经验证的 YOLO 推理统计已配置的本地类别。"""

    def __init__(
        self,
        detector: YoloDetectorSettings,
        *,
        counting: CountingSettings,
        model_store: YoloModelStore | None = None,
    ) -> None:
        self._detector = detector
        self._counting = counting
        self._store = model_store or YoloModelStore()
        self._classes = {name.casefold(): name for name in detector.classes}
        self._aliases = {
            name.casefold(): target.casefold()
            for name, target in detector.aliases.items()
        }
        self.kind: BackendKind = "yolo_obb" if detector.task == "obb" else "yolo_detect"


    @property
    def name(self) -> str:
        return self._detector.name

    @property
    def priority(self) -> int:
        return self._detector.priority

    def is_enabled(self) -> bool:
        """Return whether the detector is configured/enabled (plan-time).
        返回检测器是否已配置/启用（计划期）。"""
        return self._detector.enabled

    def is_available(self) -> bool:
        """Report configured availability without importing or loading
        Ultralytics; real weight/dependency readiness is verified at count
        time. 在不导入或加载 Ultralytics 的条件下报告配置可用性；权重与
        依赖的真实就绪状态在 count 时验证。"""
        return self._detector.enabled

    def trace_profile(self) -> dict[str, object]:
        """Return serializable detector identity without its absolute weight
        path. 返回不含绝对权重路径的可序列化检测器身份信息。"""
        return {
            "detector_name": self.name,
            "runtime": self._detector.runtime,
            "configured_device": self._detector.device,
            "require_cuda": self._detector.require_cuda,
            "allow_cpu_fallback": self._detector.allow_cpu_fallback,
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
        """Resolve one canonical leaf to audited detector classes.
        将一个 canonical 叶子仅解析为已审计的检测器类别。"""
        values = [target.canonical_label, *target.aliases]
        resolved: set[str] = set()
        for value in values:
            normalized = value.strip().casefold().replace("_", " ").replace("-", " ")
            normalized = " ".join(normalized.split())
            if normalized.endswith("s") and normalized[:-1] in self._classes:
                normalized = normalized[:-1]
            if normalized in self._aliases:
                normalized = self._aliases[normalized]
            if normalized in self._classes:
                resolved.add(normalized)
        return frozenset(resolved)

    def resolve_leaf_classes(self, leaves: tuple[str, ...]) -> frozenset[str]:
        """Map canonical execution leaves to exact configured raw classes."""
        resolved: set[str] = set()
        for leaf in leaves:
            classes = self.resolve_target_classes(
                CountTargetSpec(
                    canonical_label=leaf,
                    inclusion_rule="Count the exact canonical leaf.",
                    exclusion_rule="Exclude every other category.",
                )
            )
            if not classes:
                return frozenset()
            resolved.update(classes)
        return frozenset(resolved)

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        """Return whether the requested target resolves to at least one
        detector class. 返回请求目标是否可解析为至少一个检测器类别。"""
        return bool(self.resolve_target_classes(target))

    async def count(
        self,
        request: CountingRequest,
        context: object,
    ) -> CountingBackendOutcome:
        """Run sequential detection tiles and preserve all visible failure evidence."""
        allowed = self.resolve_leaf_classes(request.executable_leaf_categories)
        if not allowed:
            raise ValueError("YOLO backend selected for an unsupported target")
        image = request.image.convert("RGB")
        tiles = build_core_halo_tiles(
            image.width,
            image.height,
            core_size=self._counting.tile_core_size,
            halo_size=self._counting.halo_size,
            model_max_side=self._counting.model_max_side,
        )
        model = self._store.get(self._detector)
        # Shared detection seam: the audited runtime model is adapted to the
        # model-independent ObjectDetectionClient, and the provider/device
        # audit comes from the same seam so CPU is never silently claimed as
        # a required GPU. 共享检测 seam：将已审计运行时模型适配为模型无关的
        # ObjectDetectionClient；provider/device 审计来自同一 seam，绝不把
        # CPU 静默伪装成所需 GPU。
        client = RuntimeObjectDetectionClient(
            model,
            logical_model_id=self._detector.model_id,
            weights_sha256=self._detector.sha256,
        )
        provider_trace: dict[str, object] = dict(client.provider_audit)
        points: list[GlobalPointObservation] = []
        warnings: list[IssueRecord] = []
        succeeded: list[str] = []
        failed: list[str] = []
        raw_count = unrelated_count = 0
        for tile in tiles:
            try:
                tile_points, tile_raw, tile_unrelated = self._detect_tile(
                    client, crop_for_tile(image, tile), tile, request, allowed
                )
                points.extend(tile_points)
                raw_count += tile_raw
                unrelated_count += tile_unrelated
                succeeded.append(tile.tile_id)
            except Exception as exc:
                failed.append(tile.tile_id)
                warnings.append(
                    IssueRecord(
                        code="YOLO_TILE_INFERENCE_FAILED",
                        message=(
                            f"Tile {tile.tile_id} inference failed with "
                            f"{_safe_exception_type(exc)}."
                        ),
                        tile_ids=[tile.tile_id],
                    )
                )
        if failed and not succeeded:
            # Every tile failed: propagate a stable error so the agent can
            # decide on an explicit fallback instead of returning a fake zero
            # result. 所有 tile 均失败：传播稳定错误，使 Agent 能决定显式
            # 回退，而不是返回伪造的零结果。
            raise DetectorInferenceError("ALL_YOLO_TILES_FAILED")
        effective_min_confidence = max(
            self._counting.min_confidence, self._detector.confidence
        )
        raw_converted = list(points)
        ownership_rejected = sum(
            1 for point in raw_converted if not point.ownership_valid
        )
        confidence_rejected = sum(
            1
            for point in raw_converted
            if point.ownership_valid and point.confidence < effective_min_confidence
        )
        points = [
            apply_acceptance_policy(point, min_confidence=effective_min_confidence)
            for point in raw_converted
        ]
        border_fragment_ids = [
            point.global_id
            for point in points
            if point.accepted
            and _is_clipped_border_fragment(point, image.width, image.height)
        ]
        border_fragment_set = set(border_fragment_ids)
        points = [
            point.model_copy(
                update={"accepted": False, "rejection_reason": "IMAGE_BORDER_FRAGMENT"}
            )
            if point.global_id in border_fragment_set
            else point
            for point in points
        ]
        if border_fragment_ids:
            warnings.append(
                IssueRecord(
                    code="YOLO_IMAGE_BORDER_FRAGMENT_REJECTED",
                    message=(
                        f"Rejected {len(border_fragment_ids)} detector observations "
                        "clipped by the image border."
                    ),
                    point_ids=border_fragment_ids,
                )
            )
        acceptance_after_policy = sum(point.accepted for point in points)
        pairs, unresolved = _detector_duplicate_pairs(points, tiles, self._detector)
        points, merged_groups = finalize_representatives(points, pairs)
        if merged_groups:
            warnings.append(
                IssueRecord(
                    code="YOLO_DUPLICATE_MERGED",
                    message=(
                        f"Merged {len(merged_groups)} strongly matching detector "
                        "duplicate groups."
                    ),
                    point_ids=[
                        point_id for group in merged_groups for point_id in group
                    ],
                )
            )
        for first, second in unresolved:
            warnings.append(
                IssueRecord(
                    code="YOLO_BOUNDARY_CONFLICT_UNRESOLVED",
                    message=(
                        f"Possible boundary duplicate retained for review: "
                        f"{first}|{second}"
                    ),
                    point_ids=[first, second],
                )
            )
        merged_duplicate_count = sum(len(group) - 1 for group in merged_groups)
        status = (
            "failed"
            if failed and not succeeded
            else "partial"
            if failed
            else "completed_with_warnings"
            if warnings
            else "completed"
        )
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
                "backend_kind": self.kind,
                **self.trace_profile(),
                **provider_trace,
                "resolved_target_classes": sorted(allowed),
                "tile_count": len(tiles),
                "raw_detection_count": raw_count,
                "unrelated_class_rejected_count": unrelated_count,
                "ownership_rejected_count": ownership_rejected,
                "confidence_rejected_count": confidence_rejected,
                "border_fragment_rejected_count": len(border_fragment_ids),
                "merged_duplicate_count": merged_duplicate_count,
                "unresolved_conflict_count": len(unresolved),
                "accepted_after_policy_count": acceptance_after_policy,
                "accepted_count": counting.final_count,
                "effective_min_confidence": effective_min_confidence,
            },
        )

    def _detect_tile(
        self,
        client: ObjectDetectionClient,
        crop: Any,
        tile: TileSpec,
        request: CountingRequest,
        allowed: frozenset[str],
    ) -> tuple[list[GlobalPointObservation], int, int]:
        outputs = client.detect(
            crop,
            confidence=self._detector.confidence,
            iou=self._detector.iou,
            image_size=self._detector.image_size,
            device=self._detector.device,
            max_detections=self._detector.max_detections,
        )
        points: list[GlobalPointObservation] = []
        unrelated = 0
        raw_count = len(outputs)
        crop_w, crop_h = crop.size
        for index, output in enumerate(outputs):
            class_name = output.label.casefold()
            if class_name not in allowed:
                unrelated += 1
                continue
            polygon = [[x, y] for x, y in output.polygon] if output.polygon else None
            x1, y1, x2, y2 = output.xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            local_x = max(0, min(999, round(center_x / max(1, crop_w - 1) * 999)))
            local_y = max(0, min(999, round(center_y / max(1, crop_h - 1) * 999)))
            width, height = x2 - x1, y2 - y1
            radius_px = max(1.0, min(width, height) / 2.0)
            local_radius = max(
                0,
                min(250, round(radius_px / max(1, max(crop_w, crop_h)) * 999)),
            )
            confidence = output.confidence
            local = LocalPointObservation(
                local_id=f"yolo_{tile.tile_id}_{index:04d}",
                x=local_x,
                y=local_y,
                confidence=confidence,
                radius=local_radius,
                short_evidence=f"YOLO {output.label} in {tile.tile_id}",
            )
            point = convert_local_point_to_global(
                local,
                tile,
                sample_id=request.sample.sample_id,
                target=request.target.canonical_label,
                boundary_band_px=self._counting.boundary_band_px,
            )
            point = point.model_copy(
                update={
                    "provenance": PointProvenance(
                        source=("yolo_obb_center" if polygon else "yolo_box_center"),
                        backend_name=self.name,
                        model_id=self._detector.model_id,
                        source_class=output.label,
                        detector_confidence=confidence,
                        obb_polygon_local_px=polygon,
                        obb_polygon_global_px=([
                            [x + tile.crop_global.left, y + tile.crop_global.top]
                            for x, y in polygon
                        ] if polygon else None),
                        bbox_xyxy_local_px=[x1, y1, x2, y2],
                        bbox_xyxy_global_px=[
                            x1 + tile.crop_global.left, y1 + tile.crop_global.top,
                            x2 + tile.crop_global.left, y2 + tile.crop_global.top,
                        ],
                        detector_task=self._detector.task,
                        detector_source_dataset=self._detector.source_dataset,
                        weights_sha256=self._detector.sha256,
                    )
                }
            )
            points.append(point)
        return points, raw_count, unrelated


def _safe_exception_type(error: BaseException) -> str:
    """Return a bounded, identifier-only exception type name.
    返回有界、仅标识符的异常类型名。"""
    name = type(error).__name__
    if not name.isidentifier() or len(name) > 80:
        return "BackendError"
    return name


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
            if first.target != second.target or _source_class(first) != _source_class(
                second
            ):
                continue
            same_tile = first.source_tile_id == second.source_tile_id
            neighbouring_boundary = (
                first.near_core_boundary
                and second.near_core_boundary
                and cores_are_neighbours(
                    tile_by_id.get(first.source_tile_id),
                    tile_by_id.get(second.source_tile_id),
                )
            )
            if not same_tile and not neighbouring_boundary:
                continue
            distance = (
                (first.global_x_px - second.global_x_px) ** 2
                + (first.global_y_px - second.global_y_px) ** 2
            ) ** 0.5
            iou = _detection_envelope_iou(first, second)
            if iou >= detector.boundary_duplicate_iou or (
                neighbouring_boundary
                and distance <= detector.boundary_duplicate_center_px
            ):
                merged.append((first.global_id, second.global_id))
            elif (
                neighbouring_boundary
                and distance <= detector.boundary_duplicate_center_px * 2
            ):
                unresolved.append((first.global_id, second.global_id))
    return merged, unresolved


def _is_clipped_border_fragment(
    point: GlobalPointObservation,
    width: int,
    height: int,
) -> bool:
    """Reject detections whose centre and polygon are both clipped at an image
    edge. 拒绝中心与多边形同时贴近图像边缘的截断检测。"""
    provenance = point.provenance
    if provenance is None:
        return False
    polygon = provenance.obb_polygon_global_px
    if polygon:
        xs = [item[0] for item in polygon]
        ys = [item[1] for item in polygon]
    elif provenance.bbox_xyxy_global_px:
        x1, y1, x2, y2 = provenance.bbox_xyxy_global_px
        xs, ys = [x1, x2], [y1, y2]
    else:
        return False
    return (
        (min(xs) <= 0 and point.global_x_norm < 25)
        or (min(ys) <= 0 and point.global_y_norm < 25)
        or (max(xs) >= width - 1 and point.global_x_norm > 974)
        or (max(ys) >= height - 1 and point.global_y_norm > 974)
    )


def _source_class(point: GlobalPointObservation) -> str | None:
    return point.provenance.source_class if point.provenance is not None else None


def _detection_envelope_iou(
    first: GlobalPointObservation,
    second: GlobalPointObservation,
) -> float:
    boxes = []
    for point in (first, second):
        provenance = point.provenance
        if provenance is None:
            return 0.0
        if provenance.bbox_xyxy_global_px:
            boxes.append(tuple(provenance.bbox_xyxy_global_px))
        elif provenance.obb_polygon_global_px:
            xs = [item[0] for item in provenance.obb_polygon_global_px]
            ys = [item[1] for item in provenance.obb_polygon_global_px]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        else:
            return 0.0
    ax1, ay1, ax2, ay2 = boxes[0]
    bx1, by1, bx2, by2 = boxes[1]
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
    return overlap / union if union > 0 else 0.0
