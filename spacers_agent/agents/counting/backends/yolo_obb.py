"""YOLO OBB counting backend — CountingBackend implementation.
YOLO OBB 计数后端 — CountingBackend 实现。
"""

from __future__ import annotations

import logging

from spacers_agent.agents.counting.backends.base import CountingBackend, CountingRequest
from spacers_agent.agents.counting.backends.yolo_model_store import YoloModelStore
from spacers_agent.imaging import (
    build_core_halo_tiles,
    convert_local_point_to_global,
    crop_for_tile,
)
from spacers_agent.counting import apply_acceptance_policy
from spacers_agent.schemas import (
    CountTargetSpec,
    CountingResult,
    GlobalPointObservation,
    IssueRecord,
    LocalPointObservation,
    PointProvenance,
    YoloDetectorSettings,
)

logger = logging.getLogger(__name__)

# Global model store shared across all YOLO backends / 所有 YOLO 后端共享的全局模型存储
_MODEL_STORE = YoloModelStore()


class YoloOBBCountingBackend:
    """Counting backend using a local YOLO OBB model. / 使用本地 YOLO OBB 模型的计数后端。"""

    def __init__(self, detector: YoloDetectorSettings) -> None:
        self._detector = detector
        self._class_map = {c.casefold(): c for c in detector.classes}
        self._alias_map = {k.casefold(): v.casefold() for k, v in detector.aliases.items()}
        self._composite_map = {k.casefold(): [c.casefold() for c in v] for k, v in detector.composite_targets.items()}

    @property
    def name(self) -> str:
        return self._detector.name

    def is_available(self) -> bool:
        """Check weight file exists (no import, no model load). / 检查权重文件存在（不导入、不加载模型）。"""
        return self._detector.enabled and self._detector.weights.resolve().is_file()

    def supports(self, target: CountTargetSpec) -> bool:
        """Match target against classes, aliases, and composites. / 根据类别、别名和组合匹配目标。"""
        label = target.canonical_label.casefold()
        if label in self._class_map:
            return True
        if label in self._alias_map:
            return True
        if label in self._composite_map:
            return True
        return False

    async def count(self, request: CountingRequest, context: object) -> CountingResult:
        """Run YOLO OBB inference across tiles. / 跨 tile 运行 YOLO OBB 推理。"""
        settings = context.settings  # type: ignore[union-attr]
        detector = self._detector
        image = request.image.convert("RGB")
        sample = request.sample
        sample_dir = request.artifact_dir

        tiles = build_core_halo_tiles(
            *image.size,
            core_size=settings.counting.tile_core_size,
            halo_size=settings.counting.halo_size,
            model_max_side=settings.counting.model_max_side,
        )

        model = _MODEL_STORE.get(
            detector.weights,
            confidence=detector.confidence, iou=detector.iou,
            image_size=detector.image_size, device=detector.device,
            max_detections=detector.max_detections,
        )

        all_points: list[GlobalPointObservation] = []
        succeeded: list[str] = []
        failed: list[str] = []
        warnings: list[IssueRecord] = []

        for tile in tiles:
            try:
                crop = crop_for_tile(image, tile)
                points = self._detect_tile(crop, tile, sample.sample_id, request.target, settings)
                all_points.extend(points)
                succeeded.append(tile.tile_id)
            except Exception as error:
                failed.append(tile.tile_id)
                warnings.append(IssueRecord(
                    code="YOLO_TILE_INFERENCE_FAILED",
                    message=f"{tile.tile_id}: {type(error).__name__}: {error}",
                    tile_ids=[tile.tile_id],
                ))

        all_points = [apply_acceptance_policy(p, min_confidence=settings.counting.min_confidence) for p in all_points]
        accepted = sum(1 for p in all_points if p.accepted)

        status = "partial" if (failed and succeeded) else "failed" if failed else "completed_with_warnings" if warnings else "completed"

        return CountingResult(
            sample_id=sample.sample_id, target=request.target.canonical_label,
            question=sample.question, source_width=image.width, source_height=image.height,
            tile_count=len(tiles), succeeded_tiles=succeeded, failed_tiles=failed,
            global_points=all_points, merged_groups=[], unresolved_conflicts=[],
            warnings=warnings, final_count=accepted, status=status,  # type: ignore[arg-type]
        )

    def _detect_tile(self, crop, tile, sample_id, target, settings) -> list[GlobalPointObservation]:
        """Run YOLO on one tile, map classes, convert OBB centres to global points.
        在一个 tile 上运行 YOLO，映射类别，转换 OBB 中心到全局点。
        """
        model = _MODEL_STORE.get(
            self._detector.weights, confidence=self._detector.confidence,
            iou=self._detector.iou, image_size=self._detector.image_size,
            device=self._detector.device, max_detections=self._detector.max_detections,
        )

        results = model(crop, conf=self._detector.confidence, iou=self._detector.iou,
                        imgsz=self._detector.image_size, device=self._detector.device,
                        max_det=self._detector.max_detections, verbose=False)

        points: list[GlobalPointObservation] = []
        if not results or len(results) == 0:
            return points

        result = results[0]
        if result.obb is None:
            return points

        obb_boxes = getattr(result.obb, "xyxyxyxy", None)
        if obb_boxes is None or len(obb_boxes) == 0:
            return points

        classes_tensor = result.obb.cls
        confs = getattr(result.obb, "conf", None)
        crop_w, crop_h = crop.size

        target_label = self._map_target_class(request_target=target)

        for idx in range(len(obb_boxes)):
            corners = obb_boxes[idx]
            cls_id = int(classes_tensor[idx].item())
            class_name = self._detector.classes[cls_id] if cls_id < len(self._detector.classes) else str(cls_id)

            # Class filter / 类别过滤
            if class_name.casefold() not in self._class_map:
                # Try alias reverse lookup / 尝试别名反向查找
                matched = False
                for alias, mapped in self._alias_map.items():
                    if class_name.casefold() == mapped and alias in self._class_map:
                        class_name = self._class_map[alias]
                        matched = True
                        break
                if not matched:
                    continue

            cx = float(sum(c.item() for c in corners[:, 0]) / 4)
            cy = float(sum(c.item() for c in corners[:, 1]) / 4)

            local_x = max(0, min(999, round(cx / max(1, crop_w - 1) * 999)))
            local_y = max(0, min(999, round(cy / max(1, crop_h - 1) * 999)))

            conf = float(confs[idx].item()) if confs is not None and idx < len(confs) else 1.0

            polygon = tuple((float(c[0].item()), float(c[1].item())) for c in corners)

            local_point = LocalPointObservation(
                local_id=f"yolo_{tile.tile_id}_{idx:03d}",
                x=local_x, y=local_y, confidence=conf, radius=0,
                short_evidence=f"YOLO OBB {class_name} in {tile.tile_id}",
            )

            global_point = convert_local_point_to_global(
                local_point, tile, sample_id=sample_id, target=target.canonical_label,
                boundary_band_px=settings.counting.boundary_band_px,
            )
            global_point.provenance = PointProvenance(  # type: ignore[assignment]
                source="yolo_obb_center", backend_name=self._detector.name,
                source_class=class_name, detector_confidence=conf,
                obb_polygon_local_px=[[float(p[0]), float(p[1])] for p in polygon],
            )
            points.append(global_point)

        return points

    def _map_target_class(self, request_target: CountTargetSpec) -> str:
        """Map the counting target to a YOLO class. / 将计数目标映射为 YOLO 类别。"""
        label = request_target.canonical_label.casefold()
        if label in self._class_map:
            return self._class_map[label]
        if label in self._alias_map:
            return self._alias_map[label]
        if label in self._composite_map:
            return self._composite_map[label][0]
        return request_target.canonical_label
