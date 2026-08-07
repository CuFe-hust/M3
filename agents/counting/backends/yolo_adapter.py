"""YOLO OBB model adapter — protocol and ultralytics implementation.

YOLO OBB 模型适配器 — 协议与 ultralytics 实现。适配器将 Ultralytics/ONNX
输出统一为规范点/框（OBBDetection），并保留 source class。导入本模块不
加载任何模型或权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from agents.errors import OptionalDependencyMissingError


@dataclass(frozen=True)
class OBBDetection:
    """One oriented bounding-box detection. / 一个有向边界框检测。"""

    class_name: str
    confidence: float
    center_x_px: float
    center_y_px: float
    polygon_xy: tuple[tuple[float, float], ...]


class OBBModelAdapter(Protocol):
    """Adapter for running OBB inference. / 运行 OBB 推理的适配器。"""

    def predict(
        self,
        image: Image.Image,
        *,
        image_size: int,
        confidence: float,
        iou: float,
        device: str,
        max_detections: int,
    ) -> list[OBBDetection]:
        ...


class UltralyticsOBBModelAdapter:
    """Real ultralytics YOLO OBB adapter — lazy import and unified output.
    真实 ultralytics YOLO OBB 适配器 — 延迟导入与统一输出。"""

    def __init__(self) -> None:
        self._YOLO: Any = None
        self._model: Any = None

    def _get_yolo(self):
        if self._YOLO is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise OptionalDependencyMissingError(
                    "yolo",
                    dependency="ultralytics",
                    install_hint="pip install ultralytics",
                ) from exc
            self._YOLO = YOLO
        return self._YOLO

    def load(self, weights: Path) -> Any:
        """Lazily load a model from a weight path. / 从权重路径延迟加载模型。"""
        if self._model is None:
            self._model = self._get_yolo()(str(weights))
        return self._model

    def predict(
        self,
        image: Image.Image,
        *,
        image_size: int,
        confidence: float,
        iou: float,
        device: str,
        max_detections: int,
    ) -> list[OBBDetection]:
        """Run inference and normalize Ultralytics OBB output into canonical
        detections with source class names. 运行推理并将 Ultralytics OBB
        输出规范化为带 source class 的规范检测。"""
        if self._model is None:
            raise ValueError("UltralyticsOBBModelAdapter.load must be called first")
        results = self._model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            device=device,
            max_det=max_detections,
            verbose=False,
        )
        if not results or getattr(results[0], "obb", None) is None:
            return []
        obb = results[0].obb
        polygons, classes, confidences = obb.xyxyxyxy, obb.cls, obb.conf
        names = _model_names(getattr(self._model, "names", {}))
        detections: list[OBBDetection] = []
        for index in range(len(polygons)):
            class_id = int(_scalar(classes[index]))
            if class_id not in names:
                raise ValueError(f"YOLO_CLASS_ID_UNKNOWN:{class_id}")
            polygon = tuple(
                (float(_scalar(corner[0])), float(_scalar(corner[1])))
                for corner in polygons[index]
            )
            center_x = sum(point[0] for point in polygon) / len(polygon)
            center_y = sum(point[1] for point in polygon) / len(polygon)
            detections.append(
                OBBDetection(
                    class_name=names[class_id],
                    confidence=float(_scalar(confidences[index])),
                    center_x_px=center_x,
                    center_y_px=center_y,
                    polygon_xy=polygon,
                )
            )
        return detections


def _scalar(value: Any) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def _model_names(value: object) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(index): str(name).strip() for index, name in value.items()}
    if isinstance(value, (list, tuple)):
        return {index: str(name).strip() for index, name in enumerate(value)}
    return {}
