"""YOLO OBB model adapter — protocol and ultralytics implementation.
YOLO OBB 模型适配器 — 协议与 ultralytics 实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image


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

    def predict(self, image: Image.Image, *, image_size: int, confidence: float,
                iou: float, device: str, max_detections: int) -> list[OBBDetection]:
        ...


class UltralyticsOBBModelAdapter:
    """Real ultralytics YOLO OBB adapter — lazy import. / 真实 ultralytics YOLO OBB 适配器 — 延迟导入。"""

    def __init__(self) -> None:
        self._YOLO = None

    def _get_yolo(self):
        if self._YOLO is None:
            try:
                from ultralytics import YOLO  # noqa: PLC0415
            except ImportError as exc:
                from spacers_agent.agents.errors import OptionalDependencyMissingError
                raise OptionalDependencyMissingError(
                    "yolo", dependency="ultralytics", install_hint="pip install ultralytics",
                ) from exc
            self._YOLO = YOLO
        return self._YOLO

    def predict(self, image: Image.Image, *, image_size: int, confidence: float,
                iou: float, device: str, max_detections: int) -> list[OBBDetection]:
        raise NotImplementedError("UltralyticsOBBModelAdapter.predict must be used via model store")
