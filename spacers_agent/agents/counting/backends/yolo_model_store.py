"""YOLO model store — lazy loading, caching, no network, no auto-download.
YOLO 模型存储 — 延迟加载、缓存、无网络、无自动下载。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class YoloModelStore:
    """Lazy-load and cache YOLO models keyed by resolved weight path.
    按解析后权重路径为键延迟加载并缓存 YOLO 模型。

    Never downloads. Never imports ultralytics at module level.
    绝不下载。绝不在模块级导入 ultralytics。
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, weight_path: Path, *, confidence: float, iou: float,
            image_size: int, device: str, max_detections: int) -> Any:
        """Return cached model or load once. / 返回缓存模型或加载一次。"""

        resolved = str(weight_path.resolve())

        with self._lock:
            if resolved in self._models:
                return self._models[resolved]

        # Validate weight exists BEFORE import / 在导入前验证权重存在
        if not weight_path.is_file():
            from spacers_agent.agents.errors import DetectorWeightsMissingError
            raise DetectorWeightsMissingError("yolo_obb", str(weight_path))

        # Lazy import ultralytics / 延迟导入 ultralytics
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:
            from spacers_agent.agents.errors import OptionalDependencyMissingError
            raise OptionalDependencyMissingError(
                "yolo", dependency="ultralytics", install_hint="pip install ultralytics",
            ) from exc

        model = YOLO(str(weight_path))
        model.overrides["conf"] = confidence
        model.overrides["iou"] = iou
        model.overrides["imgsz"] = image_size
        model.overrides["device"] = device
        model.overrides["max_det"] = max_detections
        model.overrides["verbose"] = False

        with self._lock:
            self._models[resolved] = model

        return model

    def has(self, weight_path: Path) -> bool:
        """Return whether model is cached. / 返回模型是否已缓存。"""
        return str(weight_path.resolve()) in self._models
