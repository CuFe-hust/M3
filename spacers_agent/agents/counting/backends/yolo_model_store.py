"""Audited lazy loading for local YOLO models.
本地 YOLO 模型的可审计延迟加载。
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Callable

from spacers_agent.agents.errors import (
    DetectorClassMapMismatchError,
    DetectorTaskMismatchError,
    DetectorWeightsHashMismatchError,
    DetectorWeightsMissingError,
    OptionalDependencyMissingError,
)
from spacers_agent.schemas import YoloDetectorSettings


class YoloModelStore:
    """Load a verified OBB model once per resolved path and expected digest.
    每个解析后的权重路径和预期摘要仅加载一次已验证的 OBB 模型。
    """

    def __init__(self, loader: Callable[[str], Any] | None = None) -> None:
        self._models: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()
        self._loader = loader

    def get(self, detector: YoloDetectorSettings) -> Any:
        """Return a digest-verified cached model without network downloads.
        返回经摘要验证的缓存模型，且绝不进行网络下载。
        """
        path = detector.weights.resolve()
        key = (str(path), detector.sha256)
        with self._lock:
            cached = self._models.get(key)
        if cached is not None:
            return cached
        if not path.is_file():
            raise DetectorWeightsMissingError(detector.name, path.name)
        actual = _sha256(path)
        if actual != detector.sha256:
            raise DetectorWeightsHashMismatchError(
                f"Detector {detector.name!r} digest mismatch for {path.name}: expected "
                f"{detector.sha256}, got {actual}"
            )
        model = self._load(path, detector)
        actual_task = str(getattr(model, "task", ""))
        if actual_task != detector.task:
            raise DetectorTaskMismatchError(
                f"Detector {detector.name!r} expected task {detector.task!r}, got {actual_task!r}"
            )
        model_names = _normalized_names(getattr(model, "names", {}))
        expected_names = [value.casefold() for value in detector.classes]
        if model_names != expected_names:
            raise DetectorClassMapMismatchError(
                f"Detector {detector.name!r} class map differs from configured DOTAv1 classes"
            )
        with self._lock:
            return self._models.setdefault(key, model)

    def has(self, weight_path: Path, sha256: str | None = None) -> bool:
        """Return whether an exact verified model cache key is present.
        返回精确已验证模型缓存键是否存在。
        """
        resolved = str(weight_path.resolve())
        return any(path == resolved and (sha256 is None or digest == sha256) for path, digest in self._models)

    def _load(self, path: Path, detector: YoloDetectorSettings) -> Any:
        if self._loader is not None:
            return self._loader(str(path))
        if detector.runtime == "onnx_yolov5_obb":
            from spacers_agent.agents.counting.backends.yolov5_obb_onnx import YoloV5ObbOnnxModel  # noqa: PLC0415

            return YoloV5ObbOnnxModel(path, detector.classes)
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:
            raise OptionalDependencyMissingError(
                "yolo", dependency="ultralytics", install_hint="pip install -e '.[yolo]'"
            ) from exc
        return YOLO(str(path))


def _sha256(path: Path) -> str:
    """Hash a local weight file incrementally to avoid large-file buffering.
    增量计算本地权重文件摘要，避免缓冲整个大文件。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_names(value: object) -> list[str]:
    """Convert Ultralytics list or indexed mapping class names into one list.
    将 Ultralytics 列表或索引映射类别名转换为统一列表。
    """
    if isinstance(value, dict):
        return [str(value[index]).strip().casefold() for index in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [str(item).strip().casefold() for item in value]
    return []
