"""Lazy ONNX Runtime adapter for YOLOv5-OBB CSL models.

YOLOv5-OBB CSL 模型的延迟 ONNX Runtime 适配器。导入本模块不加载任何
运行时；缺失可选依赖时抛出专用异常。
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents.errors import DetectorInferenceError, OptionalDependencyMissingError


def _validate_device(device: str, *, require_cuda: bool) -> None:
    """CUDA mode requires a non-negative integer device id; CPU-only mode
    requires device='cpu' and is never gated by allow_cpu_fallback.
    CUDA 模式要求非负整数设备号；显式 CPU-only 模式要求 device='cpu'，且
    绝不受 allow_cpu_fallback 门控。"""
    if require_cuda:
        if not device.isdigit():
            raise ValueError(
                "CUDA mode requires a non-negative integer device id, "
                f"got {device!r}"
            )
        return
    if device != "cpu":
        raise ValueError(f"CPU mode requires device='cpu', got {device!r}")


class YoloV5ObbOnnxModel:
    """Expose DOTA CSL ONNX predictions through the existing OBB result contract.
    通过既有 OBB 结果契约暴露 DOTA CSL ONNX 预测。"""

    task = "obb"

    def __init__(
        self,
        weights: Path,
        classes: list[str],
        *,
        device: str = "0",
        require_cuda: bool = True,
        allow_cpu_fallback: bool = False,
    ) -> None:
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as exc:
            if require_cuda:
                dependency = "onnxruntime-gpu, numpy, opencv-python"
                install_hint = "pip install onnxruntime-gpu numpy opencv-python"
            else:
                dependency = "onnxruntime, numpy, opencv-python"
                install_hint = "pip install onnxruntime numpy opencv-python"
            raise OptionalDependencyMissingError(
                "yolo",
                dependency=dependency,
                install_hint=install_hint,
            ) from exc
        self._cv2 = cv2
        self._np = np
        self._device = device
        self._require_cuda = require_cuda
        self._allow_cpu_fallback = allow_cpu_fallback
        if not require_cuda and allow_cpu_fallback:
            raise ValueError("CPU-only mode must not enable CPU fallback")
        if require_cuda:
            # Preload NVIDIA site-package CUDA/cuDNN libraries only for the
            # explicitly requested CUDA execution provider.
            ort.preload_dlls(directory="")
            providers: list[object] = [
                ("CUDAExecutionProvider", {"device_id": int(device)})
            ]
            if allow_cpu_fallback:
                providers.append("CPUExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(weights), providers=providers)
        actual = tuple(self._session.get_providers())
        self.providers = actual
        self.requested_provider = (
            "CUDAExecutionProvider" if require_cuda else "CPUExecutionProvider"
        )
        self.requested_device = device
        if not require_cuda:
            if actual != ("CPUExecutionProvider",):
                raise DetectorInferenceError(
                    "CPU-only ONNX detector resolved unexpected execution provider"
                )
            self.cpu_fallback_used = False
            self.resolved_provider = "CPUExecutionProvider"
            self.resolved_device = "cpu"
        elif "CUDAExecutionProvider" not in actual:
            if not allow_cpu_fallback:
                raise DetectorInferenceError(
                    "CUDAExecutionProvider required but unavailable for ONNX detector"
                )
            self.cpu_fallback_used = True
            self.resolved_provider = "CPUExecutionProvider"
            self.resolved_device = "cpu"
        else:
            self.cpu_fallback_used = False
            self.resolved_provider = "CUDAExecutionProvider"
            self.resolved_device = device
        self.names = {index: name for index, name in enumerate(classes)}
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "YOLOv5-OBB ONNX model must expose exactly one input and one output"
            )
        input_shape = list(inputs[0].shape)
        output_shape = list(outputs[0].shape)
        if input_shape[:2] != [1, 3] or input_shape[2:] != [1024, 1024]:
            raise ValueError(
                f"YOLOv5-OBB ONNX input must be [1, 3, 1024, 1024], got {input_shape!r}"
            )
        # Actual model input size; consumed by the shared detection seam so
        # outputs report the real letterboxed resolution, not a configured one.
        # 模型实际输入尺寸；共享检测 seam 消费该值，使输出报告真实 letterbox
        # 分辨率而非配置值。
        self.model_input_size = (1024, 1024)
        expected_channels = 5 + len(classes) + 180
        if output_shape[-1] != expected_channels:
            raise ValueError(
                f"YOLOv5-OBB ONNX output must end with {expected_channels} channels, "
                f"got {output_shape!r}"
            )
        self._input_name = inputs[0].name

    def predict(
        self,
        source: Any,
        *,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        max_det: int,
        verbose: bool,
    ) -> list[SimpleNamespace]:
        """Run one fixed-size ONNX image and return an Ultralytics-like OBB
        payload. 运行一张固定尺寸 ONNX 图像并返回类似 Ultralytics 的 OBB 载荷。"""
        if imgsz != 1024:
            raise ValueError(f"YOLOv5-OBB ONNX requires image_size=1024, got {imgsz}")
        _validate_device(device, require_cuda=self._require_cuda)
        if device != self._device:
            raise ValueError(
                "predict device differs from initialized ONNX device: "
                f"initialized {self._device!r}, got {device!r}"
            )
        image = self._np.asarray(source.convert("RGB"))
        prepared, ratio, padding = self._letterbox(image)
        tensor = (
            self._np.ascontiguousarray(prepared.transpose(2, 0, 1)[None])
            .astype(self._np.float32)
            / 255.0
        )
        output = self._session.run(None, {self._input_name: tensor})[0][0]
        detections = self._nms(self._decode(output, conf), conf, iou, max_det)
        polygons = [
            self._to_source_polygon(box, ratio, padding, image.shape[1], image.shape[0])
            for _, _, box in detections
        ]
        obb = SimpleNamespace(
            xyxyxyxy=polygons,
            cls=self._np.asarray(
                [class_id for class_id, _, _ in detections], dtype=self._np.float32
            ),
            conf=self._np.asarray(
                [score for _, score, _ in detections], dtype=self._np.float32
            ),
        )
        return [SimpleNamespace(obb=obb)]

    def _letterbox(self, image: Any) -> tuple[Any, float, tuple[int, int]]:
        height, width = image.shape[:2]
        ratio = min(1024 / height, 1024 / width)
        resized = self._cv2.resize(
            image,
            (round(width * ratio), round(height * ratio)),
            interpolation=self._cv2.INTER_LINEAR,
        )
        pad_x, pad_y = (1024 - resized.shape[1]) / 2, (1024 - resized.shape[0]) / 2
        left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
        top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
        bordered = self._cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            self._cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return bordered, ratio, (left, top)

    def _decode(self, prediction: Any, confidence: float) -> list[tuple[int, float, Any]]:
        class_count = len(self.names)
        class_scores = prediction[:, 5:5 + class_count] * prediction[:, 4:5]
        selected, class_ids = self._np.where(class_scores >= confidence)
        scores = class_scores[selected, class_ids]
        angles = (
            prediction[selected, 5 + class_count:].argmax(axis=1) / 180.0 * math.pi
            - math.pi / 2.0
        )
        return [
            (
                int(class_id),
                float(score),
                self._np.concatenate((prediction[index, :4], [angle])),
            )
            for index, class_id, score, angle in zip(selected, class_ids, scores, angles)
        ]

    def _nms(
        self,
        candidates: list[tuple[int, float, Any]],
        confidence: float,
        iou: float,
        max_det: int,
    ) -> list[tuple[int, float, Any]]:
        kept: list[tuple[int, float, Any]] = []
        for class_id in sorted({item[0] for item in candidates}):
            group = [item for item in candidates if item[0] == class_id]
            rectangles = [
                (
                    (float(box[0]), float(box[1])),
                    (float(box[2]), float(box[3])),
                    float(-box[4] * 180 / math.pi),
                )
                for _, _, box in group
            ]
            indices = self._cv2.dnn.NMSBoxesRotated(
                rectangles, [score for _, score, _ in group], confidence, iou
            )
            if len(indices):
                kept.extend(group[int(index)] for index in self._np.asarray(indices).reshape(-1))
        return sorted(kept, key=lambda item: item[1], reverse=True)[:max_det]

    def _to_source_polygon(
        self,
        box: Any,
        ratio: float,
        padding: tuple[int, int],
        width: int,
        height: int,
    ) -> Any:
        center_x, center_y, long_side, short_side, theta = (float(value) for value in box)
        cosine, sine = math.cos(theta), math.sin(theta)
        long_vector = (long_side / 2 * cosine, -long_side / 2 * sine)
        short_vector = (-short_side / 2 * sine, -short_side / 2 * cosine)
        corners = self._np.asarray(
            [
                (
                    center_x + long_vector[0] + short_vector[0],
                    center_y + long_vector[1] + short_vector[1],
                ),
                (
                    center_x + long_vector[0] - short_vector[0],
                    center_y + long_vector[1] - short_vector[1],
                ),
                (
                    center_x - long_vector[0] - short_vector[0],
                    center_y - long_vector[1] - short_vector[1],
                ),
                (
                    center_x - long_vector[0] + short_vector[0],
                    center_y - long_vector[1] + short_vector[1],
                ),
            ],
            dtype=self._np.float32,
        )
        corners[:, 0] = self._np.clip((corners[:, 0] - padding[0]) / ratio, 0, width - 1)
        corners[:, 1] = self._np.clip((corners[:, 1] - padding[1]) / ratio, 0, height - 1)
        return corners
