"""Unit coverage for the lazy YOLOv5-OBB ONNX geometry adapter.
YOLOv5-OBB ONNX 几何适配器的单元测试。
"""

from __future__ import annotations

import math

import numpy as np

from spacers_agent.agents.counting.backends.yolov5_obb_onnx import YoloV5ObbOnnxModel
from spacers_agent.schemas import YoloDetectorSettings


def test_yolov5_obb_polygon_uses_image_y_axis_theta_convention() -> None:
    model = object.__new__(YoloV5ObbOnnxModel)
    model._np = np
    polygon = model._to_source_polygon(np.array([100.0, 100.0, 80.0, 20.0, math.pi / 4]), 1.0, (0, 0), 200, 200)
    long_edge = polygon[0] - polygon[3]
    assert long_edge[0] > 0
    assert long_edge[1] < 0


def test_onnx_runtime_is_an_explicit_detector_option() -> None:
    detector = YoloDetectorSettings(
        name="onnx", enabled=True, weights="model.onnx", runtime="onnx_yolov5_obb", model_id="test",
        sha256="0" * 64, classes=["small vehicle", "large vehicle"], composite_targets={"vehicle": ["small vehicle", "large vehicle"]},
    )
    assert detector.runtime == "onnx_yolov5_obb"
