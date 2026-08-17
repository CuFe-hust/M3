"""Annotate one image with the repo's YOLOv5-OBB CSL ONNX model and save to Desktop.

Variant: the original image is first resized to the model's official input size
1024x1024 via interpolation (stretch, no letterbox padding).
变体：先将原图通过插值拉伸到模型官方输入尺寸 1024x1024，再做推理。
"""

import sys
from pathlib import Path

sys.path.insert(0, "/Users/troy/Desktop/project/M3")

import cv2
import numpy as np
from PIL import Image

from agents.counting.backends.yolov5_obb_onnx import YoloV5ObbOnnxModel

CLASSES = [
    "plane", "baseball diamond", "bridge", "ground track field", "small vehicle",
    "large vehicle", "ship", "tennis court", "basketball court", "storage tank",
    "soccer ball field", "roundabout", "harbor", "swimming pool", "helicopter",
    "container crane", "airport", "helipad",
]
CONF = 0.20
IOU = 0.50
MAX_DET = 1000
MODEL_SIZE = 1024

IMAGE_PATH = Path("/Users/troy/Desktop/project/M3/data/VRSBench-full/Images_val/P0003_0002.png")
WEIGHTS = Path("/Users/troy/Desktop/project/M3/models/yolo_obb/yolov5m_obb_csl_dotav20.onnx")
OUT_PATH = Path.home() / "Desktop" / "P0003_0002_yolo_obb_1024_annotated.png"

# Deterministic per-class colors.
# 每类固定颜色。
_rng = np.random.default_rng(42)
COLORS = {name: tuple(int(c) for c in _rng.integers(0, 255, 3)) for name in CLASSES}


def main() -> None:
    model = YoloV5ObbOnnxModel(
        WEIGHTS, CLASSES, device="cpu", require_cuda=False, allow_cpu_fallback=False
    )
    print("providers:", model.providers)
    image = Image.open(IMAGE_PATH).convert("RGB")
    # Stretch-resize the original to the model's official input size.
    # 将原图直接拉伸插值到模型官方输入尺寸。
    resized = cv2.resize(
        np.asarray(image), (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR
    )
    resized_pil = Image.fromarray(resized)
    results = model.predict(
        resized_pil,
        conf=CONF,
        iou=IOU,
        imgsz=MODEL_SIZE,
        device="cpu",
        max_det=MAX_DET,
        verbose=False,
    )
    obb = results[0].obb
    polygons = obb.xyxyxyxy
    class_ids = obb.cls.astype(int).tolist()
    scores = obb.conf.astype(float).tolist()
    print(f"detections: {len(polygons)}")

    canvas = resized.copy()
    for polygon, class_id, score in zip(polygons, class_ids, scores):
        name = CLASSES[class_id]
        color = COLORS[name]
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        label = f"{name} {score:.2f}"
        x, y = int(polygon[:, 0].min()), int(polygon[:, 1].min())
        y = max(14, y - 4)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
        cv2.putText(
            canvas, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    # Canvas is already RGB; no channel conversion here (a BGR conversion
    # would swap R/B). 画布已是 RGB，此处不再做通道转换（BGR 转换会互换 R/B）。
    Image.fromarray(canvas).save(OUT_PATH)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
