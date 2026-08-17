"""Annotate one image with the repo's YOLOv5-OBB CSL ONNX model and save to Desktop."""
# 用仓库的 YOLOv5-OBB CSL ONNX 模型标注单张图片并保存到桌面。

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

IMAGE_PATH = Path("/Users/troy/Desktop/project/M3/data/VRSBench-full/Images_val/P0003_0002.png")
WEIGHTS = Path("/Users/troy/Desktop/project/M3/models/yolo_obb/yolov5m_obb_csl_dotav20.onnx")
OUT_PATH = Path.home() / "Desktop" / "P0003_0002_yolo_obb_annotated.png"

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
    results = model.predict(
        image,
        conf=CONF,
        iou=IOU,
        imgsz=1024,
        device="cpu",
        max_det=MAX_DET,
        verbose=False,
    )
    obb = results[0].obb
    polygons = obb.xyxyxyxy
    class_ids = obb.cls.astype(int).tolist()
    scores = obb.conf.astype(float).tolist()
    print(f"detections: {len(polygons)}")

    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
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

    out_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    Image.fromarray(out_bgr).save(OUT_PATH)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
