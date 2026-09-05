"""Stretch one image to 1024 square, run DOTA YOLO11m-OBB, and annotate it.

将单张图片拉伸到 1024 正方形后运行 DOTA YOLO11m-OBB 并标注。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "yolo_obb" / "dota_v2_yolo11m_obb_best.pt"
CONF = 0.20
IOU = 0.50
MAX_DET = 1000
MODEL_SIZE = 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image_path = args.image.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path.home() / "Desktop" / f"{image_path.stem}_yolo_obb_1024_annotated.png"
    )
    image = Image.open(image_path).convert("RGB")
    resized = cv2.resize(
        np.asarray(image),
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    model = YOLO(str(weights), task="obb")
    results = model.predict(
        Image.fromarray(resized),
        conf=CONF,
        iou=IOU,
        imgsz=MODEL_SIZE,
        device=args.device,
        max_det=MAX_DET,
        verbose=False,
    )
    obb = results[0].obb
    polygons = obb.xyxyxyxy.cpu().numpy()
    class_ids = obb.cls.cpu().numpy().astype(int).tolist()
    scores = obb.conf.cpu().numpy().astype(float).tolist()
    names = model.names
    print(f"detections: {len(polygons)}")

    # Use deterministic per-class colors for reproducible visual inspection.
    # 使用确定性的逐类颜色，保证可复现的视觉检查。
    rng = np.random.default_rng(42)
    colors = {
        int(class_id): tuple(int(value) for value in rng.integers(0, 255, 3))
        for class_id in sorted(names)
    }
    canvas = resized.copy()
    for polygon, class_id, score in zip(polygons, class_ids, scores, strict=True):
        name = str(names[class_id])
        color = colors[class_id]
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=2)
        label = f"{name} {score:.2f}"
        x, y = int(polygon[:, 0].min()), int(polygon[:, 1].min())
        y = max(14, y - 4)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            canvas,
            (x, y - text_height - 4),
            (x + text_width + 4, y + 2),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
