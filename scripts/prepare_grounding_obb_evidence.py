#!/usr/bin/env python3
"""Generate grounding evidence with the repository YOLOv5-OBB ONNX backend."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from agents.counting.backends.yolov5_obb_onnx import YoloV5ObbOnnxModel

MODEL_CLASSES = [
    "plane", "baseball diamond", "bridge", "ground track field",
    "small vehicle", "large vehicle", "ship", "tennis court",
    "basketball court", "storage tank", "soccer ball field",
    "roundabout", "harbor", "swimming pool", "helicopter",
    "container crane", "airport", "helipad",
]


class EvidencePreparationError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvidencePreparationError(f"file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise EvidencePreparationError(f"row is not an object at {path}:{line_no}")
        rows.append(value)
    return rows


def planner_output(row: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
    direct = row.get("planner_output")
    if isinstance(direct, Mapping):
        return dict(direct)
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise EvidencePreparationError(f"planner output missing for {episode_id}")
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(str(item.get("text", "")) for item in content if isinstance(item, Mapping))
        else:
            text = str(content or "")
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    raise EvidencePreparationError(f"planner assistant output missing for {episode_id}")


def canonical(value: Any) -> str:
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())


def model_label(value: Any) -> str:
    return canonical(value).replace(" ", "-")


def image_path(root: Path, relative: str) -> Path:
    candidates = [(root / relative).resolve()]
    if relative.startswith(("images/train/", "images/val/")):
        candidates.append((root / "Images_train" / Path(relative).name).resolve())
    for path in candidates:
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise EvidencePreparationError(f"image escapes root: {relative}") from exc
        if path.is_file():
            return path
    raise EvidencePreparationError(f"image does not exist under root: {relative}")


def normalized_aabb(polygon: Any, width: int, height: int) -> list[float]:
    points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1 = max(0.0, min(999.0, min(xs) / width * 999.0))
    y1 = max(0.0, min(999.0, min(ys) / height * 999.0))
    x2 = max(0.0, min(999.0, max(xs) / width * 999.0))
    y2 = max(0.0, min(999.0, max(ys) / height * 999.0))
    if not (x1 < x2 and y1 < y2):
        raise EvidencePreparationError("detector returned a degenerate box")
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def main(args: argparse.Namespace) -> int:
    roots = {"vrs": Path(args.vrs_image_root).resolve(), "xlrs": Path(args.xlrs_image_root).resolve()}
    targets = read_jsonl(Path(args.targets_jsonl))
    planner_rows = {str(row.get("episode_id")): row for row in read_jsonl(Path(args.planner_jsonl))}
    model = YoloV5ObbOnnxModel(
        Path(args.weights), MODEL_CLASSES, device=args.device,
        require_cuda=True, allow_cpu_fallback=False,
    )
    model_names = {int(key): model_label(value) for key, value in model.names.items()}
    supported = set(model_names.values())
    output: list[dict[str, Any]] = []
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for index, target in enumerate(targets, 1):
        episode_id = str(target.get("episode_id") or "")
        if not episode_id:
            raise EvidencePreparationError("target row is missing episode_id")
        planner_row = planner_rows.get(episode_id)
        if planner_row is None:
            raise EvidencePreparationError(f"planner row missing for {episode_id}")
        plan = planner_output(planner_row, episode_id)
        categories = plan.get("object_categories", [])
        if not isinstance(categories, list):
            raise EvidencePreparationError(f"object_categories is not a list for {episode_id}")
        requested = {canonical(category) for category in categories}
        allowed = {category for category in requested if model_label(category) in supported}
        unsupported = requested - allowed
        source = str(target.get("image_source") or args.image_source)
        if source not in roots:
            source = args.image_source
        image = Image.open(image_path(roots[source], str(target.get("image") or ""))).convert("RGB")
        result = model.predict(
            image, conf=args.conf, iou=args.iou, imgsz=1024,
            device=args.device, max_det=args.max_detections, verbose=False,
        )[0]
        polygons = result.obb.xyxyxyxy
        scores = result.obb.conf.tolist() if hasattr(result.obb.conf, "tolist") else result.obb.conf
        class_ids = result.obb.cls.tolist() if hasattr(result.obb.cls, "tolist") else result.obb.cls
        candidates: list[tuple[float, dict[str, Any]]] = []
        for polygon, score, class_id in zip(polygons, scores, class_ids):
            label = model_names[int(class_id)]
            if canonical(label) not in allowed:
                continue
            candidates.append((float(score), {"label": label, "box": normalized_aabb(polygon, image.width, image.height)}))
        candidates.sort(key=lambda item: item[0], reverse=True)
        items = [item for _, item in candidates[:args.max_evidence_per_sample]]
        output.append({"episode_id": episode_id, "evidence_items": items})
        stat = stats[source]
        stat["rows"] += 1
        stat["nonempty"] += bool(items)
        stat["candidate_items"] += len(items)
        stat["planner_categories"] += len(requested)
        stat["unsupported_categories"] += len(unsupported)
        if not unsupported:
            stat["fully_supported_plans"] += 1
        if index % 50 == 0:
            print(f"processed={index}/{len(targets)}", file=sys.stderr)
    destination = Path(args.output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(destination), "rows": len(output),
        "provider": model.resolved_provider, "device": model.resolved_device,
        "cpu_fallback": model.cpu_fallback_used,
        "stats": {key: dict(value) for key, value in stats.items()},
    }, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-jsonl", required=True)
    parser.add_argument("--planner-jsonl", required=True)
    parser.add_argument("--vrs-image-root", required=True)
    parser.add_argument("--xlrs-image-root", required=True)
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=1000)
    parser.add_argument("--max-evidence-per-sample", type=int, default=32)
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main(build_parser().parse_args()))
    except (EvidencePreparationError, json.JSONDecodeError) as exc:
        print(f"grounding evidence preparation rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)
