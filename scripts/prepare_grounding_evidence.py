#!/usr/bin/env python3
"""Generate per-sample YOLO evidence for grounding SFT preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ALIASES = {
    "baseball-diamond": "baseball_diamond",
    "basketball-court": "basketball_court",
    "ground-track-field": "ground_track_field",
    "large-vehicle": "large_vehicle",
    "roundabout": "roundabout",
    "small-vehicle": "small_vehicle",
    "soccer-ball-field": "soccer_ball_field",
    "storage-tank": "storage_tank",
    "swimming-pool": "swimming_pool",
    "tennis-court": "tennis_court",
}


class EvidencePreparationError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvidencePreparationError(f"file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidencePreparationError(f"invalid JSON at {path}:{line_no}") from exc
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
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvidencePreparationError(f"planner output is not JSON for {episode_id}") from exc
        if not isinstance(value, dict):
            raise EvidencePreparationError(f"planner output is not an object for {episode_id}")
        return value
    raise EvidencePreparationError(f"planner assistant message missing for {episode_id}")


def image_path(root: Path, relative: str, prefix: str | None) -> Path:
    if prefix:
        relative = f"{prefix.rstrip('/')}/{Path(relative).name}"
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise EvidencePreparationError(f"image escapes root: {relative}") from exc
    if not path.is_file():
        raise EvidencePreparationError(f"image does not exist: {path}")
    return path


def main(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise EvidencePreparationError("ultralytics is required for YOLO evidence generation") from exc

    targets = read_jsonl(Path(args.targets_jsonl))
    planner_rows = {str(row.get("episode_id")): row for row in read_jsonl(Path(args.planner_jsonl))}
    model = YOLO(args.weights)
    model_names = {int(key): str(value) for key, value in model.names.items()}
    supported = {value.replace("_", "-"): key for key, value in model_names.items()}
    output: list[dict[str, Any]] = []
    for index, target in enumerate(targets, 1):
        episode_id = str(target.get("episode_id") or "")
        if not episode_id:
            raise EvidencePreparationError("target row is missing episode_id")
        planner = planner_rows.get(episode_id)
        if planner is None:
            raise EvidencePreparationError(f"planner row missing for {episode_id}")
        plan = planner_output(planner, episode_id)
        categories = plan.get("object_categories", [])
        if not isinstance(categories, list):
            raise EvidencePreparationError(f"object_categories is not a list for {episode_id}")
        class_ids = []
        for category in categories:
            name = str(category).strip().lower()
            name = ALIASES.get(name, name).replace("_", "-")
            if name in supported:
                class_ids.append(supported[name])
        if not class_ids:
            output.append({"episode_id": episode_id, "evidence_items": []})
            continue
        image = image_path(Path(args.image_root).resolve(), str(target.get("image") or ""), args.image_prefix)
        result = model.predict(
            source=str(image),
            imgsz=args.imgsz,
            conf=args.conf,
            classes=sorted(set(class_ids)) or None,
            device=args.device,
            verbose=False,
        )[0]
        width, height = result.orig_shape[1], result.orig_shape[0]
        candidates: list[tuple[float, dict[str, Any]]] = []
        if result.boxes is not None:
            for box, score, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                candidates.append((float(score), {
                    "label": model_names[int(class_id)].replace("_", "-"),
                    "box": [
                        round(max(0.0, min(999.0, x1 / width * 999)), 3),
                        round(max(0.0, min(999.0, y1 / height * 999)), 3),
                        round(max(0.0, min(999.0, x2 / width * 999)), 3),
                        round(max(0.0, min(999.0, y2 / height * 999)), 3),
                    ],
                }))
        candidates.sort(key=lambda item: item[0], reverse=True)
        output.append({"episode_id": episode_id, "evidence_items": [item for _, item in candidates[:args.max_evidence_per_sample]]})
        if index % 50 == 0:
            print(f"processed={index}/{len(targets)}", file=sys.stderr)
    destination = Path(args.output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"output": str(destination), "rows": len(output)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-jsonl", required=True)
    parser.add_argument("--planner-jsonl", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-prefix")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-evidence-per-sample", type=int, default=32)
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main(build_parser().parse_args()))
    except EvidencePreparationError as exc:
        print(f"grounding evidence preparation rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)
