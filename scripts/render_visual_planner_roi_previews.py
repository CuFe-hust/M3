from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def normalized_coordinate(value: int, extent: int) -> int:
    """Map the closed 0..999 planner frame to a closed pixel extent.
    将闭区间 0..999 planner 坐标映射到闭区间像素范围。"""
    return round(value * max(0, extent - 1) / 999)


def render_roi(source: Path, destination: Path, roi: list[int]) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = (
        normalized_coordinate(roi[0], width),
        normalized_coordinate(roi[1], height),
        normalized_coordinate(roi[2], width),
        normalized_coordinate(roi[3], height),
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(3, round(min(width, height) * 0.007))
    draw.rectangle((x0, y0, x1, y1), fill=(255, 92, 32, 38))
    draw.rectangle((x0, y0, x1, y1), outline=(255, 92, 32, 255), width=line_width)
    label = f"ROI [{roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}]"
    font = ImageFont.load_default(size=max(12, round(min(width, height) * 0.022)))
    bounds = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    label_width = bounds[2] - bounds[0]
    label_height = bounds[3] - bounds[1]
    padding = max(4, line_width)
    label_x = min(max(0, x0), max(0, width - label_width - padding * 2))
    label_y = y0 - label_height - padding * 2
    if label_y < 0:
        label_y = min(height - label_height - padding * 2, y1 + line_width)
    draw.rectangle(
        (label_x, label_y, label_x + label_width + padding * 2, label_y + label_height + padding * 2),
        fill=(21, 36, 51, 225),
    )
    draw.text(
        (label_x + padding, label_y + padding),
        label,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=1,
        stroke_fill=(21, 36, 51, 255),
    )
    rendered = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(destination, format="JPEG", quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    rendered = 0
    implicit = 0
    for row in read_jsonl(root / "results.jsonl"):
        index = int(row["index"])
        sample_id = str(row["sample_id"])
        call_dir = root / "samples" / f"{index:03d}-{sample_id}" / "visual_task_plan"
        parsed = read_json(call_dir / "parsed.json")
        request = parsed.get("region_request") or {}
        roi = request.get("roi_xyxy") if request.get("explicit") else None
        if roi is None:
            implicit += 1
            continue
        source = root / "image_previews" / f"{row['image_sha256']}.jpg"
        destination = root / "roi_previews" / f"{index:03d}-{sample_id}.jpg"
        render_roi(source, destination, roi)
        rendered += 1
    print(json.dumps({"rendered_roi_previews": rendered, "implicit_previews": implicit}))


if __name__ == "__main__":
    main()
