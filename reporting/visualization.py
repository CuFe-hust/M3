"""Read-only visual asset materialization for Report V2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, UnidentifiedImageError

from agents.counting.schema import CountingResult
from agents.schema import AgentResult, VisualEvidence
from data.schema import UnifiedSample
from reporting.adapters import load_payload, load_run_request, load_sample, sample_dir_for_row
from reporting.schema import Report, VisualAssetView

_ACCEPTED_COLOR = (34, 197, 94)
_REJECTED_COLOR = (239, 68, 68)
_GT_COLOR = (56, 189, 248)
_UNRESOLVED_COLOR = (245, 158, 11)
_REVIEWER_COLOR = (168, 85, 247)


def render_counting_overlay(
    image: Image.Image, *, result: CountingResult, output_path: Path,
) -> Path:
    """Draw thin point rings and true OBB polygons without mutating source."""

    canvas = image.convert("RGB").copy()
    if canvas.size != (result.source_width, result.source_height):
        raise ValueError("counting overlay image size does not match CountingResult dimensions")
    _draw_counting_points(canvas, result, scale_x=1.0, scale_y=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=False)
    return output_path


def _draw_counting_points(
    canvas: Image.Image, result: CountingResult, *, scale_x: float, scale_y: float,
) -> None:
    draw = ImageDraw.Draw(canvas)
    width = _line_width(*canvas.size)
    unresolved = set(result.unresolved_conflicts)
    for point in result.global_points:
        color = (
            _UNRESOLVED_COLOR if point.global_id in unresolved
            else _ACCEPTED_COLOR if point.accepted
            else _REJECTED_COLOR
        )
        provenance = point.provenance
        polygon = provenance.obb_polygon_global_px if provenance is not None else None
        if polygon and len(polygon) >= 3:
            scaled = [(pair[0] * scale_x, pair[1] * scale_y) for pair in polygon]
            draw.line(scaled + [scaled[0]], fill=color, width=width)
        x = point.global_x_px * scale_x
        y = point.global_y_px * scale_y
        radius = max(3, round(point.radius_px * min(scale_x, scale_y)))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color, width=width,
        )


def materialize_report_assets(
    run_dir: Path, report: Report, report_dir: Path, *, max_visual_samples: int = 200,
) -> Report:
    """Materialize selected previews and return an updated report copy.

    Execution artifacts and dataset images are read-only. Only ``report_dir``
    is written. Missing or unsafe source paths degrade to explicit statuses.
    """

    request = load_run_request(run_dir)
    dataset_root = Path(request.dataset_root) if request is not None else None
    updated = report.model_copy(deep=True)
    ranked = sorted(
        (sample for sample in updated.samples if sample.visuals),
        key=lambda sample: (_sample_priority(sample), sample.run_task, sample.sample_id),
    )
    selected_samples = {
        (sample.run_task, sample.sample_id)
        for sample in ranked[:max(0, max_visual_samples)]
    }
    assets_dir = report_dir / "assets"
    for sample in updated.samples:
        sample_dir = sample_dir_for_row(
            run_dir, {"run_task": sample.run_task, "sample_id": sample.sample_id}
        )
        source_sample = load_sample(sample_dir) if sample_dir is not None else None
        payload = load_payload(sample_dir, sample.task) if sample_dir is not None else None
        for visual in sample.visuals:
            if (sample.run_task, sample.sample_id) not in selected_samples:
                visual.status = "omitted_by_budget"
                continue
            _materialize_visual(
                dataset_root, source_sample, payload, visual, assets_dir,
                identity=f"{sample.run_task}\0{sample.sample_id}\0{visual.image_id}",
            )
    updated.visual_total = sum(len(sample.visuals) for sample in updated.samples)
    updated.visual_materialized_count = sum(
        visual.status == "available" for sample in updated.samples for visual in sample.visuals
    )
    return updated


def _materialize_visual(
    dataset_root: Path | None, sample: UnifiedSample | None, payload: object | None,
    visual: VisualAssetView, assets_dir: Path, *, identity: str,
) -> None:
    if dataset_root is None or sample is None:
        visual.status = "missing_source"
        return
    image_ref = next((item for item in sample.images if item.image_id == visual.image_id), None)
    if image_ref is None:
        visual.status = "invalid_source"
        return
    try:
        root = dataset_root.resolve(strict=True)
        source = (root / image_ref.path).resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file():
            visual.status = "invalid_source"
            return
    except (OSError, RuntimeError):
        visual.status = "missing_source"
        return
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    original_rel = f"assets/{digest}-original.webp"
    overlay_rel = f"assets/{digest}-overlay.png"
    try:
        with Image.open(source) as opened:
            opened.load()
            image = opened.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError):
        visual.status = "invalid_source"
        return
    visual.width, visual.height = image.size
    assets_dir.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    preview.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    preview.save(assets_dir / Path(original_rel).name, format="WEBP", quality=85, method=6)
    visual.original_asset = original_rel
    if isinstance(payload, CountingResult):
        if image.size != (payload.source_width, payload.source_height):
            visual.status = "dimension_mismatch"
            return
        if len(sample.images) != 1:
            visual.status = "unsupported_geometry"
            return
        canvas = preview.copy()
        _draw_counting_points(
            canvas,
            payload,
            scale_x=canvas.width / image.width,
            scale_y=canvas.height / image.height,
        )
        canvas.save(assets_dir / Path(overlay_rel).name, format="PNG", optimize=False)
        visual.overlay_asset = overlay_rel
        visual.status = "available"
        return
    if isinstance(payload, AgentResult):
        evidence = list(_evidence_for_image(payload.evidence_items, sample, visual.image_id))
        evidence_boxes = {
            tuple(float(value) for value in item.box)
            for item in evidence if item.box is not None
        }
        top_level_boxes = (
            [list(box) for box in payload.boxes if tuple(float(value) for value in box) not in evidence_boxes]
            if len(sample.images) == 1
            else []
        )
        gt = _safe_ground_truth(sample, visual.image_id)
        if not evidence and not top_level_boxes and not gt:
            visual.status = "unsupported_geometry"
            return
        canvas = preview.copy()
        draw = ImageDraw.Draw(canvas)
        for item in evidence:
            _draw_normalized(draw, item, canvas.size, _ACCEPTED_COLOR)
        for box in top_level_boxes:
            _draw_geometry(draw, [float(value) for value in box], canvas.size, _ACCEPTED_COLOR)
        for geometry in gt:
            _draw_geometry(draw, geometry, canvas.size, _GT_COLOR)
        canvas.save(assets_dir / Path(overlay_rel).name, format="PNG", optimize=False)
        visual.overlay_asset = overlay_rel
        visual.status = "available"
        return
    visual.status = "unsupported_geometry"


def _evidence_for_image(
    items: Iterable[VisualEvidence], sample: UnifiedSample, image_id: str,
) -> Iterable[VisualEvidence]:
    for item in items:
        if item.image_id == image_id or (item.image_id is None and len(sample.images) == 1):
            yield item


def _safe_ground_truth(sample: UnifiedSample, image_id: str) -> list[list[float]]:
    if sample.ground_truth is None or sample.ground_truth.coordinate_frame != "normalized_0_999_top_left":
        return []
    if len(sample.images) != 1 or sample.images[0].image_id != image_id:
        return []
    return [list(box) for box in sample.ground_truth.boxes] + [list(point) for point in sample.ground_truth.points]


def _draw_normalized(
    draw: ImageDraw.ImageDraw, item: VisualEvidence, size: tuple[int, int], color: tuple[int, int, int],
) -> None:
    geometry = item.box if item.box is not None else item.point
    if geometry is not None:
        _draw_geometry(draw, [float(value) for value in geometry], size, color)


def _draw_geometry(
    draw: ImageDraw.ImageDraw, geometry: list[float], size: tuple[int, int], color: tuple[int, int, int],
) -> None:
    width, height = size
    line_width = _line_width(width, height)
    points = [(geometry[index] / 999 * (width - 1), geometry[index + 1] / 999 * (height - 1))
              for index in range(0, len(geometry), 2)]
    if len(points) == 1:
        x, y = points[0]
        radius = max(3, line_width * 3)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)
    elif len(points) == 2:
        draw.rectangle((*points[0], *points[1]), outline=color, width=line_width)
    elif len(points) >= 3:
        draw.line(points + [points[0]], fill=color, width=line_width)


def _sample_priority(sample: object) -> int:
    state = getattr(sample, "state", "")
    if state == "failed":
        return 0
    if state == "partial":
        return 1
    if getattr(sample, "result_quality", "") == "incorrect":
        return 2
    if getattr(sample, "fallback_used", False):
        return 3
    if getattr(sample, "warnings", []):
        return 4
    return 5


def _line_width(width: int, height: int) -> int:
    return max(1, min(2, round(min(width, height) / 900)))
