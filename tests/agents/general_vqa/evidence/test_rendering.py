"""Tests for v2 evidence rendering.

v2 证据渲染测试：裁切只接受已物化的源像素框，预览只缩小，mask overlay
保持纯内存且不推导框或计数。
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image
import pytest

from agents.general_vqa.evidence.rendering import (
    make_preview,
    overlay_mask,
    preview_from_path,
    render_roi_crop,
    stable_palette_color,
)
from agents.general_vqa.evidence.schema import RoiEvidenceRecord
from models.images import crop_image_box


def _image(size: tuple[int, int], fill: int = 7) -> Image.Image:
    return Image.new("RGB", size, (fill, fill + 1, fill + 2))


def _record(crop_size: tuple[int, int] = (600, 480)) -> RoiEvidenceRecord:
    return RoiEvidenceRecord(
        roi_id="roi-1",
        image_id="img1",
        source_size=(1000, 800),
        core_xyxy=(250, 200, 750, 600),
        expanded_xyxy=(200, 160, 800, 640),
        crop_size=crop_size,
    )


def test_render_roi_crop_uses_exact_materialized_pixel_box() -> None:
    image = _image((1000, 800))
    record = _record()
    crop = render_roi_crop(image, record)
    assert crop.size == record.crop_size
    assert crop.tobytes() == crop_image_box(image, record.expanded_xyxy).tobytes()


def test_render_roi_crop_detects_geometry_drift() -> None:
    image = _image((1000, 800))
    drift = _record().model_copy(update={"crop_size": (100, 100)})
    with pytest.raises(ValueError, match="crop drift"):
        render_roi_crop(image, drift)


def test_render_roi_crop_never_modifies_source() -> None:
    image = _image((1000, 800))
    before = image.tobytes()
    render_roi_crop(image, _record())
    assert image.tobytes() == before


def test_preview_and_preview_transport_are_shrink_only(tmp_path: Path) -> None:
    assert make_preview(_image((4000, 2000))).size == (1080, 540)
    assert make_preview(_image((400, 300))).size == (400, 300)

    path = tmp_path / "large.png"
    _image((4000, 2000)).save(path, format="PNG")
    first = preview_from_path(path)
    assert first == preview_from_path(path)
    url, digest = first
    payload = base64.b64decode(url.split(";base64,", 1)[1])
    assert url.startswith("data:image/png;base64,")
    assert hashlib.sha256(payload).hexdigest() == digest
    assert Image.open(io.BytesIO(payload)).size == (1080, 540)


def test_preview_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        preview_from_path(tmp_path / "missing.png")


def test_overlay_mask_is_pure_and_deterministic() -> None:
    source = _image((100, 100), fill=10)
    mask = Image.new("L", (100, 100), 0)
    mask.putpixel((50, 50), 255)
    before = source.tobytes()
    first = overlay_mask(source, mask, color=(255, 0, 0))
    second = overlay_mask(source, mask, color=(255, 0, 0))
    assert source.tobytes() == before
    assert first.tobytes() == second.tobytes()
    assert first.getpixel((50, 50)) != source.getpixel((50, 50))
    assert first.getpixel((0, 0)) == source.getpixel((0, 0))


def test_overlay_mask_rejects_size_mismatch_and_has_no_geometry_api() -> None:
    with pytest.raises(ValueError, match="must match"):
        overlay_mask(_image((100, 100)), Image.new("L", (50, 50)), color=(0, 255, 0))
    assert stable_palette_color("small_vehicle") == stable_palette_color("small_vehicle")
    assert stable_palette_color("small_vehicle") != stable_palette_color("building")
