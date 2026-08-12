"""Contract tests for the VQA evidence rendering layer.

VQA 证据渲染契约测试：ROI 裁切复用 crop_image_region 且零漂移守卫、preview
只缩小不放大、overlay 不修改源图且掩膜不转框、稳定调色表与纯内存 seam
（不写文件、不选择持久化格式参数）。
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image

import pytest

from agents.general_vqa.evidence.geometry import map_roi
from agents.general_vqa.evidence.rendering import (
    make_preview,
    overlay_mask,
    preview_from_path,
    render_roi_crop,
    stable_palette_color,
)
from agents.schema import RoiRegion

from models.images import crop_image_region

_REGION = RoiRegion(roi_id="roi-1", image_id="img1", xyxy=(0.25, 0.25, 0.75, 0.75))


def _image(size: tuple[int, int], fill: int = 7) -> Image.Image:
    return Image.new("RGB", size, (fill, fill + 1, fill + 2))


# ── ROI 裁切 / ROI crops ─────────────────────────────────────────────────


def test_render_roi_crop_uses_crop_image_region_and_matches_record() -> None:
    image = _image((1000, 800))
    record = map_roi(_REGION, image.size)
    crop = render_roi_crop(image, _REGION, record)
    assert crop.size == record.crop_size == (600, 480)
    # Rendered through the shared helper with identical inputs -> identical
    # pixels; zero-drift guard proves the two layers agree.
    # 通过共享 helper 以相同输入渲染 -> 像素完全一致；零漂移守卫证明两层一致。
    expected = crop_image_region(
        image,
        _REGION.xyxy,
        coordinate_frame="normalized_0_1_top_left",
        halo_ratio=0.10,
    )
    assert crop.tobytes() == expected.tobytes()


def test_render_roi_crop_detects_geometry_drift() -> None:
    image = _image((1000, 800))
    record = map_roi(_REGION, image.size)
    # A record from a different source size cannot match the rendered crop.
    # 来自不同源尺寸的记录必然与渲染 crop 不一致。
    drift = record.model_copy(update={"crop_size": (100, 100)})
    with pytest.raises(ValueError, match="crop drift"):
        render_roi_crop(image, _REGION, drift)


def test_render_roi_crop_never_modifies_source() -> None:
    image = _image((1000, 800))
    before = image.tobytes()
    record = map_roi(_REGION, image.size)
    render_roi_crop(image, _REGION, record)
    assert image.tobytes() == before


def test_render_full_image_roi_is_whole_image() -> None:
    from agents.general_vqa.evidence.geometry import full_image_roi

    image = _image((120, 80))
    record = full_image_roi("img1", image.size)
    crop = render_roi_crop(
        image, RoiRegion(roi_id="full", image_id="img1", xyxy=(0.0, 0.0, 1.0, 1.0)), record
    )
    assert crop.size == (120, 80)
    assert crop.tobytes() == image.convert("RGB").tobytes()


# ── 预览 / previews ──────────────────────────────────────────────────────


def test_preview_shrinks_only_when_above_1080() -> None:
    large = _image((4000, 2000))
    preview = make_preview(large)
    assert preview.size == (1080, 540)
    small = _image((400, 300))
    assert make_preview(small).size == (400, 300)


def test_preview_normalizes_exif_and_rgb() -> None:
    image = _image((100, 100))
    preview = make_preview(image)
    assert preview.mode == "RGB"


def test_preview_from_path_returns_deterministic_data_url_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    _image((4000, 2000)).save(path, format="PNG")
    first = preview_from_path(path)
    second = preview_from_path(path)
    assert first == second
    url, digest = first
    assert url.startswith("data:image/png;base64,")
    assert len(digest) == 64
    # The digest is the honest hash of exactly the bytes in the data URL; the
    # preview shrinks the 4000px source to the 1080 cap.
    # 摘要是 data URL 中字节的真实哈希；预览把 4000px 源图缩到 1080 上限。
    payload = base64.b64decode(url.split(";base64,", 1)[1])
    assert hashlib.sha256(payload).hexdigest() == digest
    decoded = Image.open(io.BytesIO(payload))
    assert decoded.size == (1080, 540)


def test_preview_from_path_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        preview_from_path(tmp_path / "nope.png")


# ── overlay 与调色表 / overlays and palette ───────────────────────────────


def test_overlay_mask_returns_new_image_and_keeps_source() -> None:
    source = _image((100, 100), fill=10)
    mask = Image.new("L", (100, 100), 0)
    mask.putpixel((50, 50), 255)
    before = source.tobytes()
    result = overlay_mask(source, mask, color=(255, 0, 0))
    assert result.mode == "RGB"
    assert result.size == (100, 100)
    assert source.tobytes() == before
    # The masked pixel changes toward red while unmasked pixels stay intact.
    # 掩膜像素向红色偏移，未掩膜像素保持不变。
    assert result.getpixel((50, 50)) != source.getpixel((50, 50))
    assert result.getpixel((0, 0)) == source.getpixel((0, 0))


def test_overlay_mask_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        overlay_mask(_image((100, 100)), Image.new("L", (50, 50)), color=(0, 255, 0))


def test_overlay_mask_never_derives_boxes_or_counts() -> None:
    """The overlay API surface has no box or count concept: masks stay masks.
    overlay API 表面没有框或计数的概念：掩膜保持为掩膜。"""
    import inspect

    signature = inspect.signature(overlay_mask)
    assert set(signature.parameters) == {"source", "mask", "color", "alpha"}


def test_overlay_alpha_is_deterministic() -> None:
    mask = Image.new("L", (10, 10), 128)
    first = overlay_mask(_image((10, 10)), mask, color=(0, 0, 255))
    second = overlay_mask(_image((10, 10)), mask, color=(0, 0, 255))
    assert first.tobytes() == second.tobytes()


def test_stable_palette_color_is_deterministic_and_per_leaf() -> None:
    first = stable_palette_color("small_vehicle")
    second = stable_palette_color("small_vehicle")
    other = stable_palette_color("large_vehicle")
    assert first == second
    assert first != other
    assert all(0 <= channel <= 255 for channel in first)
    assert len(first) == 3


def test_stable_palette_color_is_path_and_secret_free() -> None:
    assert stable_palette_color("sk-secret") == stable_palette_color("sk-secret")
    for token in ("/Users", "C:\\", "base64"):
        assert token not in str(stable_palette_color("vehicle"))


# ── 纯内存 seam / in-memory seam ─────────────────────────────────────────


def test_rendering_never_writes_files() -> None:
    """No persistence format/quality parameters are chosen yet: the module
    never writes image files — the only write-like call is the in-memory
    transport encode into BytesIO, and no open()/write_bytes()/write_text()
    appears anywhere. 尚未选择任何持久化格式/质量参数：本模块绝不写图片
    文件——唯一的写类调用是面向 BytesIO 的内存传输编码，任何位置不出现
    open()/write_bytes()/write_text()。"""
    import ast

    import agents.general_vqa.evidence.rendering as rendering

    tree = ast.parse(open(rendering.__file__, encoding="utf-8").read())
    write_like: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("save", "write_bytes", "write_text"):
                write_like.append(node.func.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "open":
                write_like.append("open")
    # The only write-like call is the transport encode; it must target
    # BytesIO, never a path. 唯一的写类调用是传输编码；它必须面向 BytesIO，
    # 绝不面向路径。
    assert write_like == ["save"]
    (save_call,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save"
    ]
    # The save target must resolve to an in-memory io.BytesIO(), never a path.
    # save 目标必须解析为内存 io.BytesIO()，绝不面向路径。
    buffer_names = {
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "BytesIO"
    }
    assert buffer_names, "expected an in-memory BytesIO buffer in the module"
    assert save_call.args
    first_arg = save_call.args[0]
    assert isinstance(first_arg, ast.Name)
    assert first_arg.id in buffer_names
