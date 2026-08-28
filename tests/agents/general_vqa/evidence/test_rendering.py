"""Tests for v2 evidence rendering.

v2 证据渲染测试：裁切只接受已物化的源像素框，预览只缩小；冻结三分支协议
（14.12）的 SegFormer 调色表、纯色 mask 合成与 YOLO 高对比标注保持确定性，
绝不推导框或计数。本文件同时持有旧全分辨率恢复路径的测试 oracle（26 §9.5：
旧路径只允许位于测试代码），并用它逐像素证明 preview 空间直接采样与其一致。
"""

from __future__ import annotations

import array
import base64
import hashlib
import io
from pathlib import Path

from PIL import Image
import pytest

import agents.general_vqa.evidence.rendering as rendering_module
from agents.general_vqa.evidence.geometry import (
    MODEL_INPUT_SIZE,
    compute_preview_size,
    nearest_lookup,
    partition_roi,
    segformer_model_extent,
    segformer_preview_lookups,
)
from agents.general_vqa.evidence.rendering import (
    class_id_grid_from_any,
    class_ids_in_prefix_rect,
    leaf_boolean_grid,
    make_preview,
    prepare_model_tile,
    prepare_segformer_roi,
    preview_from_path,
    render_pure_mask,
    render_roi_crop,
    render_yolo_annotation,
    sample_class_id_grid,
    segformer_palette,
)
from agents.general_vqa.evidence.schema import (
    EvidenceTileRecord,
    RoiEvidenceRecord,
    SegFormerPreprocessRecord,
)
from models.images import crop_image_box


# ── legacy full-resolution restore oracle (26 §9.5) / 旧全分辨率恢复 oracle ──
# These replicate the pre-bounded-memory runtime path exactly and exist ONLY
# as test oracles: production runtime must never call them (they materialize
# WxH / WpxHp grids).
# 以下精确复刻有界内存改造前的 runtime 路径，只作为测试 oracle 存在：
# 生产 runtime 绝不调用它们（会物化 WxH / WpxHp 网格）。


def _restore_class_id_mask_oracle(
    model_mask: Image.Image,
    tile_record: EvidenceTileRecord,
) -> Image.Image:
    """Legacy oracle: restore one 1024×1024 integer class-id model mask to
    the source tile size with NEAREST interpolation. 旧 oracle：将一个
    1024×1024 整数 class-id model mask 用 NEAREST 恢复到源 tile 尺寸。"""
    assert model_mask.mode in ("L", "I")
    assert model_mask.size == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    if tile_record.source_tile_size == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        return model_mask.copy()
    return model_mask.resize(
        tile_record.source_tile_size,
        resample=Image.Resampling.NEAREST,
    )


def _restore_segformer_class_id_mask_oracle(
    model_mask: Image.Image,
    preprocess: SegFormerPreprocessRecord,
) -> Image.Image:
    """Legacy oracle: NEAREST back to the padded canvas, then crop [0:W, 0:H]
    so the padding region can never appear in the restored grid. 旧 oracle：
    NEAREST 缩回 padded canvas 后裁切 [0:W, 0:H]，使 padding 区域绝不出现在
    恢复网格中。"""
    assert model_mask.mode in ("L", "I")
    assert model_mask.size == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    minimum, _ = model_mask.getextrema()
    assert minimum >= 0
    width, height = preprocess.source_size
    padded_width, padded_height = preprocess.padded_size
    if (padded_width, padded_height) == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        restored = model_mask.copy()
    else:
        restored = model_mask.resize(
            (padded_width, padded_height),
            resample=Image.Resampling.NEAREST,
        )
    return restored.crop((0, 0, width, height))


def _stitch_class_id_masks_oracle(
    restored_tiles: list[tuple[EvidenceTileRecord, Image.Image]],
    roi_size: tuple[int, int],
) -> Image.Image:
    """Legacy oracle: stitch restored per-tile class-id masks back into one
    ROI-local integer canvas. 旧 oracle：将恢复后的逐 tile class-id mask 拼接
    回一个 ROI 局部整数 canvas。"""
    width, height = roi_size
    canvas = Image.new("I", roi_size, 0)
    coverage = bytearray(width * height)
    for tile_record, mask in restored_tiles:
        x0, y0, x1, y1 = tile_record.source_tile_xyxy
        assert x1 <= width and y1 <= height
        assert mask.size == tile_record.source_tile_size
        for y in range(y0, y1):
            start = y * width + x0
            assert not any(coverage[start : start + (x1 - x0)])
        canvas.paste(mask, (x0, y0))
        for y in range(y0, y1):
            start = y * width + x0
            coverage[start : start + (x1 - x0)] = b"\x01" * (x1 - x0)
    assert all(coverage)
    return canvas


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


def test_final_pure_mask_shrinks_with_nearest_and_keeps_palette() -> None:
    """Final semantic masks are <=1080 and contain only palette colors.
    最终语义 mask 最长边不超过 1080，且只包含调色表颜色。"""
    size = (2000, 1200)
    mask = Image.new("L", size, 0)
    mask.paste(255, (300, 300, 1700, 900))
    palette = {"building": (100, 150, 200)}
    pure = render_pure_mask(size, [("building", mask)], palette)

    preview = make_preview(pure, resample=Image.Resampling.NEAREST)
    assert preview.size == (1080, 648)
    assert set(preview.getdata()) <= {(0, 0, 0), palette["building"]}

    # The combined branch uses the same nearest shrink before drawing boxes;
    # with no boxes this isolates the mask resampling contract.
    # combined 分支在绘框前使用同一 NEAREST 缩放；无框时可单独验证 mask
    # 的缩放契约。
    combined_mask = render_yolo_annotation(
        pure, [], resample=Image.Resampling.NEAREST
    )
    assert combined_mask.size == preview.size
    assert set(combined_mask.getdata()) <= {(0, 0, 0), palette["building"]}


def test_preview_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        preview_from_path(tmp_path / "missing.png")


def _rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _scan(image: Image.Image, window: tuple[int, int, int, int], color) -> bool:
    """True when any pixel of `color` exists inside the window (clamped to the
    image). 窗口（按图像裁剪）内是否存在任意 `color` 像素。"""
    pixels = image.load()
    x0, y0, x1, y1 = window
    for y in range(y0, min(y1, image.height)):
        for x in range(x0, min(x1, image.width)):
            if pixels[x, y] == color:
                return True
    return False


def test_segformer_palette_is_deterministic_and_obeys_distance_constraints() -> None:
    leaves = ["small_vehicle", "building", "water"]
    first = segformer_palette(leaves)
    second = segformer_palette(leaves)
    assert first == second
    assert set(first) == set(leaves)
    colors = list(first.values())
    for color in colors:
        assert _rgb_distance(color, (255, 0, 255)) >= 128.0
        assert _rgb_distance(color, (0, 0, 0)) >= 96.0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            assert _rgb_distance(colors[i], colors[j]) >= 48.0


def test_segformer_palette_reorders_leaves_deterministically() -> None:
    leaves = ["small_vehicle", "building", "water"]
    assert segformer_palette(leaves) == segformer_palette(leaves)
    # Same leaf in the same position always maps to the same color.
    # 同一叶子在同一位置永远映射到同一颜色。
    assert segformer_palette(leaves)["building"] == segformer_palette(leaves)["building"]


def test_segformer_palette_fails_stably_when_budget_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(rendering_module, "_PALETTE_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(rendering_module, "_PALETTE_MIN_DIST_MAGENTA", 1000.0)
    with pytest.raises(ValueError, match="exhausted"):
        segformer_palette(["small_vehicle"])


def test_render_pure_mask_composes_later_leaf_wins_deterministically() -> None:
    size = (100, 100)
    mask_a = Image.new("L", size, 0)
    mask_a.putpixel((10, 10), 255)
    mask_b = Image.new("L", size, 0)
    mask_b.putpixel((10, 10), 255)  # same pixel: later leaf wins
    mask_b.putpixel((80, 80), 255)
    palette = {"a": (255, 0, 0), "b": (0, 255, 0)}
    composed = render_pure_mask(size, [("a", mask_a), ("b", mask_b)], palette)
    assert composed.mode == "RGB"
    assert composed.getpixel((10, 10)) == (0, 255, 0)  # b overwrites a
    assert composed.getpixel((80, 80)) == (0, 255, 0)
    assert composed.getpixel((50, 50)) == (0, 0, 0)
    assert composed.getpixel((0, 0)) == (0, 0, 0)
    again = render_pure_mask(size, [("a", mask_a), ("b", mask_b)], palette)
    assert again.tobytes() == composed.tobytes()


def test_render_pure_mask_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        render_pure_mask(
            (100, 100),
            [("a", Image.new("L", (50, 50), 0))],
            {"a": (255, 0, 0)},
        )


def test_render_yolo_annotation_unscaled_draws_strokes_and_plate() -> None:
    source = _image((200, 100))
    before = source.tobytes()
    annotated = render_yolo_annotation(source, [("small_vehicle", (10, 10, 190, 90))])
    assert annotated.size == source.size  # never upscales
    assert source.tobytes() == before  # source untouched
    # Pillow draws both strokes inward from the boundary: the magenta width-3
    # line covers [10, 12], the black width-5 outline shows at [13, 14].
    # Pillow 从边界向内绘制描边：品红 width-3 覆盖 [10, 12]，黑色 width-5
    # 轮廓在 [13, 14] 可见。
    assert _scan(annotated, (10, 40, 13, 60), (255, 0, 255))
    assert _scan(annotated, (13, 40, 15, 60), (0, 0, 0))
    assert not _scan(annotated, (15, 40, 19, 60), (255, 0, 255))
    assert not _scan(annotated, (15, 40, 19, 60), (0, 0, 0))
    assert annotated.getpixel((100, 50)) == (7, 8, 9)  # interior untouched
    assert _scan(annotated, (10, 0, 70, 26), (255, 255, 255))  # white plate text


def test_render_yolo_annotation_shrinks_first_then_scales_boxes() -> None:
    source = _image((2160, 1080))
    annotated = render_yolo_annotation(source, [("building", (0, 0, 2000, 1000))])
    assert annotated.size == (1080, 540)  # 2160x1080 -> 1080x540, scale 0.5
    # The box scales to (0, 0, 1000, 500); strokes land on the scaled boundary.
    # 框缩放到 (0, 0, 1000, 500)；描边落在缩放后的边界上。
    assert _scan(annotated, (0, 240, 6, 260), (255, 0, 255))
    assert _scan(annotated, (0, 240, 6, 260), (0, 0, 0))
    assert annotated.getpixel((500, 300)) == (7, 8, 9)
    assert _scan(annotated, (0, 0, 130, 30), (255, 255, 255))  # plate inside top box


def test_render_yolo_annotation_is_deterministic() -> None:
    source = _image((200, 100))
    boxes = [("a", (10, 10, 90, 50)), ("b", (120, 20, 180, 80))]
    first = render_yolo_annotation(source, boxes)
    second = render_yolo_annotation(source, boxes)
    assert first.tobytes() == second.tobytes()


def test_legacy_overlay_api_is_removed() -> None:
    # The semi-transparent overlay and the order-free stable palette were
    # replaced by the frozen three-branch protocol; keep them from creeping
    # back. 半透明 overlay 与无顺序 stable palette 已被冻结三分支协议取代；
    # 防止其回归。
    assert not hasattr(rendering_module, "overlay_mask")
    assert not hasattr(rendering_module, "stable_palette_color")


# ── 1024×1024 tile materialization and mask restoration (14.8) ───────────


def _roi(size: tuple[int, int], roi_id: str = "roi-0") -> RoiEvidenceRecord:
    width, height = size
    return RoiEvidenceRecord(
        roi_id=roi_id,
        image_id="img1",
        source_size=size,
        core_xyxy=(0, 0, width, height),
        expanded_xyxy=(0, 0, width, height),
        crop_size=size,
    )


def _ramp_image(size: tuple[int, int]) -> Image.Image:
    """Coordinate-encoded RGB ramp: R = x & 255, G = y & 255 (locally linear
    everywhere except wrap points), B = 0. 坐标编码 RGB 渐变：R = x & 255、
    G = y & 255（除回绕点外处处局部线性）、B = 0。"""
    width, height = size
    row = bytes(range(256)) * (width // 256) + bytes(range(width % 256))
    red = row * height
    green = b"".join(bytes([y & 255]) * width for y in range(height))
    return Image.merge(
        "RGB",
        [
            Image.frombytes("L", (width, height), red),
            Image.frombytes("L", (width, height), green),
            Image.new("L", (width, height), 0),
        ],
    )


def _coord_mask(
    size: tuple[int, int],
    *,
    roi_width: int,
    x0: int = 0,
    y0: int = 0,
) -> Image.Image:
    """Integer class-id mask whose pixel value encodes the ROI-local position:
    value = y * roi_width + x. 像素值编码 ROI 局部位置的整数 class-id mask：
    value = y * roi_width + x。"""
    width, height = size
    values = array.array(
        "i",
        (
            (y0 + y) * roi_width + (x0 + x)
            for y in range(height)
            for x in range(width)
        ),
    )
    return Image.frombytes("I", (width, height), values.tobytes())


def test_prepare_model_tile_passes_full_tile_through_unchanged() -> None:
    roi = _ramp_image((2048, 1536))
    full = partition_roi(_roi((2048, 1536)))[0]
    assert full.source_tile_size == (1024, 1024)
    tile_image = prepare_model_tile(roi, full)
    assert tile_image.size == (1024, 1024)
    assert tile_image.tobytes() == roi.crop((0, 0, 1024, 1024)).tobytes()
    assert roi.tobytes() == _ramp_image((2048, 1536)).tobytes()  # source untouched


def test_prepare_model_tile_stretches_remainder_with_lanczos_ramp() -> None:
    roi = _ramp_image((2048, 1536))
    remainder = partition_roi(_roi((2048, 1536)))[2]  # (0, 1024, 1024, 1536)
    assert remainder.source_tile_size == (1024, 512)
    tile_image = prepare_model_tile(roi, remainder)
    assert tile_image.size == (1024, 1024)
    # scale_y = 2: model (x, y) maps to source (x, 1024 + y / 2); the ramp is
    # linear at these sample points, so LANCZOS reproduces it within rounding.
    # scale_y = 2：model (x, y) 映射到源 (x, 1024 + y / 2)；这些采样点处渐变
    # 线性，因此 LANCZOS 在取整误差内精确复现。
    for model_x, model_y, expected_r, expected_g in (
        (100, 200, 100, 100),
        (900, 200, 132, 100),
        (100, 700, 100, 94),
    ):
        pixel = tile_image.getpixel((model_x, model_y))
        assert abs(pixel[0] - expected_r) <= 2, (model_x, model_y, pixel)
        assert abs(pixel[1] - expected_g) <= 2, (model_x, model_y, pixel)
        assert pixel[2] == 0


def test_prepare_model_tile_1x1_tile_stretches_to_uniform_model_tile() -> None:
    roi = _image((1, 1), fill=5)
    single = partition_roi(_roi((1, 1)))[0]
    tile_image = prepare_model_tile(roi, single)
    assert tile_image.size == (1024, 1024)
    assert tile_image.getextrema() == ((5, 5), (6, 6), (7, 7))


def test_prepare_model_tile_rejects_box_drift_outside_roi() -> None:
    roi = _ramp_image((2000, 1024))
    foreign = partition_roi(_roi((2048, 2048)))[3]  # (1024,1024,2048,2048)
    with pytest.raises(ValueError, match="exceeds ROI image size"):
        prepare_model_tile(roi, foreign)


def test_restore_class_id_mask_oracle_full_tile_is_a_copy() -> None:
    model_mask = _coord_mask((1024, 1024), roi_width=2000)
    full = partition_roi(_roi((2000, 1024)))[0]
    restored = _restore_class_id_mask_oracle(model_mask, full)
    assert restored is not model_mask
    assert restored.size == (1024, 1024)
    assert restored.tobytes() == model_mask.tobytes()


def test_restore_class_id_mask_oracle_downscales_remainder_with_nearest() -> None:
    roi_size = (2000, 1024)
    remainder = partition_roi(_roi(roi_size))[1]  # (1024, 0, 2000, 1024)
    assert remainder.source_tile_size == (976, 1024)
    source_mask = _coord_mask((976, 1024), roi_width=2000, x0=1024)
    model_mask = source_mask.resize((1024, 1024), resample=Image.Resampling.NEAREST)
    restored = _restore_class_id_mask_oracle(model_mask, remainder)
    assert restored.size == (976, 1024)
    for x, y in ((200, 500), (488, 0), (975, 1023), (0, 511)):
        value = restored.getpixel((x, y))
        # NEAREST keeps exact class ids: the decoded ROI x stays within ±2 of
        # the scaled source position and the row decodes exactly.
        # NEAREST 保持精确 class id：解码 ROI x 与缩放后的源位置偏差不超过 ±2，
        # 行坐标精确解码。
        assert value // 2000 == y, (x, y, value)
        assert 1024 + x - 2 <= value % 2000 <= 1024 + x + 2, (x, y, value)


def test_stitch_class_id_masks_oracle_places_each_roi_pixel_exactly_once() -> None:
    roi_size = (2000, 1024)
    tiles = partition_roi(_roi(roi_size))
    full, remainder = tiles[0], tiles[1]
    full_mask = _coord_mask((1024, 1024), roi_width=2000)
    source_mask = _coord_mask((976, 1024), roi_width=2000, x0=1024)
    model_mask = source_mask.resize((1024, 1024), resample=Image.Resampling.NEAREST)
    canvas = _stitch_class_id_masks_oracle(
        [
            (full, full_mask),
            (remainder, _restore_class_id_mask_oracle(model_mask, remainder)),
        ],
        roi_size,
    )
    assert canvas.size == roi_size
    assert canvas.mode == "I"
    # Full-tile rows decode exactly; remainder rows within the NEAREST ±2 px.
    # 完整 tile 行精确解码；余块行在 NEAREST ±2 px 内。
    for y in (0, 511):
        row = [canvas.getpixel((x, y)) for x in range(2000)]
        assert all(value == y * 2000 + x for x, value in enumerate(row))
    value = canvas.getpixel((1500, 700))
    assert value // 2000 == 700
    assert 1498 <= value % 2000 <= 1502
    assert canvas.getpixel((1999, 1023)) == 1023 * 2000 + 1999


def test_preview_grid_sampling_rejects_non_integer_or_off_model_inputs() -> None:
    """The preview-space sampler keeps the same fail-closed validations as
    the removed production restore path: only strict 1024x1024 non-negative
    integer grids are accepted. preview 空间采样器保持与已移除的生产恢复路径
    相同的严格校验：只接受严格 1024×1024 非负整数网格。"""
    x_lookup = (0, 1)
    y_lookup = (0, 1)
    with pytest.raises(ValueError, match="integer class-id grid"):
        sample_class_id_grid(Image.new("F", (1024, 1024), 0.5), x_lookup, y_lookup)
    with pytest.raises(ValueError, match="strict 1024x1024"):
        sample_class_id_grid(
            _coord_mask((512, 512), roi_width=1024), x_lookup, y_lookup
        )
    negative = Image.new("I", (1024, 1024), 0)
    negative.putpixel((3, 4), -7)
    with pytest.raises(ValueError, match="non-negative"):
        sample_class_id_grid(negative, x_lookup, y_lookup)
    with pytest.raises(ValueError, match="integer class-id grid"):
        class_ids_in_prefix_rect(Image.new("F", (1024, 1024), 0.5), (10, 10))
    with pytest.raises(ValueError, match="strict 1024x1024"):
        class_ids_in_prefix_rect(_coord_mask((512, 512), roi_width=1024), (10, 10))
    with pytest.raises(ValueError, match="within the 1024 model grid"):
        class_ids_in_prefix_rect(_coord_mask((1024, 1024), roi_width=1024), (1024, 10))


# ── SegFormer pad protocol (26 §3) / SegFormer pad 协议 ───────────────────


def _pad_record(
    source_size: tuple[int, int],
    *,
    roi_id: str = "roi-0",
) -> SegFormerPreprocessRecord:
    """Build the exact pad geometry for a source size under the frozen
    minimal-ceiling protocol. 按冻结最小上取整协议构造指定源尺寸的 pad 几何。"""
    width, height = source_size
    padded_width = ((width + 1023) // 1024) * 1024
    padded_height = ((height + 1023) // 1024) * 1024
    return SegFormerPreprocessRecord(
        roi_id=roi_id,
        source_size=source_size,
        padded_size=(padded_width, padded_height),
        padding_right=padded_width - width,
        padding_bottom=padded_height - height,
        scale_x=1024 / padded_width,
        scale_y=1024 / padded_height,
    )


@pytest.mark.parametrize(
    "source_size,padded_size,padding_right,padding_bottom",
    [
        ((1024, 1024), (1024, 1024), 0, 0),
        ((1024, 2048), (1024, 2048), 0, 0),
        ((976, 1024), (1024, 1024), 48, 0),
        ((1025, 1025), (2048, 2048), 1023, 1023),
        ((2000, 1536), (2048, 2048), 48, 512),
        ((1, 1), (1024, 1024), 1023, 1023),
    ],
)
def test_prepare_segformer_roi_pads_right_bottom_black_and_resizes(
    source_size: tuple[int, int],
    padded_size: tuple[int, int],
    padding_right: int,
    padding_bottom: int,
) -> None:
    """The whole ROI is padded only on the right and bottom with constant
    black to the minimal 1024 multiples, then resized to one strict 1024x1024
    RGB model input; the source image is never modified and every original
    pixel stays at the top-left of the padded canvas.
    整张 ROI 只在右侧与底部以固定黑色 padding 到 1024 最小倍数，再缩放到单一
    严格 1024×1024 RGB 模型输入；源图像绝不修改，原始像素全部保留在 padded
    canvas 左上角。"""
    width, height = source_size
    roi = _image(source_size, fill=11)
    before = roi.tobytes()
    preprocess, model_input = prepare_segformer_roi(
        roi, roi_id="roi-0", source_size=source_size
    )
    assert roi.tobytes() == before  # source untouched / 源未修改
    assert model_input.mode == "RGB"
    assert model_input.size == (1024, 1024)
    assert preprocess.source_size == source_size
    assert preprocess.padded_size == padded_size
    assert preprocess.padding_right == padding_right
    assert preprocess.padding_bottom == padding_bottom
    assert preprocess.model_input_size == (1024, 1024)
    assert preprocess.scale_x == pytest.approx(1024 / padded_size[0])
    assert preprocess.scale_y == pytest.approx(1024 / padded_size[1])
    # Independent spec reconstruction: black padded canvas with the ROI pasted
    # at (0, 0), LANCZOS-resized, must be byte-identical to the model input.
    # 独立按规格重建：黑色 padded canvas 在 (0, 0) 粘贴 ROI 并 LANCZOS 缩放，
    # 必须与模型输入字节级一致。
    canvas = Image.new("RGB", padded_size, (0, 0, 0))
    canvas.paste(roi.convert("RGB"), (0, 0))
    expected = canvas.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
    assert model_input.tobytes() == expected.tobytes()


def test_prepare_segformer_roi_exact_1024_roi_gets_no_padding() -> None:
    """An ROI whose axes are already 1024 multiples gets zero padding and no
    extra 1024 interval. 轴已为 1024 倍数的 ROI 零 padding，绝不额外补一个
    完整 1024 区间。"""
    roi = _image((1024, 2048), fill=3)
    preprocess, model_input = prepare_segformer_roi(
        roi, roi_id="roi-0", source_size=(1024, 2048)
    )
    assert preprocess.padding_right == 0
    assert preprocess.padding_bottom == 0
    assert preprocess.padded_size == (1024, 2048)
    assert model_input.size == (1024, 1024)


def test_prepare_segformer_roi_rejects_invalid_geometry() -> None:
    roi = _image((100, 100))
    with pytest.raises(ValueError, match="positive"):
        prepare_segformer_roi(roi, roi_id="roi-0", source_size=(0, 100))
    with pytest.raises(ValueError, match="does not match"):
        prepare_segformer_roi(roi, roi_id="roi-0", source_size=(200, 100))


def test_restore_segformer_mask_oracle_crops_padding_and_keeps_coordinates() -> None:
    """NEAREST restore to the padded size followed by a deterministic
    [0:W, 0:H] crop: the final grid is exactly the source ROI size, the crop
    has no offset, and the padding region can never appear in it. This is the
    legacy oracle that the preview-space direct sampling must reproduce
    pixel-exactly. NEAREST 恢复到 padded 尺寸后确定性裁切 [0:W, 0:H]：最终网格
    恰为源 ROI 尺寸，裁切无偏移，padding 区域绝不可能出现。这是 preview 空间
    直接采样必须逐像素复刻的旧 oracle。"""
    source_size = (1500, 800)
    preprocess = _pad_record(source_size)
    assert preprocess.padded_size == (2048, 1024)
    assert preprocess.padding_right == 548
    assert preprocess.padding_bottom == 224
    padded_width = preprocess.padded_size[0]
    # Coordinate-encoded padded grid; the padding region gets a distinct id.
    # 坐标编码的 padded 网格；padding 区域使用独立 id。
    values = array.array(
        "i",
        (
            777777 if x >= 1500 or y >= 800 else y * padded_width + x
            for y in range(1024)
            for x in range(2048)
        ),
    )
    padded_grid = Image.frombytes("I", (2048, 1024), values.tobytes())
    model_mask = padded_grid.resize((1024, 1024), resample=Image.Resampling.NEAREST)
    restored = _restore_segformer_class_id_mask_oracle(model_mask, preprocess)
    assert restored.size == source_size
    for x, y in ((0, 0), (1499, 799), (700, 400), (1200, 50), (10, 790)):
        value = restored.getpixel((x, y))
        assert value // padded_width == y, (x, y, value)
        assert abs(value % padded_width - x) <= 2, (x, y, value)
        assert value != 777777
    # Padding-sourced ids never leak into the crop.
    # padding 来源的 id 绝不泄漏进裁切区域。
    for y in range(0, 800, 100):
        for x in range(0, 1500, 100):
            assert restored.getpixel((x, y)) != 777777


def test_restore_segformer_mask_oracle_exact_1024_source_is_a_copy() -> None:
    """A 1024x1024 source has no padding: restore is an exact copy of the
    strict model grid. 1024×1024 源无 padding：恢复是严格模型网格的精确副本。"""
    preprocess = _pad_record((1024, 1024))
    assert (preprocess.padding_right, preprocess.padding_bottom) == (0, 0)
    model_mask = _coord_mask((1024, 1024), roi_width=1024)
    restored = _restore_segformer_class_id_mask_oracle(model_mask, preprocess)
    assert restored is not model_mask
    assert restored.size == (1024, 1024)
    assert restored.tobytes() == model_mask.tobytes()


def test_restore_segformer_mask_oracle_crops_padding_and_keeps_coordinates() -> None:
    """NEAREST restore to the padded size followed by a deterministic
    [0:W, 0:H] crop: the final grid is exactly the source ROI size, the crop
    has no offset, and the padding region can never appear in it. This is the
    legacy oracle that the preview-space direct sampling must reproduce
    pixel-exactly. NEAREST 恢复到 padded 尺寸后确定性裁切 [0:W, 0:H]：最终网格
    恰为源 ROI 尺寸，裁切无偏移，padding 区域绝不可能出现。这是 preview 空间
    直接采样必须逐像素复刻的旧 oracle。"""
    source_size = (1500, 800)
    preprocess = _pad_record(source_size)
    assert preprocess.padded_size == (2048, 1024)
    assert preprocess.padding_right == 548
    assert preprocess.padding_bottom == 224
    padded_width = preprocess.padded_size[0]
    # Coordinate-encoded padded grid; the padding region gets a distinct id.
    # 坐标编码的 padded 网格；padding 区域使用独立 id。
    values = array.array(
        "i",
        (
            777777 if x >= 1500 or y >= 800 else y * padded_width + x
            for y in range(1024)
            for x in range(2048)
        ),
    )
    padded_grid = Image.frombytes("I", (2048, 1024), values.tobytes())
    model_mask = padded_grid.resize((1024, 1024), resample=Image.Resampling.NEAREST)
    restored = _restore_segformer_class_id_mask_oracle(model_mask, preprocess)
    assert restored.size == source_size
    for x, y in ((0, 0), (1499, 799), (700, 400), (1200, 50), (10, 790)):
        value = restored.getpixel((x, y))
        assert value // padded_width == y, (x, y, value)
        assert abs(value % padded_width - x) <= 2, (x, y, value)
        assert value != 777777
    # Padding-sourced ids never leak into the crop.
    # padding 来源的 id 绝不泄漏进裁切区域。
    for y in range(0, 800, 100):
        for x in range(0, 1500, 100):
            assert restored.getpixel((x, y)) != 777777


def test_preview_direct_sampling_parity_with_legacy_oracle() -> None:
    """26 §11.1: for a full-coverage preview (ROI <= 1080) the direct
    preview-space sampling must reproduce the legacy restore pixel-exactly,
    and for a downscaled preview it must reproduce the legacy
    restore-then-NEAREST-shrink pixel-exactly. 26 §11.1：全尺寸 preview
    （ROI <= 1080）下 preview 空间直接采样必须逐像素复刻旧恢复；缩小 preview
    下必须逐像素复刻旧“恢复后 NEAREST 缩小”。"""
    cases = [
        (1024, 1024),
        (1025, 1025),
        (976, 1024),
        (1500, 800),
        (2000, 1024),
        (300, 200),
        (1, 1),
        (1080, 1080),
    ]
    for source_size in cases:
        preprocess = _pad_record(source_size)
        padded_width, padded_height = preprocess.padded_size
        # Coordinate-encoded padded grid; padding gets a distinct id so any
        # leakage into the sampled class ids is detectable. Note: the model
        # INPUT itself is a LANCZOS blend of the padded canvas, so boundary
        # class ids influenced by padding are expected model behavior in both
        # the legacy and the direct path — parity is the guarantee, not the
        # absence of the marker at every pixel.
        # 坐标编码 padded 网格；padding 用独立 id，任何泄漏进采样 class id 都
        # 可检测。注意：模型输入本身是 padded canvas 的 LANCZOS 混合，因此
        # 边界处受 padding 影响的 class id 在旧路径与直接路径中都是预期的
        # 模型行为——保证的是 parity，而非每个像素都不含该标记。
        values = array.array(
            "i",
            (
                7000000
                if x >= source_size[0] or y >= source_size[1]
                else y * padded_width + x
                for y in range(padded_height)
                for x in range(padded_width)
            ),
        )
        padded_grid = Image.frombytes("I", (padded_width, padded_height), values.tobytes())
        model_mask = padded_grid.resize((1024, 1024), resample=Image.Resampling.NEAREST)
        restored = _restore_segformer_class_id_mask_oracle(model_mask, preprocess)
        preview_size = compute_preview_size(source_size)
        legacy_preview = restored.resize(preview_size, resample=Image.Resampling.NEAREST)
        x_lookup, y_lookup = segformer_preview_lookups(preprocess, preview_size)
        direct = sample_class_id_grid(model_mask, x_lookup, y_lookup)
        assert direct.size == preview_size
        assert direct.tobytes() == legacy_preview.tobytes(), (
            f"preview class grid drifted for source_size {source_size!r}"
        )


def test_preview_direct_sampling_never_reads_padding_region() -> None:
    """Every composed lookup index stays strictly inside the ROI region of
    the model mask: the maximum sampled model column is below
    ceil(W * 1024 / Wp) for any preview size. 每个合成查找索引都严格落在 model
    mask 的 ROI 区域内：任意 preview 尺寸下最大采样 model 列都小于
    ceil(W * 1024 / Wp)。"""
    for source_size in ((1500, 800), (2000, 1024), (1, 1), (1025, 1025)):
        preprocess = _pad_record(source_size)
        preview_size = compute_preview_size(source_size)
        x_lookup, y_lookup = segformer_preview_lookups(preprocess, preview_size)
        width, height = preprocess.source_size
        padded_width, padded_height = preprocess.padded_size
        roi_last_model_x = (width * 1024 + padded_width - 1) // padded_width
        roi_last_model_y = (height * 1024 + padded_height - 1) // padded_height
        assert max(x_lookup) < roi_last_model_x, (source_size, max(x_lookup))
        assert max(y_lookup) < roi_last_model_y, (source_size, max(y_lookup))
        # The prefix-rect extent matches the same ROI boundary.
        # 前缀矩形 extent 与同一 ROI 边界一致。
        mx, my = segformer_model_extent(preprocess)
        assert mx < roi_last_model_x
        assert my < roi_last_model_y
        assert mx == max(nearest_lookup(MODEL_INPUT_SIZE, padded_width)[:width])
        assert my == max(nearest_lookup(MODEL_INPUT_SIZE, padded_height)[:height])


def test_class_ids_in_prefix_rect_matches_legacy_restored_grid() -> None:
    """The prefix-rectangle class set is exactly the class set of the legacy
    full-resolution restored grid, for both padded and unpadded sources.
    前缀矩形类别集合与旧整分辨率恢复网格的类别集合完全一致（含 padding 与
    无 padding 源）。"""
    for source_size in ((1500, 800), (1024, 1024), (2000, 1024), (1, 1)):
        preprocess = _pad_record(source_size)
        padded_width, padded_height = preprocess.padded_size
        values = array.array(
            "i",
            (
                999
                if x >= source_size[0] or y >= source_size[1]
                else (x * 7 + y * 13) % 5
                for y in range(padded_height)
                for x in range(padded_width)
            ),
        )
        padded_grid = Image.frombytes("I", (padded_width, padded_height), values.tobytes())
        model_mask = padded_grid.resize((1024, 1024), resample=Image.Resampling.NEAREST)
        restored = _restore_segformer_class_id_mask_oracle(model_mask, preprocess)
        seen = class_ids_in_prefix_rect(model_mask, segformer_model_extent(preprocess))
        assert seen == frozenset(array.array("i", restored.tobytes()))


def test_nearest_lookup_replicates_pillow_resize() -> None:
    """The pure NEAREST lookup replicates Pillow's affine-nearest resize
    exactly (ImagingScaleAffine), including the exact-integer boundary
    behavior, for upscale, downscale and identity axes. 纯 NEAREST 查找精确
    复刻 Pillow 的仿射 nearest resize（ImagingScaleAffine），包括精确整数
    边界行为，覆盖放大、缩小与恒等轴。"""
    for src, dst in (
        (4, 10),
        (10, 3),
        (100, 37),
        (1024, 976),
        (976, 1024),
        (1024, 2048),
        (2048, 1024),
        (1080, 2000),
        (2000, 1080),
        (1024, 1024),
        (1, 5),
        (5, 1),
    ):
        probe = Image.frombytes("I", (src, 1), array.array("i", range(src)).tobytes())
        out = probe.resize((dst, 1), Image.Resampling.NEAREST)
        assert list(array.array("i", out.tobytes())) == list(nearest_lookup(src, dst)), (
            src,
            dst,
        )
    # Exhaustive small-size sweep: every axis pair in [1, 24] x [1, 24].
    # 穷举小尺寸扫描：所有 [1, 24] x [1, 24] 轴对。
    for src in range(1, 25):
        for dst in range(1, 25):
            probe = Image.frombytes("I", (src, 1), array.array("i", range(src)).tobytes())
            out = probe.resize((dst, 1), Image.Resampling.NEAREST)
            assert list(array.array("i", out.tobytes())) == list(nearest_lookup(src, dst))


def test_nearest_lookup_is_bounded_memory_for_huge_geometry() -> None:
    """26 §12.3: a 207,533,568-pixel source geometry builds its lookup with
    O(target) memory — no Pillow bomb check, no large allocation. 26 §12.3：
    2.08 亿像素源几何以 O(target) 内存生成查找——无 Pillow bomb 检查、无大
    内存分配。"""
    lookup = nearest_lookup(207533568, 1080)
    assert len(lookup) == 1080
    assert lookup[-1] == 207437487
    # The composed preview lookups over the same ROI stay in the model grid.
    # 同一 ROI 的合成 preview 查找仍保持在 model grid 内。
    preprocess = _pad_record((207533568 // 1080, 1080))
    x_lookup, y_lookup = segformer_preview_lookups(preprocess, (1080, 1080))
    assert max(x_lookup) < MODEL_INPUT_SIZE
    assert max(y_lookup) < MODEL_INPUT_SIZE


def test_render_yolo_annotation_edge_boxes_do_not_invert_plate() -> None:
    """Boxes touching the right/bottom edges or the top-left corner must not
    produce an inverted label plate (x1 < x0 / y1 < y0): the plate clamps
    inside the image instead of raising ValueError.
    贴右/下边缘或左上角的框绝不能产生反转 label plate（x1 < x0 / y1 < y0）：
    底板被限制在图像内，而不是抛 ValueError。"""
    source = _image((200, 100))
    annotated = render_yolo_annotation(
        source,
        [
            ("small_vehicle", (190, 10, 199, 90)),  # right edge
            ("building", (10, 90, 100, 99)),        # bottom edge
            ("water", (0, 0, 5, 5)),                # top-left corner
        ],
    )
    assert annotated.size == source.size
    assert _scan(annotated, (10, 60, 40, 90), (255, 255, 255))  # plate text still drawn


def test_render_yolo_annotation_pre_shrunk_image_scales_boxes_from_source_size() -> None:
    """When the caller passes an already-shrunk image (e.g. a NEAREST pure
    mask) with boxes in the source pixel frame, source_size must scale the
    boxes onto the preview; without it they render at scale 1.0 on the
    smaller canvas (off-image boxes, plates piled on the bottom edge).
    调用方传入已缩小图像（如 NEAREST 纯色 mask）而框位于源像素帧时，必须用
    source_size 把框缩放到预览上；否则框以 scale 1.0 画在更小画布上（框越界、
    标签底板堆在底部边缘）。"""
    source = Image.new("RGB", (3072, 3072), (7, 8, 9))
    preview = source.resize((1080, 1080), Image.Resampling.NEAREST)
    # Box at source (666, 152)-(780, 264) must land at preview (234, 53)-(274, 93).
    # 源坐标 (666, 152)-(780, 264) 的框必须落在预览 (234, 53)-(274, 93)。
    annotated = render_yolo_annotation(
        preview,
        [("small_vehicle", (666.0, 152.0, 780.0, 264.0))],
        source_size=(3072, 3072),
        resample=Image.Resampling.NEAREST,
    )
    assert annotated.size == (1080, 1080)
    assert _scan(annotated, (232, 50, 278, 100), (255, 0, 255))
    # Not at the unscaled position on the shrunk canvas.
    # 不得出现在缩小画布上的未缩放位置。
    assert not _scan(annotated, (660, 148, 790, 268), (255, 0, 255))
    # No plate pile at the bottom edge either.
    # 底部边缘也不得有标签底板堆积。
    assert not _scan(annotated, (0, 1040, 1080, 1080), (255, 0, 255))
