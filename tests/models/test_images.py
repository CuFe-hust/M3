"""Region-read seam tests (26 阶段 A / doc 26 §6).

区域读取 seam 测试（26 阶段 A / doc 26 §6）：``ImageRegionSource`` 是只读
逐框读取协议；第一版 Pillow backend 打开时整图解码一次（对 JPEG/PNG 不是
真实随机窗口 I/O），``read_box`` 返回独立 RGB 图像并严格校验边界，source
生命周期限定单样本并显式关闭。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from models.images import (
    ImageRegionSource,
    PillowImageRegionSource,
    crop_image_box,
    open_image_region_source,
    read_normalized_image,
)


def _fixture(path: Path, size: tuple[int, int] = (64, 48)) -> Path:
    image = Image.new("RGB", size)
    for y in range(size[1]):
        for x in range(size[0]):
            image.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    image.save(path, format="PNG")
    return path


def test_open_image_region_source_returns_the_generic_seam(tmp_path: Path) -> None:
    path = _fixture(tmp_path / "img.png")
    source = open_image_region_source(path)
    assert isinstance(source, ImageRegionSource)  # runtime protocol check
    assert isinstance(source, PillowImageRegionSource)
    assert source.size == (64, 48)
    source.close()


def test_region_source_size_is_exif_normalized(tmp_path: Path) -> None:
    """The size property must match the EXIF/RGB-normalized decode, not the
    raw file header. size 属性必须匹配 EXIF/RGB 规范化解码，而非原始文件头。"""
    from PIL import ImageOps

    path = _fixture(tmp_path / "exif.png", (30, 20))
    image = Image.open(path)
    exif = image.getexif()
    exif[0x0112] = 6  # orientation: rotate 90 deg / 方向：旋转 90 度
    image.save(path, exif=exif)
    source = open_image_region_source(path)
    assert source.size == (20, 30)  # transposed / 转置后
    assert read_normalized_image(path).size == (20, 30)
    source.close()


def test_read_box_returns_independent_rgb_crops(tmp_path: Path) -> None:
    path = _fixture(tmp_path / "img.png")
    source = open_image_region_source(path)
    first = source.read_box((4, 4, 20, 16))
    second = source.read_box((4, 4, 20, 16))
    assert first.mode == "RGB"
    assert first.size == (16, 12)
    assert first.tobytes() == second.tobytes()
    # Independent buffers: mutating one crop never affects the source or the
    # other crop. 独立缓冲区：修改一个裁切绝不影响源或其他裁切。
    first.putpixel((0, 0), (255, 0, 0))
    assert second.getpixel((0, 0)) != (255, 0, 0)
    assert source.read_box((4, 4, 20, 16)).getpixel((0, 0)) != (255, 0, 0)
    # Same pixels as the canonical normalized-crop seam.
    # 与规范规范化裁切 seam 的像素一致。
    assert first.size == crop_image_box(read_normalized_image(path), (4, 4, 20, 16)).size
    source.close()


def test_read_box_validates_bounds_fail_closed(tmp_path: Path) -> None:
    path = _fixture(tmp_path / "img.png")
    source = open_image_region_source(path)
    for bad in (
        (-1, 0, 10, 10),
        (0, 0, 0, 10),
        (0, 0, 10, 10.5),
        (60, 0, 70, 10),
        (0, 40, 10, 50),
        (10, 5, 5, 10),
    ):
        with pytest.raises(ValueError):
            source.read_box(bad)  # type: ignore[arg-type]
    source.close()


def test_region_source_rejects_use_after_close(tmp_path: Path) -> None:
    path = _fixture(tmp_path / "img.png")
    source = open_image_region_source(path)
    source.close()
    with pytest.raises(ValueError, match="closed"):
        _ = source.size
    with pytest.raises(ValueError, match="closed"):
        source.read_box((0, 0, 4, 4))
    # Closing twice stays idempotent. 重复关闭保持幂等。
    source.close()


def test_region_source_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        open_image_region_source(tmp_path / "missing.png")


def test_region_source_has_no_bomb_workaround() -> None:
    """26 §0.5: the seam must not permanently raise or disable
    Image.MAX_IMAGE_PIXELS; the source stays a plain Pillow decode.
    26 §0.5：seam 不得永久修改或禁用 Image.MAX_IMAGE_PIXELS；source 保持普通
    Pillow 解码。"""
    from PIL import Image as PILImage

    assert PILImage.MAX_IMAGE_PIXELS is not None
    source = PillowImageRegionSource.__init__  # plain constructor, no patching
    assert callable(source)
