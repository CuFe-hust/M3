"""Manual image directory collection contract. / 手动图片目录收集契约。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from spacers_agent.application import collect_images, natural_key


def _make_image(path: Path, size: tuple[int, int] = (16, 12)) -> Path:
    """Create one tiny valid image. / 创建一张微小的合法图片。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(10, 20, 30))
    image.save(path)
    return path


def test_natural_order():
    """image2.png must sort before image10.png. / image2.png 必须排在 image10.png 之前。"""

    names = ["image10.png", "image2.png", "image1.png"]
    assert sorted(names, key=lambda name: natural_key(Path(name))) == [
        "image1.png",
        "image2.png",
        "image10.png",
    ]


def test_collect_images_natural_order(tmp_path):
    _make_image(tmp_path / "image10.png")
    _make_image(tmp_path / "image2.png")
    _make_image(tmp_path / "image1.png")
    collected = collect_images(tmp_path)
    assert [item.path.name for item in collected] == [
        "image1.png",
        "image2.png",
        "image10.png",
    ]


def test_collect_images_ignores_non_images_hidden_and_subdirs(tmp_path):
    _make_image(tmp_path / "a.png")
    _make_image(tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    (tmp_path / ".hidden.png").write_bytes(b"fake")
    sub = tmp_path / "sub"
    sub.mkdir()
    _make_image(sub / "deep.png")
    collected = collect_images(tmp_path)
    assert [item.path.name for item in collected] == ["a.png", "b.jpg"]


def test_collect_images_supported_extensions(tmp_path):
    for name in ("a.jpg", "b.jpeg", "c.png", "d.webp", "e.tif", "f.tiff", "g.bmp"):
        _make_image(tmp_path / name)
    collected = collect_images(tmp_path)
    assert len(collected) == 7


def test_collect_images_records_size(tmp_path):
    _make_image(tmp_path / "wide.png", size=(64, 32))
    [item] = collect_images(tmp_path)
    assert item.width == 64
    assert item.height == 32


def test_collect_images_missing_directory_fails(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        collect_images(tmp_path / "missing")


def test_collect_images_non_directory_fails(tmp_path):
    text = tmp_path / "file.txt"
    text.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        collect_images(text)


def test_collect_images_empty_directory_fails(tmp_path):
    with pytest.raises(ValueError, match="no supported images"):
        collect_images(tmp_path)


def test_collect_images_corrupt_image_fails(tmp_path):
    (tmp_path / "broken.png").write_bytes(b"not a real png payload")
    with pytest.raises(ValueError, match="cannot open image"):
        collect_images(tmp_path)


def test_collect_images_more_than_eight_fails(tmp_path):
    for index in range(9):
        _make_image(tmp_path / f"image{index}.png")
    with pytest.raises(ValueError, match="too many images"):
        collect_images(tmp_path)
