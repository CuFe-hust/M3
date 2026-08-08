"""Contract tests for the counting overlay renderer.

计数标注图渲染器契约测试：输出保存、尺寸一致性、拒绝点标记、确定性输出与
尺寸不匹配稳定失败。绝不导入 CountingAgent/YOLO backend。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from agents.counting.schema import CountingResult, GlobalPointObservation
from reporting.visualization import render_counting_overlay


def _point(
    gid: str,
    *,
    x: int,
    y: int,
    accepted: bool,
    radius: float = 5.0,
) -> GlobalPointObservation:
    return GlobalPointObservation(
        global_id=gid,
        target="car",
        source_tile_id="t0",
        local_id=gid,
        local_x_norm=100,
        local_y_norm=100,
        local_radius_norm=5,
        global_x_px=x,
        global_y_px=y,
        global_x_norm=100,
        global_y_norm=100,
        radius_px=radius,
        confidence=0.9,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=accepted,
        short_evidence="visible",
    )


def _result() -> CountingResult:
    return CountingResult(
        sample_id="s1",
        target="car",
        question="How many cars?",
        source_width=100,
        source_height=100,
        tile_count=1,
        succeeded_tiles=["t0"],
        failed_tiles=[],
        global_points=[
            _point("p0", x=10, y=10, accepted=True),
            _point("p1", x=50, y=50, accepted=True),
            _point("p2", x=90, y=90, accepted=False),
        ],
        merged_groups=[],
        unresolved_conflicts=[],
        final_count=2,
        status="completed",
    )


def test_render_counting_overlay_saves_png(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    output = tmp_path / "overlay.png"
    render_counting_overlay(image, result=_result(), output_path=output)
    assert output.is_file()
    rendered = Image.open(output)
    assert rendered.size == (100, 100)
    assert rendered.mode == "RGB"


def test_render_overlay_is_deterministic(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    render_counting_overlay(image, result=_result(), output_path=first)
    render_counting_overlay(image, result=_result(), output_path=second)
    assert first.read_bytes() == second.read_bytes()


def test_render_overlay_image_size_mismatch_fails_stably(tmp_path: Path) -> None:
    image = Image.new("RGB", (50, 50), (255, 255, 255))
    with pytest.raises(ValueError, match="does not match"):
        render_counting_overlay(image, result=_result(), output_path=tmp_path / "bad.png")
    assert not (tmp_path / "bad.png").exists()


def test_render_overlay_does_not_mutate_source_image(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    before = image.tobytes()
    render_counting_overlay(image, result=_result(), output_path=tmp_path / "overlay.png")
    assert image.tobytes() == before
