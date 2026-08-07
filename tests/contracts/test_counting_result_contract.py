"""Contract tests for counting-domain schemas.

计数域 Schema 契约测试：几何/切片/观测/结果的全部强校验，以及计数契约
不位于 agents.schema 或 data.schema 的所有权断言。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.counting.schema import (
    CountTargetSpec,
    CountingDraft,
    CountingResult,
    GlobalPointObservation,
    IssueRecord,
    LocalPointObservation,
    PixelRect,
    PointProvenance,
    TileCountResponse,
    TileSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── PixelRect / 像素矩形 ───────────────────────────────────────────────────


def test_pixel_rect_valid_and_dimensions() -> None:
    rect = PixelRect(left=10, top=20, right=30, bottom=40)
    assert rect.width == 20
    assert rect.height == 20


@pytest.mark.parametrize("rect", [
    {"left": 10, "top": 20, "right": 10, "bottom": 40},  # zero width / 零宽
    {"left": 10, "top": 20, "right": 30, "bottom": 20},  # zero height / 零高
    {"left": 30, "top": 20, "right": 10, "bottom": 40},  # reversed / 反向
])
def test_pixel_rect_rejects_invalid(rect: dict) -> None:
    with pytest.raises(ValidationError, match="half-open"):
        PixelRect(**rect)


# ── TileSpec / 切片规格 ────────────────────────────────────────────────────


def _tile(**overrides) -> TileSpec:
    values = dict(
        tile_id="t0",
        row=0,
        col=0,
        crop_global=PixelRect(left=0, top=0, right=100, bottom=100),
        owner_core_global=PixelRect(left=10, top=10, right=90, bottom=90),
        owner_core_local=PixelRect(left=10, top=10, right=90, bottom=90),
        source_width=100,
        source_height=100,
        model_input_width=1024,
        model_input_height=1024,
    )
    values.update(overrides)
    return TileSpec(**values)


def test_tile_spec_valid() -> None:
    tile = _tile()
    assert tile.recursive_depth == 0
    assert tile.parent_tile_id is None


def test_tile_spec_rejects_crop_exceeding_source() -> None:
    with pytest.raises(ValidationError, match="exceeds source"):
        _tile(crop_global=PixelRect(left=0, top=0, right=101, bottom=100))


def test_tile_spec_rejects_core_outside_crop() -> None:
    with pytest.raises(ValidationError, match="inside crop"):
        _tile(
            owner_core_global=PixelRect(left=98, top=98, right=102, bottom=102),
            owner_core_local=PixelRect(left=98, top=98, right=102, bottom=102),
        )


def test_tile_spec_rejects_mismatched_local_core() -> None:
    with pytest.raises(ValidationError, match="relative to crop"):
        _tile(owner_core_local=PixelRect(left=0, top=0, right=80, bottom=80))


# ── 局部观测与目标 / local observations and targets ────────────────────────


def test_local_point_observation_bounds() -> None:
    with pytest.raises(ValidationError):
        LocalPointObservation(local_id="p1", x=1000, y=0, confidence=0.5, short_evidence="")
    point = LocalPointObservation(local_id="p1", x=5, y=6, confidence=0.5, short_evidence="e")
    assert point.radius == 0
    assert point.touches_crop_border is False


def test_count_target_spec_requires_rules() -> None:
    with pytest.raises(ValidationError):
        CountTargetSpec(canonical_label="car", inclusion_rule="", exclusion_rule="x")
    spec = CountTargetSpec(
        canonical_label="car", aliases=["vehicle"], inclusion_rule="r", exclusion_rule="e"
    )
    assert spec.canonical_label == "car"


def test_tile_count_response_requires_unique_matching_points() -> None:
    with pytest.raises(ValidationError, match="reported_count"):
        TileCountResponse(target="car", tile_id="t0", reported_count=1)
    with pytest.raises(ValidationError, match="duplicate local_id"):
        TileCountResponse(
            target="car",
            tile_id="t0",
            reported_count=2,
            points=[
                LocalPointObservation(local_id="a", x=1, y=1, confidence=0.9, short_evidence=""),
                LocalPointObservation(local_id="a", x=2, y=2, confidence=0.9, short_evidence=""),
            ],
        )
    response = TileCountResponse(
        target="car",
        tile_id="t0",
        reported_count=1,
        points=[LocalPointObservation(local_id="a", x=1, y=1, confidence=0.9, short_evidence="")],
    )
    assert len(response.points) == 1


# ── 全局点与来源 / global points and provenance ────────────────────────────


def _global_point(accepted: bool = True, **overrides) -> GlobalPointObservation:
    values = dict(
        global_id="g1",
        target="car",
        source_tile_id="t0",
        local_id="l1",
        local_x_norm=10,
        local_y_norm=20,
        local_radius_norm=5,
        global_x_px=10,
        global_y_px=20,
        global_x_norm=10,
        global_y_norm=20,
        radius_px=5.0,
        confidence=0.9,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=accepted,
        short_evidence="e",
    )
    values.update(overrides)
    return GlobalPointObservation(**values)


def test_global_point_with_provenance() -> None:
    point = _global_point(
        provenance=PointProvenance(
            source="yolo_obb_center",
            backend_name="det-a",
            source_class="car",
            detector_confidence=0.85,
            weights_sha256="b" * 64,
        )
    )
    assert point.provenance is not None
    assert point.provenance.source == "yolo_obb_center"


def test_point_provenance_sha256_pattern() -> None:
    with pytest.raises(ValidationError):
        PointProvenance(weights_sha256="not-a-sha256")


def test_issue_record_roundtrip() -> None:
    issue = IssueRecord(code="E1", message="m", tile_ids=["t0"], point_ids=["g1"])
    assert issue.code == "E1"
    assert issue.point_ids == ["g1"]


# ── 草稿与最终结果 / draft and final result ────────────────────────────────


def test_counting_draft_valid() -> None:
    draft = CountingDraft(
        sample_id="s1", target="car", question="Q",
        source_width=100, source_height=100, initial_tile_count=1,
    )
    assert draft.succeeded_tiles == []


def _result(**overrides) -> CountingResult:
    values = dict(
        sample_id="s1",
        target="car",
        question="Q",
        source_width=1000,
        source_height=1000,
        tile_count=1,
        final_count=0,
        status="completed",
    )
    values.update(overrides)
    return CountingResult(**values)


def test_counting_result_final_count_must_equal_accepted_points() -> None:
    """final_count is strongly tied to the accepted points.
    final_count 与接受点数量强绑定。"""
    with pytest.raises(ValidationError, match="final_count must equal accepted"):
        _result(final_count=2, global_points=[_global_point(accepted=True)])
    result = _result(final_count=1, global_points=[_global_point(accepted=True)])
    assert result.final_count == 1
    # Rejected points do not count. / 被拒绝的点不计入。
    result2 = _result(final_count=1, global_points=[_global_point(accepted=True), _global_point(accepted=False)])
    assert result2.final_count == 1


def test_counting_result_failed_tiles_require_visible_status() -> None:
    with pytest.raises(ValidationError, match="partial or failed"):
        _result(failed_tiles=["t0"], status="completed")
    partial = _result(failed_tiles=["t0"], status="partial")
    assert partial.status == "partial"


# ── 所有权 / ownership ─────────────────────────────────────────────────────


def test_counting_contracts_live_only_in_counting_schema() -> None:
    """Counting contracts must not be defined in agents.schema or data.schema
    (checked at the class-definition level, not by mentioning).
    计数契约不得定义在 agents.schema 或 data.schema（按类定义检查，而非
    文本提及）。"""
    import ast

    tokens = {"CountingResult", "PixelRect", "TileSpec", "CountTargetSpec",
              "GlobalPointObservation", "LocalPointObservation", "PointProvenance"}
    for relative in ("agents/schema.py", "data/schema.py"):
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        defined = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        assert not (tokens & defined), f"{relative} defines counting contracts"
