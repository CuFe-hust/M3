"""Contract tests for dataset-neutral spatial geometry rules.

数据集无关空间几何规则契约测试：extreme category、3x3 grid、box gap、
orientation/arrangement evidence、证据不足保留视觉答案并记录原因、
deterministic override 仅在证据完整时发生、无数据集分支。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.schema import AgentResult, VisualEvidence
from agents.spatial import (
    SpatialQuerySpec,
    apply_spatial_geometry,
    canonical_answer,
    spatial_query_from_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _evidence(label: str, box: list[int], confidence: float = 0.9) -> VisualEvidence:
    return VisualEvidence(label=label, box=box, confidence=confidence)


def _result(
    answer: str = "qwen said",
    items: list[VisualEvidence] | None = None,
    status: str = "completed",
) -> AgentResult:
    return AgentResult(
        agent_name="spatial_agent",
        answer=answer,
        evidence_items=items or [],
        status=status,  # type: ignore[arg-type]
    )


def _spec(**overrides) -> SpatialQuerySpec:
    values = dict(operation="extreme_category", target_label="small-vehicle")
    values.update(overrides)
    return SpatialQuerySpec(**values)


# ── extreme category / 极端类别 ───────────────────────────────────────────


def test_extreme_category_top_most_overrides() -> None:
    items = [
        _evidence("small-vehicle", [100, 700, 200, 800]),
        _evidence("small-vehicle", [300, 100, 400, 200]),
        _evidence("small-vehicle", [500, 400, 600, 500]),
    ]
    result = apply_spatial_geometry(
        _spec(target_hint="top"), _result(items=items)
    )
    assert result.answer == "small-vehicle"
    assert result.status == "completed"
    geometry = result.geometry
    assert geometry["answer_source"] == "deterministic_geometry"
    assert geometry["rule"] == "top_most_box_center_y"
    assert geometry["candidate_count"] == 3
    assert geometry["selected_center_y"] == 150.0
    assert geometry["evidence_complete"] is True


def test_extreme_category_bottom_most_overrides() -> None:
    items = [
        _evidence("small-vehicle", [100, 700, 200, 800]),
        _evidence("small-vehicle", [300, 100, 400, 200]),
    ]
    result = apply_spatial_geometry(
        _spec(target_hint="bottom"), _result(items=items)
    )
    assert result.answer == "small-vehicle"
    assert result.geometry["rule"] == "bottom_most_box_center_y"
    assert result.geometry["selected_center_y"] == 750.0


def test_extreme_category_insufficient_evidence_keeps_visual_answer() -> None:
    """With too few candidates the visual answer is kept and the reason is
    recorded. 候选不足时保留视觉答案并记录原因。"""
    items = [_evidence("small-vehicle", [100, 100, 200, 200])]
    result = apply_spatial_geometry(
        _spec(target_hint="top", min_candidates=2), _result(items=items)
    )
    assert result.answer == "qwen said"
    assert result.status == "partial"
    geometry = result.geometry
    assert geometry["answer_source"] == "qwen_visual_answer"
    assert geometry["rule"] == "insufficient_extreme_candidates"
    assert geometry["evidence_complete"] is False
    assert geometry["candidate_count"] == 1


def test_extreme_category_no_hint_keeps_visual_answer() -> None:
    result = apply_spatial_geometry(
        _spec(target_hint=None),
        _result(items=[_evidence("small-vehicle", [100, 100, 200, 200])]),
    )
    assert result.answer == "qwen said"
    assert result.geometry["rule"] == "missing_extreme_hint"


def test_extreme_category_filters_by_target_label() -> None:
    """Only evidence matching the spec target label participates.
    只有匹配 spec 目标标签的证据参与。"""
    items = [
        _evidence("small-vehicle", [100, 700, 200, 800]),
        _evidence("truck", [300, 100, 400, 200]),
    ]
    result = apply_spatial_geometry(
        _spec(target_hint="top", target_label="small-vehicle"), _result(items=items)
    )
    # Only one matching candidate: below min_candidates → no override.
    # 仅一个匹配候选：低于 min_candidates → 不覆盖。
    assert result.answer == "qwen said"
    assert result.geometry["candidate_count"] == 1
    assert result.geometry["evidence_complete"] is False


# ── grid position / 九宫格位置 ────────────────────────────────────────────


def test_grid_position_single_candidate_overrides() -> None:
    spec = _spec(
        operation="grid_position",
        target_label="large-vehicle",
    )
    items = [_evidence("large-vehicle", [400, 700, 500, 800])]  # bottom-middle
    result = apply_spatial_geometry(spec, _result(items=items))
    assert result.answer == "bottom-middle"
    assert result.status == "completed"
    assert result.geometry["rule"] == "three_by_three_box_center"
    assert result.geometry["evidence_complete"] is True


def test_grid_position_ambiguous_candidates_keep_visual_answer() -> None:
    spec = _spec(operation="grid_position", target_label="car")
    items = [
        _evidence("car", [100, 100, 200, 200]),
        _evidence("car", [700, 700, 800, 800]),
    ]
    result = apply_spatial_geometry(spec, _result(items=items))
    assert result.answer == "qwen said"
    assert result.status == "partial"
    assert result.geometry["rule"] == "ambiguous_position_target"
    assert result.geometry["evidence_complete"] is False


def test_grid_position_missing_target_keeps_visual_answer() -> None:
    spec = _spec(operation="grid_position", target_label="car")
    result = apply_spatial_geometry(spec, _result(items=[_evidence("truck", [100, 100, 200, 200])]))
    assert result.answer == "qwen said"
    assert result.geometry["rule"] == "missing_position_target"


def test_grid_position_custom_boundaries() -> None:
    spec = _spec(operation="grid_position", target_label="car", grid_boundaries=(500, 500))
    items = [_evidence("car", [100, 100, 200, 200])]  # left of 500 on both axes
    result = apply_spatial_geometry(spec, _result(items=items))
    assert result.answer == "top-left"


# ── box gap / 框间距 ──────────────────────────────────────────────────────


def test_box_gap_records_without_override() -> None:
    spec = _spec(operation="box_gap")
    items = [
        _evidence("car", [100, 100, 200, 200]),
        _evidence("car", [300, 100, 400, 200]),
    ]
    result = apply_spatial_geometry(spec, _result(items=items))
    assert result.answer == "qwen said"  # never overridden / 绝不覆盖
    assert result.geometry["rule"] == "box_gap_recorded_without_threshold_override"
    assert result.geometry["nearest_box_gap"] == 100.0


def test_box_gap_insufficient_boxes() -> None:
    spec = _spec(operation="box_gap")
    result = apply_spatial_geometry(spec, _result(items=[_evidence("car", [100, 100, 200, 200])]))
    assert result.geometry["rule"] == "insufficient_boxes_for_gap"


# ── orientation / arrangement evidence ────────────────────────────────────


def test_orientation_evidence_never_overrides() -> None:
    spec = _spec(operation="orientation_evidence", target_label="car")
    result = apply_spatial_geometry(
        spec, _result(items=[_evidence("car", [100, 100, 200, 200])])
    )
    assert result.answer == "qwen said"
    assert result.geometry["rule"] == "cardinal_direction_requires_dataset_north_up_assumption"
    assert result.geometry["north_metadata_available"] is False


def test_arrangement_evidence_never_overrides() -> None:
    spec = _spec(operation="arrangement_evidence", target_label="car")
    items = [
        _evidence("car", [100, 100, 200, 200]),
        _evidence("car", [300, 100, 400, 200]),
    ]
    result = apply_spatial_geometry(spec, _result(items=items))
    assert result.answer == "qwen said"
    assert result.geometry["rule"] == "arrangement_requires_instance_set"
    assert result.geometry["candidate_count"] == 2


# ── spec 构造 / spec construction ─────────────────────────────────────────


def test_spatial_query_from_metadata() -> None:
    spec = spatial_query_from_metadata(
        {
            "spatial_query": {
                "operation": "extreme_category",
                "target_label": "small-vehicle",
                "target_hint": "top",
            }
        }
    )
    assert spec is not None
    assert spec.operation == "extreme_category"
    assert spec.target_label == "small-vehicle"
    assert spatial_query_from_metadata({}) is None
    assert spatial_query_from_metadata({"spatial_query": "junk"}) is None
    assert spatial_query_from_metadata(None) is None
    # Invalid operation never guesses. / 无效操作绝不猜测。
    assert spatial_query_from_metadata(
        {"spatial_query": {"operation": "bogus"}}
    ) is None


def test_canonical_answer_normalizes_without_plural_guessing() -> None:
    """No English plural guessing: labels pass through normalized only.
    不做英语单复数猜测：标签仅归一化后原样通过。"""
    assert canonical_answer("small-vehicle") == "small-vehicle"
    assert canonical_answer("Small Vehicle") == "small-vehicle"
    assert canonical_answer("bus") == "bus"
    assert canonical_answer("glass") == "glass"
    assert canonical_answer("class") == "class"
    assert canonical_answer("trucks") == "trucks"


def test_override_only_when_evidence_complete() -> None:
    """can_override gates the operation, evidence_complete gates the override.
    can_override 门控操作，evidence_complete 门控覆盖本身。"""
    assert _spec(operation="extreme_category").can_override is True
    assert _spec(operation="grid_position").can_override is True
    assert _spec(operation="box_gap").can_override is False
    assert _spec(operation="orientation_evidence").can_override is False
    assert _spec(operation="arrangement_evidence").can_override is False


# ── 边界 / boundaries ─────────────────────────────────────────────────────


def test_spatial_modules_have_no_dataset_branch() -> None:
    for relative in ("schema.py", "geometry.py", "__init__.py"):
        source = (REPO_ROOT / "agents" / "spatial" / relative).read_text(encoding="utf-8")
        assert "VRSBench" not in source, relative
        assert "vrsbench" not in source, relative
        assert "spacers_agent" not in source, relative


def test_spatial_geometry_never_reads_question() -> None:
    """Geometry consumes only the structured spec — no question text, no
    dataset-specific regex. 几何只消费结构化 spec——无问题文本、无数据集专用
    正则。"""
    source = (REPO_ROOT / "agents" / "spatial" / "geometry.py").read_text(encoding="utf-8")
    assert "question" not in source
    assert "re.search" not in source
