"""Contract tests for the point-counting pipeline.

点式计数流水线契约测试：不切片/多 tile 两条路径、顺序/并发、失败收集与
partial/failed 状态、递归分割限制、接受点导出 final_count、回调异常不
静默丢弃、reported_count 一致性拒绝。
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from agents.counting.point_pipeline import (
    PointCountingOrchestrator,
    TileCountCallback,
    apply_acceptance_policy,
    decide_seam_pairs,
    finalize_representatives,
    find_boundary_conflicts,
)
from agents.counting.schema import (
    CountTargetSpec,
    GlobalPointObservation,
    LocalPointObservation,
    SeamDecision,
    TileCountResponse,
    TileSpec,
)
from agents.counting.settings import CountingSettings

REPO_ROOT = Path(__file__).resolve().parents[3]

_TARGET = CountTargetSpec(
    canonical_label="car",
    aliases=["vehicle"],
    inclusion_rule="visible vehicle",
    exclusion_rule="occluded more than half",
)


def _point(local_id: str, x: int, y: int, confidence: float = 0.9) -> LocalPointObservation:
    return LocalPointObservation(
        local_id=local_id, x=x, y=y, confidence=confidence, short_evidence="e"
    )


class _RecordingCallback:
    """Deterministic tile callback with optional failures and concurrency
    tracking. 带可选失败与并发跟踪的确定性 tile 回调。"""

    def __init__(
        self,
        *,
        responses: dict[str, TileCountResponse] | None = None,
        fail_tile_ids: set[str] | None = None,
        needs_split_tile_ids: set[str] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.fail_tile_ids = fail_tile_ids or set()
        self.needs_split_tile_ids = needs_split_tile_ids or set()
        self.calls: list[tuple[str, tuple[int, int]]] = []
        self.max_active = 0
        self._active = 0

    async def count_tile(self, *, tile, image, target) -> TileCountResponse:
        self.calls.append((tile.tile_id, image.size))
        if tile.tile_id in self.fail_tile_ids:
            raise RuntimeError("tile boom")
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        await asyncio.sleep(0.01)
        self._active -= 1
        cached = self.responses.get(tile.tile_id)
        if cached is not None:
            return cached
        return TileCountResponse(
            target=target.canonical_label,
            tile_id=tile.tile_id,
            reported_count=0,
            needs_split=tile.tile_id in self.needs_split_tile_ids,
        )


class _EmptyReviewCallback(_RecordingCallback):
    def __init__(self, *, review_response=None, review_error=None) -> None:
        super().__init__()
        self.review_response = review_response
        self.review_error = review_error
        self.review_calls: list[str] = []

    async def review_empty_tile(self, *, tile, image, target) -> TileCountResponse:
        self.review_calls.append(tile.tile_id)
        if self.review_error is not None:
            raise self.review_error
        if self.review_response is not None:
            return self.review_response
        return TileCountResponse(
            target=target.canonical_label,
            tile_id=tile.tile_id,
            reported_count=0,
        )


def _image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (1, 2, 3))


def _orchestrator(
    callback: TileCountCallback,
    **counting_overrides,
) -> PointCountingOrchestrator:
    return PointCountingOrchestrator(
        callback, counting=CountingSettings(**counting_overrides)
    )


async def _count(
    orchestrator: PointCountingOrchestrator,
    image: Image.Image,
    *,
    sample_id: str = "s1",
    minimum_scan_depth: int = 0,
):
    return await orchestrator.count_image(
        image,
        sample_id=sample_id,
        question="How many cars?",
        target=_TARGET,
        minimum_scan_depth=minimum_scan_depth,
    )


# ── 两条路径 / two paths ──────────────────────────────────────────────────


def test_callback_protocol_is_async() -> None:
    assert inspect.iscoroutinefunction(_RecordingCallback().count_tile)


def test_no_tiling_path_uses_single_whole_tile() -> None:
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    assert [tile_id for tile_id, _ in callback.calls] == ["whole"]
    assert result.status == "completed"
    assert result.final_count == 0
    assert result.succeeded_tiles == ["whole"]
    assert result.initial_tile_count == 1


def test_empty_tile_review_positive_result_remains_point_derived() -> None:
    callback = _EmptyReviewCallback(
        review_response=TileCountResponse(
            target="car",
            tile_id="whole",
            points=[_point("review-1", 500, 500)],
            reported_count=1,
        )
    )
    orchestrator = PointCountingOrchestrator(
        callback,
        counting=CountingSettings(),
        empty_tile_reviewer=callback,
    )
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    assert callback.review_calls == ["whole"]
    assert result.final_count == 1
    assert result.final_count == sum(point.accepted for point in result.global_points)


def test_empty_tile_review_failure_keeps_zero_with_visible_warning() -> None:
    callback = _EmptyReviewCallback(review_error=RuntimeError("review boom"))
    orchestrator = PointCountingOrchestrator(
        callback,
        counting=CountingSettings(),
        empty_tile_reviewer=callback,
    )
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    assert result.final_count == 0
    assert result.status == "completed_with_warnings"
    assert any(
        warning.code == "EMPTY_TILE_REVIEW_FAILURE"
        for warning in result.warnings
    )
    assert all("review boom" not in warning.message for warning in result.warnings)


def test_multi_tile_path_is_row_major() -> None:
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(2000, 2000)))
    tile_ids = [tile_id for tile_id, _ in callback.calls]
    assert tile_ids[0] == "r000_c000"
    assert len(tile_ids) == 9
    assert result.initial_tile_count == 9
    assert result.succeeded_tiles == tile_ids
    assert result.tile_count == 9


def test_callback_receives_tile_crop_image() -> None:
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback)
    asyncio.run(_count(orchestrator, _image(2000, 2000)))
    # The interior tile r001_c001 carries core + halo on every side.
    # 内部切片 r001_c001 四侧带 halo。
    interior_id, interior_size = callback.calls[4]
    assert interior_id == "r001_c001"
    assert interior_size == (896 + 128 * 2, 896 + 128 * 2)


# ── 失败收集 / failure collection ─────────────────────────────────────────


def test_callback_failure_marks_tile_failed_and_partial() -> None:
    callback = _RecordingCallback(fail_tile_ids={"r000_c000"})
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(2000, 2000)))
    assert "r000_c000" in result.failed_tiles
    assert len(result.succeeded_tiles) == 8
    assert result.status == "partial"
    assert any(record.code == "TILE_FAILURE" for record in result.warnings)
    assert result.final_count == 0


def test_all_tiles_failed_yields_failed_status() -> None:
    callback = _RecordingCallback(fail_tile_ids={"whole"})
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    assert result.failed_tiles == ["whole"]
    assert result.status == "failed"
    assert result.succeeded_tiles == []


def test_callback_exception_is_never_silently_dropped() -> None:
    """Callback failures surface as failed tiles and warnings — never dropped.
    回调失败以 failed tile 与 warning 呈现——绝不丢弃。"""
    callback = _RecordingCallback(fail_tile_ids={"whole"})
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    assert result.failed_tiles == ["whole"]
    assert any("RuntimeError" in record.message for record in result.warnings)


# ── 响应校验 / response validation ────────────────────────────────────────


def test_tile_id_mismatch_fails_tile() -> None:
    response = TileCountResponse(
        target="car", tile_id="wrong-tile", reported_count=0
    )
    callback = _RecordingCallback(responses={"whole": response})
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    assert result.failed_tiles == ["whole"]
    assert result.status == "failed"


def test_target_mismatch_fails_tile() -> None:
    response = TileCountResponse(
        target="airplane", tile_id="whole", reported_count=0
    )
    callback = _RecordingCallback(responses={"whole": response})
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    assert result.failed_tiles == ["whole"]


def test_reported_count_mismatch_is_rejected_by_schema() -> None:
    """reported_count must equal len(points); inconsistent responses cannot
    even be constructed. reported_count 必须等于 len(points)；不一致的响应
    根本无法构造。"""
    with pytest.raises(ValidationError, match="reported_count"):
        TileCountResponse(
            target="car", tile_id="whole", reported_count=2,
            points=[_point("p1", 100, 100)],
        )


# ── 接受策略与最终计数 / acceptance and final count ──────────────────────


def test_accepted_points_drive_final_count() -> None:
    response = TileCountResponse(
        target="car",
        tile_id="whole",
        reported_count=2,
        points=[_point("p1", 500, 500, confidence=0.9), _point("p2", 600, 600, confidence=0.9)],
    )
    callback = _RecordingCallback(responses={"whole": response})
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    assert result.final_count == 2
    assert sum(point.accepted for point in result.global_points) == 2


def test_low_confidence_points_are_rejected() -> None:
    response = TileCountResponse(
        target="car",
        tile_id="whole",
        reported_count=2,
        points=[_point("p1", 500, 500, confidence=0.9), _point("p2", 600, 600, confidence=0.05)],
    )
    callback = _RecordingCallback(responses={"whole": response})
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    assert result.final_count == 1
    rejected = [p for p in result.global_points if not p.accepted]
    assert rejected[0].rejection_reason == "LOW_CONFIDENCE"


def test_final_count_has_no_independent_model_field() -> None:
    """TileCountResponse carries no final_count field; the final count is
    derived exclusively from accepted global points.
    TileCountResponse 没有 final_count 字段；最终计数只由接受点导出。"""
    assert "final_count" not in TileCountResponse.model_fields
    assert "accepted" not in TileCountResponse.model_fields


def test_apply_acceptance_policy_validation() -> None:
    from agents.counting.schema import GlobalPointObservation

    point = GlobalPointObservation(
        global_id="g1", target="car", source_tile_id="t0", local_id="l1",
        local_x_norm=10, local_y_norm=20, local_radius_norm=0,
        global_x_px=10, global_y_px=20, global_x_norm=10, global_y_norm=20,
        radius_px=0.0, confidence=0.5, ownership_valid=True,
        near_core_boundary=False, accepted=True, short_evidence="e",
    )
    with pytest.raises(ValueError, match="min_confidence"):
        apply_acceptance_policy(point, min_confidence=1.5)


# ── 递归分割 / recursive splitting ────────────────────────────────────────


def test_needs_split_creates_children() -> None:
    callback = _RecordingCallback(needs_split_tile_ids={"r000_c000"})
    orchestrator = _orchestrator(callback)
    result = asyncio.run(_count(orchestrator, _image(2000, 2000)))
    # 9 parent tiles, the first splits into 4 children.
    # 9 个父 tile，第一个分割为 4 个子 tile。
    assert len(callback.calls) == 9 + 4
    child_ids = [tile_id for tile_id, _ in callback.calls if "d1" in tile_id]
    assert len(child_ids) == 4
    assert "r000_c000" not in result.succeeded_tiles
    assert len(result.succeeded_tiles) == 9 - 1 + 4


def test_recursive_split_respects_max_depth() -> None:
    callback = _RecordingCallback(needs_split_tile_ids={"r000_c000"})
    orchestrator = _orchestrator(callback, max_recursive_depth=0)
    result = asyncio.run(_count(orchestrator, _image(2000, 2000)))
    # Splitting is disabled entirely. / 分割被完全禁用。
    assert len(callback.calls) == 9
    assert any(record.message == "RECURSIVE_SPLIT_LIMIT" for record in result.warnings)


def test_recursive_split_respects_min_core_size() -> None:
    callback = _RecordingCallback(needs_split_tile_ids={"r000_c000"})
    # min_core_size=500 means a 896px core (896 < 1000) cannot split.
    # min_core_size=500 使 896px core（896 < 1000）无法分割。
    orchestrator = _orchestrator(callback, min_core_size=500)
    result = asyncio.run(_count(orchestrator, _image(2000, 2000)))
    assert len(callback.calls) == 9
    assert any(record.message == "RECURSIVE_SPLIT_LIMIT" for record in result.warnings)


# ── 顺序与并发 / sequential and concurrent ────────────────────────────────


def test_sequential_default_runs_in_order() -> None:
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback)
    asyncio.run(_count(orchestrator, _image(2000, 2000)))
    tile_ids = [tile_id for tile_id, _ in callback.calls]
    assert tile_ids == sorted(tile_ids)
    assert callback.max_active == 1


def test_concurrent_mode_respects_concurrency() -> None:
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback, sequential=False, concurrency=4)
    result = asyncio.run(_count(orchestrator, _image(2000, 2000)))
    assert len(result.succeeded_tiles) == 9
    assert callback.max_active == 4


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_pipeline_has_no_backend_selector_or_html() -> None:
    source = (REPO_ROOT / "agents" / "counting" / "point_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "BackendSelector" not in source
    assert "spacers_agent" not in source
    assert "VRSBench" not in source
    assert "html" not in source.casefold()


def test_find_boundary_conflicts_requires_neighbouring_cores() -> None:
    """Only adjacent-core near-boundary duplicates are candidates.
    只有相邻 core 的边界附近重复才成为候选。"""
    from agents.counting.geometry import build_core_halo_tiles, convert_local_point_to_global

    tiles = build_core_halo_tiles(1000, 1000, core_size=500, halo_size=100, model_max_side=1280)
    # One point in each of two neighbouring cores, both near the shared edge.
    # 两个相邻 core 各一个点，都靠近共享边界。
    points = []
    for tile in tiles[:2]:
        points.append(
            convert_local_point_to_global(
                _point(tile.tile_id, 500, 500, confidence=0.9),
                tile,
                sample_id="s1",
                target="car",
                boundary_band_px=0,
                tolerance_px=100,
            )
        )
    conflicts = find_boundary_conflicts(points, tiles)
    assert isinstance(conflicts, list)
    assert len(conflicts) <= 1


# ── seam finalization / seam 最终化 (25.5) ────────────────────────────────


def _seam_tile(tile_id: str, core_left: int, core_top: int, core_right: int, core_bottom: int) -> TileSpec:
    from agents.counting.schema import PixelRect

    core = PixelRect(left=core_left, top=core_top, right=core_right, bottom=core_bottom)
    local = PixelRect(
        left=0,
        top=0,
        right=core_right - core_left,
        bottom=core_bottom - core_top,
    )
    return TileSpec(
        tile_id=tile_id, row=0, col=0,
        crop_global=core,
        owner_core_global=core,
        owner_core_local=local,
        source_width=100, source_height=100,
        model_input_width=100, model_input_height=100,
    )


def _seam_point(
    global_id: str,
    tile_id: str,
    x: int,
    y: int,
    *,
    near_boundary: bool,
    source: str = "qwen_point",
) -> GlobalPointObservation:
    from agents.counting.schema import PointProvenance

    return GlobalPointObservation(
        global_id=global_id,
        target="car",
        source_tile_id=tile_id,
        local_id=global_id,
        local_x_norm=x,
        local_y_norm=y,
        local_radius_norm=0,
        global_x_px=x,
        global_y_px=y,
        global_x_norm=x,
        global_y_norm=y,
        radius_px=4.0,
        confidence=0.9,
        ownership_valid=True,
        near_core_boundary=near_boundary,
        accepted=True,
        short_evidence="e",
        provenance=PointProvenance(
            source=source,
            source_class="car",
            obb_polygon_global_px=(
                [[x - 4, y - 4], [x - 4, y + 4], [x + 4, y + 4], [x + 4, y - 4]]
                if source == "yolo_obb_center"
                else None
            ),
        ),
    )


def test_seam_finalization_merges_strong_duplicates() -> None:
    """Two adjacent tiles reporting the same boundary instance merge into one.
    两个相邻 tile 对同一边界实例的重复报告合并为一个。"""
    from agents.counting.geometry import cores_are_neighbours
    from agents.counting.point_pipeline import decide_seam_pairs, find_boundary_conflicts

    left = _seam_tile("t0", 0, 0, 50, 100)
    right = _seam_tile("t1", 50, 0, 100, 100)
    assert cores_are_neighbours(left, right) is True
    # Both points sit on the shared boundary (49/50) → very close.
    # 两个点都在共享边界（49/50）→ 距离很近。
    first = _seam_point("g0", "t0", 49, 50, near_boundary=True)
    second = _seam_point("g1", "t1", 50, 50, near_boundary=True)
    conflicts = find_boundary_conflicts([first, second], [left, right])
    assert len(conflicts) == 1
    pairs, unresolved = decide_seam_pairs(conflicts)
    assert pairs == [("g0", "g1")]
    assert unresolved == []
    final_points, merged_groups = finalize_representatives([first, second], pairs)
    assert sum(point.accepted for point in final_points) == 1
    rejected = [p for p in final_points if not p.accepted]
    assert rejected[0].rejection_reason == "MERGED_AT_SEAM"
    assert merged_groups == [["g0", "g1"]]


def test_seam_finalization_keeps_weak_conflicts_unresolved() -> None:
    """Nearby but not strongly matching boundary points stay unresolved.
    接近但证据不足的边界点保留为未解决。"""
    from agents.counting.geometry import build_core_halo_tiles
    from agents.counting.point_pipeline import decide_seam_pairs

    # Two points within threshold but beyond the merge factor.
    # 两个点在阈值内但超出合并因子。
    first = _seam_point("g0", "t0", 49, 50, near_boundary=True)
    second = _seam_point("g1", "t1", 65, 50, near_boundary=True)
    left = _seam_tile("t0", 0, 0, 50, 100)
    right = _seam_tile("t1", 50, 0, 100, 100)
    conflicts = [
        {
            "conflict_id": "g0|g1",
            "first_global_id": "g0",
            "second_global_id": "g1",
            "threshold_px": 40.0,
            "distance_px": 30.0,
        }
    ]
    pairs, unresolved = decide_seam_pairs(conflicts)
    assert pairs == []
    assert unresolved == [("g0", "g1")]


def test_seam_disabled_skips_finalization() -> None:
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback, seam_verify=False)
    result = asyncio.run(_count(orchestrator, _image(200, 200)))
    codes = {record.code for record in result.warnings}
    assert "SEAM_VERIFY_DISABLED" in codes
    assert result.merged_groups == []


def test_seam_merge_is_stable_and_reproducible() -> None:
    """merged_groups and unresolved conflicts are sorted and reproducible.
    merged_groups 与 unresolved 冲突排序稳定且可复现。"""
    from agents.counting.point_pipeline import decide_seam_pairs

    conflicts = [
        {"conflict_id": "b|a", "first_global_id": "b", "second_global_id": "a",
         "threshold_px": 10.0, "distance_px": 2.0},
        {"conflict_id": "d|c", "first_global_id": "d", "second_global_id": "c",
         "threshold_px": 10.0, "distance_px": 9.0},
    ]
    pairs1, unresolved1 = decide_seam_pairs(conflicts)
    pairs2, unresolved2 = decide_seam_pairs(list(reversed(conflicts)))
    assert pairs1 == pairs2
    assert unresolved1 == unresolved2


class _SeamReviewer:
    def __init__(self, decision: str = "uncertain", error: Exception | None = None):
        self.decision = decision
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def review(self, **kwargs) -> SeamDecision:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SeamDecision(decision=self.decision)


def _seam_conflict(distance: float, threshold: float = 10.0) -> dict[str, object]:
    return {
        "conflict_id": "g0|g1",
        "first_global_id": "g0",
        "second_global_id": "g1",
        "threshold_px": threshold,
        "distance_px": distance,
    }


def _resolve_seam(
    conflict: dict[str, object],
    *,
    reviewer: _SeamReviewer | None,
):
    first = _seam_point("g0", "t0", 49, 50, near_boundary=True)
    second = _seam_point("g1", "t1", 54, 50, near_boundary=True)
    orchestrator = PointCountingOrchestrator(
        _RecordingCallback(),
        counting=CountingSettings(seam_crop_margin_px=8),
        seam_reviewer=reviewer,
    )
    resolved = asyncio.run(
        orchestrator._resolve_seam_conflicts(
            [conflict],
            _image(100, 100),
            [first, second],
        )
    )
    return resolved, reviewer


def test_strong_seam_match_merges_without_visual_call() -> None:
    (merged, unresolved, warnings), reviewer = _resolve_seam(
        _seam_conflict(2.0),
        reviewer=_SeamReviewer("different_instances"),
    )
    assert merged == [("g0", "g1")]
    assert unresolved == []
    assert warnings == []
    assert reviewer is not None and reviewer.calls == []


def test_clear_separate_seam_pair_uses_no_visual_call() -> None:
    (merged, unresolved, warnings), reviewer = _resolve_seam(
        _seam_conflict(9.0),
        reviewer=_SeamReviewer("same_instance"),
    )
    assert merged == []
    assert unresolved == []
    assert warnings == []
    assert reviewer is not None and reviewer.calls == []


@pytest.mark.parametrize(
    ("decision", "expected_merged", "expected_unresolved"),
    [
        ("same_instance", [("g0", "g1")], []),
        ("different_instances", [], []),
        ("uncertain", [], [("g0", "g1")]),
    ],
)
def test_ambiguous_seam_pair_uses_visual_decision(
    decision: str,
    expected_merged: list[tuple[str, str]],
    expected_unresolved: list[tuple[str, str]],
) -> None:
    (merged, unresolved, warnings), reviewer = _resolve_seam(
        _seam_conflict(5.0),
        reviewer=_SeamReviewer(decision),
    )
    assert merged == expected_merged
    assert unresolved == expected_unresolved
    assert warnings == []
    assert reviewer is not None and len(reviewer.calls) == 1
    assert reviewer.calls[0]["image"].size == (22, 17)


def test_seam_reviewer_exception_is_unresolved_with_stable_warning() -> None:
    (merged, unresolved, warnings), _reviewer = _resolve_seam(
        _seam_conflict(5.0),
        reviewer=_SeamReviewer(error=RuntimeError("sensitive details")),
    )
    assert merged == []
    assert unresolved == [("g0", "g1")]
    assert warnings[0].code == "COUNTING_SEAM_REVIEW_FAILED"
    assert warnings[0].message == "Seam review failed: RuntimeError"
    assert "sensitive details" not in warnings[0].message


def test_seam_budget_exhaustion_is_unresolved_with_stable_warning() -> None:
    from workflows.call_budget import CallBudgetExceeded

    (merged, unresolved, warnings), _reviewer = _resolve_seam(
        _seam_conflict(5.0),
        reviewer=_SeamReviewer(error=CallBudgetExceeded("budget details")),
    )
    assert merged == []
    assert unresolved == [("g0", "g1")]
    assert warnings[0].message == "Seam review failed: CallBudgetExceeded"


def test_ambiguous_seam_without_callback_stays_deterministic_unresolved() -> None:
    (merged, unresolved, warnings), _reviewer = _resolve_seam(
        _seam_conflict(5.0),
        reviewer=None,
    )
    assert merged == []
    assert unresolved == [("g0", "g1")]
    assert warnings == []


def test_yolo_obb_pair_is_not_reprocessed_by_point_seam_logic() -> None:
    left = _seam_tile("t0", 0, 0, 50, 100)
    right = _seam_tile("t1", 50, 0, 100, 100)
    first = _seam_point(
        "g0", "t0", 49, 50, near_boundary=True, source="yolo_obb_center"
    )
    second = _seam_point(
        "g1", "t1", 50, 50, near_boundary=True, source="yolo_obb_center"
    )
    assert find_boundary_conflicts([first, second], [left, right]) == []


def test_semantic_centroids_use_shared_seam_candidate_logic() -> None:
    left = _seam_tile("t0", 0, 0, 50, 100)
    right = _seam_tile("t1", 50, 0, 100, 100)
    first = _seam_point(
        "g0",
        "t0",
        49,
        50,
        near_boundary=True,
        source="semantic_component_centroid",
    )
    second = _seam_point(
        "g1",
        "t1",
        50,
        50,
        near_boundary=True,
        source="semantic_component_centroid",
    )
    conflicts = find_boundary_conflicts([first, second], [left, right])
    assert len(conflicts) == 1
    assert decide_seam_pairs(conflicts)[0] == [("g0", "g1")]
