"""Contract tests for deterministic counting metrics and judge merging.

确定性计数指标与 judge 合并契约测试：exact/absolute/relative/smooth 数值、
负计数拒绝、judge 不覆盖确定性指标、inconsistency 标志、跨记录汇总、
无网络副作用。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.counting.schema import CountingResult
from data.schema import GroundTruth
from evaluation.metrics.counting import (
    aggregate_counting,
    count_deterministic_metrics,
    merge_count_evaluation,
)
from evaluation.records import EvaluationRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


def _counting(final_count: int = 5) -> CountingResult:
    """A CountingResult whose final_count matches its accepted points.
    一个 final_count 与接受点数一致的 CountingResult。"""
    from agents.counting.schema import GlobalPointObservation

    points = [
        GlobalPointObservation(
            global_id=f"p{index}",
            target="car",
            source_tile_id="t0",
            local_id=f"l{index}",
            local_x_norm=100 + index,
            local_y_norm=100,
            local_radius_norm=5,
            global_x_px=100 + index,
            global_y_px=100,
            global_x_norm=100 + index,
            global_y_norm=100,
            radius_px=5.0,
            confidence=0.9,
            ownership_valid=True,
            near_core_boundary=False,
            accepted=True,
            short_evidence="x",
        )
        for index in range(final_count)
    ]
    return CountingResult(
        sample_id="s1",
        target="car",
        question="How many cars?",
        source_width=1000,
        source_height=1000,
        tile_count=1,
        final_count=final_count,
        global_points=points,
        status="completed",
    )


def test_exact_match_and_errors() -> None:
    metrics = count_deterministic_metrics(5, 5)
    assert metrics.exact_match == 1
    assert metrics.absolute_error == 0
    assert metrics.relative_error == 0.0
    assert metrics.smooth_error_score == 1.0


def test_absolute_and_relative_errors() -> None:
    metrics = count_deterministic_metrics(5, 3)
    assert metrics.exact_match == 0
    assert metrics.absolute_error == 2
    assert metrics.relative_error == 2 / 4
    assert 0.0 < metrics.smooth_error_score < 1.0
    # The denominator is |gold| + 1, so the score is gold-anchored, not
    # symmetric. / 分母是 |gold| + 1，分数以金标为锚，因此不对称。
    mirrored = count_deterministic_metrics(3, 5)
    assert mirrored.relative_error == 2 / 6
    assert mirrored.smooth_error_score == pytest.approx(2.718281828459045 ** -1)


def test_zero_gold_count() -> None:
    metrics = count_deterministic_metrics(0, 0)
    assert metrics.exact_match == 1
    assert metrics.relative_error == 0.0
    metrics = count_deterministic_metrics(2, 0)
    assert metrics.relative_error == 2 / 1
    assert metrics.smooth_error_score == pytest.approx(2.718281828459045 ** -6)


def test_negative_counts_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        count_deterministic_metrics(-1, 5)
    with pytest.raises(ValueError, match="negative"):
        count_deterministic_metrics(5, -1)


# ── judge 合并 / judge merging ─────────────────────────────────────────────


class _CountJudge:
    def __init__(self, verdict: str = "incorrect") -> None:
        self.verdict = verdict


def test_merge_without_ground_truth_keeps_metrics_none() -> None:
    record = merge_count_evaluation(
        sample_id="s1", counting=_counting(5), ground_truth=None
    )
    assert record.deterministic_metrics is None
    assert record.judge_status == "not_requested"
    assert record.judge_inconsistency is False


def test_judge_never_overrides_deterministic_metrics() -> None:
    """A 'correct' judge verdict cannot change the deterministic mismatch.
    即使 judge 判定 correct 也不能改变确定性的不匹配结果。"""
    record = merge_count_evaluation(
        sample_id="s1",
        counting=_counting(5),
        ground_truth=GroundTruth(count=3),
        judge_raw="raw",
        judge_parsed=_CountJudge(verdict="correct"),
    )
    assert record.deterministic_metrics.exact_match == 0
    assert record.deterministic_metrics.predicted_count == 5
    assert record.judge_status == "succeeded"
    assert record.judge_inconsistency is True
    assert record.judge_raw == "raw"


def test_judge_agrees_with_deterministic_match_no_inconsistency() -> None:
    record = merge_count_evaluation(
        sample_id="s1",
        counting=_counting(3),
        ground_truth=GroundTruth(count=3),
        judge_parsed=_CountJudge(verdict="correct"),
    )
    assert record.deterministic_metrics.exact_match == 1
    assert record.judge_inconsistency is False


def test_judge_error_status() -> None:
    record = merge_count_evaluation(
        sample_id="s1",
        counting=_counting(5),
        ground_truth=GroundTruth(count=3),
        judge_error="boom",
    )
    assert record.judge_status == "failed"
    assert record.judge_error == "boom"
    assert record.deterministic_metrics.exact_match == 0


# ── 跨记录汇总 / aggregation ───────────────────────────────────────────────


def _record(count: int, gold: int) -> EvaluationRecord:
    return merge_count_evaluation(
        sample_id=f"s{count}-{gold}",
        counting=_counting(count),
        ground_truth=GroundTruth(count=gold),
    )


def test_aggregate_counting() -> None:
    summary = aggregate_counting([_record(5, 5), _record(5, 3), _record(0, 2)])
    assert summary["total"] == 3
    assert summary["samples_with_ground_truth"] == 3
    assert summary["exact_match_accuracy"] == pytest.approx(1 / 3)
    assert summary["mean_absolute_error"] == pytest.approx((0 + 2 + 2) / 3)
    assert summary["mean_relative_error"] == pytest.approx((0 + 0.5 + 2 / 3) / 3)


def test_aggregate_counting_without_ground_truth() -> None:
    record = merge_count_evaluation(sample_id="s1", counting=_counting(5), ground_truth=None)
    summary = aggregate_counting([record])
    assert summary["total"] == 1
    assert summary["samples_with_ground_truth"] == 0


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_counting_metrics_have_no_network_side_effects() -> None:
    for module in ("evaluation/records.py", "evaluation/metrics/counting.py"):
        source = (REPO_ROOT / module).read_text(encoding="utf-8")
        for token in ("urlopen", "requests", "socket", "httpx", "api.deepseek", "http://", "https://"):
            assert token not in source, (module, token)
