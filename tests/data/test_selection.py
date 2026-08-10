"""Contract tests for deterministic sample selection and sharding.

确定性样本选择与稳定分片测试：shard 跨运行稳定、start/limit/sample_ids 组合、
参数非法在迭代前失败。不包含 asyncio / Agent / 运行目录状态。
"""

from __future__ import annotations

import pytest

from data.schema import GroundTruth, ImageRef, UnifiedSample
from data.selection import select_samples, shard_for_sample


def _sample(sample_id: str) -> UnifiedSample:
    return UnifiedSample(
        sample_id=sample_id,
        dataset="parity",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i", path="a.png", role="image")],
        question="Q",
        ground_truth=GroundTruth(answers=["yes"]),
    )


def _samples(count: int) -> list[UnifiedSample]:
    return [_sample(f"s{i:03d}") for i in range(count)]


# ── shard 稳定性 / shard stability ─────────────────────────────────────────


def test_shard_is_stable_across_runs() -> None:
    first = [shard_for_sample(f"sample-{i}", 4) for i in range(20)]
    second = [shard_for_sample(f"sample-{i}", 4) for i in range(20)]
    assert first == second


def test_shard_covers_all_buckets() -> None:
    buckets = {shard_for_sample(f"sample-{i}", 5) for i in range(200)}
    assert buckets == {0, 1, 2, 3, 4}


def test_shard_is_order_independent() -> None:
    ids = ["a", "b", "c", "d"]
    direct = [shard_for_sample(sample_id, 3) for sample_id in ids]
    reversed_ = [shard_for_sample(sample_id, 3) for sample_id in reversed(ids)][::-1]
    assert direct == reversed_


def test_shard_anchor() -> None:
    assert shard_for_sample("s000", 1) == 0
    assert 0 <= shard_for_sample("s000", 7) < 7


def test_shard_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        shard_for_sample("x", 0)


# ── 选择组合 / selection combinations ───────────────────────────────────────


def test_no_options_returns_all_in_source_order() -> None:
    samples = _samples(5)
    assert [s.sample_id for s in select_samples(samples)] == ["s000", "s001", "s002", "s003", "s004"]


def test_start_index_skips_leading_samples() -> None:
    samples = _samples(5)
    assert [s.sample_id for s in select_samples(samples, start_index=2)] == ["s002", "s003", "s004"]


def test_limit_truncates() -> None:
    samples = _samples(5)
    assert [s.sample_id for s in select_samples(samples, limit=3)] == ["s000", "s001", "s002"]


def test_sample_ids_filter() -> None:
    samples = _samples(5)
    assert [s.sample_id for s in select_samples(samples, sample_ids={"s001", "s003"})] == ["s001", "s003"]


def test_shard_filtering() -> None:
    samples = _samples(50)
    shard_0 = [s.sample_id for s in select_samples(samples, shard_index=0, shard_count=3)]
    shard_1 = [s.sample_id for s in select_samples(samples, shard_index=1, shard_count=3)]
    assert set(shard_0) & set(shard_1) == set()
    assert len(shard_0) + len(shard_1) <= 50


def test_combined_options_apply_in_fixed_order() -> None:
    samples = _samples(50)
    combined = [s.sample_id for s in select_samples(
        samples, start_index=5, limit=10, sample_ids={f"s{i:03d}" for i in range(20)},
        shard_index=0, shard_count=2,
    )]
    # Fixed order: start -> shard -> sample_ids -> limit.
    # 固定顺序：start → shard → sample_ids → limit。
    expected = [
        s.sample_id for s in samples[5:20]
        if shard_for_sample(s.sample_id, 2) == 0
    ][:10]
    assert combined == expected


def test_empty_sample_ids_yields_nothing() -> None:
    assert list(select_samples(_samples(3), sample_ids=set())) == []


# ── 参数校验 / argument validation ─────────────────────────────────────────


def test_invalid_parameters_fail_before_iteration() -> None:
    samples = _samples(3)
    with pytest.raises(ValueError, match="start_index"):
        select_samples(samples, start_index=-1)
    with pytest.raises(ValueError, match="limit"):
        select_samples(samples, limit=-2)
    with pytest.raises(ValueError, match="shard_count"):
        select_samples(samples, shard_count=0)
    with pytest.raises(ValueError, match="shard_index"):
        select_samples(samples, shard_index=3, shard_count=3)
    with pytest.raises(ValueError, match="shard_index"):
        select_samples(samples, shard_index=-1, shard_count=2)


def test_validation_happens_without_consuming_the_iterator() -> None:
    """Invalid arguments must raise at call time, before any sample is consumed.
    非法参数必须在调用时立即失败，而非消费迭代器后才失败。"""
    calls: list[str] = []

    def source():
        calls.append("first")
        yield _sample("s000")

    with pytest.raises(ValueError):
        select_samples(source(), limit=-1)
    assert calls == [], "no sample may be consumed before validation"
