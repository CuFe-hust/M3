"""Deterministic sample selection and stable sharding for the data layer.

数据层确定性样本选择与稳定分片：start_index、limit、sample_ids、shard 的
确定顺序组合，从 DatasetRunner 中提取为纯能力。不包含 asyncio、Agent
或任何运行目录状态。参数非法时在开始迭代前失败。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

from data.schema import UnifiedSample


def shard_for_sample(sample_id: str, shard_count: int) -> int:
    """Stable shard index: sha256(sample_id) mod shard_count.
    稳定分片：sha256(sample_id) 对 shard_count 取模；跨运行、跨平台稳定。"""
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % shard_count


def _validate_options(
    *,
    start_index: int,
    limit: int | None,
    shard_index: int,
    shard_count: int,
) -> None:
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError(f"start_index must be a non-negative integer, got {start_index!r}")
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError(f"limit must be None or a non-negative integer, got {limit!r}")
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )


def select_samples(
    samples: Iterable[UnifiedSample],
    *,
    start_index: int = 0,
    limit: int | None = None,
    sample_ids: set[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Iterator[UnifiedSample]:
    """Apply start/shard/sample_ids/limit in a fixed deterministic order.
    按固定确定顺序组合 start/shard/sample_ids/limit：先跳过 start_index，
    再过滤分片，再过滤 sample_ids，最后以 limit 截断。
    Invalid arguments fail at call time, before any sample is consumed.
    参数非法在调用时立即失败，不消费任何样本。"""
    _validate_options(
        start_index=start_index,
        limit=limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    return _select(
        samples,
        start_index=start_index,
        limit=limit,
        sample_ids=sample_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def _select(
    samples: Iterable[UnifiedSample],
    *,
    start_index: int,
    limit: int | None,
    sample_ids: set[str] | None,
    shard_index: int,
    shard_count: int,
) -> Iterator[UnifiedSample]:
    selected = 0
    for index, sample in enumerate(samples):
        if index < start_index:
            continue
        if shard_for_sample(sample.sample_id, shard_count) != shard_index:
            continue
        if sample_ids is not None and sample.sample_id not in sample_ids:
            continue
        if limit is not None and selected >= limit:
            break
        selected += 1
        yield sample
