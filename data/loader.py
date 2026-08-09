"""Convenience dataset loading over the current Registry/Adapters.

当前 Registry/Adapters 之上的便捷数据集加载。无隐式下载/网络回退；返回
当前 UnifiedSample（绝非旧 CanonicalSample）。draft 适配器经独立类型化
迭代器（load_dataset_drafts → SampleDraft）暴露，绝不假装 draft 是
UnifiedSample。无 data/loaders.py 旧版别名。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from data.registry import build_default_registry
from data.schema import SampleDraft, UnifiedSample


def load_dataset_samples(
    dataset: str,
    *,
    root: Path,
    split: str,
    task: str | None = None,
    limit: int | None = None,
) -> Iterator[UnifiedSample]:
    """Yield UnifiedSample rows from one registered adapter: registry lookup
    → probe → iter_samples → optional deterministic task filter → limit.
    No network fallback ever happens. 从一个已注册适配器产出 UnifiedSample
    行：注册表查找 → probe → iter_samples → 可选确定性 task 过滤 → limit。
    绝不发生网络回退。"""

    adapter = build_default_registry().get(dataset)
    tasks = (task,) if task is not None else sorted(adapter.supported_tasks)
    yielded = 0
    for current_task in tasks:
        adapter.probe(root, current_task)
        for sample in adapter.iter_samples(root, split, current_task):
            if limit is not None and yielded >= limit:
                return
            yield sample
            yielded += 1


def load_dataset_drafts(
    dataset: str,
    *,
    root: Path,
    split: str,
    limit: int | None = None,
) -> Iterator[SampleDraft]:
    """Yield SampleDraft rows from a draft adapter (iter_drafts); adapters
    without draft support fail stably. Drafts are typed separately and are
    never presented as UnifiedSample. 从一个 draft 适配器（iter_drafts）
    产出 SampleDraft 行；无 draft 支持的适配器稳定失败。Draft 独立类型化，
    绝不冒充 UnifiedSample。"""

    adapter = build_default_registry().get(dataset)
    if not hasattr(adapter, "iter_drafts"):
        raise TypeError(f"adapter {adapter.name!r} does not yield drafts")
    adapter.probe(root, None)
    yielded = 0
    for draft in adapter.iter_drafts(root, split):
        if limit is not None and yielded >= limit:
            return
        yield draft
        yielded += 1
