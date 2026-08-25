"""Exact, no-weight token-length audit for ChangeAgent multimodal SFT."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .adapters import qwen_multimodal
from .image_roots import ImageRootRegistry


def _flat_ids(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise ValueError("token audit has no values")
    ordered = sorted(int(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(float(percentile) * len(ordered)) - 1))
    return ordered[index]


def _summary(values: Sequence[int], thresholds: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("token audit split has no episodes")
    return {
        "checked": len(values),
        "min": min(values),
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
        "over_threshold": {
            str(threshold): sum(value > threshold for value in values)
            for threshold in thresholds
        },
    }


def _image_size(registry: ImageRootRegistry, source: str, relative: str) -> tuple[int, int]:
    path = registry.resolve(source, relative)
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:  # noqa: BLE001 - audit must fail closed
        raise ValueError(f"IMAGE_DECODE_ERROR:{source}:{relative}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"IMAGE_DECODE_ERROR:{source}:{relative}")
    return int(width), int(height)


def _visual_delta(
    *,
    processor: Any,
    registry: ImageRootRegistry,
    episode: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    text_length: int,
) -> int:
    images = [
        registry.load_rgb(str(item["image_source"]), str(item["path"]))
        for item in episode["images"]
    ]
    try:
        text = processor.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False
        )
        encoded = processor(text=[text], images=images, return_tensors="pt")
    finally:
        for image in images:
            close = getattr(image, "close", None)
            if callable(close):
                close()
    if "mm_token_type_ids" not in encoded:
        raise ValueError("MM_TOKEN_TYPE_IDS_MISSING")
    input_ids = encoded["input_ids"]
    shape = getattr(input_ids, "shape", ())
    full_length = int(shape[-1]) if shape else len(_flat_ids(input_ids))
    delta = full_length - int(text_length)
    if delta < 0:
        raise ValueError("NEGATIVE_VISUAL_TOKEN_DELTA")
    return delta


def audit_change_agent_tokens(
    *,
    profile: Any,
    processor: Any,
    image_roots: ImageRootRegistry,
    split_episodes: Mapping[str, Iterable[Mapping[str, Any]]],
    thresholds: Sequence[int] = (4096, 8192),
    progress_every: int = 5000,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Measure exact untruncated lengths while caching shape-only visual deltas."""

    normalized_thresholds = tuple(sorted(set(int(value) for value in thresholds)))
    if not normalized_thresholds or normalized_thresholds[0] < 1:
        raise ValueError("token audit thresholds must be positive")
    size_cache: dict[tuple[str, str], tuple[int, int]] = {}
    delta_cache: dict[tuple[tuple[int, int], ...], int] = {}
    lengths_by_split: dict[str, list[int]] = {}
    supervised_by_split: dict[str, list[int]] = {}
    longest: list[dict[str, Any]] = []
    shape_counts: Counter[str] = Counter()
    total = 0

    for split, rows_iterable in split_episodes.items():
        rows = list(rows_iterable)
        lengths: list[int] = []
        supervised_counts: list[int] = []
        for row in rows:
            episode_id = str(row.get("episode_id") or "")
            if str(row.get("split")) != split:
                raise ValueError(f"SPLIT_MISMATCH:{episode_id}")
            profile.validate(row)
            messages = profile.render_messages(row)
            qwen_multimodal.validate_image_placeholder_gate(list(messages), len(row["images"]))
            text_ids, assistant_mask = qwen_multimodal.assistant_mask(processor, list(messages))
            supervised = int(sum(bool(value) for value in assistant_mask))
            if supervised <= 0:
                raise ValueError(f"EMPTY_SUPERVISION:{episode_id}")
            shape_key_parts: list[tuple[int, int]] = []
            for image in row["images"]:
                source = str(image["image_source"])
                relative = str(image["path"])
                cache_key = (source, relative)
                if cache_key not in size_cache:
                    size_cache[cache_key] = _image_size(image_roots, source, relative)
                shape_key_parts.append(size_cache[cache_key])
            shape_key = tuple(shape_key_parts)
            if shape_key not in delta_cache:
                delta_cache[shape_key] = _visual_delta(
                    processor=processor,
                    registry=image_roots,
                    episode=row,
                    messages=messages,
                    text_length=len(text_ids),
                )
            length = len(text_ids) + delta_cache[shape_key]
            lengths.append(length)
            supervised_counts.append(supervised)
            shape_counts["+".join(f"{width}x{height}" for width, height in shape_key)] += 1
            longest.append({"episode_id": episode_id, "split": split, "length": length, "supervised_tokens": supervised})
            total += 1
            if progress and progress_every > 0 and total % progress_every == 0:
                progress(f"token audit checked {total} episodes")
        lengths_by_split[split] = lengths
        supervised_by_split[split] = supervised_counts

    all_lengths = [value for values in lengths_by_split.values() for value in values]
    all_supervised = [value for values in supervised_by_split.values() for value in values]
    if not all_lengths:
        raise ValueError("token audit has no episodes")
    selected = next((value for value in normalized_thresholds if max(all_lengths) <= value), None)
    return {
        "status": "PASS",
        "algorithm": "text_tokens_plus_shape_cached_visual_delta_v1",
        "untruncated": True,
        "thresholds": list(normalized_thresholds),
        "recommended_max_seq_length": selected,
        "overall": _summary(all_lengths, normalized_thresholds),
        "splits": {
            split: {
                **_summary(lengths_by_split[split], normalized_thresholds),
                "supervised_tokens": _summary(supervised_by_split[split], ()),
            }
            for split in lengths_by_split
        },
        "supervised_tokens": _summary(all_supervised, ()),
        "empty_supervision": 0,
        "encoding_errors": 0,
        "unique_image_files": len(size_cache),
        "image_shape_episode_counts": dict(sorted(shape_counts.items())),
        "visual_token_delta_by_shape": {
            "+".join(f"{width}x{height}" for width, height in shape): delta
            for shape, delta in sorted(delta_cache.items())
        },
        "longest_episodes": sorted(longest, key=lambda item: (-item["length"], item["episode_id"]))[:20],
    }


__all__ = ["audit_change_agent_tokens"]
