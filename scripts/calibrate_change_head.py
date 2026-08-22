"""Validation-only temperature and rescue-threshold calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from training.change_head.evaluator import evaluate_probability_maps


def _samples(value: Any) -> list[np.ndarray]:
    raw = np.asarray(value)
    array = np.asarray(value, dtype=object) if raw.dtype == object else raw
    if array.dtype == object:
        return [np.asarray(item) for item in array.tolist()]
    if array.ndim >= 3:
        return [np.asarray(item) for item in array]
    return [np.asarray(array)]


def _flatten_valid(
    logits: Any,
    targets: Any,
    valid: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    logit_samples = _samples(logits)
    target_samples = _samples(targets)
    if len(logit_samples) != len(target_samples):
        raise ValueError("logits and targets sample counts differ")
    valid_samples = (
        [np.ones_like(target, dtype=bool) for target in target_samples]
        if valid is None
        else _samples(valid)
    )
    if len(valid_samples) != len(target_samples):
        raise ValueError("valid mask sample count differs")
    kept_logits: list[np.ndarray] = []
    kept_targets: list[np.ndarray] = []
    for current_logits, target, mask in zip(logit_samples, target_samples, valid_samples):
        if current_logits.shape != target.shape or target.shape != mask.shape:
            raise ValueError("calibration array shapes differ")
        current_mask = np.asarray(mask, dtype=bool)
        kept_logits.append(np.asarray(current_logits, dtype=np.float64)[current_mask])
        kept_targets.append((np.asarray(target) != 0).astype(np.float64)[current_mask])
    if not kept_logits or not any(item.size for item in kept_logits):
        raise ValueError("calibration has no valid pixels")
    return np.concatenate(kept_logits), np.concatenate(kept_targets)


def _nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    scaled = logits / float(temperature)
    return float(np.mean(np.logaddexp(0.0, scaled) - targets * scaled))


def _ece(probability: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    total = max(1, probability.size)
    for index in range(bins):
        in_bin = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if not np.any(in_bin):
            continue
        result += float(np.count_nonzero(in_bin) / total) * abs(
            float(np.mean(probability[in_bin])) - float(np.mean(targets[in_bin]))
        )
    return result


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray | None = None,
) -> float:
    flat_logits, flat_targets = _flatten_valid(logits, targets, valid)
    candidates = np.linspace(0.25, 4.0, 76)
    losses = [_nll(flat_logits, flat_targets, float(value)) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def search_rescue_threshold(
    probabilities: Sequence[Any],
    targets: Sequence[Any],
    valid_masks: Sequence[Any],
    *,
    tags: Sequence[Sequence[str]] | None = None,
    candidates: Sequence[float] = tuple(np.linspace(0.50, 0.99, 50)),
    max_nochange_scene_fp_rate: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    tags = tags or [()] * len(probabilities)
    safe: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = evaluate_probability_maps(
            list(probabilities), list(targets), list(valid_masks), threshold=float(threshold)
        )
        hard_indices = [
            index for index, item in enumerate(tags) if "hard_case" in set(item)
        ]
        if hard_indices:
            hard_metrics = evaluate_probability_maps(
                [probabilities[index] for index in hard_indices],
                [targets[index] for index in hard_indices],
                [valid_masks[index] for index in hard_indices],
                threshold=float(threshold),
            )
        else:
            hard_metrics = {"pixel_f1": 0.0}
        if metrics["scene_nochange_fp_rate"] <= max_nochange_scene_fp_rate:
            safe.append((float(hard_metrics.get("pixel_f1", 0.0)), {**metrics, "hard_case_pixel_f1": hard_metrics.get("pixel_f1", 0.0), "threshold": float(threshold)}))
    if not safe:
        raise ValueError("CALIBRATION_NO_SAFE_RESCUE_THRESHOLD")
    safe.sort(key=lambda item: (item[0], item[1]["threshold"]), reverse=True)
    return safe[0][1]["threshold"], safe[0][1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--valid", type=Path, default=None)
    parser.add_argument("--tags", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--validation-fingerprint", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logits = np.load(args.logits, allow_pickle=True)
    targets = np.load(args.targets, allow_pickle=True)
    valid = None if args.valid is None else np.load(args.valid, allow_pickle=True)
    logit_samples = _samples(logits)
    target_samples = _samples(targets)
    valid_samples = (
        [np.ones_like(item, dtype=bool) for item in target_samples]
        if valid is None
        else _samples(valid)
    )
    temperature = fit_temperature(logits, targets, valid)
    flat_logits, flat_targets = _flatten_valid(logits, targets, valid)
    pre_probability = 1.0 / (1.0 + np.exp(-np.clip(flat_logits, -80.0, 80.0)))
    post_probability = 1.0 / (1.0 + np.exp(-np.clip(flat_logits / temperature, -80.0, 80.0)))
    tags = None
    if args.tags is not None:
        tags = json.loads(args.tags.read_text(encoding="utf-8"))
    calibrated_probability_samples = [
        1.0 / (1.0 + np.exp(-np.clip(sample / temperature, -80.0, 80.0)))
        for sample in logit_samples
    ]
    threshold, threshold_metrics = search_rescue_threshold(
        calibrated_probability_samples,
        [*target_samples],
        [*valid_samples],
        tags=tags,
    )
    metrics = {
        "pre_nll": _nll(flat_logits, flat_targets, 1.0),
        "post_nll": _nll(flat_logits, flat_targets, temperature),
        "pre_ece": _ece(pre_probability, flat_targets),
        "post_ece": _ece(post_probability, flat_targets),
        "brier": float(np.mean((post_probability - flat_targets) ** 2)),
        "threshold_scene_nochange_fp_rate": threshold_metrics["scene_nochange_fp_rate"],
        "threshold_hard_case_pixel_f1": threshold_metrics["hard_case_pixel_f1"],
    }
    reliability = float(np.clip(1.0 - metrics["post_ece"], 0.0, 1.0))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "temperature": temperature,
        "rescue_probability_threshold": threshold,
        "rescue_min_component_area_ratio": 0.0005,
        "validation_reliability": reliability,
        "optional_expert_missing_reliability_factor": 0.90,
        "metrics": metrics,
        "validation_fingerprint": args.validation_fingerprint,
    }
    if args.checkpoint is not None:
        payload["created_from_checkpoint_sha256"] = _sha256_file(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
