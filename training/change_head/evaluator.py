"""Small dependency-light metrics for ChangeHead validation."""

from __future__ import annotations

from typing import Any


def evaluate_probability_maps(
    probabilities: list[Any],
    targets: list[Any],
    valid_masks: list[Any],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    import numpy as np

    tp = fp = fn = tn = 0
    for probability, target, valid in zip(probabilities, targets, valid_masks):
        mask = np.asarray(valid, dtype=bool)
        predicted = np.asarray(probability) >= threshold
        actual = np.asarray(target) != 0
        tp += int(np.count_nonzero(predicted & actual & mask))
        fp += int(np.count_nonzero(predicted & ~actual & mask))
        fn += int(np.count_nonzero(~predicted & actual & mask))
        tn += int(np.count_nonzero(~predicted & ~actual & mask))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": tp / max(1, tp + fp + fn),
        "scene_nochange_fp_rate": float(fp > 0 and tp == 0),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_negative": float(tn),
    }

