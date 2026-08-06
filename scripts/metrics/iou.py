"""Axis-aligned bounding-box IoU metrics self-implemented.

Used for VRSBench / XLRS-Bench visual grounding: Acc@0.5, Acc@0.7 and
mean IoU over unique / non-unique / all slices. Boxes are [x1, y1, x2, y2]
in any fixed coordinate system; ratios are scale-invariant.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _best_iou(prediction: Sequence[float], references: Sequence[Sequence[float]]) -> float:
    return max((_iou(prediction, reference) for reference in references), default=0.0)


def slice_metrics(
    predictions: Sequence[tuple[str, Sequence[float]]],
    references: Mapping[str, Sequence[Sequence[float]]],
    slices: Mapping[str, set[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Return {slice: {acc_0_5, acc_0_7, mean_iou}} for unique/non_unique/all.

    predictions: [(sample_id, box)]; references: {sample_id: [box, ...]}.
    slices: {slice_name: set(sample_ids)}; missing samples default to all.
    """
    if slices is None:
        slices = {"all": set(references)}
    results: dict[str, dict[str, float]] = {}
    for slice_name, sample_ids in slices.items():
        correct_05 = 0
        correct_07 = 0
        total_iou = 0.0
        count = 0
        for sample_id, box in predictions:
            if sample_id not in sample_ids:
                continue
            reference_boxes = references.get(sample_id)
            if not reference_boxes:
                continue
            iou = _best_iou(box, reference_boxes)
            if iou >= 0.5:
                correct_05 += 1
            if iou >= 0.7:
                correct_07 += 1
            total_iou += iou
            count += 1
        if count == 0:
            results[slice_name] = {"acc_0_5": 0.0, "acc_0_7": 0.0, "mean_iou": 0.0}
        else:
            results[slice_name] = {
                "acc_0_5": correct_05 / count,
                "acc_0_7": correct_07 / count,
                "mean_iou": total_iou / count,
            }
    return results
