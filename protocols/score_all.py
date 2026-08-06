"""Official scorer bridge: requests + references + predictions -> scorer JSON.

Invoked by the m3rs-eval framework as ``official_scorer_command`` with the
exact placeholders ``{requests_jsonl}`` ``{references_jsonl}``
``{predictions_jsonl}`` ``{output_json}``. Computes every official-lane
metric with the self-implemented metrics package (no M3 / pycocoevalcap
code). Task/language/variant classification comes from the requests file
(the adapter-emitted references only carry answers and boxes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from metrics import bleu, cider, iou as iou_metrics, meteor, rouge_l

SCORER_VERSION = "m3rs-self-2026-08-05"

_GROUNDING_PROMPT_SUFFIX = (
    " Return only the bounding box as [x1, y1, x2, y2] with coordinates "
    "normalized from 0 to 100."
)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number}: not an object")
            rows.append(row)
    return rows


def _sample_id(row: Mapping[str, Any]) -> str:
    value = row.get("sample_id") or row.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row missing sample_id/id: {row}")
    return value


def _reference_text(row: Mapping[str, Any]) -> str:
    value = row.get("answer") or row.get("caption")
    if not isinstance(value, str):
        raise ValueError(f"reference row missing answer: {row}")
    return value


def _valid_predictions(predictions: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in predictions:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            continue
        if row.get("status") != "ok":
            continue
        by_id[sample_id] = row
    return by_id


def _scorer_metric(
    metric_id: str, value: float, n_samples: int, n_failures: int
) -> dict[str, Any]:
    n_failures = min(n_failures, n_samples)
    return {
        "metric_id": metric_id,
        "value_canonical": value,
        "n_samples": n_samples,
        "n_failures": n_failures,
    }


def _na_metric(metric_id: str, n_samples: int, n_failures: int) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "value_canonical": None,
        "n_samples": n_samples,
        "n_failures": min(n_failures, n_samples),
        "availability": "not_applicable",
    }


def score_levir_cc(
    requests: list[Mapping[str, Any]],
    references: list[Mapping[str, Any]],
    predictions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failures = sum(1 for row in references if _sample_id(row) not in predictions)
    valid = [row for row in references if _sample_id(row) in predictions]
    output: list[dict[str, Any]] = []
    slices = {
        "all": [row for row in valid],
        "change": [row for row in valid if row.get("change") == "change"],
        "no_change": [row for row in valid if row.get("change") == "no-change"],
    }
    for scope, rows in slices.items():
        if not rows:
            continue
        candidates = [predictions[_sample_id(row)].get("prediction", "") for row in rows]
        references_batch = [[_reference_text(row)] for row in rows]
        n = len(rows)
        bleu_scores = bleu.corpus_bleu(candidates, references_batch)
        for n_gram in (1, 2, 3, 4):
            output.append(_scorer_metric(f"levir.caption.{scope}.bleu_{n_gram}", bleu_scores[n_gram], n, failures))
        output.append(_scorer_metric(f"levir.caption.{scope}.meteor", meteor.corpus_meteor(candidates, references_batch), n, failures))
        output.append(_scorer_metric(f"levir.caption.{scope}.rouge_l", rouge_l.corpus_rouge_l(candidates, references_batch), n, failures))
        if scope == "no_change":
            # registry declares no-change CIDEr-D not applicable (N/A, never 0)
            output.append(_na_metric(f"levir.caption.{scope}.cider_d", n, failures))
        else:
            output.append(_scorer_metric(f"levir.caption.{scope}.cider_d", cider.corpus_cider(candidates, references_batch, use_cider_d=True), n, failures))
    return output


def score_vrsbench(
    requests: list[Mapping[str, Any]],
    references: list[Mapping[str, Any]],
    predictions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ref_by_id = {_sample_id(row): row for row in references}
    failures = sum(1 for row in references if _sample_id(row) not in predictions)
    caption_ids = {
        req["sample_id"] for req in requests if req.get("expected_output") == "caption"
    }
    rows = [
        ref_by_id[sample_id] for sample_id in sorted(caption_ids)
        if sample_id in ref_by_id and sample_id in predictions
    ]
    if not rows:
        return []
    candidates = [predictions[_sample_id(row)].get("prediction", "") for row in rows]
    references_batch = [[_reference_text(row)] for row in rows]
    n = len(rows)
    output: list[dict[str, Any]] = []
    bleu_scores = bleu.corpus_bleu(candidates, references_batch)
    for n_gram in (1, 2, 3, 4):
        output.append(_scorer_metric(f"vrs.caption.bleu_{n_gram}", bleu_scores[n_gram], n, failures))
    output.append(_scorer_metric("vrs.caption.meteor", meteor.corpus_meteor(candidates, references_batch), n, failures))
    output.append(_scorer_metric("vrs.caption.rouge_l", rouge_l.corpus_rouge_l(candidates, references_batch), n, failures))
    output.append(_scorer_metric("vrs.caption.cider", cider.corpus_cider(candidates, references_batch, use_cider_d=False), n, failures))
    return output


def score_xlrs_bench(
    requests: list[Mapping[str, Any]],
    references: list[Mapping[str, Any]],
    predictions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ref_by_id = {_sample_id(row): row for row in references}
    failures = sum(1 for row in references if _sample_id(row) not in predictions)
    output: list[dict[str, Any]] = []
    for language in ("en", "zh"):
        caption_ids = {
            req["sample_id"] for req in requests
            if req.get("task") == "caption"
            and req.get("language") == language
            and req.get("variant") == "full"
        }
        caption_rows = [
            ref_by_id[sample_id] for sample_id in sorted(caption_ids)
            if sample_id in ref_by_id and sample_id in predictions
        ]
        if caption_rows:
            candidates = [predictions[_sample_id(row)].get("prediction", "") for row in caption_rows]
            references_batch = [[_reference_text(row)] for row in caption_rows]
            n = len(caption_rows)
            bleu_scores = bleu.corpus_bleu(candidates, references_batch)
            for n_gram in (1, 2, 3, 4):
                output.append(_scorer_metric(f"xlrs.caption.{language}.bleu_{n_gram}", bleu_scores[n_gram], n, failures))
            output.append(_scorer_metric(f"xlrs.caption.{language}.meteor", meteor.corpus_meteor(candidates, references_batch), n, failures))
            output.append(_scorer_metric(f"xlrs.caption.{language}.rouge_l", rouge_l.corpus_rouge_l(candidates, references_batch), n, failures))
        grounding_ids = {
            req["sample_id"] for req in requests
            if req.get("task") == "grounding"
            and req.get("language") == language
            and req.get("variant") == "full"
        }
        grounding_rows = [
            ref_by_id[sample_id] for sample_id in sorted(grounding_ids)
            if sample_id in ref_by_id and sample_id in predictions
        ]
        if grounding_rows:
            reference_boxes = {_sample_id(row): row.get("boxes") or [] for row in grounding_rows}
            prediction_boxes: list[tuple[str, list[float]]] = []
            for row in grounding_rows:
                boxes_field = predictions[_sample_id(row)].get("boxes")
                if not boxes_field:
                    continue
                box = boxes_field[0] if isinstance(boxes_field[0], list) else boxes_field
                prediction_boxes.append((_sample_id(row), list(box)))
            results = iou_metrics.slice_metrics(
                prediction_boxes, reference_boxes, {"all": set(reference_boxes)}
            )
            n = len(grounding_rows)
            output.append(_scorer_metric("xlrs.grounding.%s.all.acc_0_5" % language, results["all"]["acc_0_5"], n, failures))
            output.append(_scorer_metric("xlrs.grounding.%s.all.acc_0_7" % language, results["all"]["acc_0_7"], n, failures))
    return output


_DISPATCH = {
    "levir_cc": score_levir_cc,
    "vrsbench": score_vrsbench,
    "xlrs_bench": score_xlrs_bench,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="score_all")
    parser.add_argument("--dataset", required=True, choices=sorted(_DISPATCH))
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    requests = _read_jsonl_rows(args.requests)
    references = _read_jsonl_rows(args.references)
    predictions = _valid_predictions(_read_jsonl_rows(args.predictions))
    metrics = _DISPATCH[args.dataset](requests, references, predictions)
    payload = {"scorer_version": SCORER_VERSION, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
