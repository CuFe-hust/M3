"""Deterministic and opt-in DeepSeek proxy metrics.
确定性指标与可选的 DeepSeek 代理指标。
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from data.schema import CanonicalPrediction, CanonicalSample


def evaluate_records(
    records: Iterable[dict[str, Any]], use_deepseek: bool = False, deepseek_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate serialized canonical records without changing official references.
    在不修改官方参考答案的前提下评测序列化统一记录。
    """

    loaded = list(records)
    task_types = {record["sample"]["task_type"] for record in loaded}
    if len(task_types) != 1:
        raise ValueError("One result file must contain one task type.")
    task_type = task_types.pop()
    if task_type == "grounding":
        return _grounding_metrics(loaded)
    if task_type in {"caption", "change_caption"}:
        return _caption_metrics(loaded)
    metrics = _exact_match_metrics(loaded)
    if use_deepseek:
        metrics["deepseek_proxy"] = _deepseek_semantic_metrics(loaded, deepseek_config or {})
    return metrics


def _exact_match_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    matches = 0
    for record in records:
        prediction = _normalize_text(record["prediction"].get("answer") or record["prediction"]["text"])
        answers = {_normalize_text(answer) for answer in record["sample"].get("answers", [])}
        matches += int(prediction in answers)
    return {"metric": "exact_match_accuracy", "correct": matches, "total": len(records), "score": _ratio(matches, len(records))}


def _caption_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as error:
        raise RuntimeError("Install pycocoevalcap to compute caption metrics.") from error

    references = {
        record["sample"]["id"]: [
            _caption_metric_text(answer) for answer in record["sample"]["answers"]
        ]
        for record in records
    }
    candidates = {
        record["sample"]["id"]: [_caption_metric_text(record["prediction"]["text"])]
        for record in records
    }
    results: dict[str, Any] = {"total": len(records)}
    bleu, _ = Bleu(4).compute_score(references, candidates)
    for index, score in enumerate(bleu, start=1):
        results[f"BLEU_{index}"] = score
    for name, scorer in (("METEOR", Meteor()), ("ROUGE_L", Rouge()), ("CIDEr", Cider())):
        score, _ = scorer.compute_score(references, candidates)
        results[name] = score
    return results


def _grounding_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    ious = []
    for record in records:
        predicted = record["prediction"].get("boxes", [])
        expected = record["sample"].get("boxes", [])
        if predicted and expected:
            ious.append(_box_iou(predicted[0], expected[0]))
        else:
            ious.append(0.0)
    success = sum(iou >= 0.5 for iou in ious)
    return {
        "metric": "axis_aligned_iou_at_0_5",
        "total": len(ious),
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "accuracy": _ratio(success, len(ious)),
        "official_note": "Use the upstream VRSBench or XLRS-Bench evaluator for official oriented-box metrics.",
    }


def _deepseek_semantic_metrics(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY before requesting the DeepSeek proxy metric.")
    scores = []
    failures = []
    for record in records:
        try:
            scores.append(_deepseek_score(record, api_key, config))
        except (HTTPError, URLError, ValueError) as error:
            failures.append({"id": record["sample"]["id"], "error": str(error)})
    return {
        "metric": "deepseek_semantic_match_proxy",
        "score": sum(scores) / len(scores) if scores else 0.0,
        "evaluated": len(scores),
        "failed": failures,
        "comparability_note": "This is a DeepSeek proxy metric, not VRSBench's official GPT evaluation result.",
    }


def _deepseek_score(record: dict[str, Any], api_key: str, config: dict[str, Any]) -> float:
    sample = record["sample"]
    prediction = record["prediction"]
    payload = {
        "model": config.get("model", "deepseek-chat"),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "Judge whether a candidate answer is semantically correct for a visual question. Return only JSON: {\"score\": 0 or 1}.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": sample.get("meta", {}).get("question", sample["prompt"]),
                        "reference_answers": sample.get("answers", []),
                        "candidate_answer": prediction.get("answer") or prediction["text"],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = Request(
        config.get("base_url", "https://api.deepseek.com/v1").rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            result = json.loads(content)
            return float(result["score"])
        except HTTPError as error:
            if error.code < 500 or attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("DeepSeek request retry loop ended unexpectedly.")


def _box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denominator = first_area + second_area - intersection
    return intersection / denominator if denominator else 0.0


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().strip(".,;:!"))


def _caption_metric_text(value: str) -> str:
    """Make caption text safe for line-oriented metric subprocess protocols.
    将描述文本转换为适合逐行指标子进程协议的安全格式。
    """

    return re.sub(r"\s+", " ", str(value).replace("|||", " ")).strip()


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
