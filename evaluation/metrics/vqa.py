"""Deterministic VQA normalization, exact match, and judge merging.

确定性 VQA 归一化、严格匹配与 judge 合并。纯文本本地计算，无网络副作用；
judge 分数只作为旁路记录，绝不覆盖 exact_match。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evaluation.records import (
    EvaluationRecord,
    VQADeterministicMetrics,
    VQAEvaluationRecord,
)


def normalize_answer(value: str) -> str:
    """Normalize surrounding punctuation and whitespace for exact VQA
    comparison. 规范首尾标点与空白，用于 VQA 严格对比。"""

    return " ".join(str(value).strip().lower().strip(".,;:!").split())


def exact_match(candidate: str, references: Sequence[str]) -> bool:
    """Return whether the normalized candidate equals any normalized
    reference. 归一化后候选答案是否等于任一参考答案。"""

    normalized_candidate = normalize_answer(candidate)
    return normalized_candidate in {normalize_answer(answer) for answer in references}


def merge_vqa_evaluation(
    *,
    sample_id: str,
    question: str,
    reference_answers: list[str],
    candidate_answer: str,
    judge_parsed: Any | None = None,
    judge_error: str | None = None,
) -> VQAEvaluationRecord:
    """Preserve exact comparison and map a successful judge verdict to a
    binary score; judge output never replaces the deterministic match.
    保留严格对比并将成功 judge 结论映射为二值分数；judge 输出绝不替换
    确定性匹配结果。"""

    exact = exact_match(candidate_answer, reference_answers)
    if judge_error is not None:
        return VQAEvaluationRecord(
            sample_id=sample_id,
            question=question,
            reference_answers=reference_answers,
            candidate_answer=candidate_answer,
            exact_match=exact,
            judge_status="failed",
            judge_error=judge_error,
        )
    if judge_parsed is None:
        return VQAEvaluationRecord(
            sample_id=sample_id,
            question=question,
            reference_answers=reference_answers,
            candidate_answer=candidate_answer,
            exact_match=exact,
            judge_status="not_requested",
        )
    return VQAEvaluationRecord(
        sample_id=sample_id,
        question=question,
        reference_answers=reference_answers,
        candidate_answer=candidate_answer,
        exact_match=exact,
        judge_status="succeeded",
        judge_score=int(getattr(judge_parsed, "score", 0)),
        judge_parsed=judge_parsed,
    )


def vqa_deterministic_metrics(exact: bool) -> VQADeterministicMetrics:
    """Wrap one deterministic match result for the unified record.
    将单条确定性匹配结果包装进统一记录。"""

    return VQADeterministicMetrics(exact_match=exact)


def aggregate_vqa(
    records: Sequence[VQAEvaluationRecord | EvaluationRecord],
) -> dict[str, Any]:
    """Aggregate deterministic exact-match accuracy across records; both the
    compatibility wrapper and unified general_vqa records are accepted.
    跨记录汇总确定性严格匹配准确率；兼容包装与统一 general_vqa 记录均被
    接受。"""

    total = len(records)
    correct = sum(int(_exact_of(record)) for record in records)
    return {
        "metric": "exact_match_accuracy",
        "correct": correct,
        "total": total,
        "score": correct / total if total else 0.0,
    }


def _exact_of(record: VQAEvaluationRecord | EvaluationRecord) -> bool:
    if isinstance(record, VQAEvaluationRecord):
        return record.exact_match
    metrics = record.deterministic_metrics
    return bool(getattr(metrics, "exact_match", False))
