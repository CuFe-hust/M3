"""Contract tests for deterministic VQA metrics and judge merging.

确定性 VQA 指标与 judge 合并契约测试：归一化、严格匹配、judge 分数不覆盖
exact_match、跨记录汇总、无网络副作用。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.metrics.vqa import (
    aggregate_vqa,
    exact_match,
    merge_vqa_evaluation,
    normalize_answer,
)
from evaluation.records import VQAEvaluationRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_normalize_answer_lowercase_punctuation_whitespace() -> None:
    assert normalize_answer("  Yes! ") == "yes"
    assert normalize_answer("A, B.") == "a, b"
    assert normalize_answer("multi\nline") == "multi line"
    assert normalize_answer("  ") == ""


def test_exact_match_ignores_case_punctuation() -> None:
    assert exact_match("YES!", ["yes"])
    assert exact_match("the car", ["The car."])
    assert not exact_match("two cars", ["one car"])
    assert exact_match("car", ["truck", "car"])


def test_merge_vqa_not_requested() -> None:
    record = merge_vqa_evaluation(
        sample_id="s1",
        question="Q",
        reference_answers=["yes"],
        candidate_answer="yes",
    )
    assert record.exact_match is True
    assert record.judge_status == "not_requested"
    assert record.judge_score is None


def test_judge_score_never_overrides_exact_match() -> None:
    """A judge score of 1 cannot flip the deterministic mismatch.
    即使 judge 给 1 分也不能翻转确定性的不匹配结果。"""

    class _Judge:
        score = 1

    record = merge_vqa_evaluation(
        sample_id="s1",
        question="Q",
        reference_answers=["yes"],
        candidate_answer="no",
        judge_parsed=_Judge(),
    )
    assert record.exact_match is False
    assert record.judge_status == "succeeded"
    assert record.judge_score == 1
    assert record.judge_parsed is not None


def test_merge_vqa_judge_error() -> None:
    record = merge_vqa_evaluation(
        sample_id="s1",
        question="Q",
        reference_answers=["yes"],
        candidate_answer="yes",
        judge_error="timeout",
    )
    assert record.exact_match is True
    assert record.judge_status == "failed"
    assert record.judge_error == "timeout"


def test_aggregate_vqa() -> None:
    records = [
        merge_vqa_evaluation(
            sample_id="s1", question="Q", reference_answers=["yes"], candidate_answer="yes"
        ),
        merge_vqa_evaluation(
            sample_id="s2", question="Q", reference_answers=["yes"], candidate_answer="no"
        ),
    ]
    summary = aggregate_vqa(records)
    assert summary == {
        "metric": "exact_match_accuracy",
        "correct": 1,
        "total": 2,
        "score": 0.5,
    }


def test_aggregate_vqa_empty() -> None:
    summary = aggregate_vqa([])
    assert summary["total"] == 0
    assert summary["score"] == 0.0


def test_record_serialization_is_stable() -> None:
    record = merge_vqa_evaluation(
        sample_id="s1", question="Q", reference_answers=["yes"], candidate_answer="yes"
    )
    payload = record.model_dump(mode="json")
    assert payload["exact_match"] is True
    assert payload["judge_status"] == "not_requested"
    assert set(payload) == {
        "sample_id",
        "question",
        "reference_answers",
        "candidate_answer",
        "exact_match",
        "judge_status",
        "judge_score",
        "judge_parsed",
        "judge_error",
    }


def test_vqa_metrics_have_no_network_side_effects() -> None:
    for module in ("evaluation/metrics/vqa.py",):
        source = (REPO_ROOT / module).read_text(encoding="utf-8")
        for token in ("urlopen", "requests", "socket", "httpx", "api.deepseek", "http://", "https://"):
            assert token not in source, (module, token)
