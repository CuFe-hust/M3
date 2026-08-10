"""Contract tests for deterministic VQA metrics and judge merging.

确定性 VQA 指标与 judge 合并契约测试：归一化、严格匹配、judge 分数不覆盖
exact_match、跨记录汇总、无网络副作用。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.metrics.vqa import (
    aggregate_vqa,
    aggregate_vqa_semantic_judge,
    exact_match,
    merge_vqa_evaluation,
    normalize_answer,
)
from evaluation.records import (
    EvaluationRecord,
    VQADeterministicMetrics,
    VQAEvaluationRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _semantic_record(
    sample_id: str,
    *,
    exact: bool,
    judge_status: str = "not_requested",
    judge_parsed: object | None = None,
    judge_error: str | None = None,
) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=exact),
        judge_status=judge_status,  # type: ignore[arg-type]
        judge_parsed=judge_parsed,
        judge_error=judge_error,
    )


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
    assert isinstance(record, EvaluationRecord)
    assert record.task == "general_vqa"
    assert isinstance(record.deterministic_metrics, VQADeterministicMetrics)
    assert record.deterministic_metrics.exact_match is True
    assert record.judge_status == "not_requested"


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
    assert isinstance(record, EvaluationRecord)
    assert record.deterministic_metrics.exact_match is False
    assert record.judge_status == "succeeded"
    assert record.judge_parsed is not None
    assert getattr(record.judge_parsed, "score", None) == 1


def test_merge_vqa_judge_error() -> None:
    record = merge_vqa_evaluation(
        sample_id="s1",
        question="Q",
        reference_answers=["yes"],
        candidate_answer="yes",
        judge_error="timeout",
    )
    assert record.deterministic_metrics.exact_match is True
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


def test_semantic_judge_aggregate_all_exact_is_complete() -> None:
    summary = aggregate_vqa_semantic_judge(
        [_semantic_record("s1", exact=True), _semantic_record("s2", exact=True)]
    )
    assert summary == {
        "total": 2,
        "deterministic_exact_correct": 2,
        "eligible_mismatches": 0,
        "judged_mismatches": 0,
        "semantic_equivalent_mismatches": 0,
        "semantic_non_equivalent_mismatches": 0,
        "judge_failures": 0,
        "unresolved_mismatches": 0,
        "coverage": 1.0,
        "corrected_correct": 2,
        "lower_bound_score": 1.0,
        "complete": True,
        "score": 1.0,
    }


def test_semantic_judge_complete_accepts_dict_and_object() -> None:
    class _Score:
        score = 0

    records = [
        _semantic_record(
            "s1", exact=False, judge_status="succeeded", judge_parsed={"score": 1}
        ),
        _semantic_record(
            "s2", exact=False, judge_status="succeeded", judge_parsed=_Score()
        ),
    ]
    summary = aggregate_vqa_semantic_judge(records)
    assert summary["judged_mismatches"] == 2
    assert summary["semantic_equivalent_mismatches"] == 1
    assert summary["semantic_non_equivalent_mismatches"] == 1
    assert summary["coverage"] == 1.0
    assert summary["corrected_correct"] == 1
    assert summary["complete"] is True
    assert summary["score"] == 0.5


def test_semantic_judge_aggregate_partial_is_confirmed_lower_bound_only() -> None:
    records = [
        _semantic_record("exact", exact=True),
        _semantic_record(
            "equivalent",
            exact=False,
            judge_status="succeeded",
            judge_parsed={"score": 1},
        ),
        _semantic_record("unresolved", exact=False),
    ]
    summary = aggregate_vqa_semantic_judge(records)
    assert summary["eligible_mismatches"] == 2
    assert summary["judged_mismatches"] == 1
    assert summary["unresolved_mismatches"] == 1
    assert summary["coverage"] == 0.5
    assert summary["corrected_correct"] == 2
    assert summary["lower_bound_score"] == pytest.approx(2 / 3)
    assert summary["complete"] is False
    assert summary["score"] is None
    assert aggregate_vqa(records) == {
        "metric": "exact_match_accuracy",
        "correct": 1,
        "total": 3,
        "score": pytest.approx(1 / 3),
    }


def test_semantic_judge_failure_and_invalid_score_stay_unresolved() -> None:
    records = [
        _semantic_record(
            "failed", exact=False, judge_status="failed", judge_error="RuntimeError"
        ),
        _semantic_record(
            "invalid",
            exact=False,
            judge_status="succeeded",
            judge_parsed={"score": True},
        ),
        _semantic_record(
            "string",
            exact=False,
            judge_status="succeeded",
            judge_parsed={"score": "1"},
        ),
    ]
    summary = aggregate_vqa_semantic_judge(records)
    assert summary["judge_failures"] == 1
    assert summary["judged_mismatches"] == 0
    assert summary["unresolved_mismatches"] == 3
    assert summary["coverage"] == 0.0
    assert summary["complete"] is False
    assert summary["score"] is None


def test_record_serialization_is_stable() -> None:
    record = merge_vqa_evaluation(
        sample_id="s1", question="Q", reference_answers=["yes"], candidate_answer="yes"
    )
    payload = record.model_dump(mode="json")
    assert payload["task"] == "general_vqa"
    assert payload["deterministic_metrics"] == {"exact_match": True}
    assert payload["judge_status"] == "not_requested"
    assert set(payload) == {
        "sample_id",
        "task",
        "deterministic_metrics",
        "judge_status",
        "judge_raw",
        "judge_parsed",
        "judge_inconsistency",
        "judge_error",
    }


def test_vqa_metrics_have_no_network_side_effects() -> None:
    for module in ("evaluation/metrics/vqa.py",):
        source = (REPO_ROOT / module).read_text(encoding="utf-8")
        for token in ("urlopen", "requests", "socket", "httpx", "api.deepseek", "http://", "https://"):
            assert token not in source, (module, token)


# ── 统一记录不变式 / unified record invariants (33.6) ──────────────────────


def test_vqa_merge_returns_unified_record() -> None:
    record = merge_vqa_evaluation(
        sample_id="s1", question="Q", reference_answers=["yes"], candidate_answer="yes"
    )
    assert isinstance(record, EvaluationRecord)
    assert record.task == "general_vqa"
    assert isinstance(record.deterministic_metrics, VQADeterministicMetrics)


def test_legacy_wrapper_conversion_is_explicit() -> None:
    from evaluation.metrics.vqa import to_evaluation_record
    from evaluation.records import VQAEvaluationRecord

    legacy = VQAEvaluationRecord(
        sample_id="s1",
        question="Q",
        reference_answers=["yes"],
        candidate_answer="no",
        exact_match=False,
        judge_status="not_requested",
    )
    record = to_evaluation_record(legacy)
    assert isinstance(record, EvaluationRecord)
    assert record.task == "general_vqa"
    assert record.deterministic_metrics.exact_match is False


def test_aggregate_vqa_accepts_legacy_wrapper_explicitly() -> None:
    from evaluation.records import VQAEvaluationRecord

    records = [
        VQAEvaluationRecord(
            sample_id="s1",
            question="Q",
            reference_answers=["yes"],
            candidate_answer="yes",
            exact_match=True,
            judge_status="not_requested",
        )
    ]
    summary = aggregate_vqa(records)
    assert summary["correct"] == 1
