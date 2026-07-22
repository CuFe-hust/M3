"""Deterministic summaries for persisted evaluation records.
对持久化评估记录进行确定性汇总。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from spacers_agent.evaluation import EvaluationRecord


class EvaluationSummary(BaseModel):
    """Aggregate benchmark and optional judge metrics without mixing their meanings.
    汇总基准指标和可选评估器指标，但不混淆二者含义。
    """

    model_config = ConfigDict(extra="forbid")

    samples: int = Field(ge=0)
    samples_with_ground_truth: int = Field(ge=0)
    exact_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_absolute_error: float | None = Field(default=None, ge=0.0)
    mean_relative_error: float | None = Field(default=None, ge=0.0)
    judge_succeeded: int = Field(ge=0)
    judge_failed: int = Field(ge=0)
    judge_inconsistencies: int = Field(ge=0)
    mean_semantic_correctness: float | None = Field(default=None, ge=0.0, le=1.0)


def summarize_evaluations(records: Sequence[EvaluationRecord]) -> EvaluationSummary:
    """Summarize deterministic metrics separately from optional judge quality scores.
    将确定性指标与可选评估器质量分数分开汇总。
    """

    deterministic = [record.deterministic_metrics for record in records if record.deterministic_metrics is not None]
    judges = [record.judge_parsed for record in records if record.judge_parsed is not None]
    return EvaluationSummary(
        samples=len(records),
        samples_with_ground_truth=len(deterministic),
        exact_match_rate=(sum(metric.exact_match for metric in deterministic) / len(deterministic) if deterministic else None),
        mean_absolute_error=(sum(metric.absolute_error for metric in deterministic) / len(deterministic) if deterministic else None),
        mean_relative_error=(sum(metric.relative_error for metric in deterministic) / len(deterministic) if deterministic else None),
        judge_succeeded=sum(record.judge_status == "succeeded" for record in records),
        judge_failed=sum(record.judge_status == "failed" for record in records),
        judge_inconsistencies=sum(record.judge_inconsistency for record in records),
        mean_semantic_correctness=(sum(judge.semantic_correctness for judge in judges) / len(judges) if judges else None),
    )
