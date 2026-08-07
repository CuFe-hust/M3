"""Unified deterministic evaluation records.

统一确定性评估记录。纯类型契约：包含确定性指标字段与可选 judge 状态
（judge 网络调用保持在 evaluation/judges 包，本模块不导入也不发起任何
网络请求）。Judge 结果永远不能覆盖 deterministic 字段。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CountDeterministicMetrics(BaseModel):
    """Deterministic metrics for a counting sample with a known count.
    对具有已知计数真值样本的确定性指标。"""

    model_config = ConfigDict(extra="forbid")

    predicted_count: int = Field(ge=0)
    gold_count: int = Field(ge=0)
    exact_match: int = Field(ge=0, le=1)
    absolute_error: int = Field(ge=0)
    relative_error: float = Field(ge=0.0)
    smooth_error_score: float = Field(ge=0.0, le=1.0)


class VQAEvaluationRecord(BaseModel):
    """One deterministic VQA comparison plus optional text-only judge result.
    一条确定性 VQA 对比及可选的纯文本审核结果。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    question: str
    reference_answers: list[str]
    candidate_answer: str
    exact_match: bool
    judge_status: Literal["not_requested", "succeeded", "failed"]
    judge_score: int | None = Field(default=None, ge=0, le=1)
    # Judge payloads are typed by evaluation/judges; records stay decoupled.
    # judge 载荷类型由 evaluation/judges 提供；记录保持解耦。
    judge_parsed: Any | None = None
    judge_error: str | None = None


class EvaluationRecord(BaseModel):
    """One merged deterministic and optional text-only judge evaluation record.
    一条合并确定性指标与可选仅文本审核结果的评估记录。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task: str
    deterministic_metrics: CountDeterministicMetrics | None = None
    judge_status: Literal["not_requested", "succeeded", "failed"]
    judge_raw: str | None = None
    judge_parsed: Any | None = None
    judge_inconsistency: bool = False
    judge_error: str | None = None
