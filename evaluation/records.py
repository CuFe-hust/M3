"""Unified deterministic evaluation records.

统一确定性评估记录。纯类型契约：包含确定性指标字段与可选 judge 状态
（judge 网络调用保持在 evaluation/judges 包，本模块不导入也不发起任何
网络请求）。Judge 结果永远不能覆盖 deterministic 字段。
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

EvaluationTask = Literal["counting", "general_vqa", "grounding", "caption"]


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


class VQADeterministicMetrics(BaseModel):
    """Deterministic exact-match result for one general-VQA sample.
    单条通用 VQA 样本的确定性严格匹配结果。"""

    model_config = ConfigDict(extra="forbid")

    exact_match: bool


class GroundingDeterministicMetrics(BaseModel):
    """Deterministic axis-aligned IoU result for one grounding sample.
    单条接地样本的确定性轴对齐 IoU 结果。"""

    model_config = ConfigDict(extra="forbid")

    iou: float = Field(ge=0.0, le=1.0)
    iou_at_0_5: bool


class CaptionDeterministicMetrics(BaseModel):
    """Per-sample caption contract: candidate and references, from which the
    corpus-level BLEU/METEOR/ROUGE/CIDEr aggregate is computed.
    逐样本 caption 契约：候选与参考答案，语料级 BLEU/METEOR/ROUGE/CIDEr
    汇总由此计算。"""

    model_config = ConfigDict(extra="forbid")

    candidate: str
    references: list[str] = Field(min_length=1)


DeterministicMetrics = Union[
    CountDeterministicMetrics,
    VQADeterministicMetrics,
    GroundingDeterministicMetrics,
    CaptionDeterministicMetrics,
]


class VQAEvaluationRecord(BaseModel):
    """Compatibility wrapper for one deterministic VQA comparison plus
    optional text-only judge result. New code should use the unified
    EvaluationRecord with task='general_vqa'; the wrapper keeps existing
    VQA pipelines readable without a second record system.
    单条确定性 VQA 对比及可选纯文本审核结果的兼容包装。新代码应使用
    task='general_vqa' 的统一 EvaluationRecord；该包装保持既有 VQA 管线
    可读，不构成第二套公共记录体系。"""

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
    """One merged deterministic and optional text-only judge evaluation
    record for any implemented task. Judge output is recorded alongside the
    deterministic metrics and can never override them.
    任意已实现任务的单条合并确定性指标与可选仅文本审核结果记录。Judge
    输出与确定性指标并列记录，永远不能覆盖后者。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task: EvaluationTask
    deterministic_metrics: DeterministicMetrics | None = None
    judge_status: Literal["not_requested", "succeeded", "failed"]
    judge_raw: str | None = None
    judge_parsed: Any | None = None
    judge_inconsistency: bool = False
    judge_error: str | None = None
