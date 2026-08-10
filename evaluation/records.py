"""Unified deterministic evaluation records.

统一确定性评估记录。纯类型契约：包含确定性指标字段与可选 judge 状态
（judge 网络调用保持在 evaluation/judges 包，本模块不导入也不发起任何
网络请求）。Judge 结果永远不能覆盖 deterministic 字段。
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvaluationTask = Literal["counting", "general_vqa", "grounding", "caption"]

# Canonical production contract from runtime tasks to deterministic evaluation
# families and their artifact filenames. Keep both mappings immutable so every
# workflow, application command, and report reader observes the same dispatch.
# runtime task 到确定性评测族及其产物文件名的生产权威契约。两个映射均保持
# 不可变，使 workflow、应用命令和报告读取层共享同一分派。
RUNTIME_TASK_TO_EVALUATION_TASK: MappingProxyType[str, EvaluationTask] = (
    MappingProxyType(
        {
            "counting": "counting",
            "fine_grained_counting": "counting",
            "general_vqa": "general_vqa",
            "multiple_choice_vqa": "general_vqa",
            "scene_classification": "general_vqa",
            "spatial_relation": "general_vqa",
            "change_qa": "general_vqa",
            "grounding": "grounding",
            "caption": "caption",
            "change_caption": "caption",
        }
    )
)

EVALUATION_FILENAME_BY_TASK: MappingProxyType[EvaluationTask, str] = MappingProxyType(
    {
        "counting": "counting_evaluation.json",
        "general_vqa": "vqa_evaluation.json",
        "grounding": "grounding_evaluation.json",
        "caption": "caption_evaluation.json",
    }
)


def evaluation_task_for_runtime_task(task: str) -> EvaluationTask | None:
    """Return the canonical evaluation family, or None for an unsupported
    runtime task. 返回 canonical 评测族；不支持的 runtime task 返回 None。"""

    return RUNTIME_TASK_TO_EVALUATION_TASK.get(task)


def evaluation_filename_for_runtime_task(task: str) -> str | None:
    """Return the deterministic artifact filename for a runtime task without
    guessing unknown tasks. 返回 runtime task 的确定性评测产物名，未知任务
    绝不猜测。"""

    evaluation_task = evaluation_task_for_runtime_task(task)
    if evaluation_task is None:
        return None
    return EVALUATION_FILENAME_BY_TASK[evaluation_task]


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

# The only legal deterministic-metrics type for each task. Immutable and
# internal: the registry must not be modifiable through any public API.
# 每个任务唯一合法的确定性指标类型。不可变且内部：注册表不得通过任何
# 公共 API 修改。
EXPECTED_METRICS: MappingProxyType[str, type[BaseModel]] = MappingProxyType(
    {
        "counting": CountDeterministicMetrics,
        "general_vqa": VQADeterministicMetrics,
        "grounding": GroundingDeterministicMetrics,
        "caption": CaptionDeterministicMetrics,
    }
)


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
    deterministic metrics and can never override them. The deterministic
    metrics type must match the declared task.
    任意已实现任务的单条合并确定性指标与可选仅文本审核结果记录。Judge
    输出与确定性指标并列记录，永远不能覆盖后者。确定性指标类型必须与
    声明的 task 一致。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task: EvaluationTask
    deterministic_metrics: DeterministicMetrics | None = None
    judge_status: Literal["not_requested", "succeeded", "failed"]
    judge_raw: str | None = None
    judge_parsed: Any | None = None
    judge_inconsistency: bool = False
    judge_error: str | None = None

    @model_validator(mode="after")
    def validate_task_metrics(self) -> "EvaluationRecord":
        """Enforce the task/metrics invariant; the error never dumps the
        metrics payload. 强制 task/指标 不变式；错误消息不 dump 指标载荷。"""
        if self.deterministic_metrics is not None and not isinstance(
            self.deterministic_metrics, EXPECTED_METRICS[self.task]
        ):
            raise ValueError("deterministic_metrics type does not match task")
        return self
