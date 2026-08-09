"""Text-only judge schemas, protocols, and pure payload/hash builders.

仅文本 judge 的 Schema、协议与纯载荷/哈希构建。本模块只包含纯数据与纯
函数：不发起网络请求、不读写文件、不读取环境变量。所有载荷保证只含文本
与结构化证据，绝不包含图像数据或路径；哈希为稳定确定性输出（缓存键与
resume 判定依据）。依赖边界：judges 层只允许 data.schema/agents.schema 与
批准的模型契约，因此计数证据以结构子集协议（CountEvidence/CountTarget）
消费，不导入 agents.counting.schema。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from data.schema import GroundTruth
from evaluation.metrics.counting import count_deterministic_metrics
from evaluation.metrics.vqa import exact_match
from models.base import RequestMeta


class DeepSeekJudgeResult(BaseModel):
    """Structured counting-judge output that explicitly excludes visual truth
    verification. 明确排除视觉真相核验的结构化计数 judge 输出。"""

    model_config = ConfigDict(extra="forbid")

    judge_scope: Literal["text_and_structured_evidence_only"]
    can_verify_visual_truth: Literal[False]
    semantic_correctness: float = Field(ge=0.0, le=1.0)
    answer_evidence_consistency: float = Field(ge=0.0, le=1.0)
    constraint_following: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)
    verdict: Literal["correct", "mostly_correct", "incorrect", "not_judgeable"]
    issues: list[str] = Field(default_factory=list)
    concise_rationale: str = Field(max_length=500)


class VQAAnswerJudgeResult(BaseModel):
    """Minimal binary answer-validation result for text-only VQA judging.
    用于纯文本 VQA 审核的最小二值答案验证结果。"""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=1)
    concise_rationale: str = Field(default="", max_length=500)
    judge_scope: Literal["text_and_structured_evidence_only"] = (
        "text_and_structured_evidence_only"
    )
    can_verify_visual_truth: Literal[False] = False


ModelT = TypeVar("ModelT", bound=BaseModel)


class CountEvidence(Protocol):
    """Structural subset of CountingResult consumed by the counting payload
    builder; the authoritative typed contract lives in agents.counting.schema,
    which the judges layer deliberately does not import.
    计数载荷构建消费的 CountingResult 结构子集；权威类型契约位于
    agents.counting.schema，judges 层刻意不导入它。"""

    final_count: int
    tile_count: int
    succeeded_tiles: list[str]
    failed_tiles: list[str]
    global_points: list[Any]
    merged_groups: list[list[str]]
    unresolved_conflicts: list[str]


class CountTarget(Protocol):
    """Structural subset of CountTargetSpec used by the counting payload.
    计数载荷使用的 CountTargetSpec 结构子集。"""

    canonical_label: str
    inclusion_rule: str
    exclusion_rule: str


class JudgeClient(Protocol):
    """Text-only judge client protocol; implementations must never read the
    environment or load a vision model. judge service 使用的仅文本 judge 客户端
    协议；实现绝不读取环境变量或加载视觉模型。"""

    def judge(
        self,
        payload: Mapping[str, Any],
        *,
        request_meta: RequestMeta,
    ) -> DeepSeekJudgeResult: ...

    def judge_json(
        self,
        payload: Mapping[str, Any],
        *,
        response_model: type[ModelT],
        request_meta: RequestMeta,
        system_prompt: str | None = None,
    ) -> ModelT: ...


def stable_error_label(error: Exception) -> str:
    """Return a stable, content-free error label (class name only) so public
    records and artifacts never leak raw exception text or secrets.
    返回稳定、无内容的安全错误标签（仅类名），公共记录与产物绝不泄漏原始
    异常文本或密钥。"""

    return type(error).__name__


def build_vqa_judge_payload(
    *,
    question: str,
    reference_answers: list[str],
    candidate_answer: str,
) -> dict[str, Any]:
    """Build a VQA judge payload containing no image data or paths.
    构建不含图像数据或路径的 VQA judge 载荷。"""

    return {
        "question": question,
        "prediction": {"answer": candidate_answer},
        "ground_truth": {"answers": reference_answers},
        "deterministic_metrics": {
            "exact_match": int(exact_match(candidate_answer, reference_answers))
        },
    }


def build_count_judge_payload(
    *,
    question: str,
    target: CountTarget,
    display_answer: str,
    counting: CountEvidence,
    ground_truth: GroundTruth | None,
    min_confidence: float,
) -> dict[str, Any]:
    """Build a compact text-and-evidence counting payload that never includes
    image data or paths. 构建绝不包含图像数据或路径的紧凑文本与证据计数载荷。"""

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    metrics = (
        count_deterministic_metrics(counting.final_count, ground_truth.count)
        if ground_truth is not None and ground_truth.count is not None
        else None
    )
    accepted_points = [point for point in counting.global_points if point.accepted]
    return {
        "task": "counting",
        "question": question,
        "target_spec": {
            "canonical_label": target.canonical_label,
            "inclusion_rule": target.inclusion_rule,
            "exclusion_rule": target.exclusion_rule,
        },
        "prediction": {
            "display_answer": display_answer,
            "final_count": counting.final_count,
            "point_count": len(accepted_points),
            "failed_tiles": counting.failed_tiles,
            "unresolved_conflicts": counting.unresolved_conflicts,
        },
        "ground_truth": (
            {"count": ground_truth.count, "answers": ground_truth.answers}
            if ground_truth is not None
            else None
        ),
        "deterministic_metrics": (
            metrics.model_dump(mode="json") if metrics is not None else None
        ),
        "evidence_summary": {
            "tile_count": counting.tile_count,
            "succeeded_tiles": len(counting.succeeded_tiles),
            "low_confidence_points": sum(
                point.confidence < min_confidence for point in accepted_points
            ),
            "seam_merges": len(counting.merged_groups),
        },
    }


def _stable_hash(value: Any) -> str:
    """Deterministic JSON hash for stable cache keys and resume behavior.
    用于稳定缓存键与 resume 行为的确定性 JSON 哈希。"""

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_judge_request_hash(
    *,
    model: str,
    prompt_text: str,
    prompt_version: str,
    sample_id: str,
    payload: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> str:
    """Canonical judge request hash: every input that shapes the judge call —
    model identity, prompt text and version, sample id, the full payload, and
    the response schema — contributes to the cache key. Same inputs always
    produce the same hash. 规范 judge 请求哈希：塑造 judge 调用的每个输入——
    模型身份、prompt 文本与版本、sample id、完整载荷与 response schema——
    都进入缓存键。相同输入恒产生相同哈希。"""

    return _stable_hash(
        {
            "model": model,
            "prompt_sha256": _stable_hash(prompt_text),
            "prompt_version": prompt_version,
            "sample_id": sample_id,
            "payload": payload,
            "response_schema": response_schema,
        }
    )
