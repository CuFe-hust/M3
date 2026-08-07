"""Pre-sample task resolution: answer "what task is this?" without ever
touching TaskRouter's determinism.

样本前任务解析：回答“这是什么任务？”。显式 task 直接通过（不调用模型、
不消费 budget）；空问题仅两条确定性规则（1 图 caption / 2 图
change_caption）；只有缺失 task 且有 question 时才调用模型。模型调用使用
完整 ModelCacheIdentity 与 response schema 哈希；低置信度只返回结构化候选
信息，绝不执行任何业务 Agent。TaskResolver 不是业务 AgentRegistry 成员：
不接受 UnifiedSample、不返回 AgentExecution、不注册 AgentName。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field

from agents.base import CallBudget
from data.schema import TaskName
from models.base import (
    MissingModelCacheIdentityError,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
    require_model_cache_identity,
)
from routing.schema import TaskResolution, TaskResolutionRequest

# Closed set of legal task names; the model can only pick from this set.
# 合法任务名的封闭集合；模型只能从中挑选。
_ALL_TASK_NAMES = frozenset(get_args(TaskName))


class TaskResolutionError(ValueError):
    """Stable error for task-resolution failures; the public message carries
    only the stable code, never raw model text.
    任务解析失败的稳定错误；公共消息只携带稳定 code，绝不携带原始模型文本。"""

    def __init__(self, code: str) -> None:
        super().__init__(f"TASK_RESOLUTION_FAILED:{code}")
        self.code = code


class _ModelTaskResolution(BaseModel):
    """Private schema for the resolver's model call; TaskName validation
    restricts the model to legal tasks.
    解析器模型调用的私有 schema；TaskName 校验把模型限制在合法任务内。"""

    model_config = ConfigDict(extra="forbid")

    task: TaskName
    confidence: float = Field(ge=0.0, le=1.0)
    candidate_tasks: list[TaskName] = Field(default_factory=list, max_length=3)
    reason_codes: list[str] = Field(default_factory=list, max_length=6)


class TaskResolver:
    """Resolve "what task is this?" before the deterministic TaskRouter. This
    is an orchestration service, not a business agent: it never runs business
    agents and is not a member of the AgentRegistry.
    在确定性 TaskRouter 之前回答“这是什么任务？”。这是编排服务而非业务
    Agent：绝不执行业务 Agent，也不是 AgentRegistry 成员。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        system_prompt: str,
        prompt_version: str = "task-resolver-v1",
        confidence_threshold: float = 0.70,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0.0, 1.0]")
        self._client = client
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version
        self._confidence_threshold = confidence_threshold

    async def resolve(
        self,
        request: TaskResolutionRequest,
        *,
        sample_id: str,
        artifact_dir: Path,
        budget: CallBudget | None = None,
    ) -> TaskResolution:
        """Resolve the task through explicit → rule → model paths in order.
        按 explicit → rule → model 顺序解析任务。"""
        explicit = request.explicit_task
        if explicit is not None and explicit.strip():
            return self._resolve_explicit(explicit)
        if not request.question.strip():
            return self._resolve_empty_question(request.image_count)
        return await self._resolve_via_model(
            request,
            sample_id=sample_id,
            artifact_dir=artifact_dir,
            budget=budget,
        )

    def _resolve_explicit(self, explicit: str) -> TaskResolution:
        """An explicit task never calls the model; invalid tasks fail with a
        stable code instead of being guessed. 显式任务绝不调用模型；非法任务
        以稳定错误码失败而非猜测。"""
        if explicit not in _ALL_TASK_NAMES:
            raise TaskResolutionError("UNKNOWN_EXPLICIT_TASK")
        return TaskResolution(
            task=explicit,  # type: ignore[arg-type]
            confidence=1.0,
            candidate_tasks=[explicit],  # type: ignore[list-item]
            needs_candidate_fallback=False,
            source="explicit",
            reason_codes=[f"explicit_task:{explicit}"],
        )

    def _resolve_empty_question(self, image_count: int) -> TaskResolution:
        """Two narrow deterministic rules for blank questions; anything else
        fails instead of guessing general_vqa. 空问题的两条窄确定性规则；其他
        情况稳定失败而非猜测 general_vqa。"""
        if image_count == 1:
            return TaskResolution(
                task="caption",
                confidence=1.0,
                candidate_tasks=["caption"],
                needs_candidate_fallback=False,
                source="rule",
                reason_codes=["empty_question_single_image_caption"],
            )
        if image_count == 2:
            return TaskResolution(
                task="change_caption",
                confidence=1.0,
                candidate_tasks=["change_caption"],
                needs_candidate_fallback=False,
                source="rule",
                reason_codes=["empty_question_two_image_change_caption"],
            )
        raise TaskResolutionError("EMPTY_UNRESOLVABLE_REQUEST")

    async def _resolve_via_model(
        self,
        request: TaskResolutionRequest,
        *,
        sample_id: str,
        artifact_dir: Path,
        budget: CallBudget | None,
    ) -> TaskResolution:
        """One schema-validated model call; identity is required and the budget
        is only consumed once a model call is actually attempted. 一次 schema
        校验的模型调用；必须先通过身份校验，且只在真正尝试模型调用时才
        消费 budget。"""
        try:
            identity = require_model_cache_identity(
                self._client, component="task_resolver"
            )
        except MissingModelCacheIdentityError as exc:
            raise TaskResolutionError("MODEL_IDENTITY_REQUIRED") from exc
        user_payload = {
            "question": request.question,
            "image_count": request.image_count,
            "metadata_hints": request.metadata_hints,
            "allowed_tasks": sorted(_ALL_TASK_NAMES),
        }
        messages: list[dict[str, object]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=self._prompt_version,
            messages=messages,
            image_sha256=None,
            response_schema=_ModelTaskResolution.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        # The budget is consumed only when a model call is actually attempted;
        # identity failures and hash construction never reserve a call.
        # 只在真正尝试模型调用时才消费 budget；身份失败与哈希构造绝不预留
        # 调用。
        if budget is not None:
            budget.reserve_qwen()
        try:
            raw = await self._client.complete_json(
                messages=messages,
                response_model=_ModelTaskResolution,
                request_meta=RequestMeta(
                    request_id=f"{sample_id}:task_resolution",
                    request_hash=request_hash,
                    prompt_version=self._prompt_version,
                    sample_id=sample_id,
                    artifact_dir=artifact_dir / "task_resolution",
                ),
            )
        except Exception as exc:
            raise TaskResolutionError("MODEL_RESOLUTION_FAILED") from exc
        return self._build_resolution(raw)

    def _build_resolution(self, raw: _ModelTaskResolution) -> TaskResolution:
        """Stable dedupe keeps the model task first; low confidence always
        reserves a slot for general_vqa and flags a candidate fallback. The
        resolver never executes agents itself — that is left to the future
        SampleRunner. 稳定去重保持模型任务居首；低置信度恒为 general_vqa
        保留一个槽位并标记 candidate fallback。Resolver 自身绝不执行
        Agent——这留给未来的 SampleRunner。"""
        low_confidence = raw.confidence < self._confidence_threshold
        if low_confidence:
            candidates = _low_confidence_candidates(raw.task, raw.candidate_tasks)
            needs_candidate_fallback = True
        else:
            candidates = list(dict.fromkeys([raw.task, *raw.candidate_tasks]))[:3]
            needs_candidate_fallback = False
        codes = ["low_confidence" if low_confidence else "model_high_confidence"]
        if low_confidence and raw.task == "general_vqa":
            codes.append("low_confidence_general_fallback")
        reason_codes = list(dict.fromkeys([*raw.reason_codes, *codes]))
        return TaskResolution(
            task=raw.task,
            confidence=raw.confidence,
            candidate_tasks=candidates,  # type: ignore[arg-type]
            needs_candidate_fallback=needs_candidate_fallback,
            source="model",
            reason_codes=reason_codes,
        )


def _low_confidence_candidates(
    task: TaskName,
    model_candidates: list[TaskName],
) -> list[TaskName]:
    """Keep the model task first and the model's ordering, while always
    reserving one of the three slots for general_vqa when the task itself is
    not general_vqa. A full candidate list can never truncate the fallback.
    保持模型任务居首与模型顺序，同时在 task 本身不是 general_vqa 时恒为
    general_vqa 保留三个槽位之一；候选已满也绝不会截掉兜底。"""
    deduped = list(dict.fromkeys([task, *model_candidates]))
    if task == "general_vqa":
        return deduped[:3]
    non_general = [candidate for candidate in deduped if candidate != "general_vqa"]
    result = non_general[:2]
    result.append("general_vqa")
    return result
