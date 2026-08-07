"""Contract tests for the pre-sample TaskResolver.

TaskResolver 契约测试：explicit task 直接通过、空问题规则、缺失 task 时单次
模型调用、request hash 覆盖 schema/identity、低置信度候选、schema 限制合法
TaskName、身份缺失前置失败、原始异常文本隔离、TaskRouter 不受影响。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, get_args

import pytest

from agents.base import CallBudget
from data.schema import TaskName
from models.base import ModelCacheIdentity, build_request_hash
from routing.router import TaskRouter
from routing.schema import TaskResolution, TaskResolutionRequest
from workflows.task_resolver import TaskResolutionError, TaskResolver

REPO_ROOT = Path(__file__).resolve().parents[2]
_ALL_TASKS = get_args(TaskName)

_SENSITIVE_TEXT = (
    "/home/user/private/model.pt sk-test-secret Bearer abcdef base64,AAAA"
)


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _RecordingClient:
    """Records every complete_json call and validates through the response
    schema like the real client. 像真实客户端一样通过 response schema 校验，
    并记录每次 complete_json 调用。"""

    def __init__(
        self,
        model_response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._model_response = model_response or {
            "task": "general_vqa",
            "confidence": 0.95,
            "candidate_tasks": ["general_vqa"],
            "reason_codes": ["model_reason"],
        }
        self._error = error

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 64},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "response_model": response_model,
                "request_meta": request_meta,
            }
        )
        if self._error is not None:
            raise self._error
        return response_model.model_validate(self._model_response)


class _NoIdentityClient:
    """A model client without any cache identity. 无缓存身份的模型客户端。"""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(request_meta)
        return response_model.model_validate(
            {"task": "general_vqa", "confidence": 0.9}
        )


class _DuckIdentityClient(_NoIdentityClient):
    """Duck-typed identity stand-in that must be rejected. 必须被拒绝的鸭子
    类型身份替代品。"""

    class _Duck:
        model = "fake"
        generation = {}
        client_version = "1"
        revision = None

    cache_identity = _Duck()


def _resolver(client: Any, **overrides) -> TaskResolver:
    values = dict(system_prompt="Classify the task from the question.")
    values.update(overrides)
    return TaskResolver(client, **values)


def _request(**overrides) -> TaskResolutionRequest:
    values = dict(image_count=1)
    values.update(overrides)
    return TaskResolutionRequest(**values)


def _resolve(
    resolver: TaskResolver,
    request: TaskResolutionRequest,
    *,
    budget: _FakeBudget | None = None,
    sample_id: str = "s1",
    artifact_dir: Path = Path("artifacts"),
) -> TaskResolution:
    return asyncio.run(
        resolver.resolve(
            request,
            sample_id=sample_id,
            artifact_dir=artifact_dir,
            budget=budget,
        )
    )


# ── 路径 1：显式 task / explicit task ──────────────────────────────────────


def test_explicit_task_never_calls_model() -> None:
    client = _RecordingClient()
    resolution = _resolve(
        _resolver(client), _request(explicit_task="counting")
    )
    assert client.calls == []
    assert resolution.task == "counting"
    assert resolution.confidence == 1.0
    assert resolution.candidate_tasks == ["counting"]
    assert resolution.needs_candidate_fallback is False
    assert resolution.source == "explicit"
    assert resolution.reason_codes == ["explicit_task:counting"]


def test_explicit_task_consumes_no_budget() -> None:
    budget = _FakeBudget()
    _resolve(
        _resolver(_RecordingClient()),
        _request(explicit_task="caption"),
        budget=budget,
    )
    assert budget.qwen_calls == 0


def test_invalid_explicit_task_fails_without_model() -> None:
    client = _RecordingClient()
    with pytest.raises(TaskResolutionError, match="UNKNOWN_EXPLICIT_TASK") as info:
        _resolve(_resolver(client), _request(explicit_task="magic_task"))
    assert info.value.code == "UNKNOWN_EXPLICIT_TASK"
    assert client.calls == []


# ── 路径 2：空问题规则 / blank-question rules ──────────────────────────────


def test_one_image_blank_question_resolves_caption() -> None:
    client = _RecordingClient()
    resolution = _resolve(_resolver(client), _request(question="", image_count=1))
    assert client.calls == []
    assert resolution.task == "caption"
    assert resolution.source == "rule"
    assert resolution.confidence == 1.0
    assert resolution.candidate_tasks == ["caption"]
    assert resolution.needs_candidate_fallback is False


def test_two_images_blank_question_resolves_change_caption() -> None:
    client = _RecordingClient()
    resolution = _resolve(
        _resolver(client), _request(question="   ", image_count=2)
    )
    assert client.calls == []
    assert resolution.task == "change_caption"
    assert resolution.source == "rule"


def test_other_image_count_blank_question_fails() -> None:
    client = _RecordingClient()
    with pytest.raises(TaskResolutionError, match="EMPTY_UNRESOLVABLE_REQUEST"):
        _resolve(_resolver(client), _request(question="", image_count=3))
    assert client.calls == []


# ── 路径 3：模型解析 / model resolution ────────────────────────────────────


def test_missing_task_with_question_calls_model_once() -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    resolution = _resolve(
        _resolver(client),
        _request(question="How many cars?", image_count=1, metadata_hints={"split": "test"}),
        budget=budget,
    )
    assert len(client.calls) == 1
    assert budget.qwen_calls == 1
    assert resolution.source == "model"
    content = client.calls[0]["messages"][1]["content"]
    assert "How many cars?" in content
    assert '"metadata_hints": {"split": "test"}' in content
    assert client.calls[0]["request_meta"].request_id == "s1:task_resolution"


def test_request_hash_covers_schema_and_identity() -> None:
    client = _RecordingClient()
    resolver = _resolver(client, system_prompt="PROMPT-X")
    _resolve(resolver, _request(question="How many cars?"))
    call = client.calls[0]
    meta = call["request_meta"]
    assert len(meta.request_hash) == 64
    user_payload = {
        "question": "How many cars?",
        "image_count": 1,
        "metadata_hints": {},
        "allowed_tasks": sorted(_ALL_TASKS),
    }
    expected = build_request_hash(
        model="fake-model",
        generation=client.cache_identity.generation_payload(),
        prompt_version="task-resolver-v1",
        messages=[
            {"role": "system", "content": "PROMPT-X"},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        image_sha256=None,
        response_schema=call["response_model"].model_json_schema(),
        client_version="1",
        model_revision=None,
    )
    assert meta.request_hash == expected


def test_high_confidence_single_route_contract() -> None:
    client = _RecordingClient(
        model_response={
            "task": "general_vqa",
            "confidence": 0.95,
            "candidate_tasks": ["general_vqa", "caption"],
            "reason_codes": ["model_reason"],
        }
    )
    resolution = _resolve(_resolver(client), _request(question="What is this?"))
    assert resolution.task == "general_vqa"
    assert resolution.candidate_tasks == ["general_vqa", "caption"]
    assert resolution.needs_candidate_fallback is False
    assert resolution.source == "model"
    assert "model_high_confidence" in resolution.reason_codes


def test_low_confidence_adds_general_vqa_candidates() -> None:
    client = _RecordingClient(
        model_response={
            "task": "caption",
            "confidence": 0.40,
            "candidate_tasks": ["caption", "grounding"],
            "reason_codes": ["ambiguous"],
        }
    )
    resolution = _resolve(_resolver(client), _request(question="Hmm?"))
    assert resolution.task == "caption"
    assert resolution.candidate_tasks == ["caption", "grounding", "general_vqa"]
    assert resolution.needs_candidate_fallback is True
    assert "low_confidence" in resolution.reason_codes
    assert "ambiguous" in resolution.reason_codes


def test_low_confidence_general_vqa_only_candidates() -> None:
    """When the model's top task is general_vqa, the candidate list may stay
    minimal but the reason must flag a low-confidence general fallback.
    模型最可能任务为 general_vqa 时候选列表可以保持最小，但 reason 必须标记
    低置信度 general fallback。"""
    client = _RecordingClient(
        model_response={
            "task": "general_vqa",
            "confidence": 0.30,
            "candidate_tasks": ["general_vqa"],
            "reason_codes": [],
        }
    )
    resolution = _resolve(_resolver(client), _request(question="Hmm?"))
    assert resolution.candidate_tasks == ["general_vqa"]
    assert resolution.needs_candidate_fallback is True
    assert "low_confidence" in resolution.reason_codes
    assert "low_confidence_general_fallback" in resolution.reason_codes


def test_candidates_stably_deduped_task_first() -> None:
    client = _RecordingClient(
        model_response={
            "task": "caption",
            "confidence": 0.90,
            "candidate_tasks": ["caption", "grounding", "caption"],
            "reason_codes": ["r1"],
        }
    )
    resolution = _resolve(_resolver(client), _request(question="What?"))
    assert resolution.candidate_tasks == ["caption", "grounding"]
    assert resolution.candidate_tasks[0] == resolution.task
    # A model task absent from its own candidates is still first.
    # 模型任务即使不在自身候选里也保持居首。
    client2 = _RecordingClient(
        model_response={
            "task": "caption",
            "confidence": 0.90,
            "candidate_tasks": ["grounding"],
            "reason_codes": [],
        }
    )
    resolution2 = _resolve(_resolver(client2), _request(question="What?"))
    assert resolution2.candidate_tasks == ["caption", "grounding"]


def test_model_output_restricted_to_valid_task_names() -> None:
    """The private Pydantic schema rejects illegal task names before any
    resolution can materialize. 私有 Pydantic schema 在任何结果产生前拒绝
    非法任务名。"""
    client = _RecordingClient(model_response={"task": "not_a_task", "confidence": 0.9})
    with pytest.raises(TaskResolutionError, match="MODEL_RESOLUTION_FAILED"):
        _resolve(_resolver(client), _request(question="What?"))
    # The call happened; the schema rejected the illegal task name inside the
    # client, so no resolution ever materialized. 调用确实发生；schema 在客户
    # 端内部拒绝了非法任务名，任何解析结果都未曾产生。
    assert len(client.calls) == 1


def test_missing_cache_identity_fails_before_model_call() -> None:
    client = _NoIdentityClient()
    with pytest.raises(TaskResolutionError, match="MODEL_IDENTITY_REQUIRED") as info:
        _resolve(_resolver(client), _request(question="What?"))
    assert info.value.code == "MODEL_IDENTITY_REQUIRED"
    assert client.calls == []


def test_duck_typed_identity_fails_before_model_call() -> None:
    client = _DuckIdentityClient()
    with pytest.raises(TaskResolutionError, match="MODEL_IDENTITY_REQUIRED"):
        _resolve(_resolver(client), _request(question="What?"))
    assert client.calls == []


def test_raw_exception_text_never_leaks() -> None:
    client = _RecordingClient(error=RuntimeError(_SENSITIVE_TEXT))
    with pytest.raises(TaskResolutionError) as info:
        _resolve(_resolver(client), _request(question="What?"))
    public = str(info.value)
    assert public == "TASK_RESOLUTION_FAILED:MODEL_RESOLUTION_FAILED"
    assert info.value.__cause__ is not None  # chaining kept / 保留链式异常
    for token in ("/home/user/private", "sk-test-secret", "Bearer abcdef", "base64,AAAA"):
        assert token not in public, token


# ── TaskRouter 不受影响 / TaskRouter untouched ─────────────────────────────


def test_task_router_remains_unaffected() -> None:
    """TaskRouter stays synchronous, deterministic, question-free, and
    model-free. TaskRouter 保持同步、确定性、不读 question、不调用模型。"""
    import inspect

    router = TaskRouter()
    assert not inspect.iscoroutinefunction(TaskRouter.route)
    assert "question" not in inspect.signature(TaskRouter.route).parameters
    first = router.route("counting")
    assert router.route("counting") == first
    with pytest.raises(KeyError):
        router.route("no-such-task")
    for relative in ("routing/router.py", "routing/policies.py"):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "complete_json" not in source
        assert "import models" not in source
