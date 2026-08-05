"""Agent Protocol, shared execution context, and execution validation.
Agent 协议、共享执行上下文与执行校验。

Target architecture uses frozen dataclasses for runtime-constructed values
to avoid Pydantic validation cost on Protocol/object fields.
目标架构使用 frozen dataclass 保存运行时构造值，避免对 Protocol/object 字段的 Pydantic 校验开销。

The AgentRegistry lives in ``agents/registry.py`` (separated to keep base.py
free of data-structure concerns).
AgentRegistry 位于 ``agents/registry.py`` 中（分离以保持 base.py 不涉数据结构）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, Union, runtime_checkable

from models.base import VisionLanguageClient
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing.budget import CallBudget
from spacers_agent.schemas import CountingResult, ExpertResult
from spacers_agent.settings import AppSettings

if TYPE_CHECKING:
    from spacers_agent.clients.deepseek import DeepSeekJudgeClient

# ── type aliases / 类型别名 ──────────────────────────────────────────────

AgentName = Literal[
    "counting_agent",
    "change_agent",
    "grounding_agent",
    "spatial_agent",
    "general_vqa_agent",
    "caption_agent",
]

AgentPayload = Union[CountingResult, ExpertResult]

# ── legacy name normalization / 旧名规范化 ───────────────────────────────

LEGACY_AGENT_NAME_ALIASES: dict[str, AgentName] = {
    "counting_expert": "counting_agent",
    "change_expert": "change_agent",
    "grounding_expert": "grounding_agent",
    "spatial_expert": "spatial_agent",
    "general_vqa_expert": "general_vqa_agent",
    "caption_expert": "caption_agent",
}

EXPERT_TO_AGENT: dict[str, AgentName] = dict(LEGACY_AGENT_NAME_ALIASES)

AGENT_TO_EXPERT: dict[AgentName, str] = {v: k for k, v in LEGACY_AGENT_NAME_ALIASES.items()}


def normalize_agent_name(raw: str) -> AgentName:
    """Map legacy expert names to agent names; passes through valid agent names.
    将旧 expert 名映射为 agent 名；有效的 agent 名原样通过。
    """
    if raw in LEGACY_AGENT_NAME_ALIASES:
        return LEGACY_AGENT_NAME_ALIASES[raw]
    # Also try AgentName literal check
    from typing import get_args
    if raw in get_args(AgentName):
        return raw  # type: ignore[return-value]
    raise ValueError(f"Unknown agent/expert name: {raw!r}. Known: {sorted(LEGACY_AGENT_NAME_ALIASES)}")


# ── execution context / 执行上下文 ────────────────────────────────────────


@dataclass(frozen=True)
class AgentContext:
    """Immutable context injected into every agent call. / 每次 Agent 调用注入的不可变上下文。

    Never stores API keys, Base64 image data, or model weights.
    不保存 API 密钥、Base64 图像数据或模型权重。
    """

    artifact_dir: Path
    settings: AppSettings
    qwen_client: VisionLanguageClient
    call_budget: CallBudget
    prompt_catalog: PromptCatalog | None = None
    judge_client: DeepSeekJudgeClient | None = None

    def __post_init__(self) -> None:
        # Path("") evaluates to True in Python 3.11+ / Path("") 在 Python 3.11+ 中为 True
        if not self.artifact_dir or str(self.artifact_dir) in {"", "."}:
            raise ValueError("artifact_dir must not be empty")
        if not isinstance(self.artifact_dir, Path):
            raise TypeError("artifact_dir must be a pathlib.Path")


# ── execution result / 执行结果 ───────────────────────────────────────────


@dataclass(frozen=True)
class AgentExecution:
    """Runtime wrapper returned by an agent; does NOT replace persisted schemas.
    由 Agent 返回的运行时包装；不替代持久化 Schema。
    """

    agent_name: AgentName
    payload: AgentPayload
    result_filename: str
    trace: dict[str, Any] = field(default_factory=dict)
    additional_results: dict[str, AgentPayload] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_agent_execution(self)

    @property
    def status(self) -> str:
        """Delegate to payload.status for uniform readability. / 委托给 payload.status。"""
        return str(self.payload.status)  # type: ignore[union-attr]


# ── agent contract / Agent 契约 ───────────────────────────────────────────


@runtime_checkable
class Agent(Protocol):
    """Execution contract for a concrete remote-sensing agent.
    具体遥感 Agent 的执行契约。
    """

    name: AgentName
    supported_tasks: frozenset[str]

    async def run(self, sample: Any, context: AgentContext) -> AgentExecution:
        """Execute the agent pipeline for one sample. / 为单条样本执行 Agent 管线。"""
        ...


# ── execution validation / 执行校验 ───────────────────────────────────────


# Sensitive key patterns that must never appear in traces.
# trace 中绝不能出现的敏感键模式。
_SENSITIVE_TRACE_KEYS = frozenset({
    "api_key", "apikey", "authorization", "secret", "token", "password",
    "base64", "credential", "private_key",
})


def validate_agent_execution(execution: AgentExecution) -> None:
    """Validate an AgentExecution after construction. / 构造后校验 AgentExecution。

    Checks / 检查：
    - result_filename is a safe plain basename (no absolute, no ..)
    - trace does not contain sensitive key patterns
    - result_filename matches payload type expectations
    """

    # result_filename safety / result_filename 安全性
    if not isinstance(execution.result_filename, str) or not execution.result_filename:
        raise ValueError("result_filename must be a non-empty string")
    if execution.result_filename.startswith("/") or "\\" in execution.result_filename:
        raise ValueError(
            f"result_filename must be a plain basename, got {execution.result_filename!r}"
        )
    if ".." in execution.result_filename:
        raise ValueError(
            f"result_filename must not contain '..', got {execution.result_filename!r}"
        )

    # Trace sanitization check / trace 脱敏检查
    _check_trace_no_sensitive_keys(execution.trace)

    for filename in execution.additional_results:
        _validate_result_filename(filename)

    # Enforce trace is JSON-serializable (no binary / non-serializable objects)
    # 强制 trace 可 JSON 序列化
    try:
        json.dumps(execution.trace, default=str)
    except (TypeError, ValueError) as error:
        raise ValueError(f"AgentExecution trace must be JSON-serializable: {error}") from error


def _check_trace_no_sensitive_keys(trace: dict[str, Any]) -> None:
    """Recursively check that no sensitive key patterns exist in a trace dict.
    递归检查 trace dict 中不存在敏感键模式。
    """

    def _check(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, val in value.items():
                key_lower = str(key).lower().replace("-", "_").replace(" ", "_")
                for sensitive in _SENSITIVE_TRACE_KEYS:
                    if sensitive in key_lower:
                        raise ValueError(
                            f"trace contains sensitive key {key!r} at {path}"
                        )
                _check(val, f"{path}.{key}")
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                _check(item, f"{path}[{idx}]")

    _check(trace, "trace")


def _validate_result_filename(filename: str) -> None:
    """Validate a supplemental result filename with the primary rules.
    使用主结果相同的规则校验补充结果文件名。
    """

    if not isinstance(filename, str) or not filename:
        raise ValueError("result filename must be a non-empty string")
    if filename.startswith("/") or "\\" in filename or ".." in filename:
        raise ValueError(f"result filename must be a safe plain basename, got {filename!r}")
