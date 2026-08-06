"""Agent Protocol, shared execution context, and execution validation.

Agent 协议、共享执行上下文与执行校验。AgentContext 不持有完整 AppSettings
或 PromptCatalog——只保留运行一条样本所必需的轻量对象与局部服务。
AgentRegistry 位于 agents/registry.py（分离以保持 base.py 不涉数据结构）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from models.base import VisionLanguageClient
from agents.schema import AgentName, AgentResult


class CallBudget(Protocol):
    """Minimal call-budget protocol injected into agent contexts.
    注入 Agent 上下文的最小调用预算协议。"""

    def reserve_qwen(self) -> None: ...

    def reserve_deepseek(self) -> None: ...


@dataclass(frozen=True)
class AgentContext:
    """Immutable context injected into every agent call.
    每次 Agent 调用注入的不可变上下文。

    Never stores API keys, Base64 image data, or model weights; never holds a
    full AppSettings or PromptCatalog instance.
    不保存 API 密钥、Base64 图像数据或模型权重；不持有完整 AppSettings 或
    PromptCatalog 实例。
    """

    artifact_dir: Path
    qwen_client: VisionLanguageClient
    call_budget: CallBudget
    judge_client: Any | None = None
    request_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_dir or str(self.artifact_dir) in {"", "."}:
            raise ValueError("artifact_dir must not be empty")
        if not isinstance(self.artifact_dir, Path):
            raise TypeError("artifact_dir must be a pathlib.Path")


@dataclass(frozen=True)
class AgentExecution:
    """Runtime wrapper returned by an agent; does NOT replace persisted schemas.
    由 Agent 返回的运行时包装；不替代持久化 Schema。"""

    agent_name: AgentName
    payload: Any
    result_filename: str
    trace: dict[str, Any] = field(default_factory=dict)
    additional_results: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_agent_execution(self)

    @property
    def status(self) -> str:
        """Delegate to payload.status for uniform readability. / 委托给 payload.status。"""
        return str(getattr(self.payload, "status", "completed"))


@runtime_checkable
class Agent(Protocol):
    """Execution contract for a concrete remote-sensing agent.
    具体遥感 Agent 的执行契约。"""

    name: AgentName
    supported_tasks: frozenset[str]

    async def run(self, sample: Any, context: AgentContext) -> AgentExecution:
        """Execute the agent pipeline for one sample. / 为单条样本执行 Agent 管线。"""
        ...


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
    - trace is JSON-serializable
    """
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

    _check_trace_no_sensitive_keys(execution.trace)

    for filename in execution.additional_results:
        _validate_result_filename(filename)

    try:
        json.dumps(execution.trace, default=str)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"AgentExecution trace must be JSON-serializable: {error}"
        ) from error


def _check_trace_no_sensitive_keys(trace: dict[str, Any]) -> None:
    """Recursively check that no sensitive key patterns exist in a trace dict.
    递归检查 trace dict 中不存在敏感键模式。"""

    def _check(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, val in value.items():
                key_lower = str(key).lower().replace("-", "_").replace(" ", "_")
                for sensitive in _SENSITIVE_TRACE_KEYS:
                    if sensitive in key_lower:
                        raise ValueError(f"trace contains sensitive key {key!r} at {path}")
                _check(val, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                _check(item, f"{path}[{index}]")

    _check(trace, "trace")


def _validate_result_filename(filename: str) -> None:
    """Validate a supplemental result filename with the primary rules.
    使用主结果相同的规则校验补充结果文件名。"""
    if not isinstance(filename, str) or not filename:
        raise ValueError("result filename must be a non-empty string")
    if filename.startswith("/") or "\\" in filename or ".." in filename:
        raise ValueError(
            f"result filename must be a safe plain basename, got {filename!r}"
        )
