"""Agent Protocol, shared execution context, and execution validation.

Agent 协议、共享执行上下文与执行校验。AgentContext 不持有完整 AppSettings
或 PromptCatalog——只保留运行一条样本所必需的轻量对象与局部服务，并显式
携带 data_root 解析根。AgentExecution 执行严格的安全校验（纯 basename、
严格 JSON-safe、敏感内容、payload 名称一致）。AgentRegistry 位于
agents/registry.py（分离以保持 base.py 不涉数据结构）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
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
    full AppSettings or PromptCatalog instance. data_root is the explicit
    resolution root for relative ImageRef paths.
    不保存 API 密钥、Base64 图像数据或模型权重；不持有完整 AppSettings 或
    PromptCatalog 实例。data_root 是相对 ImageRef 路径的显式解析根。"""

    artifact_dir: Path
    qwen_client: VisionLanguageClient
    call_budget: CallBudget
    data_root: Path | None = None
    judge_client: Any | None = None
    request_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_dir or str(self.artifact_dir) in {"", "."}:
            raise ValueError("artifact_dir must not be empty")
        if not isinstance(self.artifact_dir, Path):
            raise TypeError("artifact_dir must be a pathlib.Path")
        if self.data_root is not None and not isinstance(self.data_root, Path):
            raise TypeError("data_root must be a pathlib.Path or None")
        _assert_json_safe(self.request_context, "request_context")
        _check_no_sensitive_values(self.request_context)


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
# High-risk sensitive value prefixes. / 高风险敏感值前缀。
_SENSITIVE_VALUE_PREFIXES = (
    "sk-",
    "Bearer ",
    "data:image/",
    "-----BEGIN PRIVATE KEY-----",
)


def validate_agent_execution(execution: AgentExecution) -> None:
    """Validate an AgentExecution after construction. / 构造后校验 AgentExecution。

    Checks / 检查：
    - result_filename is a safe plain basename on every platform
    - additional_results keys are safe plain basenames, unique, and do not
      collide with the primary filename
    - trace and additional_results values are strictly JSON-safe
    - trace and values contain no sensitive keys or values
    - payload agent_name matches the execution agent_name
    """
    _validate_plain_basename(execution.result_filename, "result_filename")

    _check_no_sensitive_keys(execution.trace)
    _check_no_sensitive_values(execution.trace)
    _assert_json_safe(execution.trace, "trace")

    seen_additional: set[str] = set()
    for filename, value in execution.additional_results.items():
        _validate_plain_basename(filename, "additional result filename")
        if filename == execution.result_filename:
            raise ValueError(
                f"additional result {filename!r} collides with the primary result filename"
            )
        if filename in seen_additional:
            raise ValueError(f"duplicate additional result filename {filename!r}")
        seen_additional.add(filename)
        _check_no_sensitive_values(value)
        _assert_json_safe(value, f"additional_results[{filename!r}]")

    payload = execution.payload
    if isinstance(payload, AgentResult) and payload.agent_name != execution.agent_name:
        raise ValueError(
            f"payload agent_name {payload.agent_name!r} does not match "
            f"execution agent_name {execution.agent_name!r}"
        )


def _validate_plain_basename(filename: str, label: str) -> None:
    """A plain basename on both POSIX and Windows semantics.
    同时满足 POSIX 与 Windows 语义的纯 basename。"""
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{label} must be a non-empty string")
    if filename in {"", ".", ".."}:
        raise ValueError(f"{label} must be a plain basename, got {filename!r}")
    if PurePosixPath(filename).name != filename or PureWindowsPath(filename).name != filename:
        raise ValueError(f"{label} must be a plain basename, got {filename!r}")


def _check_no_sensitive_keys(value: Any, path: str = "trace") -> None:
    """Recursively check that no sensitive key patterns exist in a dict.
    递归检查 dict 中不存在敏感键模式。"""
    if isinstance(value, dict):
        for key, val in value.items():
            key_lower = str(key).lower().replace("-", "_").replace(" ", "_")
            for sensitive in _SENSITIVE_TRACE_KEYS:
                if sensitive in key_lower:
                    raise ValueError(f"trace contains sensitive key {key!r} at {path}")
            _check_no_sensitive_keys(val, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_no_sensitive_keys(item, f"{path}[{index}]")


def _check_no_sensitive_values(value: Any) -> None:
    """Reject high-risk sensitive value prefixes in any string.
    拒绝字符串中的高风险敏感值前缀。"""
    if isinstance(value, str):
        for prefix in _SENSITIVE_VALUE_PREFIXES:
            if value.startswith(prefix):
                raise ValueError("trace contains a sensitive value prefix")
        return
    if isinstance(value, dict):
        for item in value.values():
            _check_no_sensitive_values(item)
    elif isinstance(value, list):
        for item in value:
            _check_no_sensitive_values(item)


def _assert_json_safe(value: Any, where: str) -> None:
    """Strict JSON-safety: no Path/set/bytes/callable/non-finite numbers.
    严格 JSON 安全：拒绝 Path/set/bytes/callable/非有限数值。"""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{where} contains a non-finite number")
        return
    if isinstance(value, Path):
        raise ValueError(f"{where} contains a Path object")
    if isinstance(value, (set, bytes, bytearray)):
        raise ValueError(f"{where} contains a {type(value).__name__}")
    if callable(value):
        raise ValueError(f"{where} contains a callable")
    if isinstance(value, list):
        for item in value:
            _assert_json_safe(item, where)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} contains a non-string key")
            _assert_json_safe(item, where)
        return
    raise ValueError(f"{where} contains unsupported type {type(value).__name__}")
