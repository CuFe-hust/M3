"""Stable error types for the multi-Agent runtime.
多 Agent 运行时的稳定错误类型。

These errors provide machine-readable codes and avoid leaking sensitive data
(API keys, Base64 payloads, absolute paths with secrets).
这些错误提供机器可读代码并避免泄露敏感数据
（API 密钥、Base64 载荷、含密钥的绝对路径）。
"""

from __future__ import annotations


class DuplicateAgentError(ValueError):
    """Raised when registering an agent with a name already in the registry.
    使用已存在于注册表中的名称注册 Agent 时抛出。
    """

    def __init__(self, name: str, *, existing_class: str | None = None, new_class: str | None = None) -> None:
        msg = f"Agent name {name!r} is already registered"
        if existing_class and new_class:
            msg += f" (existing: {existing_class}, new: {new_class})"
        super().__init__(msg)
        self.name = name


class UnsupportedAgentError(KeyError):
    """Raised when looking up an agent name not in the registry.
    查找不在注册表中的 Agent 名时抛出。
    """

    def __init__(self, name: str, *, available: list[str] | None = None) -> None:
        msg = f"UNREGISTERED_AGENT:{name}"
        if available:
            msg += f"; available: {sorted(available)}"
        super().__init__(msg)
        self.name = name


class AgentTaskMismatchError(ValueError):
    """Raised when an agent is asked to run a task it does not support.
    当 Agent 被要求执行其不支持的任务时抛出。
    """

    def __init__(self, agent_name: str, task: str, *, supported: frozenset[str] | None = None) -> None:
        msg = f"Agent {agent_name!r} does not support task {task!r}"
        if supported:
            msg += f"; supported: {sorted(supported)}"
        super().__init__(msg)
        self.agent_name = agent_name
        self.task = task


class AgentExecutionError(RuntimeError):
    """Raised when an agent's execution pipeline fails irrecoverably.
    当 Agent 执行管线不可恢复地失败时抛出。
    """

    def __init__(self, agent_name: str, sample_id: str, *, cause: str = "") -> None:
        msg = f"Agent {agent_name!r} failed on sample {sample_id!r}"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)
        self.agent_name = agent_name
        self.sample_id = sample_id


class CountingBackendUnavailableError(RuntimeError):
    """Raised when no counting backend can handle the given target.
    当没有计数后端可以处理给定目标时抛出。
    """

    def __init__(self, target_label: str, *, available: list[str] | None = None) -> None:
        msg = f"No counting backend available for target {target_label!r}"
        if available:
            msg += f"; registered backends: {available}"
        super().__init__(msg)
        self.target_label = target_label


class DetectorWeightsMissingError(FileNotFoundError):
    """Raised when a detector weight file is not found at the configured path.
    当检测器权重文件在配置路径不存在时抛出。
    """

    def __init__(self, backend_name: str, weight_path: str) -> None:
        msg = f"Detector weights missing for {backend_name!r}: {weight_path}"
        super().__init__(msg)
        self.backend_name = backend_name
        self.weight_path = weight_path


class DetectorWeightsHashMismatchError(RuntimeError):
    """Raised when a local detector weight digest differs from its declared digest.
    当本地检测器权重摘要与声明摘要不一致时抛出。
    """


class DetectorTaskMismatchError(RuntimeError):
    """Raised when a loaded detector is not the configured task type.
    当加载的检测器不是配置任务类型时抛出。
    """


class DetectorClassMapMismatchError(RuntimeError):
    """Raised when model classes differ from the audited configuration.
    当模型类别与已审计配置不一致时抛出。
    """


class DetectorInferenceError(RuntimeError):
    """Raised when one detector inference cannot produce a valid tile result.
    当一次检测器推理无法生成有效 tile 结果时抛出。
    """


class OptionalDependencyMissingError(ImportError):
    """Raised when an optional feature is requested but its dependency is not installed.
    当请求可选功能但其依赖未安装时抛出。
    """

    def __init__(self, feature: str, *, dependency: str, install_hint: str = "") -> None:
        msg = f"Optional feature {feature!r} requires {dependency!r}"
        if install_hint:
            msg += f"; install with: {install_hint}"
        super().__init__(msg)
        self.feature = feature
        self.dependency = dependency
