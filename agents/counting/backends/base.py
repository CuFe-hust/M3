"""Counting backend protocol and shared types.

计数后端协议与共享类型。本模块只定义协议与数据类型，不实现任何选择策略、
不导入旧包、不加载权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from agents.counting.schema import CountTargetSpec, CountingResult
from agents.schema import AgentResult
from data.schema import UnifiedSample


@dataclass(frozen=True)
class CountingRequest:
    """Immutable counting request for one sample.
    单条样本的不可变计数请求。"""

    sample: UnifiedSample
    image: Image.Image
    target: CountTargetSpec
    artifact_dir: Path


@dataclass(frozen=True)
class BackendSelection:
    """Reason why a specific backend was chosen.
    选择特定后端的原因。"""

    backend_name: str
    reason_codes: tuple[str, ...]
    target_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendPlan:
    """Primary and fallback counting backends selected before execution.
    在执行前选定的主计数后端与回退后端。"""

    primary_backend_name: str
    fallback_backend_names: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    target_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CountingBackendOutcome:
    """Validated backend result plus an optional VQA Agent result.
    已校验的后端结果，以及可选的 VQA Agent 结果。"""

    counting: CountingResult
    agent_result: AgentResult | None = None
    trace: dict[str, object] | None = None


class CountingBackend(Protocol):
    """Execution contract for a pluggable counting backend. Backend names are
    stable, dataset-neutral identifiers (never a concrete dataset name).
    可插拔计数后端的执行契约。后端名是稳定、与数据集无关的标识符（绝不
    使用具体数据集名）。"""

    name: str
    priority: int

    def is_available(self) -> bool:
        """Return whether the backend is ready (weights loaded, client alive).
        返回后端是否就绪（权重已加载、客户端存活）。"""
        ...

    def supports(
        self,
        target: CountTargetSpec,
        hints: Any | None = None,
    ) -> bool:
        """Return whether this backend can handle the target under optional
        neutral hints (never dataset names).
        返回后端能否在可选中性 hints（绝非数据集名）下处理目标。"""
        ...

    async def count(
        self,
        request: CountingRequest,
        context: object,
    ) -> CountingBackendOutcome:
        """Execute counting and return one validated backend outcome.
        执行计数并返回一个已校验的后端结果。"""
        ...
