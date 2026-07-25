"""Counting backend protocol and shared types. / 计数后端协议与共享类型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from spacers_agent.schemas import CountTargetSpec, CountingResult, UnifiedSample


@dataclass(frozen=True)
class CountingRequest:
    """Immutable counting request for one sample. / 单条样本的不可变计数请求。"""
    sample: UnifiedSample
    image: Image.Image
    target: CountTargetSpec
    artifact_dir: Path


@dataclass(frozen=True)
class BackendSelection:
    """Reason why a specific backend was chosen. / 选择特定后端的原因。"""
    backend_name: str
    reason_codes: tuple[str, ...]
    target_classes: tuple[str, ...] = ()


class CountingBackend(Protocol):
    """Execution contract for a pluggable counting backend. / 可插拔计数后端的执行契约。"""

    name: str

    def is_available(self) -> bool:
        """Return whether the backend is ready (weights loaded, client alive). / 返回后端是否就绪。"""
        ...

    def supports(self, target: CountTargetSpec) -> bool:
        """Return whether this backend can handle the target. / 返回后端能否处理目标。"""
        ...

    async def count(self, request: CountingRequest, context: object) -> CountingResult:
        """Execute counting and return validated CountingResult. / 执行计数并返回已验证的 CountingResult。"""
        ...
