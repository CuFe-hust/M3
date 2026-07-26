"""Counting backend registry. / 计数后端注册表。"""

from __future__ import annotations

import logging

from spacers_agent.agents.counting.backends.base import CountingBackend
from spacers_agent.schemas import CountTargetSpec

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Register and look up counting backends; never loads models. / 注册与查找计数后端；绝不加载模型。"""

    def __init__(self) -> None:
        self._backends: list[CountingBackend] = []

    def register(self, backend: CountingBackend) -> None:
        """Register a backend. / 注册后端。"""
        if any(existing.name == backend.name for existing in self._backends):
            raise ValueError(f"Duplicate counting backend: {backend.name}")
        self._backends.append(backend)

    def get(self, name: str) -> CountingBackend:
        """Return one backend by its stable name.
        按稳定名称返回一个后端。
        """

        for backend in self._backends:
            if backend.name == name:
                return backend
        raise KeyError(f"Unknown counting backend {name!r}; available={self.all_names()}")

    def list_available(self, target: CountTargetSpec) -> list[CountingBackend]:
        """Return backends that can handle the target. / 返回能处理目标的后端。"""
        return [b for b in self._backends if b.is_available() and b.supports(target)]

    def all_names(self) -> list[str]:
        """Return all registered backend names. / 返回所有注册后端名。"""
        return [b.name for b in self._backends]

    def __len__(self) -> int:
        return len(self._backends)
