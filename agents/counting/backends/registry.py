"""Counting backend registry — stable order, duplicate detection, lookups.

计数后端注册表 — 稳定注册顺序、重复检测与查找。注册表只登记后端对象，
绝不加载权重；后端名必须是中性标识符，不得含具体数据集名。
"""

from __future__ import annotations

from typing import Any

from agents.counting.backends.base import CountingBackend
from agents.counting.schema import CountTargetSpec

# Backend names must never embed a concrete dataset identity.
# 后端名绝不嵌入具体数据集身份。
_DATASET_NAME_MARKERS = ("vrsbench", "dota", "xview", "isaid", "mme")


class BackendRegistry:
    """Register and look up counting backends; never loads models.
    注册与查找计数后端；绝不加载模型。"""

    def __init__(self) -> None:
        self._backends: list[CountingBackend] = []

    def register(self, backend: CountingBackend) -> None:
        """Register a backend; duplicates and dataset-embedded names fail.
        注册后端；重复名与嵌入数据集名的名称失败。"""
        if any(existing.name == backend.name for existing in self._backends):
            raise ValueError(f"Duplicate counting backend: {backend.name}")
        if not isinstance(backend.name, str) or not backend.name.strip():
            raise ValueError("Counting backend name must be a non-empty string")
        folded = backend.name.casefold()
        if any(marker in folded for marker in _DATASET_NAME_MARKERS):
            raise ValueError(
                f"Counting backend name {backend.name!r} must not contain a dataset name"
            )
        self._backends.append(backend)

    def get(self, name: str) -> CountingBackend:
        """Return one backend by its stable name.
        按稳定名称返回一个后端。"""
        for backend in self._backends:
            if backend.name == name:
                return backend
        raise KeyError(
            f"Unknown counting backend {name!r}; available={self.all_names()}"
        )

    def list_available(
        self,
        target: CountTargetSpec,
        *,
        hints: Any | None = None,
        exclude_names: frozenset[str] = frozenset(),
    ) -> list[CountingBackend]:
        """Return backends that are available and can handle the target.
        返回可用且能处理目标的后端。"""
        return [
            backend
            for backend in self._backends
            if backend.name not in exclude_names
            and backend.is_available()
            and backend.supports(target, hints=hints)
        ]

    def items(self) -> tuple[CountingBackend, ...]:
        """Return registered backends in stable registration order without
        exposing registry storage. 按稳定注册顺序返回后端，不暴露内部存储。"""
        return tuple(self._backends)

    def all_names(self) -> list[str]:
        """Return all registered backend names in registration order.
        按注册顺序返回所有后端名。"""
        return [backend.name for backend in self._backends]

    def __len__(self) -> int:
        return len(self._backends)
