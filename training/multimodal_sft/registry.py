"""Explicit model-adapter registry with fail-closed ``auto`` selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .contracts import AdapterProbe, MultimodalModelAdapter


class UnsupportedModelAdapter(ValueError):
    """No registered adapter can safely satisfy the requested model."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


AdapterFactory = Callable[[], MultimodalModelAdapter]


@dataclass(frozen=True)
class RegisteredAdapter:
    name: str
    factory: AdapterFactory
    model_types: frozenset[str] = frozenset()


class AdapterRegistry:
    """Registry for concrete adapter factories.

    ``auto`` only chooses an adapter after a concrete identity match.  It does
    not fall back to a similar model or silently select the generic adapter.
    """

    def __init__(self, *, include_builtins: bool = True) -> None:
        self._entries: dict[str, RegisteredAdapter] = {}
        if include_builtins:
            self._register_builtins()

    def register(
        self,
        name: str,
        factory: AdapterFactory,
        *,
        model_types: Iterable[str] = (),
        replace: bool = False,
    ) -> None:
        key = name.strip()
        if not key:
            raise ValueError("adapter name must be non-empty")
        if key in self._entries and not replace:
            raise ValueError(f"adapter already registered: {key}")
        self._entries[key] = RegisteredAdapter(key, factory, frozenset(model_types))

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def get(self, name: str) -> MultimodalModelAdapter:
        try:
            adapter = self._entries[name].factory()
        except KeyError as exc:
            raise UnsupportedModelAdapter(
                f"unknown model adapter {name!r}; available: {', '.join(self.available())}"
            ) from exc
        if not isinstance(adapter, MultimodalModelAdapter):
            raise UnsupportedModelAdapter(
                "ADAPTER_CONTRACT_INCOMPLETE",
                details={"adapter": name},
            )
        return adapter

    def probe(
        self,
        model_id: str | Path,
        *,
        model_adapter: str = "auto",
        local_files_only: bool = True,
    ) -> AdapterProbe:
        if model_adapter != "auto":
            adapter = self.get(model_adapter)
            probe = adapter.probe(model_id, local_files_only=local_files_only)
            if not probe.passed:
                raise UnsupportedModelAdapter(
                    f"adapter {model_adapter!r} rejected the model: "
                    f"{', '.join(probe.missing_capabilities)}",
                    details=probe.as_dict(),
                )
            return probe

        failures: list[dict[str, Any]] = []
        for entry in self._entries.values():
            try:
                probe = self.get(entry.name).probe(model_id, local_files_only=local_files_only)
            except Exception as exc:  # adapter probes must not make auto selection guess
                failures.append({"adapter": entry.name, "error": type(exc).__name__})
                continue
            if probe.passed:
                return probe
            failures.append({"adapter": entry.name, "probe": probe.as_dict()})
        raise UnsupportedModelAdapter(
            "UNSUPPORTED_MODEL_ADAPTER: no registered adapter passed the capability probe; "
            f"available_adapters={self.available()}; probe_failures={failures!r}",
            details={"model_id": str(model_id), "available_adapters": self.available(), "failures": failures},
        )

    def resolve(
        self,
        model_id: str | Path,
        *,
        model_adapter: str = "auto",
        local_files_only: bool = True,
    ) -> tuple[MultimodalModelAdapter, AdapterProbe]:
        probe = self.probe(
            model_id, model_adapter=model_adapter, local_files_only=local_files_only
        )
        return self.get(probe.adapter_name), probe

    def _register_builtins(self) -> None:
        # Imports are intentionally delayed until a registry is constructed.
        from .adapters.qwen3_5 import Qwen35Adapter
        from .adapters.qwen3_vl import Qwen3VLAdapter
        from .adapters.hf_generic import GenericHFAdapter

        self.register("qwen3_vl", Qwen3VLAdapter, model_types={"qwen3_vl"})
        self.register("qwen3_5", Qwen35Adapter, model_types={"qwen3_5"})
        self.register("hf_generic_multimodal", GenericHFAdapter)


_DEFAULT_REGISTRY: AdapterRegistry | None = None


def default_registry() -> AdapterRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = AdapterRegistry()
    return _DEFAULT_REGISTRY
