"""Stable contracts shared by task profiles, model adapters and the trainer.

Nothing in this module imports Transformers, PyTorch, PEFT or a model family.
That boundary is intentional: a new model family must be implemented in an
adapter, not by adding branches to the generic trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable


class AdapterContractError(ValueError):
    """A model adapter cannot prove a required capability safely."""


@dataclass(frozen=True)
class ModelIdentity:
    """Logical identity of a loaded/probed model, independent of its path."""

    model_type: str
    architectures: tuple[str, ...] = ()
    processor_class: str | None = None
    revision: str | None = None
    fingerprint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "processor_class": self.processor_class,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class AdapterProbe:
    """A no-weight capability probe result."""

    adapter_name: str
    identity: ModelIdentity
    capabilities: frozenset[str] = frozenset()
    missing_capabilities: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.missing_capabilities

    def require(self, *capabilities: str) -> None:
        missing = [name for name in capabilities if name not in self.capabilities]
        if missing:
            raise AdapterContractError(
                f"adapter {self.adapter_name!r} is missing capabilities: {', '.join(missing)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "identity": self.identity.as_dict(),
            "capabilities": sorted(self.capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "details": dict(self.details),
            "passed": self.passed,
        }


CapabilityReport = AdapterProbe


@dataclass(frozen=True)
class ModelStructure:
    """Semantic module map discovered by an adapter.

    Values are exact paths in the concrete model, but the generic policy only
    consumes semantic keys such as ``language_lora_targets`` and
    ``vision_connectors``.
    """

    language_backbone: str
    vision_backbone: str
    semantic_modules: Mapping[str, tuple[str, ...]]
    details: Mapping[str, Any] = field(default_factory=dict)

    def paths_for(self, role: str) -> tuple[str, ...]:
        return tuple(self.semantic_modules.get(role, ()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "language_backbone": self.language_backbone,
            "vision_backbone": self.vision_backbone,
            "semantic_modules": {
                key: list(value) for key, value in self.semantic_modules.items()
            },
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CanonicalEpisode:
    """Model-neutral training episode.

    ``messages`` are already canonical role/content messages and ``images``
    are ordered T1/T2 or ordinary image values.  Adapters own serialization
    and processor-specific placeholder handling.
    """

    task_profile: str
    messages: tuple[Mapping[str, Any], ...]
    images: tuple[Any, ...] = ()
    target_schema: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DataProfile(Protocol):
    """Task/data contract consumed by the generic trainer."""

    name: str

    def read(self, path: str | Path) -> Iterable[CanonicalEpisode]:
        """Read canonical episodes without knowing a model family."""

    def validate(self, episode: CanonicalEpisode) -> None:
        """Fail closed when the task contract is not satisfied."""


@runtime_checkable
class MultimodalModelAdapter(Protocol):
    """Adapter seam between generic SFT code and a concrete model family."""

    name: str

    def probe(self, model_id: str | Path, *, local_files_only: bool = True) -> AdapterProbe:
        ...

    def load(
        self,
        model_id: str | Path,
        *,
        dtype: str = "auto",
        device: str = "auto",
        local_files_only: bool = True,
    ) -> tuple[Any, Any, AdapterProbe]:
        ...

    def encode(
        self,
        processor: Any,
        episode: CanonicalEpisode,
        *,
        return_tensors: str = "pt",
    ) -> Mapping[str, Any]:
        ...

    def discover_structure(self, model: Any) -> ModelStructure:
        ...

    def apply_tuning_policy(self, model: Any, parameter_plan: Any, policy: Any) -> Any:
        """Attach model-specific LoRA/projector mechanics for a semantic plan."""
        ...

    def prepare_forward_inputs(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def save_processor(self, processor: Any, output_dir: str | Path) -> None:
        ...

    def save_checkpoint(self, model: Any, processor: Any, output_dir: str | Path) -> None:
        ...

    def validate_trainable_parameters(self, model: Any, parameter_plan: Any) -> None:
        ...

    def save_trainable_state(self, model: Any, output_path: str | Path) -> None:
        ...

    def export_checkpoint(
        self,
        *,
        model_id: str | Path,
        checkpoint_dir: str | Path,
        output_dir: str | Path,
        local_files_only: bool = True,
        verify_forward: bool = False,
    ) -> Mapping[str, Any]:
        ...
