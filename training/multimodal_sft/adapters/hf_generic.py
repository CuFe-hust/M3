"""Strict opt-in Hugging Face generic adapter.

This adapter is never selected by ``auto``.  It exists as a safe extension
point for a model whose capabilities have been manually proven.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import AdapterContractError, AdapterProbe, ModelStructure
from . import _hf


class GenericHFAdapter:
    name = "hf_generic_multimodal"

    def probe(self, model_id: str | Path, *, local_files_only: bool = True) -> AdapterProbe:
        config = _hf.auto_config(model_id, local_files_only=local_files_only)
        processor = _hf.auto_processor(model_id, local_files_only=local_files_only)
        caps, missing, details = _hf.probe_processor(processor)
        architectures = tuple(str(x) for x in (getattr(config, "architectures", None) or ()))
        details.update({"architectures": architectures, "model_type": getattr(config, "model_type", None)})
        # The generic adapter deliberately cannot prove module semantics from a
        # config-only probe; loading and structure discovery must be explicit.
        missing = list(missing) + ["language_backbone", "vision_backbone", "safe_connector"]
        return AdapterProbe(self.name, _hf.identity_from_config(config, processor_class=type(processor).__name__), frozenset(caps), tuple(missing), details)

    def load(self, model_id: str | Path, *, dtype: str = "auto", device: str = "auto", local_files_only: bool = True) -> tuple[Any, Any, AdapterProbe]:
        probe = self.probe(model_id, local_files_only=local_files_only)
        raise AdapterContractError("generic adapter requires an explicit proven structure; it will not guess module paths")

    def encode(self, processor: Any, episode: Any, *, return_tensors: str = "pt") -> dict[str, Any]:
        return _hf.encode_episode(processor, episode, return_tensors=return_tensors)

    def discover_structure(self, model: Any) -> ModelStructure:
        raise AdapterContractError("generic adapter cannot infer a safe model structure")

    def prepare_forward_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        return dict(batch)

    def apply_tuning_policy(self, model: Any, parameter_plan: Any, policy: Any) -> Any:
        raise AdapterContractError("generic adapter cannot attach LoRA without a proven model-specific seam")

    def save_processor(self, processor: Any, output_dir: str | Path) -> None:
        _hf.save_processor(processor, output_dir)

    def save_checkpoint(self, model: Any, processor: Any, output_dir: str | Path) -> None:
        raise AdapterContractError("generic adapter cannot save a checkpoint without a proven model-family seam")

    def export_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        raise AdapterContractError("generic exporter requires an adapter-specific checkpoint implementation")
