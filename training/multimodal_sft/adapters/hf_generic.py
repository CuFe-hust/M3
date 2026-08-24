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

    def encode(self, processor: Any, episode: Any, *, max_seq_length: int = 4096, return_tensors: str = "pt") -> dict[str, Any]:
        raise AdapterContractError("hf_generic_multimodal cannot encode without a proven processor contract")

    def collate(self, encoded_examples: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        raise AdapterContractError("hf_generic_multimodal cannot collate without a proven processor contract")

    def processor_identity(self, processor: Any) -> dict[str, Any]:
        return {"class": f"{type(processor).__module__}.{type(processor).__name__}", "encoding_contract_version": "unproven"}

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

    def validate_trainable_parameters(self, model: Any, parameter_plan: Any) -> None:
        _hf.validate_trainable_parameters(model, parameter_plan)

    def validate_optimizer_parameters(self, model: Any, parameter_plan: Any, groups: Any) -> None:
        _hf.validate_optimizer_parameters(model, parameter_plan, groups)

    def save_trainable_state(self, model: Any, output_path: str | Path, parameter_plan: Any) -> None:
        _hf.save_trainable_state(model, output_path, parameter_plan)

    def restore_trainable_state(self, *, model: Any, checkpoint_dir: str | Path, parameter_plan: Any, manifest: dict[str, Any]) -> Any:
        return _hf.restore_trainable_state(model=model, checkpoint_dir=checkpoint_dir, parameter_plan=parameter_plan, manifest=manifest)

    def validate_checkpoint_state(self, checkpoint_dir: str | Path, parameter_plan: Any) -> dict[str, Any]:
        return _hf.validate_checkpoint_state(checkpoint_dir, parameter_plan)

    def validate_checkpoint_ownership(self, model: Any, checkpoint_dir: str | Path, parameter_plan: Any) -> dict[str, Any]:
        return _hf.validate_checkpoint_ownership(model, checkpoint_dir, parameter_plan)

    def export_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        raise AdapterContractError("generic exporter requires an adapter-specific checkpoint implementation")
