"""Qwen3-VL adapter; all Qwen3-VL knowledge is isolated here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import AdapterContractError, AdapterProbe, ModelStructure
from . import _hf
from . import qwen_multimodal


class Qwen3VLAdapter:
    name = "qwen3_vl"

    def probe(self, model_id: str | Path, *, local_files_only: bool = True) -> AdapterProbe:
        config = _hf.auto_config(model_id, local_files_only=local_files_only)
        if getattr(config, "model_type", None) != "qwen3_vl":
            identity = _hf.identity_from_config(config)
            return AdapterProbe(self.name, identity, details={"rejected_model_type": getattr(config, "model_type", None)}, missing_capabilities=("model_type=qwen3_vl",))
        processor = _hf.auto_processor(model_id, local_files_only=local_files_only)
        caps, missing, details = qwen_multimodal.probe_processor(processor)
        caps.update({"forward_labels", "language_backbone", "vision_backbone", "semantic_connectors"})
        return AdapterProbe(self.name, _hf.identity_from_config(config, processor_class=type(processor).__name__), frozenset(caps), tuple(missing), details)

    def load(self, model_id: str | Path, *, dtype: str = "auto", device: str = "auto", local_files_only: bool = True) -> tuple[Any, Any, AdapterProbe]:
        probe = self.probe(model_id, local_files_only=local_files_only)
        t = _hf.transformers()
        model_factory = getattr(t, "Qwen3VLForConditionalGeneration", None) or getattr(t, "AutoModelForImageTextToText", None)
        if model_factory is None:
            raise AdapterContractError("Transformers has no image-text model loader for qwen3_vl")
        kwargs: dict[str, Any] = {"local_files_only": local_files_only, "trust_remote_code": True}
        if dtype != "auto":
            import torch
            kwargs["dtype"] = getattr(torch, dtype)
        if device != "auto":
            kwargs["device_map"] = device
        model = model_factory.from_pretrained(model_id, **kwargs)
        processor = _hf.auto_processor(model_id, local_files_only=local_files_only)
        return model, processor, probe

    def encode(self, processor: Any, episode: Any, *, max_seq_length: int = 4096, return_tensors: str = "pt") -> dict[str, Any]:
        return qwen_multimodal.encode_multimodal_episode(processor, episode, max_seq_length=max_seq_length, return_tensors=return_tensors)

    def collate(self, encoded_examples: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return qwen_multimodal.collate(encoded_examples)

    def processor_identity(self, processor: Any) -> dict[str, Any]:
        return qwen_multimodal.processor_identity(processor)

    def discover_structure(self, model: Any) -> ModelStructure:
        modules = _hf.module_map(model)
        language = _hf.find_root(modules, ("model.language_model", "language_model", "model.text_model", "text_model"))
        vision = _hf.find_root(modules, ("model.visual", "visual", "model.vision_tower", "vision_tower"))
        targets = []
        connectors = []
        for name, module in _hf.child_paths(modules, language):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
                targets.append(name)
        for name, module in _hf.child_paths(modules, vision):
            leaf = name.rsplit(".", 1)[-1].lower()
            if _hf.has_parameters(module) and ("merger" in leaf or "projector" in leaf or "connector" in leaf):
                connectors.append(name)
        if not targets:
            raise AdapterContractError("qwen3_vl adapter found no language LoRA targets")
        if not connectors:
            raise AdapterContractError("qwen3_vl adapter found no vision connectors")
        return ModelStructure(language, vision, {"language_lora_targets": tuple(targets), "vision_connectors": tuple(connectors)}, {"adapter": self.name})

    def prepare_forward_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        return dict(batch)

    def apply_tuning_policy(self, model: Any, parameter_plan: Any, policy: Any) -> Any:
        return _hf.apply_tuning_policy(model, parameter_plan, policy)

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

    def save_processor(self, processor: Any, output_dir: str | Path) -> None:
        _hf.save_processor(processor, output_dir)

    def save_checkpoint(self, model: Any, processor: Any, output_dir: str | Path) -> None:
        _hf.save_checkpoint(model, processor, output_dir)

    def export_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        return _hf.export_peft_checkpoint(self, **kwargs)
