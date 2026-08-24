"""Qwen3.5 adapter with structure discovery isolated from generic training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import AdapterContractError, AdapterProbe, ModelStructure
from . import _hf


class Qwen35Adapter:
    name = "qwen3_5"
    model_types = frozenset({"qwen3_5", "qwen3_5_moe"})

    def probe(self, model_id: str | Path, *, local_files_only: bool = True) -> AdapterProbe:
        config = _hf.auto_config(model_id, local_files_only=local_files_only)
        model_type = str(getattr(config, "model_type", ""))
        identity = _hf.identity_from_config(config)
        if model_type not in self.model_types:
            return AdapterProbe(self.name, identity, details={"rejected_model_type": model_type}, missing_capabilities=("model_type=qwen3_5",))
        processor = _hf.auto_processor(model_id, local_files_only=local_files_only)
        caps, missing, details = _hf.probe_processor(processor)
        caps.update({"forward_labels", "language_backbone", "vision_backbone", "structure_discovery"})
        return AdapterProbe(self.name, _hf.identity_from_config(config, processor_class=type(processor).__name__), frozenset(caps), tuple(missing), details)

    def load(self, model_id: str | Path, *, dtype: str = "auto", device: str = "auto", local_files_only: bool = True) -> tuple[Any, Any, AdapterProbe]:
        probe = self.probe(model_id, local_files_only=local_files_only)
        t = _hf.transformers()
        model_factory = getattr(t, "AutoModelForImageTextToText", None)
        if model_factory is None:
            raise AdapterContractError("Transformers has no generic image-text model loader for qwen3_5")
        kwargs: dict[str, Any] = {"local_files_only": local_files_only, "trust_remote_code": True}
        if dtype != "auto":
            import torch
            kwargs["dtype"] = getattr(torch, dtype)
        if device != "auto":
            kwargs["device_map"] = device
        model = model_factory.from_pretrained(model_id, **kwargs)
        processor = _hf.auto_processor(model_id, local_files_only=local_files_only)
        return model, processor, probe

    def encode(self, processor: Any, episode: Any, *, return_tensors: str = "pt") -> dict[str, Any]:
        return _hf.encode_episode(processor, episode, return_tensors=return_tensors)

    def discover_structure(self, model: Any) -> ModelStructure:
        modules = _hf.module_map(model)
        language = _find_language_root(modules)
        vision = _find_vision_root(modules)
        language_prefix = language + "." if language else ""
        target_leaf_names = {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj", "out_proj", "z_proj", "b_proj", "dt_proj",
        }
        targets = []
        for name, module in _hf.child_paths(modules, language):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in target_leaf_names and _hf.has_parameters(module):
                targets.append(name)
        connectors = []
        for name, module in _hf.child_paths(modules, vision):
            leaf = name.rsplit(".", 1)[-1].lower()
            if _hf.has_parameters(module) and any(token in leaf for token in ("merger", "projector", "connector")):
                connectors.append(name)
        if not targets:
            raise AdapterContractError("qwen3_5 adapter could not discover safe language targets")
        if not connectors:
            raise AdapterContractError("qwen3_5 adapter could not discover a safe vision connector/projector")
        return ModelStructure(language, vision, {"language_lora_targets": tuple(targets), "vision_connectors": tuple(connectors)}, {"adapter": self.name, "discovery": "semantic-module-scan"})

    def prepare_forward_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        return dict(batch)

    def apply_tuning_policy(self, model: Any, parameter_plan: Any, policy: Any) -> Any:
        return _hf.apply_tuning_policy(model, parameter_plan, policy)

    def validate_trainable_parameters(self, model: Any, parameter_plan: Any) -> None:
        _hf.validate_trainable_parameters(model, parameter_plan)

    def save_trainable_state(self, model: Any, output_path: str | Path) -> None:
        _hf.save_trainable_state(model, output_path)

    def save_processor(self, processor: Any, output_dir: str | Path) -> None:
        _hf.save_processor(processor, output_dir)

    def save_checkpoint(self, model: Any, processor: Any, output_dir: str | Path) -> None:
        _hf.save_checkpoint(model, processor, output_dir)

    def export_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        return _hf.export_peft_checkpoint(self, **kwargs)


def _find_language_root(modules: dict[str, Any]) -> str:
    preferred = ("model.language_model", "language_model", "model.text_model", "text_model", "model.decoder", "decoder")
    for name in preferred:
        module = modules.get(name)
        if module is not None and _has_layer_and_embedding(module):
            return name
    for name, module in sorted(modules.items(), key=lambda item: (item[0].count("."), item[0])):
        if _has_layer_and_embedding(module):
            return name
    raise AdapterContractError("qwen3_5 adapter could not locate a language backbone")


def _find_vision_root(modules: dict[str, Any]) -> str:
    preferred = ("model.visual", "visual", "model.vision_tower", "vision_tower", "model.vision_model", "vision_model")
    for name in preferred:
        if name in modules:
            return name
    for name in modules:
        if any(token in name.lower().split(".")[-1] for token in ("visual", "vision")):
            return name
    raise AdapterContractError("qwen3_5 adapter could not locate a vision backbone")


def _has_layer_and_embedding(module: Any) -> bool:
    try:
        children = dict(module.named_children())
        return "layers" in children and ("embed_tokens" in children or "embeddings" in children)
    except AttributeError:
        return False


# Public spelling retained for callers that mirror the model-family name.
Qwen3_5Adapter = Qwen35Adapter
