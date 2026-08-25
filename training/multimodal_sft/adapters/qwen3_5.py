"""Qwen3.5 adapter with structure discovery isolated from generic training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..contracts import AdapterContractError, AdapterProbe, CanonicalEpisode, ModelStructure, PreparedMultimodalEpisode
from . import _hf
from . import qwen_multimodal


class Qwen35Adapter:
    name = "qwen3_5"
    model_types = frozenset({"qwen3_5"})

    def probe(self, model_id: str | Path, *, local_files_only: bool = True) -> AdapterProbe:
        config = _hf.auto_config(model_id, local_files_only=local_files_only)
        model_type = str(getattr(config, "model_type", ""))
        identity = _hf.identity_from_config(config)
        if model_type not in self.model_types:
            return AdapterProbe(self.name, identity, details={"rejected_model_type": model_type}, missing_capabilities=("model_type=qwen3_5",))
        processor = _hf.auto_processor(model_id, local_files_only=local_files_only)
        caps, missing, details = qwen_multimodal.probe_processor(processor)
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

    def encode(self, processor: Any, episode: PreparedMultimodalEpisode | CanonicalEpisode, *, max_seq_length: int = 4096, return_tensors: str = "pt") -> Mapping[str, Any]:
        return qwen_multimodal.encode_multimodal_episode(processor, episode, max_seq_length=max_seq_length, return_tensors=return_tensors)

    def collate(self, encoded_examples: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return qwen_multimodal.collate(encoded_examples)

    def processor_identity(self, processor: Any) -> dict[str, Any]:
        return qwen_multimodal.processor_identity(processor)

    def load_processor(self, processor_dir: str | Path, *, local_files_only: bool = True) -> Any:
        return _hf.auto_processor(processor_dir, local_files_only=local_files_only)

    def saved_processor_identity(self, processor: Any, processor_dir: str | Path) -> dict[str, Any]:
        return qwen_multimodal.saved_processor_identity(processor, str(processor_dir))

    def discover_structure(self, model: Any) -> ModelStructure:
        modules = _hf.module_map(model)
        language = _find_language_root(modules)
        vision = _find_vision_root(modules)
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", None)
        layer_types = list(getattr(text_config, "layer_types", ()) or ())
        num_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
        if not layer_types or len(layer_types) != num_layers:
            raise AdapterContractError("QWEN35_LAYER_TYPES_MISMATCH")
        unsupported = sorted(set(layer_types) - {"linear_attention", "full_attention"})
        if unsupported:
            raise AdapterContractError(f"QWEN35_UNSUPPORTED_LAYER_TYPE: {unsupported}")
        language_module = modules[language]
        layers = getattr(language_module, "layers", None)
        if layers is None or len(list(layers)) != num_layers:
            raise AdapterContractError("QWEN35_DECODER_COUNT_MISMATCH")
        targets: list[str] = []
        full_indexes: list[int] = []
        linear_indexes: list[int] = []
        for index, layer_type in enumerate(layer_types):
            prefix = f"{language}.layers.{index}"
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                path = f"{prefix}.mlp.{leaf}"
                self._require_parameter_module(modules, path, "QWEN35_MLP_TOPOLOGY_MISMATCH")
                targets.append(path)
            if layer_type == "full_attention":
                full_indexes.append(index)
                leaves = ("q_proj", "k_proj", "v_proj", "o_proj")
                code = "QWEN35_FULL_ATTENTION_TOPOLOGY_MISMATCH"
                root = "self_attn"
            else:
                linear_indexes.append(index)
                leaves = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
                code = "QWEN35_LINEAR_ATTENTION_TOPOLOGY_MISMATCH"
                root = "linear_attn"
            for leaf in leaves:
                path = f"{prefix}.{root}.{leaf}"
                self._require_parameter_module(modules, path, code)
                targets.append(path)
        connector = f"{vision}.merger"
        self._require_parameter_module(modules, connector, "QWEN35_VISION_MERGER_MISSING")
        deepstack_paths = tuple(name for name in modules if name.startswith(vision + ".") and "deepstack" in name.lower())
        if deepstack_paths:
            raise AdapterContractError("QWEN35_UNEXPECTED_DEEPSTACK")
        layer_sha = hashlib.sha256(json.dumps(layer_types, separators=(",", ":")).encode("utf-8")).hexdigest()
        expected = 3 * num_layers + 4 * len(full_indexes) + 5 * len(linear_indexes)
        if len(targets) != expected or len(set(targets)) != expected:
            raise AdapterContractError("QWEN35_TARGET_COUNT_MISMATCH")
        details = {
            "adapter_contract_version": "qwen35_strict_v1",
            "discovery": "layer_types_exact",
            "num_hidden_layers": num_layers,
            "layer_types_sha256": layer_sha,
            "full_attention_layer_indexes": full_indexes,
            "linear_attention_layer_indexes": linear_indexes,
            "expected_target_count": expected,
            "actual_target_count": len(targets),
            "exact_vision_connector": connector,
            "deepstack_expected": False,
            "deepstack_present": False,
        }
        return ModelStructure(language, vision, {"language_lora_targets": tuple(targets), "vision_connectors": (connector,)}, details)

    @staticmethod
    def _require_parameter_module(modules: dict[str, Any], path: str, code: str) -> None:
        module = modules.get(path)
        if module is None or not _hf.has_parameters(module):
            raise AdapterContractError(f"{code}: {path}")

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

    def export_checkpoint(
        self,
        *,
        model_id: str | Path,
        checkpoint_dir: str | Path,
        output_dir: str | Path,
        local_files_only: bool = True,
        verify_forward: bool = False,
        change_fixture: str | Path | None = None,
    ) -> Mapping[str, Any]:
        return _hf.export_peft_checkpoint(
            self,
            model_id=model_id,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            local_files_only=local_files_only,
            verify_forward=verify_forward,
            change_fixture=change_fixture,
        )

    def reload_exported(self, output_dir: str | Path, *, local_files_only: bool = True) -> tuple[Any, Any]:
        t = _hf.transformers()
        factory = getattr(t, "AutoModelForImageTextToText", None)
        if factory is None:
            raise AdapterContractError("Transformers has no generic image-text model loader for exported qwen3_5")
        model = factory.from_pretrained(output_dir, local_files_only=local_files_only, trust_remote_code=True)
        processor = _hf.auto_processor(output_dir, local_files_only=local_files_only)
        return model, processor

    def verify_export_forward(self, model: Any, processor: Any, *, change_fixture: str | Path | None = None) -> dict[str, Any]:
        return qwen_multimodal.verify_export_forward(self, model, processor, change_fixture=change_fixture)


def _find_language_root(modules: dict[str, Any]) -> str:
    candidates = [name for name in ("model.language_model", "language_model") if name in modules]
    if len(candidates) != 1:
        raise AdapterContractError(f"QWEN35_LANGUAGE_ROOT_AMBIGUOUS: {candidates}")
    return candidates[0]


def _find_vision_root(modules: dict[str, Any]) -> str:
    candidates = [name for name in ("model.visual", "visual") if name in modules]
    if len(candidates) != 1:
        raise AdapterContractError(f"QWEN35_VISION_ROOT_AMBIGUOUS: {candidates}")
    return candidates[0]


def _has_layer_and_embedding(module: Any) -> bool:
    try:
        children = dict(module.named_children())
        return "layers" in children and ("embed_tokens" in children or "embeddings" in children)
    except AttributeError:
        return False


# Public spelling retained for callers that mirror the model-family name.
Qwen3_5Adapter = Qwen35Adapter
