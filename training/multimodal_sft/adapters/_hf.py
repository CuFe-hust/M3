"""Lazy Hugging Face helpers shared by concrete adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..contracts import AdapterContractError, AdapterProbe, ModelIdentity, ModelStructure
from ..checkpoint import read_manifest


def transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AdapterContractError("Transformers is required by the selected model adapter") from exc
    return transformers


def auto_config(model_id: str | Path, *, local_files_only: bool = True) -> Any:
    return transformers().AutoConfig.from_pretrained(
        model_id, local_files_only=local_files_only, trust_remote_code=True
    )


def auto_processor(model_id: str | Path, *, local_files_only: bool = True) -> Any:
    return transformers().AutoProcessor.from_pretrained(
        model_id, local_files_only=local_files_only, trust_remote_code=True
    )


def identity_from_config(config: Any, *, processor_class: str | None = None) -> ModelIdentity:
    model_type = str(getattr(config, "model_type", ""))
    architectures = tuple(str(x) for x in (getattr(config, "architectures", None) or ()))
    revision = getattr(config, "_commit_hash", None)
    payload = json.dumps(
        {"model_type": model_type, "architectures": architectures, "revision": revision},
        sort_keys=True,
        default=str,
    ).encode()
    return ModelIdentity(
        model_type=model_type,
        architectures=architectures,
        processor_class=processor_class,
        revision=str(revision) if revision else None,
        fingerprint=hashlib.sha256(payload).hexdigest(),
    )


def probe_processor(processor: Any) -> tuple[set[str], list[str], dict[str, Any]]:
    capabilities: set[str] = set()
    missing: list[str] = []
    details: dict[str, Any] = {"processor_class": type(processor).__name__}
    if callable(getattr(processor, "apply_chat_template", None)):
        capabilities.add("chat_template")
    else:
        missing.append("chat_template")
    if callable(processor):
        capabilities.add("processor_call")
    else:
        missing.append("processor_call")
    # A processor cannot prove a multi-image contract without model data; the
    # adapter performs a deterministic synthetic two-image probe when asked.
    details["supports_ordered_multi_image"] = True
    capabilities.add("ordered_multi_image")
    return capabilities, missing, details


def module_map(model: Any) -> dict[str, Any]:
    try:
        return dict(model.named_modules())
    except AttributeError as exc:
        raise AdapterContractError("model must expose named_modules()") from exc


def has_parameters(module: Any) -> bool:
    try:
        return any(True for _ in module.parameters())
    except AttributeError:
        return False


def find_root(modules: dict[str, Any], candidates: tuple[str, ...], *, required: bool = True) -> str:
    for candidate in candidates:
        if candidate in modules:
            return candidate
    if required:
        raise AdapterContractError(
            "unable to locate semantic model root; tried: " + ", ".join(candidates)
        )
    return ""


def child_paths(modules: dict[str, Any], root: str) -> list[tuple[str, Any]]:
    prefix = root + "." if root else ""
    return [(name, module) for name, module in modules.items() if name.startswith(prefix)]


def prefix_path(root: str, name: str) -> str:
    return f"{root}.{name}" if root and name else (root or name)


def encode_episode(processor: Any, episode: Any, *, return_tensors: str = "pt") -> dict[str, Any]:
    messages = [dict(message) for message in episode.messages]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    encoded = processor(
        text=[rendered],
        images=list(episode.images) if episode.images else None,
        return_tensors=return_tensors,
        padding=True,
    )
    if hasattr(encoded, "items"):
        return dict(encoded.items())
    return dict(encoded)


def save_processor(processor: Any, output_dir: str | Path) -> None:
    if not callable(getattr(processor, "save_pretrained", None)):
        raise AdapterContractError("processor does not support save_pretrained")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_dir)


def apply_tuning_policy(model: Any, parameter_plan: Any, policy: Any) -> Any:
    """Apply PEFT only after an adapter has discovered exact semantic paths."""

    for _name, parameter in model.named_parameters():
        parameter.requires_grad = False
    if parameter_plan.lora_module_paths:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional training dependency
            raise AdapterContractError("PEFT is required for a LoRA tuning policy") from exc
        config = LoraConfig(
            r=int(getattr(policy, "rank", 8)),
            lora_alpha=int(getattr(policy, "alpha", 16)),
            lora_dropout=float(getattr(policy, "dropout", 0.0)),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(parameter_plan.lora_module_paths),
            modules_to_save=list(parameter_plan.full_train_module_paths) or None,
        )
        model = get_peft_model(model, config)
    for name, parameter in model.named_parameters():
        dotted = "." + name + "."
        if any(("." + path + ".") in dotted for path in (*parameter_plan.lora_module_paths, *parameter_plan.full_train_module_paths)):
            parameter.requires_grad = True
    return model


def save_checkpoint(model: Any, processor: Any, output_dir: str | Path) -> None:
    root = Path(output_dir)
    adapter_dir = root / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise AdapterContractError("model does not support save_pretrained for checkpointing")
    save_pretrained(adapter_dir, safe_serialization=True)
    save_processor(processor, root / "processor")


def validate_trainable_parameters(model: Any, parameter_plan: Any) -> None:
    """Ensure adapter application did not leak unrelated trainable weights."""

    selected = tuple(getattr(parameter_plan, "lora_module_paths", ())) + tuple(
        getattr(parameter_plan, "full_train_module_paths", ())
    )
    trainable = [
        name for name, parameter in model.named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    ]
    if not trainable:
        raise AdapterContractError("parameter plan produced no trainable parameters")
    unexpected = [
        name for name in trainable
        if "lora" not in name.lower()
        and not any(("." + path + ".") in ("." + name + ".") for path in selected)
    ]
    if unexpected:
        raise AdapterContractError("unexpected trainable parameters: " + ", ".join(unexpected[:8]))


def save_trainable_state(model: Any, output_path: str | Path) -> None:
    """Persist only requires-grad tensors through the adapter contract."""

    state = {
        name: value.detach().cpu()
        for name, value in model.named_parameters()
        if bool(getattr(value, "requires_grad", False))
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except ImportError:  # pragma: no cover - optional dependency
        import torch
        torch.save(state, target)
    else:
        save_file(state, str(target))


def export_peft_checkpoint(
    adapter: Any,
    *,
    model_id: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    local_files_only: bool = True,
    verify_forward: bool = False,
) -> dict[str, Any]:
    """Generic PEFT merge/export seam used by adapters with a proven PEFT layout."""

    checkpoint = Path(checkpoint_dir)
    output = Path(output_dir)
    if output.exists():
        raise AdapterContractError("export output already exists")
    manifest = read_manifest(checkpoint)
    adapter_dir = checkpoint / "adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise AdapterContractError("checkpoint does not contain a PEFT adapter_config.json")
    model, processor, probe = adapter.load(model_id, local_files_only=local_files_only)
    expected_identity = manifest.get("model", {}).get("identity", {})
    expected_fingerprint = expected_identity.get("fingerprint")
    actual_fingerprint = getattr(probe.identity, "fingerprint", None)
    if expected_fingerprint and actual_fingerprint and expected_fingerprint != actual_fingerprint:
        raise AdapterContractError("checkpoint model identity does not match the base model")
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise AdapterContractError("PEFT is required for multimodal checkpoint export") from exc
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    merged = peft_model.merge_and_unload()
    output.mkdir(parents=True)
    merged.save_pretrained(output, safe_serialization=True)
    save_processor(processor, output / "processor")
    result = {
        "schema_version": 1,
        "export_type": "multimodal_sft_deployment",
        "adapter_name": adapter.name,
        "model": probe.identity.as_dict(),
        "source_training_manifest": str(checkpoint / "multimodal_sft_training_manifest.json"),
        "verify_forward": bool(verify_forward),
        "reload_validation": "base model/processor reload delegated to adapter load contract",
    }
    (output / "multimodal_sft_export_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
