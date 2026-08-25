"""Lazy Hugging Face helpers shared by concrete adapters."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ..contracts import AdapterContractError, AdapterProbe, ModelIdentity, ModelStructure
from ..checkpoint import (
    PARAMETER_PLAN_FILENAME,
    TRAINING_MANIFEST_FILENAME,
    checkpoint_complete,
    file_sha256,
    identity_fingerprint,
    read_manifest,
)
from ..identity import (
    base_weight_identity,
    processor_content_identity,
    processor_semantic_equal,
    processor_semantic_identity,
)


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
    from . import qwen_multimodal

    return qwen_multimodal.probe_processor(processor)


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
            modules_to_save=None,
        )
        model = get_peft_model(model, config)
    lora_names = collect_lora_parameter_names(model)
    full_parameters = resolve_full_train_parameters(model, parameter_plan)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in lora_names
    for _canonical, (_actual, parameter) in full_parameters.items():
        parameter.requires_grad = True
    validate_trainable_parameters(model, parameter_plan)
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


def canonicalize_model_parameter_name(name: str) -> str:
    """Map PEFT-wrapped parameter names back to pre-PEFT semantic names."""

    value = str(name)
    while value.startswith("base_model.model.") or value.startswith("base_model."):
        for prefix in ("base_model.model.", "base_model."):
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
    value = value.replace(".modules_to_save.default", "")
    value = value.replace(".lora_A.default.", ".lora_A.")
    value = value.replace(".lora_B.default.", ".lora_B.")
    value = value.replace(".lora_embedding_A.default.", ".lora_embedding_A.")
    value = value.replace(".lora_embedding_B.default.", ".lora_embedding_B.")
    return value


def _canonical_parameter_name(name: str) -> str:
    return canonicalize_model_parameter_name(name)


def _is_lora_parameter_name(name: str) -> bool:
    segments = canonicalize_model_parameter_name(name).lower().split(".")
    return any(segment.startswith("lora_") for segment in segments)


def collect_lora_parameter_names(model: Any, *, trainable_only: bool = False) -> set[str]:
    """Return actual PEFT LoRA parameter names, not target-module parents."""

    return {
        name
        for name, parameter in model.named_parameters()
        if _is_lora_parameter_name(name)
        and (not trainable_only or bool(getattr(parameter, "requires_grad", False)))
    }


def _canonical_parameter_map(model: Any) -> dict[str, tuple[str, Any]]:
    result: dict[str, tuple[str, Any]] = {}
    for actual, parameter in model.named_parameters():
        canonical = canonicalize_model_parameter_name(actual)
        if canonical in result and result[canonical][0] != actual:
            raise AdapterContractError(f"ambiguous canonical parameter name: {canonical}")
        result[canonical] = (actual, parameter)
    return result


def resolve_full_train_parameters(model: Any, parameter_plan: Any) -> dict[str, tuple[str, Any]]:
    """Resolve exact pre-PEFT full-train names to one post-PEFT parameter."""

    by_canonical = _canonical_parameter_map(model)
    resolved: dict[str, tuple[str, Any]] = {}
    for planned in getattr(parameter_plan, "full_train_parameter_names", ()):
        canonical = canonicalize_model_parameter_name(planned)
        match = by_canonical.get(canonical)
        if match is None:
            raise AdapterContractError(f"full-train parameter cannot be resolved: {planned}")
        resolved[canonical] = match
    return resolved


def _expected_trainable_names(model: Any, parameter_plan: Any) -> tuple[set[str], set[str]]:
    lora = collect_lora_parameter_names(model)
    full = {actual for actual, _parameter in resolve_full_train_parameters(model, parameter_plan).values()}
    if lora & full:
        raise AdapterContractError("LoRA/full-train trainable sets overlap")
    return lora, full


def validate_trainable_parameters(model: Any, parameter_plan: Any) -> dict[str, Any]:
    """Require exactly LoRA tensors plus exact planned full-train tensors."""

    expected_lora, expected_full = _expected_trainable_names(model, parameter_plan)
    actual = {
        name
        for name, parameter in model.named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    }
    expected = expected_lora | expected_full
    leakage = sorted(
        name for name in actual - expected
        if ".base_layer." in ("." + name + ".") or not _is_lora_parameter_name(name)
    )
    if leakage:
        raise AdapterContractError("base_weight_trainable_leakage: " + ", ".join(leakage[:8]))
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise AdapterContractError(
            f"trainable-set mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if not actual:
        raise AdapterContractError("parameter plan produced no trainable parameters")
    return {
        "lora_parameter_names": sorted(expected_lora),
        "full_train_parameter_names": sorted(expected_full),
        "base_layer_trainable_leakage": 0,
    }


def validate_optimizer_parameters(model: Any, parameter_plan: Any, groups: Any) -> None:
    """Ensure optimizer groups contain exactly the adapter-owned trainables."""

    expected_lora, expected_full = _expected_trainable_names(model, parameter_plan)
    expected_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name in expected_lora or name in expected_full
    }
    optimizer_ids = {id(parameter) for group in groups for parameter in group.get("params", ())}
    if optimizer_ids != expected_ids:
        raise AdapterContractError("optimizer trainable-set mismatch")
    for name, parameter in model.named_parameters():
        if id(parameter) in optimizer_ids and ".base_layer." in ("." + name + "."):
            raise AdapterContractError("base_weight_trainable_leakage in optimizer")


def _load_tensor_file(path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError:  # pragma: no cover - optional dependency
        import torch
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    return dict(load_file(str(path), device="cpu"))


def canonicalize_peft_lora_state_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize only known LoRA keys; reject all non-LoRA tensors."""

    canonical: dict[str, Any] = {}
    for name, value in state.items():
        key = _canonical_parameter_name(name)
        if ".lora_" not in key.lower():
            raise AdapterContractError(f"UNEXPECTED_NON_LORA_ADAPTER_STATE: {name}")
        if key in canonical:
            raise AdapterContractError(f"duplicate canonical LoRA key: {key}")
        canonical[key] = value
    return canonical


def _adapter_state_path(root: Path) -> Path:
    for candidate in (root / "adapter" / "adapter_model.safetensors", root / "adapter" / "adapter_model.bin"):
        if candidate.is_file():
            return candidate
    raise AdapterContractError("resume checkpoint is missing adapter weights")


def validate_checkpoint_state(checkpoint_dir: str | Path, parameter_plan: Any) -> dict[str, Any]:
    """Audit state ownership and exact persisted key topology after saving."""

    root = Path(checkpoint_dir)
    adapter_state = canonicalize_peft_lora_state_keys(_load_tensor_file(_adapter_state_path(root)))
    full_state = _load_tensor_file(root / "model_trainable_state.safetensors")
    expected_full = set(getattr(parameter_plan, "full_train_parameter_names", ()))
    if set(full_state) != expected_full:
        raise AdapterContractError("full-train state keys do not match ParameterPlan.full_train_parameter_names")
    overlap = set(adapter_state) & set(full_state)
    if overlap:
        raise AdapterContractError(f"CHECKPOINT_STATE_OWNERSHIP_CONFLICT: {sorted(overlap)[:8]}")
    parents = {key.rsplit(".lora_", 1)[0] for key in adapter_state}
    planned = {_canonical_parameter_name(path) for path in getattr(parameter_plan, "lora_module_paths", ())}
    if parents != planned:
        raise AdapterContractError(f"LoRA parent mismatch: planned={sorted(planned)[:8]} persisted={sorted(parents)[:8]}")
    return {"adapter_keys": sorted(adapter_state), "full_train_keys": sorted(full_state), "overlap_count": len(overlap)}


def save_trainable_state(model: Any, output_path: str | Path, parameter_plan: Any) -> None:
    """Persist only full-train connector tensors with stable semantic names."""

    state = {
        canonical: parameter.detach().cpu()
        for canonical, (_actual, parameter) in resolve_full_train_parameters(model, parameter_plan).items()
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


def validate_checkpoint_ownership(model: Any, checkpoint_dir: str | Path, parameter_plan: Any) -> dict[str, Any]:
    """Verify every actual trainable tensor has a persisted owner."""

    audit = validate_checkpoint_state(checkpoint_dir, parameter_plan)
    adapter_state = set(audit["adapter_keys"])
    full_state = set(audit["full_train_keys"])
    owned = adapter_state | full_state
    actual = {
        canonicalize_model_parameter_name(name)
        for name, parameter in model.named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    }
    if actual != owned:
        raise AdapterContractError(
            "checkpoint-owned trainable mismatch: "
            f"missing={sorted(actual - owned)[:8]} unexpected={sorted(owned - actual)[:8]}"
        )
    return audit


def restore_trainable_state(
    *,
    model: Any,
    checkpoint_dir: str | Path,
    parameter_plan: Any,
    manifest: dict[str, Any],
) -> Any:
    """Restore adapter and full-train tensors with strict semantic checks."""

    root = Path(checkpoint_dir)
    model_state = model.state_dict()
    actual_by_canonical: dict[str, str] = {}
    for name in model_state:
        canonical = canonicalize_model_parameter_name(name)
        if canonical in actual_by_canonical and actual_by_canonical[canonical] != name:
            raise AdapterContractError(f"ambiguous canonical model state name: {canonical}")
        actual_by_canonical[canonical] = name

    adapter_path = root / "adapter" / "adapter_model.safetensors"
    if not adapter_path.is_file():
        adapter_path = root / "adapter" / "adapter_model.bin"
    if not adapter_path.is_file():
        raise AdapterContractError("resume checkpoint is missing adapter weights")
    adapter_state = _load_tensor_file(adapter_path)
    adapter_canonical = canonicalize_peft_lora_state_keys(adapter_state)
    restored_parents = {
        name.rsplit(".lora_", 1)[0]
        for name in adapter_canonical
        if ".lora_" in name
    }
    planned_parents = {
        _canonical_parameter_name(path)
        for path in getattr(parameter_plan, "lora_module_paths", ())
    }
    if restored_parents != planned_parents:
        raise AdapterContractError(
            f"adapter restore parent mismatch: planned={sorted(planned_parents)[:8]} restored={sorted(restored_parents)[:8]}"
        )
    expected_adapter = {
        canonical
        for canonical in actual_by_canonical
        if "lora_" in canonical.lower()
        and any(("." + path + ".") in ("." + canonical + ".") for path in getattr(parameter_plan, "lora_module_paths", ()))
    }
    if set(adapter_canonical) != expected_adapter:
        missing = sorted(expected_adapter - set(adapter_canonical))
        unexpected = sorted(set(adapter_canonical) - expected_adapter)
        raise AdapterContractError(f"adapter restore key mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    mapped: dict[str, Any] = {}
    for canonical, value in adapter_canonical.items():
        target_name = actual_by_canonical.get(canonical)
        if target_name is None:
            raise AdapterContractError(f"adapter restore key is not present in model: {canonical}")
        expected_value = model_state[target_name]
        if tuple(expected_value.shape) != tuple(value.shape):
            raise AdapterContractError(f"adapter restore shape mismatch: {canonical}")
        mapped[target_name] = value.to(dtype=expected_value.dtype)
    result = model.load_state_dict(mapped, strict=False)
    if getattr(result, "unexpected_keys", ()):
        raise AdapterContractError("unexpected adapter keys after restore")

    full_path = root / "model_trainable_state.safetensors"
    if not full_path.is_file():
        raise AdapterContractError("resume checkpoint is missing model_trainable_state.safetensors")
    full_state = _load_tensor_file(full_path)
    expected_full = set(getattr(parameter_plan, "full_train_parameter_names", ()))
    if set(full_state) != expected_full:
        missing = sorted(expected_full - set(full_state))
        unexpected = sorted(set(full_state) - expected_full)
        raise AdapterContractError(f"full-train restore key mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    mapped = {}
    for canonical, value in full_state.items():
        target_name = actual_by_canonical.get(_canonical_parameter_name(canonical))
        if target_name is None:
            raise AdapterContractError(f"full-train restore key is not present in model: {canonical}")
        expected_value = model_state[target_name]
        if tuple(expected_value.shape) != tuple(value.shape):
            raise AdapterContractError(f"full-train restore shape mismatch: {canonical}")
        if getattr(expected_value, "dtype", None) != getattr(value, "dtype", None):
            raise AdapterContractError(f"full-train restore dtype mismatch: {canonical}")
        mapped[target_name] = value.to(dtype=expected_value.dtype)
    model.load_state_dict(mapped, strict=False)
    for canonical, value in {**adapter_canonical, **full_state}.items():
        target_name = actual_by_canonical.get(_canonical_parameter_name(canonical))
        if target_name is None or not model_state[target_name].equal(value.to(dtype=model_state[target_name].dtype)):
            raise AdapterContractError(f"restored tensor equality check failed: {canonical}")
    return model


def restore_full_train_state_for_export(
    *,
    model: Any,
    checkpoint_dir: str | Path,
    parameter_plan: Any,
) -> Any:
    """Restore only the persisted full-train tensors during export.

    PEFT has already loaded the LoRA state at this point.  Re-entering the
    resume path would load the adapter a second time, so export uses this
    deliberately narrow helper for connector/projector state.
    """

    root = Path(checkpoint_dir)
    full_path = root / "model_trainable_state.safetensors"
    if not full_path.is_file():
        raise AdapterContractError("EXPORT_MISSING_FULL_TRAIN_STATE")
    full_state = _load_tensor_file(full_path)
    expected_full = set(getattr(parameter_plan, "full_train_parameter_names", ()))
    if set(full_state) != expected_full:
        missing = sorted(expected_full - set(full_state))
        unexpected = sorted(set(full_state) - expected_full)
        raise AdapterContractError(
            "EXPORT_FULL_TRAIN_STATE_KEY_MISMATCH: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    model_state = model.state_dict()
    actual_by_canonical: dict[str, str] = {}
    for name in model_state:
        canonical = canonicalize_model_parameter_name(name)
        if canonical in actual_by_canonical and actual_by_canonical[canonical] != name:
            raise AdapterContractError(f"ambiguous canonical model state name: {canonical}")
        actual_by_canonical[canonical] = name

    mapped: dict[str, Any] = {}
    for canonical, value in full_state.items():
        target_name = actual_by_canonical.get(canonicalize_model_parameter_name(canonical))
        if target_name is None:
            raise AdapterContractError(f"EXPORT_FULL_TRAIN_STATE_MISSING_TARGET: {canonical}")
        expected_value = model_state[target_name]
        if tuple(expected_value.shape) != tuple(value.shape):
            raise AdapterContractError(f"EXPORT_FULL_TRAIN_STATE_SHAPE_MISMATCH: {canonical}")
        if getattr(expected_value, "dtype", None) != getattr(value, "dtype", None):
            raise AdapterContractError(f"EXPORT_FULL_TRAIN_STATE_DTYPE_MISMATCH: {canonical}")
        mapped[target_name] = value.to(dtype=expected_value.dtype)
    result = model.load_state_dict(mapped, strict=False)
    if getattr(result, "unexpected_keys", ()):
        raise AdapterContractError("EXPORT_FULL_TRAIN_STATE_UNEXPECTED_KEYS")
    for canonical, value in full_state.items():
        target_name = actual_by_canonical[canonicalize_model_parameter_name(canonical)]
        if not model_state[target_name].equal(value):
            raise AdapterContractError(f"EXPORT_FULL_TRAIN_STATE_RESTORE_FAILED: {canonical}")
    return model


def snapshot_full_train_state(model: Any, parameter_plan: Any) -> dict[str, Any]:
    """Clone exact full-train tensors using their canonical semantic names."""

    by_canonical = _canonical_parameter_map(model)
    snapshot: dict[str, Any] = {}
    for planned in getattr(parameter_plan, "full_train_parameter_names", ()):
        canonical = canonicalize_model_parameter_name(planned)
        match = by_canonical.get(canonical)
        if match is None:
            raise AdapterContractError(f"EXPORT_FULL_TRAIN_PARAMETER_MISSING: {canonical}")
        snapshot[canonical] = match[1].detach().clone()
    return snapshot


def compare_tensor_snapshots(before: dict[str, Any], after: dict[str, Any]) -> None:
    if set(before) != set(after):
        raise AdapterContractError("EXPORT_CONNECTOR_MUTATED_DURING_MERGE")
    for name in before:
        if not before[name].equal(after[name]):
            raise AdapterContractError("EXPORT_CONNECTOR_MUTATED_DURING_MERGE")


def export_peft_checkpoint(
    adapter: Any,
    *,
    model_id: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    local_files_only: bool = True,
    verify_forward: bool = False,
    change_fixture: str | Path | None = None,
) -> dict[str, Any]:
    """Validate, restore, merge, export and offline-verify a PEFT checkpoint."""

    checkpoint = Path(checkpoint_dir)
    output = Path(output_dir)
    if output.exists():
        raise AdapterContractError("EXPORT_OUTPUT_EXISTS")
    if not checkpoint_complete(checkpoint, required_files=("adapter/adapter_config.json",)):
        raise AdapterContractError("EXPORT_CHECKPOINT_INCOMPLETE")
    checkpoint_processor_dir = checkpoint / "processor"
    if not checkpoint_processor_dir.is_dir():
        raise AdapterContractError("EXPORT_PROCESSOR_MISSING")
    manifest_path = checkpoint / TRAINING_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise AdapterContractError("EXPORT_CANONICAL_MANIFEST_MISSING")
    manifest = read_manifest(checkpoint)
    expected_weight = (
        manifest.get("base_model", {}).get("weight_identity")
        or manifest.get("model", {}).get("identity", {}).get("base_weight_identity")
    )
    if not expected_weight:
        raise AdapterContractError("EXPORT_BASE_WEIGHT_IDENTITY_MISSING")
    try:
        actual_weight = base_weight_identity(model_id)
    except ValueError as exc:
        raise AdapterContractError(str(exc)) from exc
    if actual_weight != expected_weight:
        raise AdapterContractError("EXPORT_BASE_WEIGHT_IDENTITY_MISMATCH")
    load_checkpoint_processor = getattr(adapter, "load_processor", None)
    if callable(load_checkpoint_processor):
        checkpoint_processor = load_checkpoint_processor(checkpoint_processor_dir, local_files_only=local_files_only)
    else:
        checkpoint_processor = auto_processor(checkpoint_processor_dir, local_files_only=local_files_only)
    saved_processor_identity_fn = getattr(adapter, "saved_processor_identity", None)
    if callable(saved_processor_identity_fn):
        actual_processor_identity = dict(saved_processor_identity_fn(checkpoint_processor, checkpoint_processor_dir))
    else:
        actual_processor_identity = dict(processor_content_identity(checkpoint_processor_dir, checkpoint_processor))
    expected_processor_identity = manifest.get("processor", {})
    if not expected_processor_identity or not processor_semantic_equal(expected_processor_identity, actual_processor_identity):
        raise AdapterContractError("EXPORT_PROCESSOR_IDENTITY_MISMATCH")
    if expected_processor_identity.get("content_sha256") != actual_processor_identity.get("content_sha256"):
        raise AdapterContractError("EXPORT_PROCESSOR_IDENTITY_MISMATCH")
    plan_path = checkpoint / PARAMETER_PLAN_FILENAME
    try:
        parameter_plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterContractError("EXPORT_PARAMETER_PLAN_UNREADABLE") from exc
    if manifest.get("parameter_plan_sha256") != identity_fingerprint(parameter_plan_payload):
        raise AdapterContractError("EXPORT_PARAMETER_PLAN_MANIFEST_MISMATCH")
    try:
        from ..parameter_plan import ParameterPlan

        parameter_plan = ParameterPlan(**parameter_plan_payload)
    except Exception as exc:  # noqa: BLE001 - convert malformed plans to the export contract
        raise AdapterContractError("EXPORT_PARAMETER_PLAN_INVALID") from exc
    if manifest.get("adapter_name") != adapter.name:
        raise AdapterContractError("EXPORT_ADAPTER_IDENTITY_MISMATCH")
    adapter_dir = checkpoint / "adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise AdapterContractError("EXPORT_MISSING_PEFT_CONFIG")
    adapter_state_path = _adapter_state_path(checkpoint)
    full_state_path = checkpoint / "model_trainable_state.safetensors"
    validate_state = getattr(adapter, "validate_checkpoint_state", None)
    if callable(validate_state):
        validate_state(checkpoint, parameter_plan)
    model, _model_processor, probe = adapter.load(model_id, local_files_only=local_files_only)
    expected_identity = manifest.get("model", {}).get("identity", {})
    expected_fingerprint = expected_identity.get("fingerprint")
    actual_fingerprint = getattr(probe.identity, "fingerprint", None)
    if expected_fingerprint and actual_fingerprint and expected_fingerprint != actual_fingerprint:
        raise AdapterContractError("EXPORT_BASE_IDENTITY_MISMATCH")
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise AdapterContractError("PEFT is required for multimodal checkpoint export") from exc
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    restore_full_train_state_for_export(
        model=peft_model, checkpoint_dir=checkpoint, parameter_plan=parameter_plan
    )
    connector_before = snapshot_full_train_state(peft_model, parameter_plan)
    merged = peft_model.merge_and_unload()
    connector_after = snapshot_full_train_state(merged, parameter_plan)
    compare_tensor_snapshots(connector_before, connector_after)
    output.mkdir(parents=True)
    merged.save_pretrained(output, safe_serialization=True)
    # The root is the canonical deployment/reload directory.  Keep a nested
    # copy for old callers that still expect a processor/ subdirectory.
    source_files = [item["path"] for item in actual_processor_identity.get("files", ())]
    for relative in source_files:
        source = checkpoint_processor_dir / relative
        root_target = output / relative
        root_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, root_target)
    nested = output / "processor"
    for relative in source_files:
        source = checkpoint_processor_dir / relative
        nested_target = nested / relative
        nested_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, nested_target)
    exported_processor_identity = dict(
        processor_content_identity(output, checkpoint_processor, include_paths=source_files)
    )
    if not processor_semantic_equal(actual_processor_identity, exported_processor_identity):
        raise AdapterContractError("EXPORT_PROCESSOR_IDENTITY_MISMATCH")
    if exported_processor_identity.get("content_sha256") != actual_processor_identity.get("content_sha256"):
        raise AdapterContractError("EXPORT_PROCESSOR_IDENTITY_MISMATCH")
    reload_export = getattr(adapter, "reload_exported", None)
    if not callable(reload_export):
        raise AdapterContractError("EXPORT_ADAPTER_MISSING_OFFLINE_RELOAD")
    reloaded_model, reloaded_processor = reload_export(
        output, local_files_only=local_files_only
    )
    if not processor_semantic_equal(actual_processor_identity, processor_semantic_identity(reloaded_processor)):
        raise AdapterContractError("EXPORT_PROCESSOR_IDENTITY_MISMATCH")
    verification = {
        "offline_processor_reload": "PASS",
        "offline_model_reload": "PASS",
        "synthetic_two_image_forward": "NOT_RUN",
        "change_fixture_forward": "NOT_REQUESTED",
    }
    verify = getattr(adapter, "verify_export_forward", None)
    if verify_forward:
        if not callable(verify):
            raise AdapterContractError("EXPORT_ADAPTER_MISSING_FORWARD_VERIFIER")
        verification.update(
            dict(
                verify(
                    reloaded_model,
                    reloaded_processor,
                    change_fixture=change_fixture,
                )
            )
        )
    result = {
        "schema_version": 2,
        "export_type": "multimodal_sft_deployment",
        "adapter_name": adapter.name,
        "model": probe.identity.as_dict(),
        "source_training_manifest": str(manifest_path),
        "training_manifest_sha256": file_sha256(manifest_path),
        "parameter_plan_sha256": file_sha256(plan_path),
        "adapter_state_sha256": file_sha256(adapter_state_path),
        "full_train_state_sha256": file_sha256(full_state_path),
        "source": {
            "base_weight_sha256": actual_weight["sha256"],
            "processor_content_sha256": actual_processor_identity["content_sha256"],
            "checkpoint_completion_sha256": file_sha256(checkpoint / "checkpoint_complete.json"),
        },
        "exported": {
            "processor_content_sha256": exported_processor_identity["content_sha256"],
        },
        "base_model_identity": probe.identity.as_dict(),
        "processor_identity": dict(getattr(adapter, "processor_identity")(checkpoint_processor)),
        **verification,
        "verify_forward": bool(verify_forward),
    }
    (output / "training_export_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Legacy filename remains an alias; training_manifest.json remains the
    # only canonical source identity on the input checkpoint.
    (output / "multimodal_sft_export_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
