"""Model-neutral checkpoint identity, manifest and resume validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .identity import artifact_tree_identity, sha256_file


SCHEMA_VERSION = 2
TRAINING_MANIFEST_FILENAME = "training_manifest.json"
COMPATIBILITY_MANIFEST_FILENAME = "multimodal_sft_training_manifest.json"
PARAMETER_PLAN_FILENAME = "parameter_plan.json"
TRAINER_STATE_FILENAME = "trainer_state.json"
OPTIMIZER_FILENAME = "optimizer.pt"
SCHEDULER_FILENAME = "scheduler.pt"
RNG_STATE_FILENAME = "rng_state.pt"
COMPLETION_MARKER_FILENAME = "checkpoint_complete.json"


class CheckpointContractError(ValueError):
    """Checkpoint is incomplete or incompatible with the requested run."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def identity_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def build_training_manifest(
    *,
    adapter_name: str,
    model_identity: Mapping[str, Any],
    task_profile: str,
    data_contract: Mapping[str, Any],
    tuning_policy: Mapping[str, Any],
    parameter_plan: Mapping[str, Any],
    processor_identity: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    training_plan: Mapping[str, Any] | None = None,
    base_model_id: str | None = None,
    base_weight_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not adapter_name or not task_profile:
        raise CheckpointContractError("adapter_name and task_profile are required")
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_type": "multimodal_sft_composite",
        "adapter_name": adapter_name,
        "model": {"identity": dict(model_identity), "fingerprint": identity_fingerprint(model_identity)},
        "base_model": {
            "model_id": str(base_model_id) if base_model_id is not None else None,
            "model_type": model_identity.get("model_type"),
            "architectures": list(model_identity.get("architectures", ())),
            "config_fingerprint": model_identity.get("fingerprint"),
            "weight_identity": dict(base_weight_identity or model_identity.get("base_weight_identity", {})),
        },
        "task": {"profile": task_profile, "contract": dict(data_contract)},
        "tuning_policy": dict(tuning_policy),
        "parameter_plan": dict(parameter_plan),
        "parameter_plan_sha256": identity_fingerprint(parameter_plan),
        "training_plan": dict(training_plan or {}),
        "training_plan_sha256": identity_fingerprint(training_plan or {}),
        "processor": dict(processor_identity or {}),
        "training": dict(training or {}),
    }


def validate_training_manifest(manifest: Mapping[str, Any]) -> None:
    required = {"schema_version", "checkpoint_type", "adapter_name", "model", "task", "tuning_policy", "parameter_plan"}
    missing = sorted(required - set(manifest))
    if missing:
        raise CheckpointContractError(f"manifest missing required fields: {', '.join(missing)}")
    if manifest.get("schema_version") not in (1, SCHEMA_VERSION):
        raise CheckpointContractError("unsupported multimodal SFT manifest schema")
    if manifest.get("checkpoint_type") != "multimodal_sft_composite":
        raise CheckpointContractError("checkpoint_type is not multimodal_sft_composite")
    if not isinstance(manifest["model"], Mapping) or not manifest["model"].get("identity"):
        raise CheckpointContractError("model identity is required")


def write_manifest(checkpoint_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    validate_training_manifest(manifest)
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / TRAINING_MANIFEST_FILENAME
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, target)
    # Keep the first generic name readable by the previous Phase 1 adapter
    # tests and callers while making training_manifest.json canonical.
    compatibility = root / COMPATIBILITY_MANIFEST_FILENAME
    compatibility.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def read_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    path = root / TRAINING_MANIFEST_FILENAME
    if not path.is_file():
        path = root / COMPATIBILITY_MANIFEST_FILENAME
    if not path.is_file():
        raise CheckpointContractError(f"missing manifest: {path.name}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointContractError("manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise CheckpointContractError("manifest must be a JSON object")
    validate_training_manifest(manifest)
    return manifest


def read_compatible_manifest(
    checkpoint_dir: str | Path,
    *,
    legacy_manifest_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Read the canonical manifest or an explicitly named legacy manifest.

    Legacy names are supplied by the caller so this generic layer does not
    encode a task/model-family filename.  Legacy payloads are returned as-is;
    callers must apply their own compatibility validation before resume.
    """

    root = Path(checkpoint_dir)
    try:
        return read_manifest(root)
    except CheckpointContractError as canonical_error:
        for name in legacy_manifest_names:
            candidate = root / str(name)
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointContractError(f"legacy manifest is not valid JSON: {candidate.name}") from exc
            if not isinstance(payload, dict):
                raise CheckpointContractError("legacy manifest must be a JSON object")
            return payload
        raise canonical_error


def checkpoint_complete(
    checkpoint_dir: str | Path,
    *,
    required_files: Iterable[str] = (),
) -> bool:
    """Return whether the canonical composite checkpoint is resumable."""

    root = Path(checkpoint_dir)
    required = {
        TRAINING_MANIFEST_FILENAME,
        PARAMETER_PLAN_FILENAME,
        TRAINER_STATE_FILENAME,
        OPTIMIZER_FILENAME,
        SCHEDULER_FILENAME,
        RNG_STATE_FILENAME,
        "model_trainable_state.safetensors",
        COMPLETION_MARKER_FILENAME,
        "training_log.jsonl",
    }
    required.update(str(item) for item in required_files)
    if not all((root / item).exists() for item in required):
        return False
    if not (root / "adapter").is_dir() or not (root / "processor").is_dir():
        return False
    try:
        read_manifest(root)
    except CheckpointContractError:
        return False
    try:
        marker = json.loads((root / COMPLETION_MARKER_FILENAME).read_text(encoding="utf-8"))
        manifest = read_manifest(root)
        plan = json.loads((root / PARAMETER_PLAN_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, CheckpointContractError):
        return False
    if marker.get("schema_version") != 2 or marker.get("complete") is not True:
        return False
    if manifest.get("parameter_plan_sha256") != identity_fingerprint(plan):
        return False
    files = marker.get("files")
    directories = marker.get("directories")
    if not isinstance(files, Mapping) or not isinstance(directories, Mapping):
        return False
    core_names = (
        TRAINING_MANIFEST_FILENAME,
        PARAMETER_PLAN_FILENAME,
        "model_trainable_state.safetensors",
        OPTIMIZER_FILENAME,
        SCHEDULER_FILENAME,
        TRAINER_STATE_FILENAME,
        RNG_STATE_FILENAME,
        "training_log.jsonl",
    )
    for name in core_names:
        path = root / name
        expected = files.get(name)
        if not path.is_file() or not isinstance(expected, Mapping):
            return False
        if dict(expected) != {"sha256": file_sha256(path), "size": path.stat().st_size}:
            return False
    for directory in ("adapter", "processor"):
        expected = directories.get(directory)
        if not isinstance(expected, Mapping) or not (root / directory).is_dir():
            return False
        actual = artifact_tree_identity(root / directory)
        if dict(expected) != actual:
            return False
    return True


def file_sha256(path: str | Path) -> str:
    return sha256_file(path)


def write_completion_marker(
    checkpoint_dir: str | Path,
    *,
    global_step: int,
) -> Path:
    root = Path(checkpoint_dir)
    manifest = root / TRAINING_MANIFEST_FILENAME
    plan = root / PARAMETER_PLAN_FILENAME
    if not manifest.is_file() or not plan.is_file():
        raise CheckpointContractError("cannot mark incomplete checkpoint")
    target = root / COMPLETION_MARKER_FILENAME
    required_names = (
        TRAINING_MANIFEST_FILENAME,
        PARAMETER_PLAN_FILENAME,
        "model_trainable_state.safetensors",
        OPTIMIZER_FILENAME,
        SCHEDULER_FILENAME,
        TRAINER_STATE_FILENAME,
        RNG_STATE_FILENAME,
        "training_log.jsonl",
    )
    if not all((root / name).is_file() for name in required_names):
        raise CheckpointContractError("cannot mark incomplete checkpoint state")
    payload = {
        "schema_version": 2,
        "global_step": int(global_step),
        "complete": True,
        "files": {
            name: {"sha256": file_sha256(root / name), "size": (root / name).stat().st_size}
            for name in required_names
        },
        "directories": {
            name: artifact_tree_identity(root / name) for name in ("adapter", "processor")
        },
    }
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, target)
    return target


def validate_resume_compatibility(
    manifest: Mapping[str, Any],
    *,
    adapter_name: str,
    model_identity: Mapping[str, Any],
    task_profile: str,
    tuning_policy: Mapping[str, Any],
    parameter_plan: Mapping[str, Any] | None = None,
    data_contract: Mapping[str, Any] | None = None,
    training_plan: Mapping[str, Any] | None = None,
    processor_identity: Mapping[str, Any] | None = None,
) -> None:
    validate_training_manifest(manifest)
    if manifest.get("adapter_name") != adapter_name:
        raise CheckpointContractError("resume adapter mismatch")
    if manifest.get("model", {}).get("fingerprint") != identity_fingerprint(model_identity):
        raise CheckpointContractError("resume model identity mismatch")
    expected_weight = (
        manifest.get("base_model", {}).get("weight_identity")
        or manifest.get("model", {}).get("identity", {}).get("base_weight_identity")
    )
    actual_weight = model_identity.get("base_weight_identity")
    if expected_weight and expected_weight != actual_weight:
        raise CheckpointContractError("RESUME_BASE_WEIGHT_IDENTITY_MISMATCH")
    if manifest.get("task", {}).get("profile") != task_profile:
        raise CheckpointContractError("resume task profile mismatch")
    if dict(manifest.get("tuning_policy", {})) != dict(tuning_policy):
        raise CheckpointContractError("resume tuning policy mismatch")
    if parameter_plan is not None:
        found_plan = manifest.get("parameter_plan", {})
        if identity_fingerprint(found_plan) != identity_fingerprint(parameter_plan):
            raise CheckpointContractError("resume parameter plan mismatch")
        if manifest.get("parameter_plan_sha256") not in (None, identity_fingerprint(parameter_plan)):
            raise CheckpointContractError("resume parameter plan fingerprint mismatch")
    if data_contract is not None and dict(manifest.get("task", {}).get("contract", {})) != dict(data_contract):
        raise CheckpointContractError("resume data contract mismatch")
    if training_plan is not None:
        expected = identity_fingerprint(training_plan)
        if manifest.get("training_plan_sha256") != expected or dict(manifest.get("training_plan", {})) != dict(training_plan):
            raise CheckpointContractError("RESUME_TRAINING_PLAN_MISMATCH")
    if processor_identity:
        expected_processor = dict(manifest.get("processor", {}))
        semantic_keys = ("class", "tokenizer_class", "chat_template_sha256", "special_tokens_sha256", "special_token_ids")
        if any(expected_processor.get(key) != processor_identity.get(key) for key in semantic_keys):
            raise CheckpointContractError("RESUME_PROCESSOR_IDENTITY_MISMATCH")
