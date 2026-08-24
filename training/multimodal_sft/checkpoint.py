"""Model-neutral checkpoint identity, manifest and resume validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
TRAINING_MANIFEST_FILENAME = "multimodal_sft_training_manifest.json"


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
) -> dict[str, Any]:
    if not adapter_name or not task_profile:
        raise CheckpointContractError("adapter_name and task_profile are required")
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_type": "multimodal_sft_composite",
        "adapter_name": adapter_name,
        "model": {"identity": dict(model_identity), "fingerprint": identity_fingerprint(model_identity)},
        "task": {"profile": task_profile, "contract": dict(data_contract)},
        "tuning_policy": dict(tuning_policy),
        "parameter_plan": dict(parameter_plan),
        "processor": dict(processor_identity or {}),
        "training": dict(training or {}),
    }


def validate_training_manifest(manifest: Mapping[str, Any]) -> None:
    required = {"schema_version", "checkpoint_type", "adapter_name", "model", "task", "tuning_policy", "parameter_plan"}
    missing = sorted(required - set(manifest))
    if missing:
        raise CheckpointContractError(f"manifest missing required fields: {', '.join(missing)}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
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
    return target


def read_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / TRAINING_MANIFEST_FILENAME
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


def validate_resume_compatibility(
    manifest: Mapping[str, Any],
    *,
    adapter_name: str,
    model_identity: Mapping[str, Any],
    task_profile: str,
    tuning_policy: Mapping[str, Any],
) -> None:
    validate_training_manifest(manifest)
    if manifest.get("adapter_name") != adapter_name:
        raise CheckpointContractError("resume adapter mismatch")
    if manifest.get("model", {}).get("fingerprint") != identity_fingerprint(model_identity):
        raise CheckpointContractError("resume model identity mismatch")
    if manifest.get("task", {}).get("profile") != task_profile:
        raise CheckpointContractError("resume task profile mismatch")
    if dict(manifest.get("tuning_policy", {})) != dict(tuning_policy):
        raise CheckpointContractError("resume tuning policy mismatch")


