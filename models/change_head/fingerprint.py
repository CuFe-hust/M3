"""Canonical preprocessing/runtime fingerprint for ChangeHead checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import ModelCacheIdentity

CHANGE_PREPARED_PAIR_CONTRACT_VERSION = "change-prepared-pair-v1"


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_payload(identity: ModelCacheIdentity) -> dict[str, Any]:
    return {
        "model": identity.model,
        "generation": identity.generation_payload(),
        "client_version": identity.client_version,
        "revision": identity.revision,
    }


def build_change_input_pipeline_fingerprint(
    *,
    settings: Any,
    semantic_client_identities: Mapping[str, ModelCacheIdentity],
    calibration_file_sha256: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return digest and a path-free canonical payload."""

    registration = settings.registration.model_dump(mode="json")
    registration.pop("save_artifacts", None)
    harmonization = settings.harmonization.model_dump(mode="json")
    harmonization.pop("save_artifacts", None)
    harmonization.pop("calibration_file", None)
    harmonization["calibration_file_sha256"] = (
        calibration_file_sha256
        if calibration_file_sha256 is not None
        else _file_sha256(settings.harmonization.calibration_file)
    )
    payload: dict[str, Any] = {
        "prepared_pair_contract_version": CHANGE_PREPARED_PAIR_CONTRACT_VERSION,
        "registration": registration,
        "harmonization": harmonization,
        "semantic_inference": {
            "tile_size": settings.semantic.tile_size,
            "tile_overlap": settings.semantic.tile_overlap,
            "feature_stages": list(settings.semantic.feature_stages),
            "feature_stage_weights": settings.semantic.feature_stage_weights,
        },
        "per_expert_client": {
            expert_id: _identity_payload(identity)
            for expert_id, identity in sorted(semantic_client_identities.items())
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload
