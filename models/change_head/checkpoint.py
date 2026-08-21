"""Safe, path-sanitized ChangeHead checkpoint loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.base import (
    ModelAssetHashMismatchError,
    ModelAssetMissingError,
    ModelAssetPointerError,
    validate_local_model_asset,
)
from models.change_head.calibration import ChangeHeadCalibration
from models.change_head.manifest import ChangeHeadManifest, ChangeHeadManifestError


class ChangeHeadCheckpointError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class LoadedChangeHeadCheckpoint:
    root: Path
    manifest: ChangeHeadManifest
    calibration: ChangeHeadCalibration
    state_dict: Mapping[str, Any]


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ChangeHeadCheckpointError(code, path.name) from error
    if not isinstance(value, dict):
        raise ChangeHeadCheckpointError(code, path.name)
    return value


def load_change_head_checkpoint(root: Path) -> LoadedChangeHeadCheckpoint:
    """Load metadata and safetensors only after asset/hash validation."""

    if not root.is_dir():
        raise ChangeHeadCheckpointError("LEARNED_CHANGE_MANIFEST_MISSING", root.name)
    manifest_path = root / "manifest.json"
    calibration_path = root / "calibration.json"
    weights_path = root / "model.safetensors"
    try:
        manifest = ChangeHeadManifest.model_validate(
            _read_json(manifest_path, code="LEARNED_CHANGE_MANIFEST_INVALID")
        )
        calibration = ChangeHeadCalibration.model_validate(
            _read_json(calibration_path, code="LEARNED_CHANGE_CALIBRATION_INVALID")
        )
    except ChangeHeadManifestError as error:
        raise ChangeHeadCheckpointError(error.code, str(error)) from error
    except ValueError as error:
        raise ChangeHeadCheckpointError(
            "LEARNED_CHANGE_MANIFEST_INVALID", manifest_path.name
        ) from error
    try:
        validate_local_model_asset(
            weights_path, expected_sha256=manifest.model_weights_sha256
        )
    except (ModelAssetMissingError, FileNotFoundError, ModelAssetPointerError) as error:
        raise ChangeHeadCheckpointError("LEARNED_CHANGE_WEIGHT_MISSING", weights_path.name) from error
    except (ModelAssetHashMismatchError, ValueError) as error:
        raise ChangeHeadCheckpointError("LEARNED_CHANGE_WEIGHT_HASH_MISMATCH", weights_path.name) from error
    except RuntimeError as error:
        raise ChangeHeadCheckpointError("LEARNED_CHANGE_WEIGHT_MISSING", weights_path.name) from error
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ChangeHeadCheckpointError(
            "LEARNED_CHANGE_INFERENCE_FAILED", "safetensors dependency missing"
        ) from error
    try:
        state_dict = load_file(str(weights_path), device="cpu")
    except Exception as error:
        raise ChangeHeadCheckpointError("LEARNED_CHANGE_INFERENCE_FAILED", weights_path.name) from error
    return LoadedChangeHeadCheckpoint(
        root=root,
        manifest=manifest,
        calibration=calibration,
        state_dict=state_dict,
    )
