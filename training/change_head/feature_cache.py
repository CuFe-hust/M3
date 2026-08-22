"""Strict, content-addressed ChangeHead feature cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import LEARNED_CHANGE_INPUT_CONTRACT_VERSION


CACHE_SCHEMA_VERSION = "change-head-feature-cache-v2"


@dataclass(frozen=True)
class CachedExpertFeaturePair:
    """Serialized, executable-free output of one production expert pair."""

    expert_id: str
    logical_model_id: str
    weights_sha256: str
    class_map_sha256: str | None
    feature_stages: tuple[int, ...]
    first_features: Mapping[int, Any]
    second_features: Mapping[int, Any]
    first_semantic_probabilities: Any | None
    second_semantic_probabilities: Any | None


@dataclass(frozen=True)
class CachedChangeTrainingSample:
    """Pure arrays and scalar metadata needed by the ChangeHead trainer."""

    sample_id: str
    image_size: tuple[int, int]
    experts: Mapping[str, CachedExpertFeaturePair]
    target_change_mask: Any
    loss_valid_mask: Any
    pif_mask: Any | None
    comparison_t1: Any | None
    comparison_t2: Any | None
    dataset_name: str
    split: str
    tags: tuple[str, ...]
    input_pipeline_fingerprint: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_feature_cache_key(
    *,
    sample_id: str,
    t1_path: Path,
    t2_path: Path,
    pipeline_fingerprint: str,
    experts: list[dict[str, Any]],
    contract_version: str = LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
) -> str:
    payload = {
        "sample_id": sample_id,
        "t1_sha256": _sha256_file(t1_path),
        "t2_sha256": _sha256_file(t2_path),
        "pipeline_fingerprint": pipeline_fingerprint,
        "contract_version": contract_version,
        "experts": [dict(item) for item in experts],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class FeatureCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _paths(self, key: str) -> tuple[Path, Path]:
        directory = self.root / key[:2]
        return directory / f"{key}.npz", directory / f"{key}.json"

    def write(self, key: str, arrays: dict[str, Any], metadata: dict[str, Any]) -> None:
        import numpy as np

        data_path, metadata_path = self._paths(key)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        temp_data = data_path.with_suffix(".npz.tmp")
        temp_metadata = metadata_path.with_suffix(".json.tmp")
        with temp_data.open("wb") as file:
            np.savez_compressed(file, **arrays)
        temp_metadata.write_text(
            json.dumps({**metadata, "cache_key": key}, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_data.replace(data_path)
        temp_metadata.replace(metadata_path)

    def write_sample(self, key: str, sample: CachedChangeTrainingSample) -> None:
        arrays, metadata = serialize_cached_sample(sample)
        self.write(key, arrays, metadata)

    def read_with_metadata(
        self,
        key: str,
        *,
        expected_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        import numpy as np

        data_path, metadata_path = self._paths(key)
        if not data_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cache_key") != key:
                return None
            if expected_metadata is not None and any(
                metadata.get(k) != v for k, v in expected_metadata.items()
            ):
                return None
            with np.load(data_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            return arrays, metadata
        except Exception:
            return None

    def read_sample(self, key: str) -> CachedChangeTrainingSample | None:
        loaded = self.read_with_metadata(key)
        if loaded is None:
            return None
        arrays, metadata = loaded
        try:
            return deserialize_cached_sample(arrays, metadata)
        except (KeyError, TypeError, ValueError):
            return None

    def read(self, key: str, *, expected_metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        loaded = self.read_with_metadata(key, expected_metadata=expected_metadata)
        return None if loaded is None else loaded[0]


def _array(value: Any, *, name: str) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(f"{name} must be numeric")
    if array.size and not np.isfinite(array.astype(np.float32, copy=False)).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def serialize_cached_sample(
    sample: CachedChangeTrainingSample,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Flatten a sample into NPZ arrays plus JSON-only metadata."""

    arrays: dict[str, Any] = {
        "target_change_mask": _array(sample.target_change_mask, name="target_change_mask"),
        "loss_valid_mask": _array(sample.loss_valid_mask, name="loss_valid_mask"),
    }
    optional_arrays: dict[str, str] = {}
    for field_name in ("pif_mask", "comparison_t1", "comparison_t2"):
        value = getattr(sample, field_name)
        if value is not None:
            key = field_name
            arrays[key] = _array(value, name=field_name)
            optional_arrays[field_name] = key
    expert_metadata: list[dict[str, Any]] = []
    for index, expert_id in enumerate(sorted(sample.experts)):
        expert = sample.experts[expert_id]
        prefix = f"expert_{index}"
        stages = [int(stage) for stage in expert.feature_stages]
        first_keys: dict[str, str] = {}
        second_keys: dict[str, str] = {}
        for stage in stages:
            first_key = f"{prefix}_first_stage_{stage}"
            second_key = f"{prefix}_second_stage_{stage}"
            arrays[first_key] = _array(
                expert.first_features[stage], name=first_key
            )
            arrays[second_key] = _array(
                expert.second_features[stage], name=second_key
            )
            first_keys[str(stage)] = first_key
            second_keys[str(stage)] = second_key
        semantic_keys: dict[str, str] = {}
        if expert.first_semantic_probabilities is not None:
            semantic_keys["first"] = f"{prefix}_first_semantic"
            arrays[semantic_keys["first"]] = _array(
                expert.first_semantic_probabilities,
                name=semantic_keys["first"],
            )
        if expert.second_semantic_probabilities is not None:
            semantic_keys["second"] = f"{prefix}_second_semantic"
            arrays[semantic_keys["second"]] = _array(
                expert.second_semantic_probabilities,
                name=semantic_keys["second"],
            )
        expert_metadata.append(
            {
                "expert_id": expert.expert_id,
                "logical_model_id": expert.logical_model_id,
                "weights_sha256": expert.weights_sha256,
                "class_map_sha256": expert.class_map_sha256,
                "feature_stages": stages,
                "first_feature_keys": first_keys,
                "second_feature_keys": second_keys,
                "semantic_keys": semantic_keys,
            }
        )
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "image_size": list(sample.image_size),
        "dataset_name": sample.dataset_name,
        "split": sample.split,
        "tags": list(sample.tags),
        "input_pipeline_fingerprint": sample.input_pipeline_fingerprint,
        "optional_arrays": optional_arrays,
        "experts": expert_metadata,
    }
    return arrays, metadata


def deserialize_cached_sample(
    arrays: Mapping[str, Any], metadata: Mapping[str, Any]
) -> CachedChangeTrainingSample:
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported feature cache schema")
    def optional(name: str) -> Any | None:
        key = metadata.get("optional_arrays", {}).get(name)
        return None if key is None else arrays[key]

    experts: dict[str, CachedExpertFeaturePair] = {}
    for item in metadata.get("experts", []):
        stages = tuple(int(stage) for stage in item["feature_stages"])
        first = {stage: arrays[item["first_feature_keys"][str(stage)]] for stage in stages}
        second = {stage: arrays[item["second_feature_keys"][str(stage)]] for stage in stages}
        semantic_keys = item.get("semantic_keys", {})
        expert = CachedExpertFeaturePair(
            expert_id=str(item["expert_id"]),
            logical_model_id=str(item["logical_model_id"]),
            weights_sha256=str(item["weights_sha256"]),
            class_map_sha256=item.get("class_map_sha256"),
            feature_stages=stages,
            first_features=first,
            second_features=second,
            first_semantic_probabilities=(
                None if "first" not in semantic_keys else arrays[semantic_keys["first"]]
            ),
            second_semantic_probabilities=(
                None if "second" not in semantic_keys else arrays[semantic_keys["second"]]
            ),
        )
        if expert.expert_id in experts:
            raise ValueError("duplicate cached expert")
        experts[expert.expert_id] = expert
    image_size = tuple(int(value) for value in metadata["image_size"])
    if len(image_size) != 2:
        raise ValueError("invalid cached image size")
    return CachedChangeTrainingSample(
        sample_id=str(metadata["sample_id"]),
        image_size=(image_size[0], image_size[1]),
        experts=experts,
        target_change_mask=arrays["target_change_mask"],
        loss_valid_mask=arrays["loss_valid_mask"],
        pif_mask=optional("pif_mask"),
        comparison_t1=optional("comparison_t1"),
        comparison_t2=optional("comparison_t2"),
        dataset_name=str(metadata["dataset_name"]),
        split=str(metadata["split"]),
        tags=tuple(str(tag) for tag in metadata.get("tags", [])),
        input_pipeline_fingerprint=str(metadata["input_pipeline_fingerprint"]),
    )
