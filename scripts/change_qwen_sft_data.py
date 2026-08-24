#!/usr/bin/env python3
"""ChangeAgent Qwen SFT data pipeline for ordered multi-image episodes.

This module consumes only canonical Change SFT JSONL.  It deliberately
reuses Phase2's safe paths, processor encoding and collator rather than
duplicating the model-facing data path.
"""

from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.change.prompt_contract import INITIAL_RESPONSE_SUFFIX, evidence_label
from agents.change.schema import ChangeInitialResult
from training.multimodal_sft.image_roots import ImageRootRegistry
from training.multimodal_sft.profiles.change_agent import ChangeAgentDataProfile


def _phase2() -> Any:
    """Load the shared heavy data implementation only when data is used."""
    return importlib.import_module("scripts.qwen3vl_phase2_data")


def DatasetRootConfig(*args: Any, **kwargs: Any) -> Any:
    return _phase2().DatasetRootConfig(*args, **kwargs)


def AugmentationConfig(*args: Any, **kwargs: Any) -> Any:
    return _phase2().AugmentationConfig(*args, **kwargs)


CHANGE_SFT_SCHEMA_VERSION = 1
CHANGE_PAIR_AUGMENTATION_UNSUPPORTED = "CHANGE_PAIR_AUGMENTATION_UNSUPPORTED"
_ALLOWED_TASKS = {"change_caption", "change_qa"}
_ALLOWED_CONTRACTS = {"semantic_pair_v1", "runtime_initial_v1"}


def prepare_change_episode(
    episode: Mapping[str, Any],
    *,
    image_roots: ImageRootRegistry | Mapping[str, str],
    data_manifest: str | Path,
    prompt_ref: str | None = None,
    prompt_file: str | Path | None = None,
    split: str = "train",
    epoch: int = 0,
    seed: int | str = 0,
) -> Any:
    """Compatibility façade for the generic ChangeAgent data profile."""

    profile = ChangeAgentDataProfile(data_manifest=data_manifest, prompt_ref=prompt_ref, prompt_file=prompt_file)
    registry = image_roots if isinstance(image_roots, ImageRootRegistry) else ImageRootRegistry(image_roots)
    return profile.prepare(episode, image_roots=registry, split=split, epoch=epoch, seed=seed)


class ChangeSFTDataError(ValueError):
    """Stable contract error with a public rejection code."""

    def __init__(self, code: str, episode_id: str = "") -> None:
        super().__init__(f"{code}: {episode_id}" if episode_id else code)
        self.code = code
        self.episode_id = episode_id


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _is_safe_relative_path(value: str) -> bool:
    if not value or "\\" in value or "\x00" in value or value.startswith(("/", "//")):
        return False
    if len(value) >= 2 and value[1] == ":":
        return False
    return all(part not in {"", ".", ".."} for part in value.split("/"))


def validate_change_episode(episode: Mapping[str, Any]) -> None:
    """Validate the canonical Change SFT episode before image I/O."""
    episode_id = str(episode.get("episode_id") or "")
    if episode.get("schema_version") != CHANGE_SFT_SCHEMA_VERSION:
        raise ChangeSFTDataError("schema_version", episode_id)
    task = episode.get("task")
    if task not in _ALLOWED_TASKS:
        raise ChangeSFTDataError("unknown_task", episode_id)
    question = episode.get("question")
    if not isinstance(question, str) or (task == "change_qa" and not question.strip()):
        raise ChangeSFTDataError("missing_question", episode_id)
    if episode.get("input_contract") not in _ALLOWED_CONTRACTS:
        raise ChangeSFTDataError("invalid_input_contract", episode_id)
    images = episode.get("images")
    if not isinstance(images, list) or len(images) < 2:
        raise ChangeSFTDataError("missing_t2", episode_id)
    expected_roles = ("raw_full_t1", "raw_full_t2")
    if tuple(item.get("role") for item in images[:2] if isinstance(item, dict)) != expected_roles:
        raise ChangeSFTDataError("invalid_role_order", episode_id)
    for item in images:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("image_source"), str)
            or not isinstance(item.get("path"), str)
            or not _is_safe_relative_path(item["path"])
        ):
            raise ChangeSFTDataError("unsafe_image_path", episode_id)
    payload = episode.get("request_payload")
    manifest = payload.get("image_manifest") if isinstance(payload, dict) else None
    expected_manifest = [{"index": str(i), "role": item["role"]} for i, item in enumerate(images)]
    if manifest != expected_manifest:
        raise ChangeSFTDataError("image_manifest_mismatch", episode_id)
    target = episode.get("target")
    if not isinstance(target, dict) or target.get("response_schema") != "ChangeInitialResult":
        raise ChangeSFTDataError("invalid_target_schema", episode_id)
    try:
        ChangeInitialResult.model_validate(target.get("result"))
    except Exception as error:  # pydantic error is not a stable public contract.
        raise ChangeSFTDataError("invalid_target_schema", episode_id) from error


def render_change_messages(episode: Mapping[str, Any], prompt_text: str) -> list[dict]:
    """Render the initial runtime-shaped ChangeAgent conversation."""
    validate_change_episode(episode)
    images = episode["images"]
    user: list[dict] = [
        {"type": "text", "text": "Decision stage: initial. Compare the next two authoritative raw images first."},
    ]
    for image in images:
        user.append({"type": "text", "text": evidence_label(image["role"])})
        user.append({"type": "image"})
    user.append({"type": "text", "text": _safe_json(episode["request_payload"])})
    return [
        {"role": "system", "content": prompt_text + "\n\n" + INITIAL_RESPONSE_SUFFIX},
        {"role": "user", "content": user},
        {"role": "assistant", "content": [{"type": "text", "text": _safe_json(episode["target"]["result"])}]},
    ]


class ChangeEpisodeDataset:
    """Ordered ChangeAgent episode dataset; pair augmentation is unsupported."""

    def __init__(
        self,
        episode_jsonl: str | Path,
        roots: Any,
        processor: Any,
        aug_config: Any,
        max_seq_length: int,
        seed: str,
        split: str = "train",
        start_epoch: int = 0,
        prompt_text: str = "",
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"invalid split: {split!r}")
        if aug_config.enabled:
            raise ChangeSFTDataError(CHANGE_PAIR_AUGMENTATION_UNSUPPORTED)
        phase2 = _phase2()
        self._store = phase2.LazyJsonLines(Path(episode_jsonl))
        self._roots = roots if hasattr(roots, "image_path") else phase2.DatasetRootConfig(dict(roots))
        self._processor = processor
        self._max_seq_length = int(max_seq_length)
        self._seed = str(seed)
        self._split = split
        self._epoch = int(start_epoch)
        self._prompt_text = str(prompt_text)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._store)

    def _feature(self, episode: Mapping[str, Any], truncate: bool = True) -> dict:
        validate_change_episode(episode)
        loaded = []
        for item in episode["images"]:
            path = self._roots.image_path(item["image_source"], item["path"])
            loaded.append(_phase2().load_image_rgb(path, item["path"]))
        messages = render_change_messages(episode, self._prompt_text)
        return _phase2().encode_multimodal_episode(
            self._processor, loaded, messages, self._max_seq_length,
            str(episode["episode_id"]), truncate=truncate,
        )

    def __getitem__(self, index: int) -> dict:
        episode = self._store[index]
        feature = self._feature(episode)
        feature.update({
            "episode_id": episode["episode_id"],
            "parent_sample_id": episode["parent_sample_id"],
            "task": episode["task"],
            "input_contract": episode["input_contract"],
            "image_roles": [item["role"] for item in episode["images"]],
            "augmentation": {"group_seed": _phase2().group_seed_hex(self._seed, self._epoch, episode["parent_sample_id"])[:16], "kind": "identity_pair_locked"},
        })
        return feature

    def preflight(self, limit: int | None = None) -> dict:
        counts = {"checked": 0, "too_long": 0, "episode_too_long": 0, "image_errors": 0, "other_errors": 0}
        count = len(self) if limit is None else min(len(self), int(limit))
        for index in range(count):
            counts["checked"] += 1
            try:
                feature = self._feature(self._store[index], truncate=False)
                if int(feature["input_ids"].shape[0]) > self._max_seq_length:
                    counts["episode_too_long"] += 1
            except OSError:
                counts["image_errors"] += 1
            except Exception as error:
                if "episode_too_long" in str(error):
                    counts["episode_too_long"] += 1
                else:
                    counts["other_errors"] += 1
        return counts


class Phase2DataCollator:
    """Phase2-compatible collator that keeps Change metadata out of kwargs."""

    def __call__(self, features: Sequence[dict]) -> tuple[dict, list[dict]]:
        batch, meta = _phase2().Phase2DataCollator()(features)
        for item, feature in zip(meta, features):
            item.update({key: feature.get(key) for key in ("parent_sample_id", "task", "input_contract", "image_roles")})
        return batch, meta


# Trainer compatibility symbols. / 训练器兼容符号。
Phase2EpisodeDataset = ChangeEpisodeDataset
