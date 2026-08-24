"""Model-neutral bridge for the existing Phase2 source schema."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.multimodal_sft.contracts import ImageRef, PreparedMultimodalEpisode
from training.multimodal_sft.image_roots import ImageRootRegistry


class Phase2DataError(ValueError):
    pass


class Phase2DataProfile:
    name = "phase2"

    def __init__(self, *, augmentation_enabled: bool = True) -> None:
        self.augmentation_enabled = bool(augmentation_enabled)

    def read(self, path: str | Path) -> Iterable[dict[str, Any]]:
        source = Path(path)
        if not source.is_file():
            raise Phase2DataError(f"data file does not exist: {source}")
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise Phase2DataError("phase2 row must be an object")
                rows.append(row)
        return iter(rows)

    def validate(self, episode: Mapping[str, Any]) -> None:
        required = ("episode_id", "image_source", "image", "turns", "task_kind", "augmentation_policy")
        missing = [key for key in required if key not in episode]
        if missing:
            raise Phase2DataError("missing phase2 fields: " + ",".join(missing))
        if not isinstance(episode["turns"], list) or not episode["turns"]:
            raise Phase2DataError("phase2 turns must be non-empty")

    def prepare(self, episode: Mapping[str, Any], *, image_roots: Any, split: str, epoch: int, seed: int | str) -> PreparedMultimodalEpisode:
        self.validate(episode)
        registry = image_roots if isinstance(image_roots, ImageRootRegistry) else ImageRootRegistry(dict(image_roots or {}))
        try:
            phase2 = __import__("scripts.qwen3vl_phase2_data", fromlist=["*"])
        except ImportError:
            phase2 = None
        image_source = str(episode["image_source"])
        relative = str(episode["image"])
        image = registry.load_rgb(image_source, relative)
        group_seed = hashlib.sha256(f"{seed}|{epoch}|{episode.get('parent_episode_id', episode['episode_id'])}".encode("utf-8")).hexdigest()
        box_entries = [box for turn in episode["turns"] for box in turn.get("input_boxes", []) + turn.get("target_boxes", [])]
        geometry = str(episode["augmentation_policy"]["geometry"])
        if phase2 is not None and split == "train" and self.augmentation_enabled:
            image, augmentation = phase2.apply_augmentation(image, box_entries, geometry, phase2.AugmentationConfig(enabled=True), group_seed)
        else:
            augmentation = {"group_seed": group_seed[:16], "geometry": {"kind": "identity", "ops": [], "reason": geometry}, "degradations": {"selected": []}}
        if phase2 is not None:
            messages, _ = phase2.render_messages(dict(episode))
        else:
            messages = self._render_fallback(episode)
        return PreparedMultimodalEpisode(
            episode_id=str(episode["episode_id"]), task_profile=self.name,
            messages=tuple(messages), images=(image,), image_roles=("image",),
            target_schema=str(episode.get("task_kind", "")),
            metadata={"image_refs": (ImageRef(image_source, str(registry.resolve(image_source, relative)), "image"),), "augmentation": augmentation, "parent_episode_id": episode.get("parent_episode_id")},
        )

    @staticmethod
    def _render_fallback(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for index, turn in enumerate(episode["turns"]):
            user = re.sub(r"<[^>]+>", "", str(turn.get("user_text", ""))).strip()
            assistant = re.sub(r"<[^>]+>", "", str(turn.get("assistant_text", ""))).strip()
            content: list[dict[str, Any]] = [{"type": "text", "text": user}]
            if index == 0:
                content.insert(0, {"type": "image"})
            messages.extend([
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
            ])
        return messages

    def identity_contract(self, image_roots: Any) -> dict[str, Any]:
        registry = image_roots if isinstance(image_roots, ImageRootRegistry) else ImageRootRegistry(dict(image_roots or {}))
        return {"data_profile": self.name, "image_root_contract": registry.contract(), "augmentation_enabled": self.augmentation_enabled}
