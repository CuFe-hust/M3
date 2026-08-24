"""Small model-neutral JSONL data profiles.

Existing task-specific preparation scripts may continue to produce canonical
episodes.  This module only defines the generic boundary; it does not infer a
task from a model name or filename.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import CanonicalEpisode


class DataProfileError(ValueError):
    """Canonical episode data does not satisfy its declared contract."""


class JsonlDataProfile:
    """Read canonical JSON/JSONL rows with explicit task-profile metadata."""

    def __init__(self, name: str, *, required_target_schema: str | None = None) -> None:
        if not name.strip():
            raise ValueError("data profile name must be non-empty")
        self.name = name
        self.required_target_schema = required_target_schema

    def _row_to_episode(self, row: dict[str, Any], index: int) -> CanonicalEpisode:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise DataProfileError(f"row {index}: messages must be a non-empty list")
        images = row.get("images", ())
        if not isinstance(images, (list, tuple)):
            raise DataProfileError(f"row {index}: images must be a list")
        profile = str(row.get("task_profile", self.name))
        target = str(row.get("target_schema", ""))
        if self.required_target_schema and target != self.required_target_schema:
            raise DataProfileError(
                f"row {index}: target_schema={target!r} expected {self.required_target_schema!r}"
            )
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise DataProfileError(f"row {index}: metadata must be an object")
        return CanonicalEpisode(
            task_profile=profile,
            messages=tuple(messages),
            images=tuple(images),
            target_schema=target,
            metadata=metadata,
        )

    def read(self, path: str | Path) -> Iterable[CanonicalEpisode]:
        source = Path(path)
        if not source.is_file():
            raise DataProfileError(f"data file does not exist: {source}")
        if source.suffix.lower() == ".jsonl":
            rows = (json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip())
        elif source.suffix.lower() == ".json":
            obj = json.loads(source.read_text(encoding="utf-8"))
            rows = obj if isinstance(obj, list) else obj.get("records", obj.get("data", []))
        else:
            raise DataProfileError("canonical data must be .json or .jsonl")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise DataProfileError(f"row {index}: expected an object")
            yield self._row_to_episode(row, index)

    def validate(self, episode: CanonicalEpisode) -> None:
        if episode.task_profile != self.name:
            raise DataProfileError(
                f"episode task_profile={episode.task_profile!r} does not match {self.name!r}"
            )
        if not episode.messages:
            raise DataProfileError("episode messages cannot be empty")
        if self.required_target_schema and episode.target_schema != self.required_target_schema:
            raise DataProfileError("episode target schema does not match profile")


def profile_for(name: str) -> JsonlDataProfile:
    """Return an explicit task profile; model adapters are not consulted."""

    if name == "change_agent":
        return JsonlDataProfile(name, required_target_schema="ChangeInitialResult")
    if name == "phase2":
        return JsonlDataProfile(name)
    raise DataProfileError(f"unsupported data profile: {name}")


