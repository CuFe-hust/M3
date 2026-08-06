"""Atomic persistence for the immutable M3-RS run-state contract."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from m3rs_eval.contracts import RunManifest, validate_persisted_record


class InvalidTransition(ValueError):
    """Raised when a run attempts to leave the fixed state graph."""


_TRANSITIONS = {
    "created": frozenset({"preflight_passed"}),
    "preflight_passed": frozenset({"inference_running"}),
    "inference_running": frozenset({"evaluating"}),
    "evaluating": frozenset({"complete", "incomplete", "failed"}),
    "complete": frozenset(),
    "incomplete": frozenset(),
    "failed": frozenset(),
}
_TERMINAL = frozenset({"complete", "incomplete", "failed"})


class RunStateStore:
    """Own one run manifest and replace it atomically for each legal update."""

    filename = "run_manifest.json"

    def __init__(self, run_dir: Path, manifest: RunManifest) -> None:
        self.run_dir = Path(run_dir)
        self.manifest = manifest

    @classmethod
    def create(cls, run_dir: Path, manifest: RunManifest) -> "RunStateStore":
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / cls.filename
        if target.exists():
            raise FileExistsError(f"run manifest already exists: {target}")
        store = cls(run_dir, manifest)
        store._write(manifest)
        return store

    @classmethod
    def load(cls, run_dir: Path) -> "RunStateStore":
        run_dir = Path(run_dir)
        target = run_dir / cls.filename
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load run manifest: {target}") from error
        validate_persisted_record(raw, RunManifest)
        return cls(run_dir, RunManifest.from_dict(raw))

    def transition(self, next_status: str) -> RunManifest:
        allowed = _TRANSITIONS.get(self.manifest.status, frozenset())
        if next_status not in allowed:
            raise InvalidTransition(
                f"illegal run transition: {self.manifest.status} -> {next_status}"
            )
        return self._replace(status=next_status)

    def update(self, **changes: Any) -> RunManifest:
        if self.manifest.status in _TERMINAL:
            raise InvalidTransition(f"terminal run is immutable: {self.manifest.status}")
        if "status" in changes:
            raise InvalidTransition("use transition() to change run status")
        return self._replace(**changes)

    def _replace(self, **changes: Any) -> RunManifest:
        manifest = replace(self.manifest, **changes)
        self._write(manifest)
        self.manifest = manifest
        return manifest

    def _write(self, manifest: RunManifest) -> None:
        payload = manifest.to_dict()
        validate_persisted_record(payload, RunManifest)
        target = self.run_dir / self.filename
        temporary = self.run_dir / f".{self.filename}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
