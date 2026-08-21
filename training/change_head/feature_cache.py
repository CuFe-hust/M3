"""Strict, content-addressed ChangeHead feature cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


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
) -> str:
    payload = {
        "sample_id": sample_id,
        "t1_sha256": _sha256_file(t1_path),
        "t2_sha256": _sha256_file(t2_path),
        "pipeline_fingerprint": pipeline_fingerprint,
        "experts": experts,
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

    def read(self, key: str, *, expected_metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        import numpy as np

        data_path, metadata_path = self._paths(key)
        if not data_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cache_key") != key:
                return None
            if expected_metadata is not None and any(metadata.get(k) != v for k, v in expected_metadata.items()):
                return None
            with np.load(data_path, allow_pickle=False) as archive:
                return {name: archive[name] for name in archive.files}
        except Exception:
            return None
