"""Build a deterministic rendered Change fixture for generic Qwen export smoke."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from training.multimodal_sft.image_roots import ImageRootRegistry
from training.multimodal_sft.profiles.change_agent import ChangeAgentDataProfile


CHANGE_EXPORT_FIXTURE_SCHEMA_VERSION = 1
CHANGE_EXPORT_FIXTURE_SELECTION_POLICY = "lexical_min_episode_id_v1"


class ChangeExportFixtureError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_change_export_fixture(
    *,
    validation: str | Path,
    manifest: str | Path,
    image_roots: ImageRootRegistry | Mapping[str, str | Path],
    output: str | Path,
    prompt_ref: str | None = None,
    prompt_file: str | Path | None = None,
) -> dict[str, Any]:
    """Render the lexical-min validation episode with canonical messages/images."""

    validation_path = Path(validation)
    manifest_path = Path(manifest)
    output_path = Path(output)
    if output_path.exists():
        raise ChangeExportFixtureError("EXPORT_FIXTURE_OUTPUT_EXISTS")
    registry = (
        image_roots
        if isinstance(image_roots, ImageRootRegistry)
        else ImageRootRegistry(dict(image_roots))
    )
    profile = ChangeAgentDataProfile(
        data_manifest=manifest_path,
        prompt_ref=prompt_ref,
        prompt_file=prompt_file,
    )
    rows = list(profile.read(validation_path))
    if not rows:
        raise ChangeExportFixtureError("EXPORT_FIXTURE_VALIDATION_EMPTY")
    episode = min(rows, key=lambda row: str(row.get("episode_id") or ""))
    episode_id = str(episode.get("episode_id") or "")
    if not episode_id:
        raise ChangeExportFixtureError("EXPORT_FIXTURE_EPISODE_ID_MISSING")
    images = episode.get("images")
    if not isinstance(images, list) or len(images) != 2:
        raise ChangeExportFixtureError("EXPORT_FIXTURE_REQUIRES_TWO_IMAGES")
    roles = [str(item.get("role") or "") for item in images if isinstance(item, dict)]
    if roles != ["raw_full_t1", "raw_full_t2"]:
        raise ChangeExportFixtureError("EXPORT_FIXTURE_ROLE_ORDER_INVALID")
    rendered_images = []
    for item in images:
        source, relative = str(item["image_source"]), str(item["path"])
        rendered_images.append(
            {
                "path": str(registry.resolve(source, relative)),
                "role": str(item["role"]),
                "image_source": source,
            }
        )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture = {
        "schema_version": CHANGE_EXPORT_FIXTURE_SCHEMA_VERSION,
        "selection_policy": CHANGE_EXPORT_FIXTURE_SELECTION_POLICY,
        "task_profile": "change_agent",
        "target_schema": "ChangeInitialResult",
        "messages": list(profile.render_messages(episode)),
        "images": rendered_images,
        "metadata": {
            "episode_id": episode_id,
            "canonical_pair_id": str(episode.get("parent_sample_id") or ""),
            "source_validation_sha256": _sha256(validation_path),
            "manifest_sha256": _sha256(manifest_path),
            "prompt_sha256": str(manifest_payload["change_prompt"]["sha256"]),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output_path)
    return fixture
