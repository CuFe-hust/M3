"""Content identities for local model and processor artifacts.

The helpers in this module are deliberately model-family agnostic.  They
hash bytes and canonical artifact metadata; they never inspect model module
names or tokenizer-specific implementation details.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


_IGNORED_PARTS = {"__pycache__", ".DS_Store"}
_IGNORED_NAMES = re.compile(r"(?:\.lock$|\.tmp$|\.temp$)", re.IGNORECASE)
_WEIGHT_NAMES = (
    "model.safetensors",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
)
_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included_file(path: Path) -> bool:
    return not any(part in _IGNORED_PARTS for part in path.parts) and not _IGNORED_NAMES.search(path.name)


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def artifact_tree_identity(root: str | Path, *, include_paths: Iterable[str] | None = None) -> dict[str, Any]:
    """Return a deterministic recursive file identity for an artifact tree."""

    base = Path(root)
    if not base.is_dir():
        raise ValueError(f"artifact directory is missing: {base}")
    allowed = {str(item).replace("\\", "/") for item in include_paths} if include_paths is not None else None
    records: list[dict[str, Any]] = []
    for path in sorted((item for item in base.rglob("*") if item.is_file()), key=lambda item: item.relative_to(base).as_posix()):
        if not _included_file(path):
            continue
        relative = path.relative_to(base).as_posix()
        if allowed is not None and relative not in allowed:
            continue
        records.append(_file_record(base, path))
    return {
        "files": records,
        "sha256": hashlib.sha256(_canonical_json(records)).hexdigest(),
    }


def _referenced_weight_files(model_dir: Path) -> set[Path]:
    files: set[Path] = set()
    for index_name in _INDEX_NAMES:
        index = model_dir / index_name
        if not index.is_file():
            continue
        files.add(index)
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map", {})
            if not isinstance(weight_map, Mapping):
                raise ValueError("weight_map is not an object")
            for relative in set(str(value) for value in weight_map.values()):
                candidate = (model_dir / relative).resolve()
                candidate.relative_to(model_dir.resolve())
                if not candidate.is_file():
                    raise ValueError(f"referenced weight shard is missing: {relative}")
                files.add(candidate)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"BASE_WEIGHT_IDENTITY_INVALID_INDEX: {index}") from exc
    if files:
        return files
    for pattern in _WEIGHT_NAMES:
        files.update(path for path in model_dir.glob(pattern) if path.is_file())
    return files


def base_weight_identity(model_id: str | Path, *, resolved_revision: str | None = None) -> dict[str, Any]:
    """Fingerprint actual local HF weight bytes, or fail closed remotely."""

    root = Path(model_id)
    if not root.is_dir():
        if resolved_revision:
            return {
                "scheme": "hf_resolved_revision_v1",
                "sha256": str(resolved_revision),
                "files": [],
            }
        raise ValueError("BASE_WEIGHT_IDENTITY_UNPROVEN")
    files = _referenced_weight_files(root)
    if not files:
        raise ValueError(f"BASE_WEIGHT_IDENTITY_MISSING_WEIGHTS: {root}")
    records = sorted((_file_record(root, path) for path in files), key=lambda item: item["path"])
    return {
        "scheme": "hf_local_weight_files_v1",
        "sha256": hashlib.sha256(_canonical_json(records)).hexdigest(),
        "files": records,
    }


def _object_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _object_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_object_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def processor_semantic_identity(processor: Any) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(processor, "chat_template", None)
    if template is None and tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None)
    template_json = json.dumps(template or "", ensure_ascii=False, sort_keys=True, default=str)
    special_tokens = getattr(tokenizer, "special_tokens_map", None) if tokenizer is not None else None
    special_json = json.dumps(_object_value(special_tokens or {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    token_ids = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "image_token_id", "video_token_id"):
        value = getattr(processor, name, None)
        if value is None and tokenizer is not None:
            value = getattr(tokenizer, name, None)
        if value is not None:
            token_ids[name] = value
    return {
        "class": f"{type(processor).__module__}.{type(processor).__name__}",
        "tokenizer_class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__name__}" if tokenizer is not None else None
        ),
        "chat_template_sha256": hashlib.sha256(template_json.encode("utf-8")).hexdigest(),
        "special_tokens_sha256": hashlib.sha256(special_json.encode("utf-8")).hexdigest(),
        "special_token_ids": _object_value(token_ids),
    }


def processor_content_identity(
    processor_dir: str | Path,
    processor: Any | None = None,
    *,
    include_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Combine processor semantic identity with a canonical saved-file tree."""

    root = Path(processor_dir)
    tree = artifact_tree_identity(root, include_paths=include_paths)
    semantic = processor_semantic_identity(processor) if processor is not None else {
        "class": None,
        "tokenizer_class": None,
        "chat_template_sha256": None,
        "special_tokens_sha256": None,
        "special_token_ids": {},
    }
    return {**semantic, "content_sha256": tree["sha256"], "files": tree["files"]}


def processor_semantic_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("class", "tokenizer_class", "chat_template_sha256", "special_tokens_sha256", "special_token_ids")
    return all(left.get(key) == right.get(key) for key in keys)
