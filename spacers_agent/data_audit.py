"""Read-only generic dataset audit for local layout discovery.
用于本地布局发现的只读通用数据审计。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
MANIFEST_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".yaml", ".yml"}
FIELD_TOKENS = ("split", "question", "answer", "caption", "bbox", "box", "point", "image_a", "image_b", "before", "after")


class DatasetAuditReport(BaseModel):
    """Serializable evidence from a read-only dataset layout inspection.
    可序列化的只读数据布局检查证据。
    """

    model_config = ConfigDict(extra="forbid")

    root: Path
    file_count: int
    extension_counts: dict[str, int]
    candidate_manifests: list[str]
    image_count: int
    image_samples: list[dict[str, Any]]
    damaged_images: list[str]
    discovered_field_names: list[str]
    split_hints: list[str]
    duplicate_ids: list[str]
    encoding_errors: list[str]
    missing_referenced_images: list[str]
    notes: list[str] = Field(default_factory=list)


def inspect_dataset_root(root: Path, *, image_sample_limit: int = 10, json_record_limit: int = 200) -> DatasetAuditReport:
    """Inspect local files without mutating dataset contents or annotations.
    检查本地文件而不修改数据集内容或标注。
    """

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    files = [path for path in root.rglob("*") if path.is_file()]
    extension_counts = Counter(path.suffix.lower() or "<none>" for path in files)
    manifests = [path for path in files if path.suffix.lower() in MANIFEST_SUFFIXES]
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    image_samples, damaged_images = _inspect_images(root, images, image_sample_limit)
    fields, split_hints, duplicate_ids, encoding_errors, missing_images = _inspect_json_records(
        root, manifests, json_record_limit
    )
    return DatasetAuditReport(
        root=root,
        file_count=len(files),
        extension_counts=dict(sorted(extension_counts.items())),
        candidate_manifests=[path.relative_to(root).as_posix() for path in manifests],
        image_count=len(images),
        image_samples=image_samples,
        damaged_images=damaged_images,
        discovered_field_names=sorted(fields),
        split_hints=sorted(split_hints),
        duplicate_ids=sorted(duplicate_ids),
        encoding_errors=encoding_errors,
        missing_referenced_images=sorted(missing_images),
        notes=["Read-only audit; source dataset files were not modified."],
    )


def write_dataset_audit(report: DatasetAuditReport, output: Path) -> None:
    """Write a separate JSON report without touching dataset source files.
    写入独立 JSON 报告而不接触数据集源文件。
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _inspect_images(root: Path, images: list[Path], limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    damaged: list[str] = []
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if len(samples) < limit:
                    samples.append(
                        {
                            "path": image_path.relative_to(root).as_posix(),
                            "width": image.width,
                            "height": image.height,
                            "format": image.format,
                        }
                    )
        except (OSError, UnidentifiedImageError):
            damaged.append(image_path.relative_to(root).as_posix())
    return samples, damaged


def _inspect_json_records(
    root: Path, manifests: list[Path], limit: int
) -> tuple[set[str], set[str], set[str], list[str], set[str]]:
    fields: set[str] = set()
    split_hints: set[str] = set()
    duplicate_ids: set[str] = set()
    encoding_errors: list[str] = []
    missing_images: set[str] = set()
    seen_ids: set[str] = set()
    inspected = 0
    for manifest in manifests:
        if manifest.suffix.lower() not in {".json", ".jsonl"} or inspected >= limit:
            continue
        try:
            records = _read_json_records(manifest)
        except UnicodeDecodeError:
            encoding_errors.append(manifest.relative_to(root).as_posix())
            continue
        except (json.JSONDecodeError, ValueError):
            continue
        for record in records:
            if inspected >= limit:
                break
            inspected += 1
            fields.update(record)
            split = record.get("split") or record.get("Split")
            if split:
                split_hints.add(str(split))
            source_id = record.get("id") or record.get("ID") or record.get("image_id")
            if source_id is not None:
                normalized_id = str(source_id)
                if normalized_id in seen_ids:
                    duplicate_ids.add(normalized_id)
                seen_ids.add(normalized_id)
            for key, value in record.items():
                if key.lower() in {"image", "image_path", "img_path", "image_a", "image_b", "before", "after"}:
                    candidate = root / str(value)
                    if value and not candidate.exists():
                        missing_images.add(str(value))
    fields.update(token for token in FIELD_TOKENS if token in {field.lower() for field in fields})
    return fields, split_hints, duplicate_ids, encoding_errors, missing_images


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in file if line.strip()]
        payload = json.load(file)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "annotations", "samples", "items", "images"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Unsupported JSON record layout")
