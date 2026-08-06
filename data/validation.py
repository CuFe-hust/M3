"""Sample-level input-fact validation for the data layer.

数据层样本级输入事实校验：图片存在、路径不逃逸数据根、角色唯一、
change pair 数量检查。所有验证函数只读，不自动修复源数据，
不做任何隐式图片猜测或全数据根扫描。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from data.adapters.base import DatasetProbeError, read_json_rows
from data.schema import ImageRef, TaskName, UnifiedSample, ValidationIssue

CHANGE_TASKS = frozenset({"change_caption", "change_qa"})


class ValidationReport(BaseModel):
    """Serializable aggregate result of one validation pass.
    一次校验的可序列化聚合结果。"""

    model_config = ConfigDict(extra="forbid")

    source: str
    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


def _is_absolute_like(value: str) -> bool:
    """Detect absolute paths on both Windows and POSIX spellings.
    同时识别 Windows 与 POSIX 写法的绝对路径（与 data.schema 一致）。"""
    if value.startswith(("/", "\\")):
        return True
    return len(value) >= 3 and value[1] == ":" and value[2] in "/\\"


def validate_roles(task: TaskName, roles: Sequence[str]) -> list[ValidationIssue]:
    """Role uniqueness and change-pair count checks.
    角色唯一性与 change pair 数量检查（不修改任何数据）。"""
    issues: list[ValidationIssue] = []
    counts = Counter(roles)
    if task in CHANGE_TASKS:
        if counts.get("t1", 0) != 1:
            issues.append(
                ValidationIssue(
                    code="MISSING_T1",
                    message="change samples require exactly one t1 image",
                )
            )
        if counts.get("t2", 0) != 1:
            issues.append(
                ValidationIssue(
                    code="MISSING_T2",
                    message="change samples require exactly one t2 image",
                )
            )
    elif counts.get("image", 0) != 1:
        issues.append(
            ValidationIssue(
                code="MISSING_PRIMARY_IMAGE",
                message="non-change samples require exactly one primary image",
            )
        )
    return issues


def validate_image_facts(images: Sequence[ImageRef], data_root: Path) -> list[ValidationIssue]:
    """Image existence and data-root escape checks, read-only and explicit.
    图片存在与数据根逃逸检查；只读、只校验显式路径，不做隐式猜测。"""
    issues: list[ValidationIssue] = []
    root = data_root.resolve()
    for ref in images:
        text = str(ref.path)
        if _is_absolute_like(text):
            issues.append(
                ValidationIssue(
                    code="PATH_ESCAPES_DATA_ROOT",
                    message=f"image path escapes data root: {text}",
                )
            )
            continue
        candidate = (data_root / text).resolve()
        if not candidate.is_relative_to(root):
            issues.append(
                ValidationIssue(
                    code="PATH_ESCAPES_DATA_ROOT",
                    message=f"image path escapes data root: {text}",
                )
            )
            continue
        if not candidate.is_file():
            issues.append(
                ValidationIssue(
                    code="MISSING_IMAGE_FILE",
                    message=f"image file does not exist: {text}",
                )
            )
    return issues


def validate_sample(sample: UnifiedSample, data_root: Path) -> ValidationReport:
    """Validate one sample's roles and image facts; never mutates inputs.
    校验单条样本的角色与图片事实；绝不修改输入。"""
    issues = validate_roles(sample.task, [image.role for image in sample.images])
    issues += validate_image_facts(sample.images, data_root)
    report = ValidationReport(source=sample.sample_id, ok=not issues, issues=issues)
    if sample.sample_id:
        report = report.model_copy(
            update={
                "issues": [
                    issue.model_copy(update={"sample_id": sample.sample_id})
                    for issue in report.issues
                ]
            }
        )
    return report


# ── Read-only dataset root audit / 只读数据根审计 ──────────────────────────

MANIFEST_SUFFIXES = (".json", ".jsonl")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")
_IMAGE_REF_KEYS = ("image", "Image", "image_path", "img_path", "img", "image_name", "file_name", "filename", "image_id")
_SPLIT_KEYS = ("split", "Split", "subset", "partition")
_ID_KEYS = ("id", "ID", "question_id", "Question_id", "image_id", "sample_id")


class DatasetAuditReport(BaseModel):
    """Serializable evidence from a read-only dataset layout inspection.
    可序列化的只读数据布局检查证据。"""

    model_config = ConfigDict(extra="forbid")

    root: str
    file_count: int
    extension_counts: dict[str, int]
    candidate_manifests: list[str]
    image_count: int
    image_samples: list[dict[str, object]]
    damaged_images: list[str]
    discovered_field_names: list[str]
    split_hints: list[str]
    duplicate_ids: list[str]
    encoding_errors: list[str]
    missing_referenced_images: list[str]
    notes: list[str] = Field(default_factory=list)


def audit_dataset_root(
    root: Path,
    *,
    image_sample_limit: int = 10,
    json_record_limit: int = 200,
) -> DatasetAuditReport:
    """Inspect a local dataset root without mutating contents or annotations.
    检查本地数据根而不修改任何内容或标注；全部输出可 JSON 序列化。"""
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
        root=str(root),
        file_count=len(files),
        extension_counts=dict(sorted(extension_counts.items())),
        candidate_manifests=[path.relative_to(root).as_posix() for path in manifests],
        image_count=len(images),
        image_samples=image_samples,
        damaged_images=sorted(damaged_images),
        discovered_field_names=sorted(fields),
        split_hints=sorted(split_hints),
        duplicate_ids=sorted(duplicate_ids),
        encoding_errors=encoding_errors,
        missing_referenced_images=sorted(missing_images),
        notes=["Read-only audit; source dataset files were not modified."],
    )


def _inspect_images(
    root: Path,
    images: list[Path],
    limit: int,
) -> tuple[list[dict[str, object]], list[str]]:
    """Sample image dimensions; verify decodability without full loads.
    抽样图片尺寸；在不整图加载的前提下校验可解码性。"""
    samples: list[dict[str, object]] = []
    damaged: list[str] = []
    for path in images[:limit]:
        relative = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            samples.append({"path": relative, "width": width, "height": height})
        except Exception as error:  # noqa: BLE001 - audit records any decode failure
            damaged.append(f"{relative}: {type(error).__name__}: {error}")
    return samples, damaged


def _inspect_json_records(
    root: Path,
    manifests: list[Path],
    limit: int,
) -> tuple[set[str], set[str], set[str], list[str], set[str]]:
    """Collect field names, split hints, duplicate ids, encoding errors, and
    missing referenced images from annotation manifests (read-only).
    从标注清单收集字段名、split 提示、重复 id、编码错误与缺失引用图片。"""
    fields: set[str] = set()
    split_hints: set[str] = set()
    duplicate_ids: set[str] = set()
    encoding_errors: list[str] = []
    missing_images: set[str] = set()
    seen_ids: dict[str, str] = {}
    for manifest in manifests:
        try:
            rows = read_json_rows(manifest)
        except (DatasetProbeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            encoding_errors.append(f"{manifest.name}: {type(error).__name__}: {error}")
            continue
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            fields.update(str(key) for key in row)
            for key in _SPLIT_KEYS:
                if row.get(key) is not None:
                    split_hints.add(str(row[key]))
            for key in _ID_KEYS:
                if row.get(key) is not None:
                    value = str(row[key])
                    if value in seen_ids:
                        duplicate_ids.add(value)
                    seen_ids[value] = manifest.name
            for key in _IMAGE_REF_KEYS:
                value = row.get(key)
                if value is None:
                    continue
                candidates = value if isinstance(value, list) else [value]
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        candidate = candidate.get("path") or candidate.get("file_name")
                    if not isinstance(candidate, str) or not candidate:
                        continue
                    if not (root / candidate).is_file():
                        missing_images.add(candidate)
    return fields, split_hints, duplicate_ids, encoding_errors, missing_images
