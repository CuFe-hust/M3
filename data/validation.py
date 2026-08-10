"""Sample-level input-fact validation and read-only dataset audit.

数据层样本级输入事实校验与只读数据根审计：图片存在、路径不逃逸数据根、
角色唯一、change pair 数量检查；只读目录审计（扩展名、候选标注、图片
尺寸/损坏、字段、split、按字段语义分组的重复 ID、缺失/未决/歧义引用
图片、扫描计数与截断标记）。所有验证与审计函数只读，不自动修复源数据，
不做隐式图片猜测或全数据根扫描（除显式审计命令）。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal, Sequence

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
    ImageRef paths are already schema-guaranteed to be relative and
    non-escape; resolution still verifies they stay inside the resolved root.
    图片存在与数据根逃逸检查；只读、只校验显式路径，不做隐式猜测。
    ImageRef 路径已由 Schema 保证相对且无逃逸段；解析仍验证其位于解析根内。"""
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
# Image-reference fields recognized by the generic audit. / 通用审计识别的图片引用字段。
_IMAGE_REF_KEYS = (
    "image", "Image", "image_path", "img_path", "img", "image_name",
    "file_name", "filename", "image_id", "images", "image_A", "image_B",
    "before", "after",
)
_SPLIT_KEYS = ("split", "Split", "subset", "partition")
# Duplicate-ID fields grouped by semantics: ids only compare within their own
# semantic field. image_id is not a sample-unique id and is never reported.
# 重复 ID 字段按语义分组比较；image_id 不是样本唯一 ID，不报告。
_DUPLICATE_ID_FIELDS = ("id", "ID", "question_id", "Question_id", "sample_id")


class DatasetAuditReport(BaseModel):
    """Serializable evidence from a read-only dataset layout inspection.
    可序列化的只读数据布局检查证据。"""

    model_config = ConfigDict(extra="forbid")

    root: str
    file_count: int
    extension_counts: dict[str, int]
    candidate_manifests: list[str]
    image_count: int
    images_scanned: int
    image_scan_truncated: bool
    image_samples: list[dict[str, object]]
    damaged_images: list[str]
    discovered_field_names: list[str]
    split_hints: list[str]
    duplicate_ids: list[str]
    encoding_errors: list[str]
    missing_referenced_images: list[str]
    unresolved_referenced_images: list[str]
    ambiguous_referenced_images: list[str]
    escaped_referenced_images: list[str]
    records_scanned: int
    record_scan_truncated: bool
    manifest_record_counts: dict[str, int]
    scan_mode: Literal["quick", "full"]
    notes: list[str] = Field(default_factory=list)


def audit_dataset_root(
    root: Path,
    *,
    image_sample_limit: int = 10,
    json_record_limit: int = 200,
    scan_mode: Literal["quick", "full"] = "quick",
) -> DatasetAuditReport:
    """Inspect a local dataset root without mutating contents or annotations;
    all outputs are JSON-serializable. quick mode samples; full mode checks
    every image and record and reports exact counts.
    检查本地数据根而不修改任何内容或标注；全部输出可 JSON 序列化。
    quick 模式抽样；full 模式检查全部图片与记录并报告精确计数。"""
    if scan_mode not in {"quick", "full"}:
        raise ValueError(f"scan_mode must be 'quick' or 'full', got {scan_mode!r}")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    # Deterministic ordering: audit results must not depend on directory
    # iteration order. 确定性排序：审计结果不依赖目录迭代顺序。
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    extension_counts = Counter(path.suffix.lower() or "<none>" for path in files)
    manifests = sorted(
        (path for path in files if path.suffix.lower() in MANIFEST_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    images = sorted(
        (path for path in files if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix(),
    )

    full = scan_mode == "full"
    image_limit = None if full else image_sample_limit
    record_limit = None if full else json_record_limit
    image_samples, damaged_images, images_scanned = _inspect_images(
        root, images, image_limit
    )
    (
        fields, split_hints, duplicate_ids, encoding_errors,
        missing, unresolved, ambiguous, escaped, records_scanned, manifest_counts,
    ) = _inspect_json_records(root, manifests, record_limit)
    return DatasetAuditReport(
        root=str(root),
        file_count=len(files),
        extension_counts=dict(sorted(extension_counts.items())),
        candidate_manifests=[path.relative_to(root).as_posix() for path in manifests],
        image_count=len(images),
        images_scanned=images_scanned,
        image_scan_truncated=full is False and images_scanned < len(images),
        image_samples=image_samples,
        damaged_images=sorted(damaged_images),
        discovered_field_names=sorted(fields),
        split_hints=sorted(split_hints),
        duplicate_ids=sorted(duplicate_ids),
        encoding_errors=encoding_errors,
        missing_referenced_images=sorted(missing),
        unresolved_referenced_images=sorted(unresolved),
        ambiguous_referenced_images=sorted(ambiguous),
        escaped_referenced_images=sorted(escaped),
        records_scanned=records_scanned,
        record_scan_truncated=full is False and records_scanned < sum(manifest_counts.values()),
        manifest_record_counts=dict(sorted(manifest_counts.items())),
        scan_mode=scan_mode,
        notes=["Read-only audit; source dataset files were not modified."],
    )


def _inspect_images(
    root: Path,
    images: list[Path],
    limit: int | None,
) -> tuple[list[dict[str, object]], list[str], int]:
    """Sample image dimensions; verify decodability without full loads.
    Returns (samples, damaged, scanned_count).
    抽样图片尺寸；在不整图加载的前提下校验可解码性。返回（样本、损坏、扫描数）。"""
    samples: list[dict[str, object]] = []
    damaged: list[str] = []
    scanned = 0
    inspected = images if limit is None else images[:limit]
    for path in inspected:
        scanned += 1
        relative = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            samples.append({"path": relative, "width": width, "height": height})
        except Exception as error:  # noqa: BLE001 - audit records any decode failure
            damaged.append(f"{relative}: {type(error).__name__}: {error}")
    return samples, damaged, scanned


def _inspect_json_records(
    root: Path,
    manifests: list[Path],
    limit: int | None,
) -> tuple[
    set[str], set[str], set[str], list[str], set[str], set[str], set[str], set[str], int, dict[str, int]
]:
    """Collect field names, split hints, duplicate ids (grouped by field
    semantics), encoding errors, and missing/unresolved/ambiguous/escaped
    referenced images. 收集字段名、split 提示、按字段语义分组的重复 id、
    编码错误，以及缺失/未决/歧义/逃逸的引用图片。"""
    fields: set[str] = set()
    split_hints: set[str] = set()
    duplicate_ids: set[str] = set()
    encoding_errors: list[str] = []
    missing: set[str] = set()
    unresolved: set[str] = set()
    ambiguous: set[str] = set()
    escaped: set[str] = set()
    seen_ids: dict[str, dict[str, str]] = {}
    manifest_counts: dict[str, int] = {}
    records_scanned = 0
    for manifest in manifests:
        try:
            rows = read_json_rows(manifest)
        except (DatasetProbeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            encoding_errors.append(f"{manifest.name}: {type(error).__name__}: {error}")
            manifest_counts[manifest.name] = 0
            continue
        manifest_counts[manifest.name] = len(rows)
        inspected = rows if limit is None else rows[:limit]
        records_scanned += len(inspected)
        for row in inspected:
            if not isinstance(row, dict):
                continue
            fields.update(str(key) for key in row)
            for key in _SPLIT_KEYS:
                if row.get(key) is not None:
                    split_hints.add(str(row[key]))
            for key in _DUPLICATE_ID_FIELDS:
                if row.get(key) is not None:
                    value = str(row[key])
                    bucket = seen_ids.setdefault(key, {})
                    if value in bucket:
                        duplicate_ids.add(f"{key}:{value}")
                    bucket[value] = manifest.name
            for key in _IMAGE_REF_KEYS:
                value = row.get(key)
                if value is None:
                    continue
                candidates = value if isinstance(value, list) else [value]
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        candidate = (
                            candidate.get("path")
                            or candidate.get("file_name")
                            or candidate.get("filename")
                        )
                    if not isinstance(candidate, str) or not candidate:
                        continue
                    _classify_image_reference(
                        root, candidate, missing, unresolved, ambiguous, escaped
                    )
    return (
        fields, split_hints, duplicate_ids, encoding_errors,
        missing, unresolved, ambiguous, escaped, records_scanned, manifest_counts,
    )


def _classify_image_reference(
    root: Path,
    reference: str,
    missing: set[str],
    unresolved: set[str],
    ambiguous: set[str],
    escaped: set[str],
) -> None:
    """Classify a referenced image with a fixed priority: escaped, ambiguous,
    resolved, missing, unresolved. A reference belongs to exactly one class.
    按固定优先级分类引用图片：escaped、ambiguous、resolved、missing、
    unresolved；一条引用只属于一个类别。"""
    normalized = reference.replace("\\", "/")
    root_resolved = root.resolve()
    if _is_absolute_like(reference) or ".." in normalized.split("/"):
        escaped.add(reference)
        return
    resolved_candidates: set[Path] = set()
    for base in ("", "images", "Images_val"):
        candidate = (root / base / normalized).resolve()
        if not candidate.is_relative_to(root_resolved):
            escaped.add(reference)
            return
        if candidate.is_file():
            resolved_candidates.add(candidate)
    if len(resolved_candidates) > 1:
        ambiguous.add(reference)
        return
    if resolved_candidates:
        return  # resolved / 已解析
    if "/" in normalized:
        missing.add(reference)
    else:
        unresolved.add(reference)
