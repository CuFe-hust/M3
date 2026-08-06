"""Contract tests for the data-layer input-fact validation and audit.

data/validation.py 输入事实校验与只读审计测试：
- 角色唯一、change pair 数量、图片存在、路径不逃逸、聚合报告、只读性；
- 只读目录审计：扩展名、候选标注、图片尺寸/损坏、字段、split、重复 ID、
  缺图、编码错误；审计不修改数据源；输出可 JSON 序列化。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from data.schema import ImageRef, UnifiedSample
from data.validation import (
    DatasetAuditReport,
    ValidationReport,
    audit_dataset_root,
    validate_image_facts,
    validate_roles,
    validate_sample,
)


def _make_image(path: Path, seed: int = 3, size: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (seed, seed * 2, seed * 3)).save(path)


def _ref(relative: str, role: str = "image") -> ImageRef:
    return ImageRef(image_id=relative, path=relative, role=role, width=4, height=4)


def _sample(task: str = "general_vqa", images=None, sample_id: str = "s1") -> UnifiedSample:
    return UnifiedSample(
        sample_id=sample_id,
        dataset="parity",
        split="test",
        task=task,
        images=images if images is not None else [_ref("img.png")],
        question="Q",
    )


# ── 角色唯一与 change pair / role uniqueness and change pair ───────────────


def test_change_pair_requires_one_t1_and_one_t2() -> None:
    issues = validate_roles("change_caption", ["t1", "t2"])
    assert issues == []
    only_t1 = validate_roles("change_caption", ["t1"])
    assert [issue.code for issue in only_t1] == ["MISSING_T2"]
    only_t2 = validate_roles("change_qa", ["t2"])
    assert [issue.code for issue in only_t2] == ["MISSING_T1"]
    none_roles = validate_roles("change_caption", ["context"])
    assert {issue.code for issue in none_roles} == {"MISSING_T1", "MISSING_T2"}
    duplicated = validate_roles("change_caption", ["t1", "t2", "t1"])
    assert [issue.code for issue in duplicated] == ["MISSING_T1"]


def test_non_change_requires_exactly_one_primary_image() -> None:
    assert validate_roles("general_vqa", ["image"]) == []
    assert validate_roles("general_vqa", ["image", "context"]) == []
    no_primary = validate_roles("general_vqa", ["context"])
    assert [issue.code for issue in no_primary] == ["MISSING_PRIMARY_IMAGE"]
    duplicated = validate_roles("counting", ["image", "image"])
    assert [issue.code for issue in duplicated] == ["MISSING_PRIMARY_IMAGE"]


# ── 图片存在与逃逸 / image existence and escape ────────────────────────────


def test_image_facts_accept_existing_images(tmp_path: Path) -> None:
    _make_image(tmp_path / "img.png")
    issues = validate_image_facts([_ref("img.png")], tmp_path)
    assert issues == []


def test_missing_image_file_is_reported(tmp_path: Path) -> None:
    issues = validate_image_facts([_ref("missing.png")], tmp_path)
    assert len(issues) == 1
    assert issues[0].code == "MISSING_IMAGE_FILE"
    assert "missing.png" in issues[0].message


def test_path_escape_is_reported(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.png"
    _make_image(outside)
    issues = validate_image_facts([_ref("../outside.png")], tmp_path)
    assert len(issues) == 1
    assert issues[0].code == "PATH_ESCAPES_DATA_ROOT"
    assert "outside.png" in issues[0].message


def test_absolute_path_is_reported_as_escape(tmp_path: Path) -> None:
    for bad in ("C:/data/img.png", r"C:\data\img.png", "/abs/img.png", r"\\server\share\img.png"):
        issues = validate_image_facts([_ref(bad)], tmp_path)
        assert len(issues) == 1, bad
        assert issues[0].code == "PATH_ESCAPES_DATA_ROOT", bad


def test_nested_relative_path_within_root_is_accepted(tmp_path: Path) -> None:
    _make_image(tmp_path / "A" / "01_t1.png")
    issues = validate_image_facts([_ref("A/01_t1.png")], tmp_path)
    assert issues == []


# ── validate_sample 聚合 / aggregate validation ─────────────────────────────


def test_validate_sample_ok(tmp_path: Path) -> None:
    _make_image(tmp_path / "img.png")
    report = validate_sample(_sample(images=[_ref("img.png")]), tmp_path)
    assert report.ok is True
    assert report.issues == []
    assert report.source == "s1"


def test_validate_sample_aggregates_role_and_image_issues(tmp_path: Path) -> None:
    report = validate_sample(
        _sample(task="change_caption", images=[_ref("t1.png", "t1"), _ref("missing_t2.png", "t2")]),
        tmp_path,
    )
    assert report.ok is False
    codes = {issue.code for issue in report.issues}
    assert codes == {"MISSING_IMAGE_FILE"}
    assert all(issue.sample_id == "s1" for issue in report.issues)


def test_validation_report_is_json_serializable(tmp_path: Path) -> None:
    _make_image(tmp_path / "img.png")
    report = validate_sample(_sample(images=[_ref("img.png")]), tmp_path)
    payload = report.model_dump(mode="json")
    json.dumps(payload)
    assert payload["ok"] is True
    rebuilt = ValidationReport.model_validate(payload)
    assert rebuilt.source == "s1"


def test_validation_is_read_only(tmp_path: Path) -> None:
    _make_image(tmp_path / "img.png")
    before = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    content_before = (tmp_path / "img.png").read_bytes()
    validate_sample(_sample(images=[_ref("img.png")]), tmp_path)
    after = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    assert before == after
    assert (tmp_path / "img.png").read_bytes() == content_before


def test_validation_does_not_scan_the_data_root(tmp_path: Path) -> None:
    """No implicit guessing: only explicitly referenced paths are checked.
    无隐式猜测：只校验显式引用的路径。"""
    _make_image(tmp_path / "unreferenced.png")
    issues = validate_image_facts([_ref("missing.png")], tmp_path)
    assert [issue.code for issue in issues] == ["MISSING_IMAGE_FILE"]


# ── 只读目录审计 / read-only dataset-root audit ────────────────────────────


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_audit_root(tmp_path: Path) -> Path:
    root = tmp_path / "audit"
    _make_image(root / "images" / "a.png", size=6)
    _make_image(root / "images" / "b.jpg", size=6)
    _write_json(root / "ann.json", [
        {"id": "dup", "split": "test", "image": "images/a.png"},
        {"id": "dup", "split": "test", "image": "images/missing.png"},
        {"id": "ok", "split": "train", "question": "Q", "image": "images/b.jpg"},
    ])
    _write(root / "notes.txt", "not a manifest")
    return root


def test_audit_counts_extensions_and_manifests(tmp_path: Path) -> None:
    root = _build_audit_root(tmp_path)
    report = audit_dataset_root(root)
    assert report.file_count == 4
    assert report.extension_counts[".png"] == 1
    assert report.extension_counts[".jpg"] == 1
    assert report.extension_counts[".json"] == 1
    assert report.extension_counts[".txt"] == 1
    assert report.candidate_manifests == ["ann.json"]
    assert report.image_count == 2


def test_audit_discovers_fields_split_hints_and_duplicate_ids(tmp_path: Path) -> None:
    report = audit_dataset_root(_build_audit_root(tmp_path))
    assert "question" in report.discovered_field_names
    assert "id" in report.discovered_field_names
    assert "test" in report.split_hints and "train" in report.split_hints
    assert report.duplicate_ids == ["dup"]


def test_audit_reports_missing_referenced_images(tmp_path: Path) -> None:
    report = audit_dataset_root(_build_audit_root(tmp_path))
    assert "images/missing.png" in report.missing_referenced_images


def test_audit_reports_encoding_errors(tmp_path: Path) -> None:
    root = tmp_path / "audit_bad"
    _write(root / "broken.json", "{not valid json")
    report = audit_dataset_root(root)
    assert len(report.encoding_errors) == 1
    assert "broken.json" in report.encoding_errors[0]


def test_audit_reports_damaged_images(tmp_path: Path) -> None:
    root = tmp_path / "audit_damaged"
    (root / "images").mkdir(parents=True)
    (root / "images" / "bad.png").write_bytes(b"not a png at all")
    report = audit_dataset_root(root)
    assert len(report.damaged_images) == 1
    assert "bad.png" in report.damaged_images[0]


def test_audit_samples_image_dimensions(tmp_path: Path) -> None:
    report = audit_dataset_root(_build_audit_root(tmp_path))
    assert len(report.image_samples) == 2
    dimensions = {(item["path"], item["width"], item["height"]) for item in report.image_samples}
    assert ("images/a.png", 6, 6) in dimensions
    assert ("images/b.jpg", 6, 6) in dimensions


def test_audit_is_read_only(tmp_path: Path) -> None:
    root = _build_audit_root(tmp_path)
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    audit_dataset_root(root)
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    assert before == after


def test_audit_report_is_json_serializable(tmp_path: Path) -> None:
    root = _build_audit_root(tmp_path)
    report = audit_dataset_root(root)
    payload = report.model_dump(mode="json")
    json.dumps(payload)
    rebuilt = DatasetAuditReport.model_validate(payload)
    assert rebuilt.file_count == report.file_count
    assert rebuilt.root == str(root)


def test_audit_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        audit_dataset_root(tmp_path / "nope")
