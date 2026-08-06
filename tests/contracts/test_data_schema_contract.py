"""Contract tests for the final data-layer unified sample models.

data.schema 统一样本契约测试：合法任务、时相顺序、路径序列化、
extra=forbid、PIL 对象拒绝，以及旧 Golden fixture 到新 Schema 的无损映射。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from data import __all__ as data_exports
from data.schema import (
    GroundTruth,
    ImageRef,
    TaskNormalization,
    UnifiedSample,
    ValidationIssue,
    stable_sample_id,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "migration"

ALL_TASKS = (
    "counting",
    "fine_grained_counting",
    "change_caption",
    "change_qa",
    "grounding",
    "spatial_relation",
    "scene_classification",
    "general_vqa",
    "caption",
    "multiple_choice_vqa",
)


def _image(role: str = "image", image_id: str = "img") -> ImageRef:
    return ImageRef(image_id=image_id, path="image.png", role=role, width=4, height=4)


def _sample(task: str = "general_vqa", images: list[ImageRef] | None = None) -> UnifiedSample:
    return UnifiedSample(
        sample_id="sample-1",
        dataset="VRSBench",
        split="validation",
        task=task,
        images=images if images is not None else [_image()],
        question="Is the statement correct?",
    )


# ── TaskName / 任务名 ──────────────────────────────────────────────────────


def test_all_public_task_names_are_accepted() -> None:
    for task in ALL_TASKS:
        images = [_image("t1"), _image("t2")] if task in {"change_caption", "change_qa"} else [_image()]
        sample = _sample(task=task, images=images)
        assert sample.task == task


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValidationError):
        UnifiedSample.model_validate(
            {
                "sample_id": "s",
                "dataset": "VRSBench",
                "split": "validation",
                "task": "detection",
                "images": [_image().model_dump()],
                "question": "q",
            }
        )


# ── ImageRef / 图像引用 ─────────────────────────────────────────────────────


def test_image_ref_accepts_str_and_path_inputs() -> None:
    from pathlib import Path as PathType

    assert ImageRef(image_id="i", path="a/b.png", role="image").path == PathType("a/b.png")
    assert ImageRef(image_id="i", path=PathType("a/b.png"), role="image").path == PathType("a/b.png")


def test_image_ref_rejects_non_path_objects() -> None:
    with pytest.raises(ValidationError):
        ImageRef(image_id="i", path=object(), role="image")
    try:
        from PIL import Image as PILImage
    except ImportError:
        PILImage = None
    if PILImage is not None:
        with pytest.raises(ValidationError):
            ImageRef(image_id="i", path=PILImage.new("RGB", (4, 4)), role="image")


def test_image_ref_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ImageRef.model_validate(
            {"image_id": "i", "path": "a.png", "role": "image", "pixels": "base64..."}
        )


def test_image_ref_rejects_empty_image_id() -> None:
    with pytest.raises(ValidationError):
        ImageRef(image_id="", path="a.png", role="image")


# ── Path serialization / 路径序列化 ─────────────────────────────────────────


def test_path_serializes_with_forward_slashes() -> None:
    ref = ImageRef(image_id="i", path=Path("dir") / "sub" / "image.png", role="image")
    dumped = ref.model_dump(mode="json")
    assert dumped["path"] == "dir/sub/image.png"


def test_windows_style_path_input_serializes_as_posix() -> None:
    ref = ImageRef(image_id="i", path=r"C:\data\images\a.png", role="image")
    assert ref.model_dump(mode="json")["path"] == "C:/data/images/a.png"


# ── GroundTruth / 真值 ──────────────────────────────────────────────────────


def test_ground_truth_rejects_negative_count() -> None:
    with pytest.raises(ValidationError):
        GroundTruth(count=-1)


def test_ground_truth_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        GroundTruth.model_validate({"answers": ["yes"], "score": 0.9})


def test_ground_truth_defaults() -> None:
    truth = GroundTruth()
    assert truth.answers == [] and truth.count is None and truth.raw == {}


# ── UnifiedSample / 统一样本 ────────────────────────────────────────────────


def test_empty_images_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _sample(images=[])


def test_change_task_requires_ordered_t1_t2() -> None:
    ordered = _sample(task="change_caption", images=[_image("t1"), _image("t2")])
    assert [item.role for item in ordered.images] == ["t1", "t2"]
    with pytest.raises(ValidationError, match="t1 before t2"):
        _sample(task="change_caption", images=[_image("t2"), _image("t1")])
    with pytest.raises(ValidationError, match="t1 before t2"):
        _sample(task="change_qa", images=[_image("image")])
    # Extra context after the ordered pair remains valid. / 有序时相图像后的额外 context 仍合法。
    extra = _sample(task="change_caption", images=[_image("t1"), _image("t2"), _image("context")])
    assert len(extra.images) == 3


def test_non_change_tasks_allow_single_image() -> None:
    sample = _sample(task="general_vqa", images=[_image("image")])
    assert sample.task == "general_vqa"


def test_unified_sample_rejects_extra_fields() -> None:
    payload = _sample().model_dump()
    payload["sample_id"] = "s"
    payload["router"] = "extra"
    with pytest.raises(ValidationError):
        UnifiedSample.model_validate(payload)


def test_metadata_preserves_arbitrary_json_values() -> None:
    sample = _sample()
    sample = sample.model_copy(update={"metadata": {"question_type": "quantity", "nested": {"a": 1}}})
    assert sample.metadata["question_type"] == "quantity"
    assert sample.metadata["nested"]["a"] == 1


# ── TaskNormalization / 任务规范化 ──────────────────────────────────────────


def test_task_normalization_contract() -> None:
    norm = TaskNormalization(
        source_task="vrsbench_vqa",
        normalized_task="counting",
        semantic_subtype="quantity",
        confidence=0.95,
        normalizer="vrsbench_task_normalizer",
        version="1",
        reason_codes=["quantity_question"],
        spatial_query="target",
        answer_constraints=["0..999"],
        count_target_hint="building",
    )
    assert norm.normalized_task == "counting"
    assert norm.confidence == 0.95
    assert norm.answer_constraints == ["0..999"]


def test_task_normalization_rejects_invalid_task_and_confidence() -> None:
    with pytest.raises(ValidationError):
        TaskNormalization(source_task="x", normalized_task="detection", normalizer="n", version="1")
    with pytest.raises(ValidationError):
        TaskNormalization(source_task="x", normalized_task="counting", normalizer="n", version="1", confidence=1.5)


# ── ValidationIssue / 校验问题 ──────────────────────────────────────────────


def test_validation_issue_contract() -> None:
    issue = ValidationIssue(code="MISSING_IMAGE", message="image missing", sample_id="s1")
    assert issue.severity == "error"
    warning = ValidationIssue(code="W", message="m", severity="warning")
    assert warning.severity == "warning"
    with pytest.raises(ValidationError):
        ValidationIssue(code="W", message="m", severity="fatal")


# ── stable_sample_id / 稳定样本 ID ──────────────────────────────────────────


def test_stable_sample_id_matches_baseline_digest() -> None:
    assert stable_sample_id("", Path("img.png"), "How many buildings?", 3) == "dc7c628994b424e07336"
    assert stable_sample_id("qid-42", Path("img.png"), "q", 0) == "qid-42"


# ── data package exports / data 包导出 ──────────────────────────────────────


def test_data_init_only_exports_stable_types() -> None:
    expected = {
        "GroundTruth",
        "ImageRef",
        "ImageRole",
        "TaskName",
        "TaskNormalization",
        "UnifiedSample",
        "ValidationIssue",
        "stable_sample_id",
    }
    assert set(data_exports) == expected


def test_data_init_has_no_import_side_effects() -> None:
    tree = ast.parse((REPO_ROOT / "data" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        assert not isinstance(node, (ast.Call, ast.Try, ast.With, ast.If)), (
            f"data/__init__.py contains forbidden top-level {type(node).__name__}"
        )


# ── dependency boundary / 依赖边界 ──────────────────────────────────────────


def test_schema_imports_only_data_and_stdlib() -> None:
    source = (REPO_ROOT / "data" / "schema.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "agents",
        "workflows",
        "application",
        "routing",
        "evaluation",
        "reporting",
        "models",
        "spacers_agent",
        "eval",
    }
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    assert not (tops & forbidden), f"data/schema.py imports forbidden packages: {tops & forbidden}"


# ── Golden fixture round-trip / 旧 fixture 无损映射 ─────────────────────────


def test_golden_fixtures_map_losslessly_into_new_schema() -> None:
    case_dirs = [p for p in FIXTURE_ROOT.iterdir() if p.is_dir()]
    assert len(case_dirs) >= 9, "migration golden fixtures missing"
    for case_dir in sorted(case_dirs):
        sample_json = json.loads((case_dir / "sample.json").read_text(encoding="utf-8"))
        sample = UnifiedSample.model_validate(sample_json)
        assert sample.sample_id == sample_json["sample_id"], case_dir.name
        assert sample.dataset == sample_json["dataset"]
        assert sample.task == sample_json["task"]
        assert sample.split == sample_json["split"]
        assert len(sample.images) == len(sample_json["images"])
        for new_ref, old_ref in zip(sample.images, sample_json["images"]):
            assert new_ref.image_id == old_ref["image_id"]
            assert new_ref.role == old_ref["role"]
            assert new_ref.path.as_posix() == old_ref["path"], f"{case_dir.name}: path mutated"
            assert new_ref.width == old_ref["width"] and new_ref.height == old_ref["height"]
        if sample_json.get("ground_truth") is not None:
            assert sample.ground_truth is not None
            assert sample.ground_truth.answers == sample_json["ground_truth"]["answers"]
        # Re-dump must preserve the fixture's stable fields. / 重新序列化必须保留稳定字段。
        redump = sample.model_dump(mode="json")
        assert redump["task"] == sample_json["task"]
        assert [item["path"] for item in redump["images"]] == [
            item["path"] for item in sample_json["images"]
        ]
