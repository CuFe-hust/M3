"""Contract tests for the final data-layer unified sample models.

data.schema 统一样本契约测试：合法任务、时相顺序、JSON-safe 值、路径规范化、
frozen ImageRef、结构化 normalization、几何校验、question 语义，
以及旧 Golden fixture 到新 Schema 的无损映射。所有用例通过构造器或
model_validate 构造输入，不使用 model_copy 绕过校验。
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
    SampleDraft,
    SampleMaterializationError,
    TaskNormalization,
    UnifiedSample,
    ValidationIssue,
    materialize_sample,
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


def _image(role: str = "image", image_id: str | None = None) -> ImageRef:
    return ImageRef(
        image_id=image_id or f"img-{role}",
        path="image.png",
        role=role,
        width=4,
        height=4,
    )


def _sample(task: str = "general_vqa", images: list[ImageRef] | None = None, **overrides) -> UnifiedSample:
    payload = {
        "sample_id": "sample-1",
        "dataset": "VRSBench",
        "split": "validation",
        "task": task,
        "images": [item.model_dump() for item in images] if images is not None else [_image().model_dump()],
        "question": "Is the statement correct?",
    }
    payload.update(overrides)
    return UnifiedSample.model_validate(payload)


def _valid_payload(**overrides) -> dict:
    payload = {
        "sample_id": "sample-1",
        "dataset": "VRSBench",
        "split": "validation",
        "task": "general_vqa",
        "images": [_image().model_dump()],
        "question": "Is the statement correct?",
    }
    payload.update(overrides)
    return payload


# ── TaskName / 任务名 ──────────────────────────────────────────────────────


def test_all_public_task_names_are_accepted() -> None:
    for task in ALL_TASKS:
        images = [_image("t1"), _image("t2")] if task in {"change_caption", "change_qa"} else [_image()]
        sample = _sample(task=task, images=images)
        assert sample.task == task


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValidationError):
        UnifiedSample.model_validate(_valid_payload(task="detection"))


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


def test_image_ref_is_frozen() -> None:
    ref = _image()
    with pytest.raises(ValidationError):
        ref.path = Path("other.png")


# ── sha256 / 摘要校验 ───────────────────────────────────────────────────────


def test_sha256_accepts_hex_digests_and_lowercases() -> None:
    upper = "A" * 64
    ref = ImageRef(image_id="i", path="a.png", role="image", sha256=upper)
    assert ref.sha256 == "a" * 64
    ref = ImageRef(image_id="i", path="a.png", role="image", sha256="a" * 64)
    assert ref.sha256 == "a" * 64


def test_sha256_rejects_wrong_length_or_chars() -> None:
    with pytest.raises(ValidationError):
        ImageRef(image_id="i", path="a.png", role="image", sha256="a" * 63)
    with pytest.raises(ValidationError):
        ImageRef(image_id="i", path="a.png", role="image", sha256="z" * 64)


# ── Path serialization / 路径序列化 ─────────────────────────────────────────


def test_path_serializes_with_forward_slashes() -> None:
    ref = ImageRef(image_id="i", path=Path("dir") / "sub" / "image.png", role="image")
    assert ref.model_dump(mode="json")["path"] == "dir/sub/image.png"


def test_windows_style_path_input_is_rejected() -> None:
    """Absolute machine paths are rejected by the schema; paths must be
    dataset-root-relative. 绝对机器路径被 Schema 拒绝；路径必须相对 dataset root。"""
    for bad in (r"C:\data\images\a.png", "C:/data/images/a.png", "/abs/a.png", r"\\server\share\a.png"):
        with pytest.raises(ValidationError, match="relative"):
            ImageRef(image_id="i", path=bad, role="image")
    for bad in ("../outside.png", "images/../../outside.png", "a/./b.png"):
        with pytest.raises(ValidationError, match="segment"):
            ImageRef(image_id="i", path=bad, role="image")


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


def test_ground_truth_box_length_must_be_4_or_8() -> None:
    GroundTruth(boxes=[[1, 2, 3, 4]])
    GroundTruth(boxes=[[1, 2, 3, 4, 5, 6, 7, 8]])
    with pytest.raises(ValidationError, match="4 or 8"):
        GroundTruth(boxes=[[1, 2, 3]])
    with pytest.raises(ValidationError, match="4 or 8"):
        GroundTruth(boxes=[[1, 2, 3, 4, 5]])


def test_ground_truth_point_length_must_be_2() -> None:
    GroundTruth(points=[[1.0, 2.0]])
    with pytest.raises(ValidationError, match="2 coordinates"):
        GroundTruth(points=[[1.0, 2.0, 3.0]])


def test_ground_truth_rejects_non_finite_coordinates() -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        GroundTruth(boxes=[[float("nan"), 1, 2, 3]])
    with pytest.raises(ValidationError, match="non-finite"):
        GroundTruth(boxes=[[float("inf"), 1, 2, 3]])
    with pytest.raises(ValidationError, match="non-finite"):
        GroundTruth(points=[[1.0, float("inf")]])


def test_ground_truth_labels_match_boxes_plus_points() -> None:
    GroundTruth(boxes=[[1, 2, 3, 4]], points=[[5.0, 6.0]], labels=["b", "p"])
    with pytest.raises(ValidationError, match="labels must match"):
        GroundTruth(boxes=[[1, 2, 3, 4]], labels=["a", "b"])


# ── JSON-safe metadata/raw / JSON 安全值 ────────────────────────────────────


def test_metadata_accepts_json_values() -> None:
    sample = _sample(metadata={"question_type": "quantity", "nested": {"a": 1}, "flags": [True, None, 2.5]})
    assert sample.metadata["nested"]["a"] == 1


def test_metadata_rejects_non_json_values() -> None:
    for bad in (Path("x.png"), object(), lambda: 1, {1, 2}, b"bytes", {"k": {1, 2}}):
        with pytest.raises(ValidationError):
            _sample(metadata={"bad": bad})
    with pytest.raises(ValidationError, match="non-finite"):
        _sample(metadata={"bad": float("nan")})


def test_ground_truth_raw_rejects_non_json_values() -> None:
    for bad in (Path("x.png"), object(), lambda: 1, {1, 2}, b"bytes"):
        with pytest.raises(ValidationError):
            GroundTruth(raw={"bad": bad})


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
    with pytest.raises(ValidationError, match="t1 before t2"):
        _sample(task="change_caption", images=[_image("t1")])


def test_change_extra_roles_must_be_context() -> None:
    _sample(task="change_caption", images=[_image("t1"), _image("t2"), _image("context")])
    _sample(task="change_caption", images=[
        _image("t1"), _image("t2"), _image("context"), _image("context", image_id="ctx2"),
    ])
    with pytest.raises(ValidationError, match="only context"):
        _sample(task="change_caption", images=[_image("t1"), _image("t2"), _image("t1")])
    with pytest.raises(ValidationError, match="only context"):
        _sample(task="change_caption", images=[_image("t1"), _image("t2"), _image("t2")])
    with pytest.raises(ValidationError, match="only context"):
        _sample(task="change_caption", images=[_image("t1"), _image("t2"), _image("image")])


def test_non_change_tasks_reject_temporal_roles() -> None:
    _sample(task="general_vqa", images=[_image("image")])
    _sample(task="general_vqa", images=[_image("image"), _image("context")])
    with pytest.raises(ValidationError, match="start with an image"):
        _sample(task="general_vqa", images=[_image("t1")])


def test_materialize_sample_assigns_canonical_roles() -> None:
    draft = SampleDraft(
        sample_id="draft-1",
        dataset="demo",
        split="val",
        images=[
            ImageRef(image_id="a", path="a.png", role="image"),
            ImageRef(image_id="b", path="b.png", role="image"),
            ImageRef(image_id="c", path="c.png", role="image"),
        ],
        question="What changed?",
    )
    change = materialize_sample(draft, "change_qa")
    assert [item.role for item in change.images] == ["t1", "t2", "context"]

    single = materialize_sample(draft.model_copy(update={"images": draft.images[:1]}), "general_vqa")
    assert [item.role for item in single.images] == ["image"]


def test_materialize_sample_preserves_canonical_choice_facts() -> None:
    normalization = TaskNormalization(
        source_task="source_vqa",
        normalized_task="multiple_choice_vqa",
        normalizer="test",
        version="1",
        choices=["(A) Road", "(B) Water"],
        allow_multiple=False,
    )
    draft = SampleDraft(
        sample_id="draft-mc",
        dataset="demo",
        split="val",
        images=[ImageRef(image_id="a", path="a.png", role="image")],
        question="Which class is shown?",
        normalization=normalization,
    )
    sample = materialize_sample(draft, "multiple_choice_vqa")
    assert sample.normalization is not None
    assert sample.normalization.choices == ["(A) Road", "(B) Water"]
    assert sample.normalization.allow_multiple is False


def test_materialize_sample_fails_closed_for_unknown_or_invalid_task() -> None:
    draft = SampleDraft(
        sample_id="draft-1",
        dataset="demo",
        split="val",
        images=[ImageRef(image_id="a", path="a.png", role="image")],
        question="How many?",
    )
    with pytest.raises(SampleMaterializationError, match="UNKNOWN_TASK"):
        materialize_sample(draft, "not-a-task")
    with pytest.raises(SampleMaterializationError, match="CHANGE_TASK_NEEDS_TWO_IMAGES"):
        materialize_sample(draft, "change_qa")
    with pytest.raises(ValidationError, match="start with an image"):
        _sample(task="counting", images=[_image("t1"), _image("t2")])
    with pytest.raises(ValidationError, match="only context"):
        _sample(task="general_vqa", images=[_image("image"), _image("image")])


def test_unified_sample_rejects_extra_fields() -> None:
    payload = _valid_payload(router="extra")
    with pytest.raises(ValidationError):
        UnifiedSample.model_validate(payload)


# ── Question semantics / question 语义 ──────────────────────────────────────


def test_caption_tasks_allow_empty_question() -> None:
    _sample(task="caption", question="")
    _sample(task="change_caption", images=[_image("t1"), _image("t2")], question="  ")


def test_other_tasks_require_nonempty_question() -> None:
    for task in ("general_vqa", "counting", "spatial_relation", "change_qa",
                 "grounding", "multiple_choice_vqa", "fine_grained_counting"):
        images = [_image("t1"), _image("t2")] if task == "change_qa" else [_image()]
        with pytest.raises(ValidationError, match="non-empty question"):
            _sample(task=task, images=images, question="   ")


# ── TaskNormalization / 任务规范化 ──────────────────────────────────────────


def test_task_normalization_uses_structured_fields() -> None:
    norm = TaskNormalization(
        source_task="vrsbench_vqa",
        normalized_task="counting",
        semantic_subtype="quantity",
        confidence=0.95,
        normalizer="vrsbench_task_normalizer",
        version="1",
        reason_codes=["quantity_question"],
        spatial_query={"target": "vehicle"},
        answer_constraints={"vocabulary": ["0..999"], "closed": False},
        count_target_hint={"canonical_label": "small_vehicle"},
    )
    assert norm.normalized_task == "counting"
    assert norm.spatial_query == {"target": "vehicle"}
    assert norm.answer_constraints == {"vocabulary": ["0..999"], "closed": False}
    assert norm.count_target_hint == {"canonical_label": "small_vehicle"}


def test_multiple_choice_normalization_requires_canonical_choices() -> None:
    with pytest.raises(ValidationError, match="at least two choices"):
        TaskNormalization(
            source_task="source_vqa",
            normalized_task="multiple_choice_vqa",
            normalizer="test",
            version="1",
        )
    with pytest.raises(ValidationError, match="must be unique"):
        TaskNormalization(
            source_task="source_vqa",
            normalized_task="multiple_choice_vqa",
            normalizer="test",
            version="1",
            choices=["Road", " road "],
        )


def test_legacy_choice_constraints_are_promoted_for_read_compatibility() -> None:
    normalization = TaskNormalization.model_validate(
        {
            "source_task": "legacy_vqa",
            "normalized_task": "multiple_choice_vqa",
            "normalizer": "legacy",
            "version": "1",
            "answer_constraints": {
                "choices": ["(A) Road", "(B) Water"],
                "allow_multiple": True,
            },
        }
    )
    assert normalization.choices == ["(A) Road", "(B) Water"]
    assert normalization.allow_multiple is True


def test_task_normalization_rejects_invalid_task_and_confidence() -> None:
    with pytest.raises(ValidationError):
        TaskNormalization(source_task="x", normalized_task="detection", normalizer="n", version="1")
    with pytest.raises(ValidationError):
        TaskNormalization(source_task="x", normalized_task="counting", normalizer="n", version="1", confidence=1.5)


def test_task_normalization_rejects_non_json_structured_values() -> None:
    with pytest.raises(ValidationError):
        TaskNormalization(
            source_task="x", normalized_task="counting", normalizer="n", version="1",
            count_target_hint={"label": Path("x")},
        )


# ── Normalization first-class field / 一等规范化字段 ────────────────────────


def test_normalization_is_first_class_and_must_match_task() -> None:
    norm = TaskNormalization(
        source_task="vrsbench_vqa", normalized_task="counting",
        normalizer="vrsbench_task_normalizer", version="1",
    )
    sample = _sample(task="counting", normalization=norm.model_dump())
    assert sample.normalization is not None
    assert sample.normalization.normalized_task == "counting"
    with pytest.raises(ValidationError, match="does not match sample task"):
        _sample(task="general_vqa", normalization=norm.model_dump())


def test_normalization_absent_is_allowed() -> None:
    sample = _sample(task="counting")
    assert sample.normalization is None


# ── ValidationIssue / 校验问题 ──────────────────────────────────────────────


def test_validation_issue_contract() -> None:
    issue = ValidationIssue(code="MISSING_IMAGE", message="image missing", sample_id="s1")
    assert issue.severity == "error"
    warning = ValidationIssue(code="W", message="m", severity="warning")
    assert warning.severity == "warning"
    with pytest.raises(ValidationError):
        ValidationIssue(code="W", message="m", severity="fatal")


# ── stable_sample_id / 稳定样本 ID ──────────────────────────────────────────


def test_stable_sample_id_returns_safe_source_id() -> None:
    assert stable_sample_id(
        dataset="VRSBench", split="validation", source_id="qid-42",
        relative_image_paths=[Path("images/a.png")], question="Q?", source_index=0,
    ) == "qid-42"


def test_stable_sample_id_digest_anchors() -> None:
    assert stable_sample_id(
        dataset="VRSBench", split="validation", source_id=None,
        relative_image_paths=[Path("images/a.png")],
        question="How many buildings?", source_index=3,
    ) == "84f0286d7b7b7fd2d476"
    assert stable_sample_id(
        dataset="LEVIR-CC", split="test", source_id="",
        relative_image_paths=[Path("A/01_t1.png"), Path("B/02_t2.png")],
        question="Describe the change.", source_index=1,
    ) == "21401c92a969e878c2cd"


def test_stable_sample_id_is_order_sensitive() -> None:
    a = stable_sample_id(dataset="D", split="s", source_id=None,
                         relative_image_paths=[Path("x.png"), Path("y.png")],
                         question="q", source_index=0)
    b = stable_sample_id(dataset="D", split="s", source_id=None,
                         relative_image_paths=[Path("y.png"), Path("x.png")],
                         question="q", source_index=0)
    assert a != b


def test_stable_sample_id_hashes_unsafe_source_ids() -> None:
    for unsafe in ("../evil", "a/b", "a\\b", ".", "..", "has..dots", "line\nbreak", "x" * 121):
        result = stable_sample_id(
            dataset="D", split="s", source_id=unsafe,
            relative_image_paths=[Path("a.png")], question="q", source_index=0,
        )
        assert result != unsafe
        assert len(result) == 20
    assert stable_sample_id(
        dataset="VRSBench", split="validation", source_id="../evil",
        relative_image_paths=[Path("images/a.png")], question="Q?", source_index=0,
    ) == "bc17b5837ee4ae633c39"


def test_stable_sample_id_backslash_and_posix_forms_agree() -> None:
    """The same logical sample must yield the same ID for backslash and forward
    slash path spellings on every platform.
    同一逻辑样本在反斜杠与正斜杠写法下必须得到相同 ID（跨平台稳定）。"""
    posix = stable_sample_id(
        dataset="LEVIR-CC", split="test", source_id=None,
        relative_image_paths=["A/01_t1.png", "B/02_t2.png"],
        question="Describe the change.", source_index=1,
    )
    windows = stable_sample_id(
        dataset="LEVIR-CC", split="test", source_id=None,
        relative_image_paths=[r"A\01_t1.png", r"B\02_t2.png"],
        question="Describe the change.", source_index=1,
    )
    assert posix == windows == "21401c92a969e878c2cd"


def test_stable_sample_id_accepts_path_and_str_inputs() -> None:
    a = stable_sample_id(
        dataset="D", split="s", source_id=None,
        relative_image_paths=[Path("x/y.png")], question="q", source_index=0,
    )
    b = stable_sample_id(
        dataset="D", split="s", source_id=None,
        relative_image_paths=["x/y.png"], question="q", source_index=0,
    )
    assert a == b


def test_stable_sample_id_rejects_absolute_paths() -> None:
    """Machine-specific absolute paths must never enter the sample ID.
    机器相关绝对路径不得进入样本 ID。"""
    for bad in ("C:/data/a.png", r"C:\data\a.png", "/abs/a.png", r"\\server\share\a.png", "/"):
        with pytest.raises(ValueError, match="relative"):
            stable_sample_id(
                dataset="D", split="s", source_id=None,
                relative_image_paths=[bad], question="q", source_index=0,
            )
    with pytest.raises(ValueError, match="relative"):
        stable_sample_id(
            dataset="D", split="s", source_id=None,
            relative_image_paths=[Path("ok.png"), "/abs/b.png"],
            question="q", source_index=0,
        )


def test_stable_sample_id_rejects_empty_paths() -> None:
    with pytest.raises(ValueError, match="empty"):
        stable_sample_id(
            dataset="D", split="s", source_id=None,
            relative_image_paths=[""], question="q", source_index=0,
        )


def test_stable_sample_id_rejects_escape_segments() -> None:
    with pytest.raises(ValueError, match="segments"):
        stable_sample_id(
            dataset="D", split="s", source_id=None,
            relative_image_paths=["images/../a.png"], question="q", source_index=0,
        )


def test_stable_sample_id_parameter_validation() -> None:
    kwargs = dict(dataset="D", split="s", source_id=None,
                  relative_image_paths=["a.png"], question="q", source_index=0)
    with pytest.raises(ValueError, match="dataset"):
        stable_sample_id(**{**kwargs, "dataset": "  "})
    with pytest.raises(ValueError, match="split"):
        stable_sample_id(**{**kwargs, "split": ""})
    with pytest.raises(ValueError, match="at least one"):
        stable_sample_id(**{**kwargs, "relative_image_paths": []})
    with pytest.raises(ValueError, match="source_index"):
        stable_sample_id(**{**kwargs, "source_index": -1})


def test_stable_sample_id_windows_reserved_names_fall_back_to_hash() -> None:
    for unsafe in ("CON", "con.txt", "PRN", "AUX", "NUL", "COM1", "COM9",
                   "LPT1", "LPT9", "a<b", "a>b", "a:b", 'a"b', "a|b", "a?b",
                   "a*b", "trailing.", "trailing "):
        result = stable_sample_id(
            dataset="D", split="s", source_id=unsafe,
            relative_image_paths=["a.png"], question="q", source_index=0,
        )
        assert result != unsafe
        assert len(result) == 20


def test_duplicate_image_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate image_id"):
        UnifiedSample.model_validate(
            {
                "sample_id": "s",
                "dataset": "VRSBench",
                "split": "validation",
                "task": "general_vqa",
                "images": [
                    {"image_id": "same", "path": "a.png", "role": "image"},
                    {"image_id": "same", "path": "b.png", "role": "context"},
                ],
                "question": "Q",
            }
        )


# ── GroundTruth label binding / 标签绑定 (A5) ───────────────────────────────


def test_label_binding_boxes_and_points() -> None:
    GroundTruth(boxes=[[1, 2, 3, 4]], labels=["a"], label_binding="boxes")
    GroundTruth(points=[[1.0, 2.0]], labels=["a"], label_binding="points")
    GroundTruth(boxes=[[1, 2, 3, 4]], points=[[5.0, 6.0]], labels=["a", "b"],
                label_binding="all_geometry")
    GroundTruth(labels=["a", "b", "c"], label_binding="unbound")
    GroundTruth()  # no labels, no binding / 无 labels 无 binding
    GroundTruth(labels=[], label_binding="unbound")
    with pytest.raises(ValidationError, match="label_binding='boxes'"):
        GroundTruth(boxes=[[1, 2, 3, 4]], labels=["a", "b"], label_binding="boxes")
    with pytest.raises(ValidationError, match="ambiguous"):
        GroundTruth(boxes=[[1, 2, 3, 4]], points=[[5.0, 6.0]], labels=["a"])
    with pytest.raises(ValidationError, match="requires labels"):
        GroundTruth(label_binding="boxes")


# ── data package exports / data 包导出 ──────────────────────────────────────


def test_data_init_only_exports_stable_types() -> None:
    expected = {
        "GroundTruth",
        "ImageRef",
        "ImageRole",
        "JsonScalar",
        "JsonValue",
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
    case_dirs = [p for p in FIXTURE_ROOT.iterdir() if p.is_dir() and p.name != "adapters"]
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
        redump = sample.model_dump(mode="json")
        assert redump["task"] == sample_json["task"]
        assert [item["path"] for item in redump["images"]] == [
            item["path"] for item in sample_json["images"]
        ]
