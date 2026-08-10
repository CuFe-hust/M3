"""Final internal unified sample contracts for the data layer.

数据层最终内部统一样本契约。本模块不依赖 agents / workflows / application，
不包含 AgentResult、CountingResult 或任何运行状态；CanonicalSample /
CanonicalPrediction 属于外部兼容记录，不在本模块定义。

All free-form fields are JSON-safe (no Path, PIL image, callable, set, bytes,
or non-finite number). ImageRef is frozen; paths serialize with forward slashes
on every platform; change tasks enforce strict t1/t2/context roles.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator
from typing_extensions import TypeAliasType

# Public task names accepted by the runtime. / 运行时接受的公开任务名。
TaskName = Literal[
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
]

# Valid image roles; change tasks require an ordered t1/t2 pair, then context.
# 合法图像角色；变化任务要求有序 t1/t2 时相图像对，之后只允许 context。
ImageRole = Literal["image", "t1", "t2", "context"]

CHANGE_TASKS = frozenset({"change_caption", "change_qa"})
# Tasks whose question may be empty. / 允许 question 为空的任务。
QUESTION_OPTIONAL_TASKS = frozenset({"caption", "change_caption"})

# How GroundTruth.labels bind to geometry. / GroundTruth.labels 与几何的绑定方式。
LabelBinding = Literal["boxes", "points", "all_geometry", "unbound"]

# JSON-safe value types for free-form fields. / 自由字段的 JSON 安全类型。
JsonScalar = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)

_MAX_SOURCE_ID_LENGTH = 120


def _assert_json_safe(value: Any, where: str) -> None:
    """Reject non-JSON values and non-finite numbers recursively.
    递归拒绝非 JSON 值与非有限数值。"""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{where} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _assert_json_safe(item, where)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} contains a non-string key {type(key).__name__}")
            _assert_json_safe(item, where)
        return
    raise ValueError(f"{where} contains non-JSON value of type {type(value).__name__}")


class ImageRef(BaseModel):
    """One immutable image reference in a unified sample.
    统一样本中一条不可变的图像引用。

    path must be a non-empty, relative, non-escape path resolved against the
    dataset root; backslashes are normalized to forward slashes.
    path 必须是相对 dataset root 的非空相对路径（禁止绝对路径与 .. 逃逸段）；
    反斜杠统一规范化为正斜杠。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: str = Field(min_length=1)
    path: Path
    role: ImageRole
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("path", mode="before")
    @classmethod
    def normalize_path_input(cls, value: Any) -> Any:
        """Normalize backslashes and reject escape segments before Path
        construction (Path collapses '.' segments silently).
        构造 Path 前统一反斜杠并拒绝逃逸段（Path 会静默折叠 '.' 段）。"""
        if isinstance(value, str):
            text = value.replace("\\", "/")
            segments = text.split("/")
            if any(segment in {".", ".."} for segment in segments):
                raise ValueError(
                    f"image path must not contain '.' or '..' segments, got {value!r}"
                )
            return text
        return value

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: Path) -> Path:
        """Reject absolute paths and escape segments; file existence is not
        checked here. 拒绝绝对路径与逃逸段；文件存在性不在此检查。"""
        text = value.as_posix()
        if not text.strip():
            raise ValueError("image path must not be empty")
        if _is_absolute_like(text):
            raise ValueError(f"image path must be relative, got {text!r}")
        segments = text.split("/")
        if any(segment in {".", ".."} for segment in segments):
            raise ValueError(f"image path must not contain '.' or '..' segments, got {text!r}")
        return value

    @field_validator("sha256")
    @classmethod
    def lowercase_sha256(cls, value: str | None) -> str | None:
        """Store SHA256 digests in lowercase. / 统一以小写保存 SHA256 摘要。"""
        return value.lower() if value else value

    @field_serializer("path")
    def _serialize_path(self, value: Path) -> str:
        """Serialize paths with forward slashes on every platform.
        所有平台统一使用正斜杠序列化路径。"""
        return value.as_posix()


class GroundTruth(BaseModel):
    """Preserved ground truth without changing source annotations.
    在不改变源标注的前提下保留真值。"""

    model_config = ConfigDict(extra="forbid")

    answers: list[str] = Field(default_factory=list)
    count: int | None = Field(default=None, ge=0)
    boxes: list[list[float]] = Field(default_factory=list)
    points: list[list[float]] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    raw: dict[str, JsonValue] = Field(default_factory=dict)
    coordinate_frame: str | None = None
    label_binding: LabelBinding | None = None

    @field_validator("raw", mode="before")
    @classmethod
    def _reject_non_json_raw(cls, value: Any) -> Any:
        """Reject non-JSON values before Pydantic coercion.
        在 Pydantic 类型转换前拒绝非 JSON 值（如 Path、bytes、set）。"""
        if isinstance(value, dict):
            _assert_json_safe(value, "ground_truth.raw")
        return value

    @model_validator(mode="after")
    def validate_geometry(self) -> "GroundTruth":
        """Boxes are length 4 (xyxy) or 8 (polygon); points are length 2;
        coordinates must be finite; labels bind to geometry according to the
        declared label_binding (or the unambiguous default).
        框长度为 4 或 8；点长度为 2；坐标必须有限；labels 按声明的
        label_binding（或无歧义默认）与几何绑定。"""
        for box in self.boxes:
            if len(box) not in (4, 8):
                raise ValueError(f"ground-truth box must have 4 or 8 coordinates, got {len(box)}")
            if not all(math.isfinite(float(value)) for value in box):
                raise ValueError("ground-truth box contains a non-finite coordinate")
        for point in self.points:
            if len(point) != 2:
                raise ValueError(f"ground-truth point must have 2 coordinates, got {len(point)}")
            if not all(math.isfinite(float(value)) for value in point):
                raise ValueError("ground-truth point contains a non-finite coordinate")
        self._validate_label_binding()
        _assert_json_safe(self.raw, "ground_truth.raw")
        return self

    def _validate_label_binding(self) -> None:
        labels = self.labels
        binding = self.label_binding
        if not labels:
            if binding not in (None, "unbound"):
                raise ValueError(
                    f"label_binding={binding!r} requires labels; got none"
                )
            return
        if binding == "boxes":
            if len(labels) != len(self.boxes):
                raise ValueError(
                    f"label_binding='boxes' requires len(labels)==len(boxes): "
                    f"{len(labels)} labels for {len(self.boxes)} boxes"
                )
        elif binding == "points":
            if len(labels) != len(self.points):
                raise ValueError(
                    f"label_binding='points' requires len(labels)==len(points): "
                    f"{len(labels)} labels for {len(self.points)} points"
                )
        elif binding == "all_geometry":
            if len(labels) != len(self.boxes) + len(self.points):
                raise ValueError(
                    f"label_binding='all_geometry' requires len(labels)==len(boxes)+len(points): "
                    f"{len(labels)} labels for {len(self.boxes)}+{len(self.points)}"
                )
        elif binding == "unbound":
            return
        else:  # binding is None: accept only unambiguous defaults.
            if self.boxes and self.points:
                if len(labels) != len(self.boxes) + len(self.points):
                    raise ValueError(
                        "labels are ambiguous with both boxes and points; "
                        "declare label_binding explicitly"
                    )
            elif self.boxes:
                if len(labels) != len(self.boxes):
                    raise ValueError(
                        f"labels must match boxes count: {len(labels)} labels "
                        f"for {len(self.boxes)} boxes"
                    )
            elif self.points:
                if len(labels) != len(self.points):
                    raise ValueError(
                        f"labels must match points count: {len(labels)} labels "
                        f"for {len(self.points)} points"
                    )
            else:
                raise ValueError("labels require at least one box or point")


class TaskNormalization(BaseModel):
    """Adapter task-normalization metadata, attached to UnifiedSample.
    适配器任务规范化元数据，作为 UnifiedSample 的一等字段。"""

    model_config = ConfigDict(extra="forbid")

    source_task: str
    normalized_task: TaskName
    semantic_subtype: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    normalizer: str
    version: str
    reason_codes: list[str] = Field(default_factory=list)
    spatial_query: dict[str, JsonValue] | None = None
    answer_constraints: dict[str, JsonValue] = Field(default_factory=dict)
    count_target_hint: dict[str, JsonValue] | None = None

    @field_validator("spatial_query", "answer_constraints", "count_target_hint", mode="before")
    @classmethod
    def _reject_non_json_structured(cls, value: Any) -> Any:
        """Reject non-JSON values before Pydantic coercion.
        在 Pydantic 类型转换前拒绝非 JSON 值。"""
        if isinstance(value, dict):
            _assert_json_safe(value, "normalization.structured")
        return value

    @model_validator(mode="after")
    def validate_structured_fields(self) -> "TaskNormalization":
        """Keep all free-form fields JSON-safe. / 所有自由字段保持 JSON 安全。"""
        if self.spatial_query is not None:
            _assert_json_safe(self.spatial_query, "normalization.spatial_query")
        _assert_json_safe(self.answer_constraints, "normalization.answer_constraints")
        if self.count_target_hint is not None:
            _assert_json_safe(self.count_target_hint, "normalization.count_target_hint")
        return self


class ValidationIssue(BaseModel):
    """One deterministic dataset-validation issue for read-only audits.
    只读审计中的一条确定性数据集校验问题。"""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    sample_id: str | None = None
    severity: Literal["error", "warning"] = "error"


class UnifiedSample(BaseModel):
    """Dataset-neutral sample consumed by agents and workflows.
    Agent 与工作流消费的与数据集无关的样本。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    task: TaskName
    images: list[ImageRef] = Field(min_length=1)
    question: str
    ground_truth: GroundTruth | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    normalization: TaskNormalization | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def _reject_non_json_metadata(cls, value: Any) -> Any:
        """Reject non-JSON values before Pydantic coercion.
        在 Pydantic 类型转换前拒绝非 JSON 值（如 Path、bytes、set）。"""
        if isinstance(value, dict):
            _assert_json_safe(value, "metadata")
        return value

    @model_validator(mode="after")
    def validate_roles_and_temporal_order(self) -> "UnifiedSample":
        """Change tasks require ordered t1/t2 then context; other tasks require
        an image role first, then context only.
        变化任务要求 t1 在 t2 之前且其后只允许 context；其他任务首图为 image，之后只允许 context。"""
        roles = [image.role for image in self.images]
        if self.task in CHANGE_TASKS:
            if roles[:2] != ["t1", "t2"]:
                raise ValueError("change samples must place t1 before t2")
            if any(role != "context" for role in roles[2:]):
                raise ValueError("change samples allow only context roles after t1/t2")
        else:
            if roles[0] != "image":
                raise ValueError("non-change samples must start with an image role")
            if any(role != "context" for role in roles[1:]):
                raise ValueError("non-change samples allow only context roles after the first image")
        return self

    @model_validator(mode="after")
    def validate_question_requirement(self) -> "UnifiedSample":
        """Caption tasks may have an empty question; all other tasks require one.
        caption 类任务 question 可为空；其余任务必须非空。"""
        if self.task not in QUESTION_OPTIONAL_TASKS and not self.question.strip():
            raise ValueError(f"task {self.task} requires a non-empty question")
        return self

    @model_validator(mode="after")
    def validate_normalization_consistency(self) -> "UnifiedSample":
        """The attached normalization must agree with the sample task.
        附加的任务规范化必须与样本任务一致。"""
        if self.normalization is not None and self.normalization.normalized_task != self.task:
            raise ValueError(
                f"normalization task {self.normalization.normalized_task} "
                f"does not match sample task {self.task}"
            )
        _assert_json_safe(self.metadata, "metadata")
        return self

    @model_validator(mode="after")
    def validate_unique_image_ids(self) -> "UnifiedSample":
        """Image IDs must be unique within one sample. / 样本内 image_id 必须唯一。"""
        ids = [image.image_id for image in self.images]
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate image_id: {duplicates}")
        return self


class SampleDraft(BaseModel):
    """Pre-sample contract for datasets without an explicit per-sample task.
    No task-specific roles validation happens here: image roles are
    placeholder 'image' and are rebuilt when the draft is materialized after
    task resolution. UnifiedSample.task stays mandatory — the draft exists
    only before resolution. 无显式逐样本 task 数据集的样本前契约。此处不做
    task 相关角色校验：图像角色为占位 'image'，在任务解析后物化样本时重建。
    UnifiedSample.task 保持必填——draft 只存在于解析之前。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    images: list[ImageRef] = Field(min_length=1)
    question: str = ""
    explicit_task: TaskName | None = None
    ground_truth: GroundTruth | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _reject_non_json_metadata(cls, value: Any) -> Any:
        """Reject non-JSON values before Pydantic coercion.
        在 Pydantic 类型转换前拒绝非 JSON 值（如 Path、bytes、set）。"""
        if isinstance(value, dict):
            _assert_json_safe(value, "metadata")
        return value



def stable_sample_id(
    *,
    dataset: str,
    split: str,
    source_id: str | None,
    relative_image_paths: Sequence[Path | str],
    question: str,
    source_index: int,
) -> str:
    """Return the source ID when it is a safe directory name, otherwise a stable
    20-character digest.

    Hash input fields (in order): dataset, split, source ID, ordered relative
    image paths (POSIX form), question, source index. Encoding is UTF-8; the
    digest is the first 20 hex characters of SHA-256.
    哈希输入字段（按序）：dataset、split、源 ID、有序相对图片路径（POSIX 形式）、
    question、源索引。编码 UTF-8；摘要为 SHA-256 前 20 个十六进制字符。

    Path normalization: every image path is rendered with forward slashes
    (backslashes are converted), so the same logical sample yields the same ID
    on Windows and POSIX. Absolute paths are rejected because machine-specific
    paths must never enter the ID.
    路径规范化：所有图片路径统一为正斜杠（反斜杠被转换），同一逻辑样本在
    Windows/POSIX 下得到相同 ID。绝对路径被拒绝——机器相关路径不得进入 ID。

    The original unsafe source ID is not returned; adapters that need it must
    preserve it in sample metadata.
    不安全源 ID 不返回原值，需要保留时由适配器放入 sample metadata。
    """
    if not dataset.strip():
        raise ValueError("dataset must not be empty")
    if not split.strip():
        raise ValueError("split must not be empty")
    if not isinstance(source_index, int) or source_index < 0:
        raise ValueError(f"source_index must be a non-negative integer, got {source_index!r}")
    if not relative_image_paths:
        raise ValueError("relative_image_paths must contain at least one path")
    # ALL inputs are validated before the safe source ID is reused, so a safe
    # source ID can never bypass image-path validation.
    # 所有输入先校验，再决定复用安全 source ID——安全 ID 不能绕过路径校验。
    posix_paths = _normalize_relative_image_paths(relative_image_paths)
    if _source_id_is_safe(source_id):
        return source_id  # type: ignore[return-value]
    parts = [
        dataset,
        split,
        source_id or "",
        "\n".join(posix_paths),
        question,
        str(source_index),
    ]
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _normalize_relative_image_paths(values: Sequence[Path | str]) -> list[str]:
    """Normalize and validate every image path (POSIX form, relative,
    non-escape). 规范化并校验每条图片路径（POSIX 形式、相对、无逃逸段）。"""
    posix_paths: list[str] = []
    for value in values:
        text = str(value)
        if not text.strip():
            raise ValueError("relative_image_paths must not be empty")
        if _is_absolute_like(text):
            raise ValueError(f"relative_image_paths must be relative, got {text!r}")
        normalized = text.replace("\\", "/")
        segments = normalized.split("/")
        if any(segment in {".", ".."} for segment in segments):
            raise ValueError(
                f"relative_image_paths must not contain '.' or '..' segments, got {text!r}"
            )
        posix_paths.append(normalized)
    return posix_paths


_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
_WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"/\\|?*')


def _is_absolute_like(value: str) -> bool:
    """Detect absolute paths on both Windows and POSIX spellings.
    同时识别 Windows 与 POSIX 写法的绝对路径。"""
    if value.startswith(("/", "\\")):
        return True
    return len(value) >= 3 and value[1] == ":" and value[2] in "/\\"


def _source_id_is_safe(source_id: str | None) -> bool:
    """A safe source ID is non-empty, short, and usable as a directory name on
    Windows and POSIX: no reserved device names, no forbidden characters, no
    control characters, no dot/space suffix, no escape segments.
    安全源 ID：非空、长度受限、Windows/POSIX 均可作为目录名：无保留设备名、
    无非法字符、无控制字符、无尾随点/空格、无逃逸段。"""
    if not source_id:
        return False
    if len(source_id) > _MAX_SOURCE_ID_LENGTH:
        return False
    if any(ord(char) < 32 for char in source_id):
        return False
    if any(char in _WINDOWS_FORBIDDEN_CHARS for char in source_id):
        return False
    if source_id in {".", ".."} or ".." in source_id:
        return False
    if source_id.endswith((" ", ".")):
        return False
    basename = source_id.split("/")[-1].split("\\")[-1]
    stem = basename.split(".")[0].casefold()
    if stem in _WINDOWS_RESERVED_BASENAMES:
        return False
    return True
