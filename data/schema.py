"""Final internal unified sample contracts for the data layer.

数据层最终内部统一样本契约。本模块不依赖 agents / workflows / application，
不包含 AgentResult、CountingResult 或任何运行状态；CanonicalSample /
CanonicalPrediction 属于外部兼容记录，不在本模块定义。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

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

# Valid image roles; change tasks require an ordered t1/t2 pair.
# 合法图像角色；变化任务要求有序的 t1/t2 时相图像对。
ImageRole = Literal["image", "t1", "t2", "context"]

CHANGE_TASKS = frozenset({"change_caption", "change_qa"})


class ImageRef(BaseModel):
    """One immutable image reference in a unified sample.
    统一样本中一条不可变的图像引用。
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    path: Path
    role: ImageRole
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    sha256: str | None = None

    @field_serializer("path")
    def _serialize_path(self, value: Path) -> str:
        """Serialize paths with forward slashes on every platform.
        所有平台统一使用正斜杠序列化路径。
        """
        return value.as_posix()


class GroundTruth(BaseModel):
    """Preserved ground truth without changing source annotations.
    在不改变源标注的前提下保留真值。
    """

    model_config = ConfigDict(extra="forbid")

    answers: list[str] = Field(default_factory=list)
    count: int | None = Field(default=None, ge=0)
    boxes: list[list[float]] = Field(default_factory=list)
    points: list[list[float]] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskNormalization(BaseModel):
    """Adapter task-normalization metadata; stored under metadata['normalization'].
    适配器任务规范化元数据；存放于 metadata['normalization']。
    """

    model_config = ConfigDict(extra="forbid")

    source_task: str
    normalized_task: TaskName
    semantic_subtype: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    normalizer: str
    version: str
    reason_codes: list[str] = Field(default_factory=list)
    spatial_query: str | None = None
    answer_constraints: list[str] = Field(default_factory=list)
    count_target_hint: str | None = None


class ValidationIssue(BaseModel):
    """One deterministic dataset-validation issue for read-only audits.
    只读审计中的一条确定性数据集校验问题。
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    sample_id: str | None = None
    severity: Literal["error", "warning"] = "error"


class UnifiedSample(BaseModel):
    """Dataset-neutral sample consumed by agents and workflows.
    Agent 与工作流消费的与数据集无关的样本。
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    task: TaskName
    images: list[ImageRef] = Field(min_length=1)
    question: str
    ground_truth: GroundTruth | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_temporal_order(self) -> "UnifiedSample":
        """Require ordered temporal images for change tasks.
        要求变化任务的时相图像顺序正确（t1 在 t2 之前）。
        """
        if self.task in CHANGE_TASKS:
            roles = [image.role for image in self.images]
            if roles[:2] != ["t1", "t2"]:
                raise ValueError("change samples must place t1 before t2")
        return self


def stable_sample_id(
    source_id: str | None,
    relative_image_path: Path,
    question: str,
    source_index: int,
) -> str:
    """Return the source ID when present or a stable 20-character digest.
    存在源 ID 时返回源 ID，否则返回稳定的 20 字符摘要。
    """

    if source_id:
        return source_id
    payload = f"{relative_image_path.as_posix()}\n{question}\n{source_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]
