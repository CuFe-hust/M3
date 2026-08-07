"""Dataset-neutral spatial query contracts.

数据集无关的空间查询契约。SpatialQuerySpec 由数据层从样本元数据提取，
几何模块只消费结构化 spec——绝不读取 question、绝不做数据集专用正则。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SpatialOperation = Literal[
    "extreme_category",
    "grid_position",
    "box_gap",
    "orientation_evidence",
    "arrangement_evidence",
]

# Operations that may deterministically override the visual answer when the
# evidence is complete and the rule is reproducible.
# 当证据完整且规则可复现时，可确定性覆盖视觉答案的操作。
_OVERRIDABLE_OPERATIONS = frozenset({"extreme_category", "grid_position"})


class SpatialQuerySpec(BaseModel):
    """Structured spatial query extracted from sample metadata.
    从样本元数据提取的结构化空间查询。"""

    model_config = ConfigDict(extra="forbid")

    operation: SpatialOperation
    # Canonical target label (normalized) that evidence labels must match.
    # 证据标签必须匹配的规范目标标签（已归一化）。
    target_label: str | None = None
    # Direction hint for extreme_category ("top"/"bottom").
    # extreme_category 的方向提示（"top"/"bottom"）。
    target_hint: str | None = None
    # Grid boundaries for grid_position (horizontal and vertical splits).
    # grid_position 的网格边界（水平与垂直分割线）。
    grid_boundaries: tuple[int, int] = (333, 666)
    # Minimum number of matching candidates required for a deterministic
    # override. 确定性覆盖所需的最小匹配候选数。
    min_candidates: int = Field(default=2, ge=1)
    # Closed answer vocabulary when the query entails one; empty otherwise.
    # 查询本身蕴含封闭答案空间时的词表；否则为空。
    answer_vocabulary: list[str] = Field(default_factory=list)

    @property
    def can_override(self) -> bool:
        """Whether this operation may deterministically override the answer.
        该操作是否可确定性覆盖答案。"""
        return self.operation in _OVERRIDABLE_OPERATIONS


def spatial_query_from_metadata(metadata: dict[str, Any] | None) -> SpatialQuerySpec | None:
    """Build a SpatialQuerySpec from sample.metadata["spatial_query"]; absent
    or invalid values yield None (never guessed).
    从 sample.metadata["spatial_query"] 构建 SpatialQuerySpec；缺失或无效值
    返回 None（绝不猜测）。"""
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("spatial_query")
    if not isinstance(raw, dict):
        return None
    try:
        return SpatialQuerySpec.model_validate(raw)
    except Exception:
        return None
