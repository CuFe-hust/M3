"""Runtime schemas for auditable bi-temporal change preprocessing.

可审计双时相变化预处理的运行时 Schema。全部类型可序列化，不含任何语义
变化描述（不生成语义结论）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.counting.schema import IssueRecord


class PairValidationReport(BaseModel):
    """Validation facts recorded before any pixel transform.
    像素变换前记录的校验事实。"""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    temporal_roles_valid: bool
    same_size: bool
    alignment_status: Literal[
        "assumed_dataset_aligned", "metadata_aligned", "weakly_aligned", "unreliable"
    ]
    original_sizes: list[list[int]] = Field(default_factory=list)
    warnings: list[IssueRecord] = Field(default_factory=list)


class HarmonizationMetrics(BaseModel):
    """Pair metrics; every ratio is represented on the inclusive 0..1 scale.
    图对指标；所有比例统一使用闭区间 0..1。"""

    model_config = ConfigDict(extra="forbid")

    pif_ratio: float = Field(ge=0.0, le=1.0)
    mad_full_before: float
    mad_full_after: float
    mad_pif_before: float
    mad_pif_after: float
    corr_full_before: float | None = None
    corr_full_after: float | None = None
    corr_pif_before: float | None = None
    corr_pif_after: float | None = None
    pct_diff_gt20_before: float = Field(ge=0.0, le=1.0)
    pct_diff_gt20_after: float = Field(ge=0.0, le=1.0)
    lapvar_t1_before: float
    lapvar_t2_before: float
    lapvar_t1_after: float
    lapvar_t2_after: float


class HarmonizationDecision(BaseModel):
    """Stable quality-gate decision. / 稳定的质量门控决策。"""

    model_config = ConfigDict(extra="forbid")

    version: str
    status: Literal["applied", "skipped", "rejected", "failed"]
    reason_codes: list[str]
    metrics: HarmonizationMetrics | None
    used_for_proposal: bool
    raw_retained: bool = True


class ChangeProposal(BaseModel):
    """Explainable attention proposal, not a semantic change claim.
    可解释关注候选，不代表语义变化结论。"""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    box: list[int]
    pixel_box: list[int]
    score: float = Field(ge=0.0, le=1.0)
    area_ratio: float = Field(ge=0.0, le=1.0)
    source: Literal["difference_map_v1"] = "difference_map_v1"
    evidence_filenames: list[str] = Field(default_factory=list)
