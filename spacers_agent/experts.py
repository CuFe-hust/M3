"""Structured non-counting experts and deterministic geometry checks. / 结构化非计数专家和确定性几何校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash, image_to_data_url
from spacers_agent.schemas import IssueRecord, TaskName, UnifiedSample


class VisualEvidence(BaseModel):
    """Normalized whole-image visual evidence. / 归一化整图视觉证据。"""
    model_config = ConfigDict(extra="forbid")
    image_id: str
    label: str
    box: list[int] | None = None
    point: list[int] | None = None
    source_tile: str | None = None
    confidence: float = Field(ge=0, le=1)


class ExpertResult(BaseModel):
    """Common persisted expert result. / 通用持久化专家结果。"""
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    task: TaskName
    answer: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[VisualEvidence] = Field(default_factory=list)
    status: Literal["completed", "partial", "failed"] = "completed"
    issues: list[IssueRecord] = Field(default_factory=list)
    raw_structured: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ExpertContext:
    """Shared Qwen request context. / 共享 Qwen 请求上下文。"""
    client: VisionLanguageClient
    model: str
    prompt: str
    artifact_dir: Path


class Expert(Protocol):
    """Execution contract for a routed expert. / 路由专家的执行契约。"""
    async def run(self, sample: UnifiedSample, context: ExpertContext) -> ExpertResult: ...


class GroundingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    boxes: list[dict[str, Any]]
    answer: str
    uncertainty: list[str] = Field(default_factory=list)


class ChangeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    changes: list[dict[str, Any]]
    answer: str
    uncertainty: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_roles(self) -> "ChangeOutput":
        for change in self.changes:
            if change.get("type") == "appeared" and not change.get("t2_box"):
                raise ValueError("appeared change requires t2_box")
            if change.get("type") == "disappeared" and not change.get("t1_box"):
                raise ValueError("disappeared change requires t1_box")
        return self


def relation_is_valid(first: list[int], predicate: str, second: list[int]) -> bool | None:
    """Check 2D box relations or return None for semantic-only relations. / 校验二维框关系，语义关系返回 None。"""
    ax, ay = (first[0]+first[2])/2, (first[1]+first[3])/2; bx, by = (second[0]+second[2])/2, (second[1]+second[3])/2
    if predicate == "north_of": return ay < by
    if predicate == "south_of": return ay > by
    if predicate == "east_of": return ax > bx
    if predicate == "west_of": return ax < bx
    if predicate == "overlap": return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])
    if predicate == "contains": return first[0] <= second[0] and first[1] <= second[1] and first[2] >= second[2] and first[3] >= second[3]
    return None
