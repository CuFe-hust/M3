"""Spatial candidate review — independent localization completeness pass.
空间候选复查 — 独立的定位完整性复核。

Provides the concrete spatial Agent's candidate-review logic. Not a top-level Agent.
提供具体空间 Agent 的候选复查逻辑，不是顶层 Agent。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from models.base import RequestMeta, VisionLanguageClient, build_request_hash, image_to_data_url
from spacers_agent.schemas import AgentResult, UnifiedSample, VisualEvidence
from spacers_agent.vqa_geometry import vrsbench_answer_vocabulary, vrsbench_question_subtype
from spacers_agent.agents.spatial.evidence_merge import (
    is_corner_anchored_box,
    matches_position_target,
    maximum_repair_severity,
    merge_visual_evidence,
    needs_candidate_review,
)


class SpatialCandidateReviewResult(BaseModel):
    """Compact review contract without duplicated prose or geometry fields.
    不含重复文字与几何字段的紧凑复核契约。
    """

    model_config = ConfigDict(extra="forbid")

    boxes: list[tuple[str, int, int, int, int]] = Field(default_factory=list, max_length=200)
    complete: bool = True
    _local_recoveries: list[str] = PrivateAttr(default_factory=list)

    @classmethod
    def recover_json_payload(cls, value: str) -> dict[str, Any] | None:
        """Recover complete candidate tuples from Qwen's missing inner brackets.
        从 Qwen 缺失内层方括号的输出中恢复完整候选元组。
        """

        if not value.lstrip().startswith('{"boxes"'):
            return None
        pattern = re.compile(
            r'"([^"\\]+)"\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)'
        )
        boxes = [
            [label, int(x1), int(y1), int(x2), int(y2)]
            for label, x1, y1, x2, y2 in pattern.findall(value)
        ]
        if not boxes:
            return None
        complete_match = re.search(r'"complete"\s*:\s*(true|false)', value, re.IGNORECASE)
        return {
            "boxes": boxes,
            "complete": bool(complete_match and complete_match.group(1).casefold() == "true"),
        }

    @field_validator("boxes", mode="before")
    @classmethod
    def clamp_one_step_coordinate_drift(cls, value: Any) -> Any:
        """Clamp only the model's common -1/1000 normalization drift.
        仅裁剪模型常见的 -1/1000 归一化越界。
        """

        if not isinstance(value, list):
            return value
        normalized: list[Any] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 5:
                normalized.append(item)
                continue
            label, *coordinates = item
            normalized.append([
                label,
                *[
                    min(999, max(0, coordinate))
                    if isinstance(coordinate, int) and -1 <= coordinate <= 1000
                    else coordinate
                    for coordinate in coordinates
                ],
            ])
        return normalized

    @model_validator(mode="after")
    def validate_boxes(self) -> "SpatialCandidateReviewResult":
        for label, x1, y1, x2, y2 in self.boxes:
            VisualEvidence(label=label, box=[x1, y1, x2, y2])
        return self


class SpatialCandidateReviewer:
    """Review incomplete spatial instance enumeration independently.
    独立复查不完整的空间实例枚举。
    """

    def __init__(
        self,
        client: VisionLanguageClient,
        model: str,
        *,
        review_prompt: str,
        review_prompt_version: str,
        grid_review_prompt: str = "",
        grid_review_prompt_version: str = "",
        review_max_tokens: int = 128,
    ) -> None:
        self._client = client
        self._model = model
        self._review_prompt = review_prompt
        self._review_prompt_version = review_prompt_version
        self._grid_review_prompt = grid_review_prompt
        self._grid_review_prompt_version = grid_review_prompt_version
        self._review_max_tokens = review_max_tokens

    def needs_review(self, sample: UnifiedSample, result: AgentResult) -> bool:
        """Return whether a spatial result requires candidate review.
        返回空间结果是否需要候选复查。
        """
        return needs_candidate_review(sample, result)

    async def review(
        self, sample: UnifiedSample, first_result: AgentResult, artifact_dir: Path
    ) -> AgentResult:
        """Run candidate review and merge evidence into the final result.
        运行候选复查并将证据合并到最终结果。
        """

        # -- fetch review prompt / 获取复查 Prompt --
        subtype = vrsbench_question_subtype(
            sample.question,
            str(sample.metadata.get("question_type", "")),
        )
        is_grid = subtype == "grid_position"
        review_prompt = self._grid_review_prompt if is_grid and self._grid_review_prompt else self._review_prompt
        review_version = (
            self._grid_review_prompt_version if is_grid and self._grid_review_prompt else self._review_prompt_version
        )

        if not review_prompt or not self.needs_review(sample, first_result):
            return first_result

        # -- run review / 执行复查 --
        try:
            review_result = await self._request_review(sample, review_prompt, review_version, subtype, artifact_dir)
        except Exception as error:
            geometry = dict(first_result.geometry)
            geometry.update({
                "candidate_review_used": True,
                "candidate_review_added": 0,
                "candidate_review_error": f"{type(error).__name__}: {error}",
            })
            return first_result.model_copy(update={"geometry": geometry, "status": "partial"})

        # -- merge evidence / 合并证据 --
        first_evidence = list(first_result.evidence_items)
        replaced_evidence = 0
        if is_grid:
            first_evidence = [
                item for item in first_result.evidence_items
                if not (matches_position_target(sample.question, item) and is_corner_anchored_box(item))
            ]
            replaced_evidence = len(first_result.evidence_items) - len(first_evidence)

        review_evidence = [
            VisualEvidence(label=label, box=[x1, y1, x2, y2], confidence=0.0)
            for label, x1, y1, x2, y2 in review_result.boxes
        ]
        labeled_review_boxes = len(review_evidence) if is_grid else 0
        merged = merge_visual_evidence(first_evidence, review_evidence)

        # -- geometry audit / 几何审计 --
        geometry = dict(first_result.geometry)
        merged_quality = ["trusted_box" if item.box is not None else "trusted_point" for item in merged]
        geometry.update({
            "candidate_review_used": True,
            "candidate_review_added": len(merged) - len(first_evidence),
            "candidate_review_replaced": replaced_evidence,
            "candidate_review_labeled_boxes": labeled_review_boxes,
            "candidate_review_geometry": {
                "review_contract": "compact-boxes-v1",
                "enumeration_complete": review_result.complete,
            },
            "evidence_quality": merged_quality,
            "repair_severity": maximum_repair_severity(
                str(first_result.geometry.get("repair_severity", "none")),
                "none",
            ),
        })

        # -- finalise answer / 最终化答案 --
        reviewed_result = first_result.model_copy(update={
            "boxes": [list(item.box) for item in merged if item.box is not None],
            "evidence_items": merged,
            "geometry": geometry,
        })
        status = (
            "partial"
            if not review_result.complete or needs_candidate_review(sample, reviewed_result)
            else "completed"
        )
        return reviewed_result.model_copy(update={"status": status})

    async def _request_review(
        self, sample: UnifiedSample, prompt: str, version: str, subtype: str, artifact_dir: Path
    ) -> SpatialCandidateReviewResult:
        """Issue the review request. / 发起复查请求。"""

        content: list[dict[str, Any]] = []
        image_hashes: list[str] = []
        for image_ref in sample.images:
            data = image_ref.path.read_bytes()
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(data, "image/png")}})
            image_hashes.append(hashlib.sha256(data).hexdigest())

        answer_vocabulary = (
            [] if subtype == "grid_position" else vrsbench_answer_vocabulary(subtype, sample.question)
        )
        content.append({
            "type": "text",
            "text": json.dumps({
                "question": sample.question,
                "dataset_question_type": sample.metadata.get("question_type"),
                "semantic_subtype": subtype,
                "answer_vocabulary": answer_vocabulary,
                "coordinate_frame": "normalized_0_999_top_left",
                "review_mode": "independent_candidate_enumeration",
            }, ensure_ascii=False),
        })

        system_prompt = (
            prompt
            + "\n\nReturn valid JSON only using the compact response schema."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        request_hash = build_request_hash(
            model=self._model,
            generation={"temperature": 0.0, "max_tokens": self._review_max_tokens},
            prompt_version=version,
            messages=messages,
            image_sha256="|".join(image_hashes),
        )
        return await self._client.complete_json(
            messages=messages,
            response_model=SpatialCandidateReviewResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:spatial-candidate-review",
                request_hash=request_hash,
                prompt_version=version,
                sample_id=sample.sample_id,
                artifact_dir=artifact_dir / "spatial_agent_candidate_review",
            ),
            max_tokens=self._review_max_tokens,
        )
