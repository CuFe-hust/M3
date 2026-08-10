"""Spatial candidate review — independent localization completeness pass.

空间候选复查 — 独立的定位完整性复核。提供具体空间 Agent 的候选复查逻辑；
复查最多执行一次且受 CallBudget 限制；失败时保留初次结果并记录稳定错误
类型，绝不丢失已有证据。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from agents.counting.backends.base import (
    MissingModelCacheIdentityError,
    require_model_cache_identity,
)
from agents.schema import AgentResult, VisualEvidence
from data.schema import UnifiedSample
from agents.spatial.evidence_merge import (
    is_corner_anchored_box,
    matches_target_label,
    maximum_repair_severity,
    merge_visual_evidence,
    needs_candidate_review,
    position_review_evidence,
)
from models.base import RequestMeta, VisionLanguageClient, build_request_hash
from models.images import detect_image_mime, image_to_data_url


class SpatialCandidateReviewResult(BaseModel):
    """Compact review contract without duplicated prose or geometry fields.
    不含重复文字与几何字段的紧凑复核契约。"""

    model_config = ConfigDict(extra="forbid")

    boxes: list[tuple[str, int, int, int, int]] = Field(default_factory=list, max_length=200)
    complete: bool = True
    _local_recoveries: list[str] = PrivateAttr(default_factory=list)

    @classmethod
    def recover_json_payload(cls, value: str) -> dict[str, Any] | None:
        """Recover complete candidate tuples from Qwen's missing inner brackets.
        从 Qwen 缺失内层方括号的输出中恢复完整候选元组。"""
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
        仅裁剪模型常见的 -1/1000 归一化越界。"""
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
    独立复查不完整的空间实例枚举。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        review_prompt: str,
        review_prompt_version: str,
        grid_review_prompt: str = "",
        grid_review_prompt_version: str = "",
        review_max_tokens: int = 128,
    ) -> None:
        self._client = client
        self._review_prompt = review_prompt
        self._review_prompt_version = review_prompt_version
        self._grid_review_prompt = grid_review_prompt
        self._grid_review_prompt_version = grid_review_prompt_version
        self._review_max_tokens = review_max_tokens

    def needs_review(
        self,
        result: AgentResult,
        *,
        operation: str,
        target_label: str | None,
    ) -> bool:
        """Return whether a spatial result requires candidate review.
        返回空间结果是否需要候选复查。"""
        return needs_candidate_review(
            result, operation=operation, target_label=target_label
        )

    async def review(
        self,
        sample: UnifiedSample,
        first_result: AgentResult,
        artifact_dir: Path,
        *,
        operation: str,
        target_label: str | None,
        data_root: Path,
        budget: Any = None,
    ) -> AgentResult:
        """Run candidate review (at most once) and merge evidence into the
        final result; review failure never loses the first result.
        运行候选复查（最多一次）并将证据合并到最终结果；复查失败绝不丢失
        初次结果。"""
        is_grid = operation == "grid_position"
        review_prompt = (
            self._grid_review_prompt
            if is_grid and self._grid_review_prompt
            else self._review_prompt
        )
        review_version = (
            self._grid_review_prompt_version
            if is_grid and self._grid_review_prompt
            else self._review_prompt_version
        )
        if not review_prompt or not self.needs_review(
            first_result, operation=operation, target_label=target_label
        ):
            return first_result

        try:
            review_result = await self._request_review(
                sample,
                review_prompt,
                review_version,
                operation=operation,
                target_label=target_label,
                artifact_dir=artifact_dir,
                data_root=data_root,
                budget=budget,
            )
        except MissingModelCacheIdentityError:
            # A missing cache identity is a configuration error, never a
            # review failure. 缺失缓存身份是配置错误，绝不是复核失败。
            raise
        except Exception as error:
            # Never lose the first result; record a stable error type only.
            # 绝不丢失初次结果；只记录稳定错误类型。
            geometry = dict(first_result.geometry)
            geometry.update(
                {
                    "candidate_review_used": True,
                    "candidate_review_added": 0,
                    "candidate_review_replaced": 0,
                    "candidate_review_labeled_boxes": 0,
                    "candidate_review_error_type": type(error).__name__,
                }
            )
            return first_result.model_copy(
                update={"geometry": geometry, "status": "partial"}
            )

        # Merge evidence / 合并证据
        first_evidence = list(first_result.evidence_items)
        replaced_evidence = 0
        if is_grid:
            first_evidence = [
                item
                for item in first_result.evidence_items
                if not (
                    matches_target_label(item, target_label)
                    and is_corner_anchored_box(item)
                )
            ]
            replaced_evidence = len(first_result.evidence_items) - len(first_evidence)

        review_evidence, labeled_review_boxes = position_review_evidence(
            AgentResult(
                agent_name="spatial_agent",
                answer="",
                boxes=[list(box) for box in review_result.boxes],
            ),
            is_grid=is_grid,
            target_label=target_label,
        )
        merged = merge_visual_evidence(first_evidence, review_evidence)

        geometry = dict(first_result.geometry)
        merged_quality = [
            "trusted_box" if item.box is not None else "trusted_point"
            for item in merged
        ]
        geometry.update(
            {
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
            }
        )

        reviewed_result = first_result.model_copy(
            update={
                "boxes": [list(item.box) for item in merged if item.box is not None],
                "evidence_items": merged,
                "geometry": geometry,
            }
        )
        status = (
            "partial"
            if not review_result.complete
            or needs_candidate_review(
                reviewed_result, operation=operation, target_label=target_label
            )
            else "completed"
        )
        return reviewed_result.model_copy(update={"status": status})

    async def _request_review(
        self,
        sample: UnifiedSample,
        prompt: str,
        version: str,
        *,
        operation: str,
        target_label: str | None,
        artifact_dir: Path,
        data_root: Path,
        budget: Any,
    ) -> SpatialCandidateReviewResult:
        """Issue the review request. / 发起复查请求。"""
        image_path = _resolve_sample_image(sample, data_root)
        image_bytes = image_path.read_bytes()
        # Detect the real MIME from content, never from the file suffix.
        # 从真实内容检测 MIME，绝不按文件后缀猜测。
        mime = detect_image_mime(image_path)
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image_bytes, mime)},
            },
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "question": sample.question,
                        "operation": operation,
                        "target_label": target_label,
                        "coordinate_frame": "normalized_0_999_top_left",
                        "review_mode": "independent_candidate_enumeration",
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        system_prompt = (
            prompt + "\n\nReturn valid JSON only using the compact response schema."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        identity = require_model_cache_identity(
            self._client, component="spatial_candidate_review"
        )
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=version,
            messages=messages,
            image_sha256=image_hash,
            response_schema=SpatialCandidateReviewResult.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        if budget is not None:
            budget.reserve_qwen()
        return await self._client.complete_json(
            messages=messages,
            response_model=SpatialCandidateReviewResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:spatial-candidate-review",
                request_hash=request_hash,
                prompt_version=version,
                sample_id=sample.sample_id,
                image_sha256=image_hash,
                artifact_dir=artifact_dir / "spatial_agent_candidate_review",
            ),
            max_tokens=self._review_max_tokens,
        )


def _resolve_sample_image(sample: UnifiedSample, data_root: Path) -> Path:
    """Resolve the first sample image against data_root with escape
    protection. 按 data_root 解析样本首图并防逃逸。"""
    root = data_root.resolve()
    candidate = (root / sample.images[0].path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("image path escapes data root")
    if not candidate.is_file():
        raise ValueError("image file does not exist")
    return candidate
