"""Spatial candidate review — independent localization completeness pass.
空间候选复查 — 独立的定位完整性复核。

Extracted from ``workflow.SpatialExpert._review_candidates`` and
``workflow.SpatialExpert.run`` review logic. Not a top-level Agent.
从 ``workflow.SpatialExpert._review_candidates`` 和
``workflow.SpatialExpert.run`` 复查逻辑中提取。不是顶层 Agent。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash, image_to_data_url
from spacers_agent.schemas import ExpertResult, UnifiedSample, VisualEvidence
from spacers_agent.vqa_geometry import vrsbench_answer_vocabulary, vrsbench_question_subtype, vrsbench_vehicle_class
from spacers_agent.workflow import (
    _is_corner_anchored_box,
    _is_status_answer_placeholder,
    _matches_position_target,
    _maximum_repair_severity,
    _merge_visual_evidence,
    _needs_spatial_candidate_review,
    _position_review_evidence,
)


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
    ) -> None:
        self._client = client
        self._model = model
        self._review_prompt = review_prompt
        self._review_prompt_version = review_prompt_version
        self._grid_review_prompt = grid_review_prompt
        self._grid_review_prompt_version = grid_review_prompt_version

    def needs_review(self, sample: UnifiedSample, result: ExpertResult) -> bool:
        """Return whether a spatial result requires candidate review.
        返回空间结果是否需要候选复查。
        """
        return _needs_spatial_candidate_review(sample, result)

    async def review(
        self, sample: UnifiedSample, first_result: ExpertResult, artifact_dir: Path
    ) -> ExpertResult:
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
                if not (_matches_position_target(sample.question, item) and _is_corner_anchored_box(item))
            ]
            replaced_evidence = len(first_result.evidence_items) - len(first_evidence)

        review_evidence, labeled_review_boxes = _position_review_evidence(sample.question, subtype, review_result)
        merged = _merge_visual_evidence(first_evidence, review_evidence)

        # -- geometry audit / 几何审计 --
        geometry = dict(first_result.geometry)
        merged_quality = ["trusted_box" if item.box is not None else "trusted_point" for item in merged]
        geometry.update({
            "candidate_review_used": True,
            "candidate_review_added": len(merged) - len(first_evidence),
            "candidate_review_replaced": replaced_evidence,
            "candidate_review_labeled_boxes": labeled_review_boxes,
            "candidate_review_geometry": review_result.geometry,
            "evidence_quality": merged_quality,
            "repair_severity": _maximum_repair_severity(
                str(first_result.geometry.get("repair_severity", "none")),
                str(review_result.geometry.get("repair_severity", "none")),
            ),
        })

        # -- finalise answer / 最终化答案 --
        reviewed_answer = first_result.answer
        if _is_status_answer_placeholder(reviewed_answer) and not _is_status_answer_placeholder(review_result.answer):
            reviewed_answer = review_result.answer

        status = "partial" if _needs_spatial_candidate_review(sample, first_result) else "completed"
        return first_result.model_copy(update={
            "answer": reviewed_answer,
            "boxes": [list(item.box) for item in merged if item.box is not None],
            "evidence_items": merged,
            "geometry": geometry,
            "status": status,
        })

    async def _request_review(
        self, sample: UnifiedSample, prompt: str, version: str, subtype: str, artifact_dir: Path
    ) -> ExpertResult:
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
            + "\n\nReturn valid JSON only. Set expert to 'spatial_expert', keep answer concise, "
            "copy every evidence box into boxes, and set status to 'completed' only when enumeration is complete."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        request_hash = build_request_hash(
            model=self._model,
            generation={"temperature": 0.0},
            prompt_version=version,
            messages=messages,
            image_sha256="|".join(image_hashes),
        )
        return await self._client.complete_json(
            messages=messages,
            response_model=ExpertResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:spatial-candidate-review",
                request_hash=request_hash,
                prompt_version=version,
                sample_id=sample.sample_id,
                artifact_dir=artifact_dir / "spatial_expert_candidate_review",
            ),
        )
