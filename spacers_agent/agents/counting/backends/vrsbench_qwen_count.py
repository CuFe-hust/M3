"""VRSBench quantity counting backend — proposal + localizer path.
VRSBench 数量计数后端 — proposal + localizer 路径。

Preserves the existing 200-sample-comparable VRSBench vehicle counting pipeline.
保留现有 200 条样本可比 VRSBench 车辆计数管线。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spacers_agent.agents.counting.backends.base import CountingBackend, CountingRequest
from spacers_agent.agents.counting.evidence import (
    accepted_count_evidence,
    box_evidence,
    global_count_point,
    parse_count_answer,
    recover_count_proposal_header,
)
from spacers_agent.clients.base import RequestMeta, build_request_hash, image_to_data_url
from spacers_agent.schemas import CountTargetSpec, CountingResult, IssueRecord
from spacers_agent.vqa_geometry import vrsbench_count_target


class _CountProposalResult(BaseModel):
    """Compact whole-image count proposal. / 紧凑整图计数提议。"""
    model_config = ConfigDict(extra="forbid")
    expert: str
    answer: str
    boxes: list[list[float]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    status: str = "completed"


class VRSBenchQwenCountBackend:
    """VRSBench vehicle counting via proposal + optional localizer. / 通过 proposal + 可选 localizer 的 VRSBench 车辆计数。"""

    name = "vrsbench_qwen_count"
    priority = 5  # above qwen_point for VRSBench / 高于 qwen_point

    def __init__(self, client, *, settings, prompts: dict[str, str]) -> None:
        self._client = client
        self._settings = settings
        self._prompts = prompts

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec) -> bool:
        # Only for VRSBench vehicle counting / 仅 VRSBench 车辆计数
        from spacers_agent.vqa_geometry import vrsbench_vehicle_class
        cls = vrsbench_vehicle_class(target.canonical_label)
        return cls in {"small-vehicle", "large-vehicle"}

    async def count(self, request: CountingRequest, context: object) -> CountingResult:
        sample = request.sample
        target = vrsbench_count_target(sample.question)
        sample_dir = request.artifact_dir

        # 1. Proposal / 提议
        proposal, recovery = await self._proposal(sample, sample_dir)
        proposal_count = parse_count_answer(proposal.answer)
        issues: list[IssueRecord] = []
        if recovery:
            issues.append(IssueRecord(code="COUNT_PROPOSAL_HEADER_RECOVERED", message=recovery))

        proposal_evidence = box_evidence(proposal.boxes, target.canonical_label, sample.images[0].image_id)
        points, supporting_boxes, dropped = accepted_count_evidence(
            proposal_evidence, target.canonical_label, sample.images[0].image_id,
        )

        # 2. Localize if mismatch / 不匹配则定位
        localization_used = proposal_count == 0 or len(points) != proposal_count
        localizer_answer: int | None = None
        if localization_used:
            issues.append(IssueRecord(
                code="COUNT_PROPOSAL_EVIDENCE_MISMATCH",
                message=f"proposal={proposal_count}, proposal_boxes={len(points)}",
            ))
            localized = await self._localize(sample, sample_dir, target, proposal_count)
            try:
                localizer_answer = parse_count_answer(localized.answer)
            except ValueError:
                localizer_answer = None
            localized_evidence = localized.evidence_items or box_evidence(
                localized.boxes, target.canonical_label, sample.images[0].image_id,
            )
            points, supporting_boxes, dropped = accepted_count_evidence(
                localized_evidence, target.canonical_label, sample.images[0].image_id,
            )

        if dropped:
            issues.append(IssueRecord(
                code="COUNT_BORDER_OR_DUPLICATE_EVIDENCE_DROPPED",
                message=f"Dropped {dropped} duplicate or tiny border-fragment observations.",
            ))

        complete = len(points) == proposal_count and (localizer_answer is None or localizer_answer == len(points) or dropped > 0)
        if not complete:
            issues.append(IssueRecord(
                code="COUNT_LOCALIZATION_EVIDENCE_MISMATCH",
                message=f"proposal={proposal_count}, localizer={localizer_answer}, accepted_points={len(points)}",
            ))

        global_points = [
            global_count_point(sample.sample_id, target.canonical_label, item, idx,
                               request.image.width, request.image.height)
            for idx, item in enumerate(points, start=1)
        ]

        status = "partial" if not complete else "completed_with_warnings" if issues else "completed"
        return CountingResult(
            sample_id=sample.sample_id, target=target.canonical_label,
            question=sample.question, source_width=request.image.width,
            source_height=request.image.height, tile_count=1,
            initial_tile_count=1, leaf_tile_count=1,
            succeeded_tiles=["whole_image_overview"], failed_tiles=[],
            global_points=global_points, merged_groups=[], unresolved_conflicts=[],
            warnings=issues, final_count=len(global_points), status=status,  # type: ignore[arg-type]
        )

    async def _proposal(self, sample, sample_dir) -> tuple[_CountProposalResult, str | None]:
        image_bytes = sample.images[0].path.read_bytes()
        system_prompt = self._prompts.get("count_proposal", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_bytes, "image/png")}},
                {"type": "text", "text": sample.question},
            ]},
        ]
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        request_hash = build_request_hash(
            model=self._settings.models.qwen.model,
            generation={"temperature": 0.0, "max_tokens": self._settings.models.qwen.max_tokens},
            prompt_version="general-vqa-v1-count-proposal", messages=messages, image_sha256=image_hash,
        )
        artifact_dir = sample_dir / "counting_expert" / "count_proposal"
        try:
            proposal = await self._client.complete_json(
                messages=messages, response_model=_CountProposalResult,
                request_meta=RequestMeta(
                    request_id=f"{sample.sample_id}:count-proposal", request_hash=request_hash,
                    prompt_version="general-vqa-v1-count-proposal", sample_id=sample.sample_id,
                    image_sha256=image_hash, artifact_dir=artifact_dir,
                ),
            )
            return proposal, None
        except Exception:
            raw_path = artifact_dir / "raw_response.txt"
            recovered = recover_count_proposal_header(raw_path.read_text(encoding="utf-8") if raw_path.is_file() else "")
            if recovered is None:
                raise
            return _CountProposalResult(expert="general_vqa_expert", answer=str(recovered), status="partial"), (
                "Recovered a complete integer answer header; malformed geometry was discarded."
            )

    async def _localize(self, sample, sample_dir, target, proposal_count):
        from spacers_agent.schemas import ExpertResult
        image_bytes = sample.images[0].path.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        messages = [
            {"role": "system", "content": self._prompts.get("count_localize", "")},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_bytes, "image/png")}},
                {"type": "text", "text": json.dumps({
                    "question": sample.question,
                    "target_spec": target.model_dump(mode="json"),
                    "independent_count_proposal": proposal_count,
                    "image_scope": "complete_image",
                }, ensure_ascii=False)},
            ]},
        ]
        request_hash = build_request_hash(
            model=self._settings.models.qwen.model,
            generation={"temperature": 0.0, "max_tokens": self._settings.models.qwen.max_tokens},
            prompt_version="count-localize-v1", messages=messages, image_sha256=image_hash,
            target_spec=target.model_dump(mode="json"),
        )
        return await self._client.complete_json(
            messages=messages, response_model=ExpertResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:count-localizer", request_hash=request_hash,
                prompt_version="count-localize-v1", sample_id=sample.sample_id,
                image_sha256=image_hash, artifact_dir=sample_dir / "counting_expert" / "count_localizer",
            ),
        )
