"""VRSBench quantity counting through proposal and grounded localization.
通过数量提议与落地定位执行 VRSBench 数量计数。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from spacers_agent.agents.base import AgentContext
from spacers_agent.agents.counting.backends.base import CountingBackendOutcome, CountingRequest
from spacers_agent.agents.counting.evidence import (
    accepted_count_evidence,
    box_evidence,
    global_count_point,
    parse_count_answer,
    recover_count_proposal_header,
)
from models.base import RequestMeta, build_request_hash, image_to_data_url
from spacers_agent.schemas import AgentName, AgentResult, CountTargetSpec, CountingResult, IssueRecord
from spacers_agent.vqa_geometry import vrsbench_vehicle_class


class _CountProposalResult(BaseModel):
    """Frozen response model for the whole-image count proposal.
    整图数量提议的冻结响应模型。
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName
    answer: str
    boxes: list[list[float]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    status: Literal["completed", "partial", "failed"] = "completed"


class VRSBenchQwenCountBackend:
    """Preserve the audited VRSBench proposal/localizer counting behavior.
    保持经审计的 VRSBench 提议/定位计数行为。
    """

    name = "vrsbench_qwen_count"
    priority = 5

    def __init__(self, client, *, settings, prompts: dict[str, str]) -> None:
        self._client = client
        self._settings = settings
        self._proposal_prompt = prompts["count_proposal"]
        self._localizer_prompt = prompts["count_localize"]

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec) -> bool:
        return vrsbench_vehicle_class(target.canonical_label) in {
            "vehicle",
            "small-vehicle",
            "large-vehicle",
        }

    async def count(
        self,
        request: CountingRequest,
        context: AgentContext,
    ) -> CountingBackendOutcome:
        sample = request.sample
        target = request.target
        proposal, recovery = await self._proposal(sample, request.artifact_dir, context)
        proposal_count = parse_count_answer(proposal.answer)
        issues: list[IssueRecord] = []
        if recovery is not None:
            issues.append(IssueRecord(code="COUNT_PROPOSAL_HEADER_RECOVERED", message=recovery))

        proposal_evidence = box_evidence(
            proposal.boxes,
            target.canonical_label,
            sample.images[0].image_id,
        )
        points, supporting_boxes, dropped = accepted_count_evidence(
            proposal_evidence,
            target.canonical_label,
            sample.images[0].image_id,
        )
        localization_used = proposal_count == 0 or len(points) != proposal_count
        localizer_answer: int | None = None
        if localization_used:
            issues.append(
                IssueRecord(
                    code="COUNT_PROPOSAL_EVIDENCE_MISMATCH",
                    message=f"proposal={proposal_count}, proposal_boxes={len(points)}",
                )
            )
            localized = await self._localize(
                sample,
                request.artifact_dir,
                target,
                proposal_count,
                context,
            )
            try:
                localizer_answer = parse_count_answer(localized.answer)
            except ValueError:
                localizer_answer = None
            localized_evidence = localized.evidence_items or box_evidence(
                localized.boxes,
                target.canonical_label,
                sample.images[0].image_id,
            )
            points, supporting_boxes, dropped = accepted_count_evidence(
                localized_evidence,
                target.canonical_label,
                sample.images[0].image_id,
            )

        if dropped:
            issues.append(
                IssueRecord(
                    code="COUNT_BORDER_OR_DUPLICATE_EVIDENCE_DROPPED",
                    message=f"Dropped {dropped} duplicate or tiny border-fragment observations.",
                )
            )
        complete = len(points) == proposal_count and (
            localizer_answer is None or localizer_answer == len(points) or dropped > 0
        )
        if not complete:
            issues.append(
                IssueRecord(
                    code="COUNT_LOCALIZATION_EVIDENCE_MISMATCH",
                    message=(
                        f"proposal={proposal_count}, localizer={localizer_answer}, "
                        f"accepted_points={len(points)}"
                    ),
                )
            )

        global_points = [
            global_count_point(
                sample.sample_id,
                target.canonical_label,
                item,
                index,
                request.image.width,
                request.image.height,
            )
            for index, item in enumerate(points, start=1)
        ]
        counting_status: Literal["completed", "completed_with_warnings", "partial"] = (
            "partial" if not complete else "completed_with_warnings" if issues else "completed"
        )
        counting = CountingResult(
            sample_id=sample.sample_id,
            target=target.canonical_label,
            question=sample.question,
            source_width=request.image.width,
            source_height=request.image.height,
            tile_count=1,
            initial_tile_count=1,
            leaf_tile_count=1,
            succeeded_tiles=["whole_image_overview"],
            failed_tiles=[],
            global_points=global_points,
            merged_groups=[],
            unresolved_conflicts=[],
            warnings=issues,
            final_count=len(global_points),
            status=counting_status,
        )
        geometry: dict[str, Any] = {
            "version": "accepted-point-count-v3",
            "prompt_version": "vrsbench-count-hybrid-v1",
            "coordinate_frame": "normalized_0_999_top_left",
            "rule": "final_count_equals_accepted_points",
            "accepted_point_count": len(points),
            "final_count": counting.final_count,
            "proposal_count": proposal_count,
            "proposal_status": proposal.status,
            "proposal_recovery": recovery,
            "localization_used": localization_used,
            "localizer_answer": localizer_answer,
            "supporting_box_count": len(supporting_boxes),
            "pipeline": "general_vqa_v1_proposal_then_grounded_localization",
            "counting_status": counting.status,
            "warnings": [item.model_dump(mode="json") for item in counting.warnings],
        }
        agent_result = AgentResult(
            agent_name="counting_agent",
            answer=(
                str(counting.final_count)
                if complete
                else f"Confirmed {counting.final_count} localized instances; the count is incomplete."
            ),
            boxes=[[float(value) for value in box] for box in supporting_boxes],
            evidence=[
                f"Accepted point {index + 1}: {item.point}" for index, item in enumerate(points)
            ],
            evidence_items=points,
            geometry=geometry,
            status="completed" if complete else "partial",
        )
        return CountingBackendOutcome(
            counting=counting,
            agent_result=agent_result,
            trace={
                "prompt_version": "vrsbench-count-hybrid-v1",
                "geometry": agent_result.geometry,
                "localization_used": localization_used,
            },
        )

    async def _proposal(
        self,
        sample,
        sample_dir,
        context: AgentContext,
    ) -> tuple[_CountProposalResult, str | None]:
        image_bytes = sample.images[0].path.read_bytes()
        system_prompt = self._proposal_prompt + (
            "\n\nReturn valid JSON only. Set agent_name to 'counting_agent'; put the concise "
            "final answer in answer, use empty boxes/evidence when they are not needed, and set "
            "status to 'completed'."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_bytes, "image/png")},
                    },
                    {"type": "text", "text": sample.question},
                ],
            },
        ]
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        request_hash = build_request_hash(
            model=self._settings.models.qwen.model,
            generation={"temperature": 0.0, "max_tokens": self._settings.models.qwen.max_tokens},
            prompt_version="general-vqa-v1-count-proposal",
            messages=messages,
            image_sha256=image_hash,
        )
        artifact_dir = sample_dir / "counting_agent" / "count_proposal"
        try:
            context.call_budget.reserve_qwen()
            proposal = await self._client.complete_json(
                messages=messages,
                response_model=_CountProposalResult,
                request_meta=RequestMeta(
                    request_id=f"{sample.sample_id}:count-proposal",
                    request_hash=request_hash,
                    prompt_version="general-vqa-v1-count-proposal",
                    sample_id=sample.sample_id,
                    image_sha256=image_hash,
                    artifact_dir=artifact_dir,
                ),
            )
            return proposal, None
        except Exception:
            raw_path = artifact_dir / "raw_response.txt"
            recovered = recover_count_proposal_header(
                raw_path.read_text(encoding="utf-8") if raw_path.is_file() else ""
            )
            if recovered is None:
                raise
            return (
                _CountProposalResult(
                    agent_name="counting_agent",
                    answer=str(recovered),
                    boxes=[],
                    evidence=[],
                    status="partial",
                ),
                "Recovered a complete integer answer header; malformed geometry was discarded.",
            )

    async def _localize(
        self,
        sample,
        sample_dir,
        target: CountTargetSpec,
        proposal_count: int,
        context: AgentContext,
    ) -> AgentResult:
        image_bytes = sample.images[0].path.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._localizer_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_bytes, "image/png")},
                    },
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "question": sample.question,
                                "target_spec": target.model_dump(mode="json"),
                                "independent_count_proposal": proposal_count,
                                "image_scope": "complete_image",
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            },
        ]
        request_hash = build_request_hash(
            model=self._settings.models.qwen.model,
            generation={"temperature": 0.0, "max_tokens": self._settings.models.qwen.max_tokens},
            prompt_version="count-localize-v1",
            messages=messages,
            image_sha256=image_hash,
            target_spec=target.model_dump(mode="json"),
        )
        context.call_budget.reserve_qwen()
        return await self._client.complete_json(
            messages=messages,
            response_model=AgentResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:count-localizer",
                request_hash=request_hash,
                prompt_version="count-localize-v1",
                sample_id=sample.sample_id,
                image_sha256=image_hash,
                artifact_dir=sample_dir / "counting_agent" / "count_localizer",
            ),
        )
