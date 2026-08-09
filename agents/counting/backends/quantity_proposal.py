"""Quantity proposal backend — whole-image proposal then grounded localization.

数量提议后端 — 整图数量提议 + 落地定位。仅根据 target/hints 判断能力，
绝不做数据来源判断；没有可靠 hint 时拒绝 supports 而不是猜测。不实现
Agent 回退、不自行创建提示词目录（所有 prompt text/version 由构造参数
注入）。
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.base import CallBudget
from agents.counting.backends.base import (
    BackendKind,
    CountingBackendOutcome,
    CountingRequest,
    require_model_cache_identity,
)
from agents.counting.evidence import (
    accepted_count_evidence,
    box_evidence,
    global_count_point,
    parse_count_answer,
    recover_count_proposal_header,
)
from agents.counting.schema import CountTargetSpec, CountingResult, IssueRecord
from agents.counting.settings import CountingSettings
from agents.schema import AgentName, AgentResult
from models.base import RequestMeta, VisionLanguageClient, build_request_hash
from models.images import image_to_data_url


def _is_recoverable_proposal_error(error: Exception) -> bool:
    """Only parse/validation/repair failures may trigger raw-response recovery;
    network, filesystem, permission, and budget errors must propagate
    unchanged. The models layer is the single stable source of model-client
    errors (e.g. QwenTransformersError), so an error whose class is defined in
    a models.* module is treated as a repairable model failure without
    importing any concrete client module into the agent layer.
    只有解析/校验/修复类失败可触发 raw response 恢复；网络、文件系统、权限
    与 budget 错误必须原样传播。models.* 是模型客户端错误的唯一稳定来源
    （如 QwenTransformersError），因此类定义在 models.* 模块中的错误被视为
    可修复的模型失败，无需把具体客户端模块 import 进 Agent 层。"""
    if isinstance(error, ValueError):
        return True
    return (
        isinstance(error, RuntimeError)
        and error.__class__.__module__.startswith("models.")
    )


class _CountProposalResult(BaseModel):
    """Frozen response model for the whole-image count proposal.
    整图数量提议的冻结响应模型。"""

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName
    answer: str
    boxes: list[list[float]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    status: Literal["completed", "partial", "failed"] = "completed"


class QuantityProposalBackend:
    """Neutral quantity proposal/localizer counting backend.
    中性的数量提议/定位计数后端。"""

    name = "quantity_proposal"
    kind: BackendKind = "quantity_proposal"
    priority = 5

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        counting: CountingSettings,
        proposal_prompt: str,
        localizer_prompt: str,
        proposal_prompt_version: str,
        localizer_prompt_version: str,
        supported_targets: tuple[str, ...] = ("small-vehicle", "large-vehicle"),
    ) -> None:
        self._client = client
        self._counting = counting
        self._proposal_prompt = proposal_prompt
        self._localizer_prompt = localizer_prompt
        self._proposal_prompt_version = proposal_prompt_version
        self._localizer_prompt_version = localizer_prompt_version
        self._supported_targets = frozenset(
            _normalize_target(value) for value in supported_targets
        )

    def is_enabled(self) -> bool:
        return True  # always configured / 始终配置

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        """Refuse without a reliable hint instead of guessing.
        没有可靠 hint 时拒绝，而不是猜测。"""
        if not isinstance(hints, dict) or not hints.get("quantity_estimation"):
            return False
        if hints.get("countable") is False:
            return False
        hinted_label = hints.get("canonical_label")
        label = (
            hinted_label
            if isinstance(hinted_label, str) and hinted_label.strip()
            else target.canonical_label
        )
        return _normalize_target(label) in self._supported_targets

    async def count(
        self,
        request: CountingRequest,
        context: object,
    ) -> CountingBackendOutcome:
        sample = request.sample
        target = request.target
        budget = getattr(context, "call_budget", None)
        proposal, recovery = await self._proposal(
            request, sample_id=sample.sample_id, budget=budget
        )
        proposal_count = parse_count_answer(proposal.answer)
        issues: list[IssueRecord] = []
        if recovery is not None:
            issues.append(
                IssueRecord(code="COUNT_PROPOSAL_HEADER_RECOVERED", message=recovery)
            )

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
                request,
                sample_id=sample.sample_id,
                target=target,
                proposal_count=proposal_count,
                budget=budget,
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
                    message=(
                        f"Dropped {dropped} duplicate or tiny border-fragment observations."
                    ),
                )
            )
        complete = len(points) == proposal_count and (
            localizer_answer is None
            or localizer_answer == len(points)
            or dropped > 0
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
            "partial"
            if not complete
            else "completed_with_warnings"
            if issues
            else "completed"
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
            "prompt_version": self._proposal_prompt_version,
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
            "pipeline": "quantity_proposal_then_grounded_localization",
            "counting_status": counting.status,
            "warnings": [item.model_dump(mode="json") for item in counting.warnings],
        }
        agent_result = AgentResult(
            agent_name="counting_agent",
            answer=(
                str(counting.final_count)
                if complete
                else f"Confirmed {counting.final_count} localized instances; "
                "the count is incomplete."
            ),
            boxes=[[float(value) for value in box] for box in supporting_boxes],
            evidence=[
                f"Accepted point {index + 1}: {item.point}"
                for index, item in enumerate(points)
            ],
            evidence_items=points,
            geometry=geometry,
            status="completed" if complete else "partial",
        )
        return CountingBackendOutcome(
            counting=counting,
            agent_result=agent_result,
            trace={
                "backend": self.name,
                "pipeline": "quantity_proposal_then_grounded_localization",
                "proposal_prompt_version": self._proposal_prompt_version,
                "localizer_prompt_version": self._localizer_prompt_version,
                "localization_used": localization_used,
            },
        )

    async def _proposal(
        self,
        request: CountingRequest,
        *,
        sample_id: str,
        budget: CallBudget | None,
    ) -> tuple[_CountProposalResult, str | None]:
        image_bytes = _encode_image(request.image)
        system_prompt = self._proposal_prompt + (
            "\n\nReturn valid JSON only. Set agent_name to 'counting_agent'; put the "
            "concise final answer in answer, use empty boxes/evidence when they are not "
            "needed, and set status to 'completed'."
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
                    {"type": "text", "text": request.sample.question},
                ],
            },
        ]
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        identity = require_model_cache_identity(
            self._client, component="quantity_proposal"
        )
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=self._proposal_prompt_version,
            messages=messages,
            image_sha256=image_hash,
            response_schema=_CountProposalResult.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        artifact_dir = (
            request.artifact_dir / "counting_agent" / "count_proposal" / request_hash
        )
        if budget is not None:
            budget.reserve_qwen()
        try:
            proposal = await self._client.complete_json(
                messages=messages,
                response_model=_CountProposalResult,
                request_meta=RequestMeta(
                    request_id=f"{sample_id}:count-proposal",
                    request_hash=request_hash,
                    prompt_version=self._proposal_prompt_version,
                    sample_id=sample_id,
                    image_sha256=image_hash,
                    artifact_dir=artifact_dir,
                ),
            )
            return proposal, None
        except Exception as error:
            if not _is_recoverable_proposal_error(error):
                raise
            # Recovery reads ONLY the current request's hash-scoped artifact
            # directory, so stale raw responses from previous requests can
            # never be reused. 恢复只读取当前请求的 hash 作用域产物目录，
            # 旧请求遗留的 raw response 绝不可能被复用。
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
                "Recovered a complete integer answer header; malformed geometry "
                "was discarded.",
            )

    async def _localize(
        self,
        request: CountingRequest,
        *,
        sample_id: str,
        target: CountTargetSpec,
        proposal_count: int,
        budget: CallBudget | None,
    ) -> AgentResult:
        image_bytes = _encode_image(request.image)
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
                                "question": request.sample.question,
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
        identity = require_model_cache_identity(
            self._client, component="quantity_proposal"
        )
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=self._localizer_prompt_version,
            messages=messages,
            image_sha256=image_hash,
            target_spec=target.model_dump(mode="json"),
            response_schema=AgentResult.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        if budget is not None:
            budget.reserve_qwen()
        return await self._client.complete_json(
            messages=messages,
            response_model=AgentResult,
            request_meta=RequestMeta(
                request_id=f"{sample_id}:count-localizer",
                request_hash=request_hash,
                prompt_version=self._localizer_prompt_version,
                sample_id=sample_id,
                image_sha256=image_hash,
                artifact_dir=request.artifact_dir / "counting_agent" / "count_localizer",
            ),
        )


def _encode_image(image: Any) -> bytes:
    """Encode an opened image as PNG bytes without touching the filesystem.
    将已打开的图片编码为 PNG 字节，不触碰文件系统。"""
    with io.BytesIO() as buffer:
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()


def _normalize_target(value: str) -> str:
    separators_normalized = value.strip().casefold().replace("_", " ").replace("-", " ")
    return "-".join(separators_normalized.split())


