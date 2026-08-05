"""Auditable dual-path ChangeAgent on the existing Agent runtime.
现有 Agent Runtime 上的可审计双路径 ChangeAgent。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from spacers_agent.agents.base import AgentContext, AgentExecution, AgentName
from spacers_agent.agents.change.preprocess import preprocess_pair
from spacers_agent.agents.change.reviewer import review_result
from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash, image_to_data_url
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.schemas import ExpertResult, UnifiedSample
from spacers_agent.settings import AgentChangeSettings


class ChangeAgent:
    """Discover changes on harmonized evidence and confirm semantics on raw evidence.
    在一致化证据上发现变化，并在原始证据上确认语义。
    """

    name: AgentName = "change_agent"
    supported_tasks: frozenset[str] = frozenset({"change_caption", "change_qa"})

    def __init__(
        self,
        client: VisionLanguageClient,
        prompt: PromptAsset,
        model: str,
        *,
        settings: AgentChangeSettings | None = None,
    ) -> None:
        self._client = client
        self._prompt = prompt
        self._model = model
        self._settings = settings or AgentChangeSettings()

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        settings = context.settings.agents.change if context.settings is not None else self._settings
        if not settings.dual_path_enabled:
            settings = settings.model_copy(update={"input_mode": "raw_only"})
        preprocess = preprocess_pair(sample, settings, context.artifact_dir)
        content, image_hashes, image_manifest = self._build_evidence(sample, context.artifact_dir, preprocess, settings)
        payload: dict[str, Any] = {
            "question": sample.question,
            "dataset": sample.dataset,
            "task": sample.task,
            "coordinate_frame": "normalized_0_999_top_left",
            "input_mode": settings.input_mode,
            "temporal_roles": [ref.role for ref in sample.images],
            "image_manifest": image_manifest,
            "harmonization": {
                "status": preprocess.decision.status,
                "reason_codes": preprocess.decision.reason_codes,
                "used_for_proposal": preprocess.decision.used_for_proposal,
            },
            "proposals": [
                {"proposal_id": item.proposal_id, "box": item.box, "score": round(item.score, 6)}
                for item in preprocess.proposals
            ],
            "empty_proposals_instruction": "Inspect raw full T1/T2; do not infer no change from an empty proposal list.",
        }
        content.append({"type": "text", "text": json.dumps(payload, ensure_ascii=False)})
        structured_prompt = (
            self._prompt.text
            + "\n\nReturn valid JSON only. Set expert to 'change_expert'; put the concise final answer in answer, "
            "retain relevant labeled boxes in evidence_items and boxes, use concise factual evidence strings, "
            "record proposal ids and evidence path types in geometry, and set status to 'completed'."
        )
        messages = [{"role": "system", "content": structured_prompt}, {"role": "user", "content": content}]
        request_hash = build_request_hash(
            model=self._model,
            generation={"temperature": 0.0},
            prompt_version=self._prompt.version,
            messages=messages,
            image_sha256="|".join(image_hashes),
        )
        if context.call_budget is not None:
            context.call_budget.reserve_qwen()
        result = await self._client.complete_json(
            messages=messages,
            response_model=ExpertResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:change_expert",
                request_hash=request_hash,
                prompt_version=self._prompt.version,
                sample_id=sample.sample_id,
                artifact_dir=context.artifact_dir / "change_expert",
            ),
        )
        reviewed, review_warnings = review_result(result, preprocess.proposals, settings.review)
        metrics = preprocess.decision.metrics
        trace = {
            "agent_class": "spacers_agent.agents.change.agent.ChangeAgent",
            "route": f"ChangeAgent.run -> preprocess_pair -> {type(self._client).__name__}.complete_json",
            "prompt_version": self._prompt.version,
            "image_roles": [ref.role for ref in sample.images],
            "input_mode": settings.input_mode,
            "harmonization_version": preprocess.decision.version,
            "harmonization_status": preprocess.decision.status,
            "harmonization_reason_codes": preprocess.decision.reason_codes,
            "pif_ratio": metrics.pif_ratio if metrics else None,
            "mad_pif_before": metrics.mad_pif_before if metrics else None,
            "mad_pif_after": metrics.mad_pif_after if metrics else None,
            "raw_fallback_used": not preprocess.decision.used_for_proposal,
            "sharpness_adjustment_used": bool(preprocess.transform_summary.get("sharpness_adjustment_used", False)),
            "proposal_count": len(preprocess.proposals),
            "proposal_source": "difference_map_v1",
            "review_used": settings.review.enabled,
            "review_warnings": review_warnings,
            "preprocess_artifacts": preprocess.artifact_files,
        }
        return AgentExecution(
            agent_name=self.name,
            payload=reviewed,
            result_filename="expert_result.json",
            trace=trace,
        )

    def _build_evidence(self, sample: UnifiedSample, artifact_dir: Path, preprocess: object, settings: AgentChangeSettings) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
        """Build mode-specific image content with an explicit ordered manifest.
        构建带显式顺序清单的模式专属图像内容。
        """

        paths: list[tuple[str, Path]] = []
        raw = [("raw_full_t1", sample.images[0].path), ("raw_full_t2", sample.images[1].path)]
        harmonized = []
        for key in ("harmonized_t1", "harmonized_t2"):
            relative = preprocess.artifact_files.get(key)
            if relative:
                harmonized.append((key, artifact_dir / relative))
        if settings.input_mode == "raw_only" or not harmonized:
            paths.extend(raw)
        elif settings.input_mode == "harmonized_only":
            paths.extend(harmonized)
        else:
            paths.extend(raw)
            paths.extend(harmonized)
            overlay = preprocess.artifact_files.get("proposal_overlay")
            if overlay:
                paths.append(("proposal_overlay", artifact_dir / overlay))
            for proposal in preprocess.proposals[: settings.proposals.max_proposals]:
                for relative in proposal.evidence_filenames:
                    paths.append((f"{proposal.proposal_id}:{Path(relative).stem}", artifact_dir / relative))
        content: list[dict[str, Any]] = []
        hashes: list[str] = []
        manifest: list[dict[str, str]] = []
        for index, (role, path) in enumerate(paths):
            data = path.read_bytes()
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(data, "image/png")}})
            hashes.append(hashlib.sha256(data).hexdigest())
            manifest.append({"index": str(index), "role": role})
        return content, hashes, manifest
