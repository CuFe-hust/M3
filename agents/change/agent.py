"""Auditable dual-path ChangeAgent.

可审计双路径 ChangeAgent。组合 raw/harmonized/proposal 证据构造三种有序
image manifest（raw_only / harmonized_only / dual_path），一次结构化 Qwen
调用，模型结果经保守规则复核（review_result）后输出。不接入评测器、不做
评测、不读取数据集标注格式；输入模式由既有设置确定，不按数据集名分支。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from agents.base import AgentContext, AgentExecution
from agents.change.perception import (
    ChangePerceptionError,
    ChangePerceptionPipeline,
)
from agents.change.preprocess import prepare_pair, publish_change_proposals
from agents.change.registration import RegistrationError
from agents.change.reviewer import meaningful_proposals, review_outcome, review_result
from agents.change.schema import ChangeAdjudicationResult, ChangeInitialResult, ChangePreprocessResult, SemanticTransition
from agents.change.settings import AgentChangeSettings
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.schema import AgentName, AgentResult
from agents.visual_base import PromptBinding
from data.schema import UnifiedSample
from models.base import (
    DenseSemanticClient,
    LearnedChangeClient,
    ModelCacheIdentity,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
)
from models.images import UnsupportedImageFormatError, detect_image_mime, image_to_data_url

# Runtime prompt authority is injected from the versioned PromptCatalog.
InputMode = Literal["raw_only", "harmonized_only", "dual_path"]
EvidenceStage = Literal["initial", "adjudication"]


def resolve_input_mode(settings: AgentChangeSettings) -> InputMode:
    """Derive the ordered image manifest mode from the enabled preprocessing
    stages. Harmonization disabled leaves only raw evidence; enabled without
    proposals uses harmonized comparison images; enabled with proposals uses
    the full dual-path evidence set. Deterministic, dataset-neutral.
    根据启用的预处理阶段推导有序图像 manifest 模式。一致化关闭时只剩原始
    证据；开启但无提议时使用一致化对比图；两者均开启时使用完整双路径证据
    集。确定性、与数据集无关。"""
    if not settings.harmonization.enabled:
        return "raw_only"
    if not settings.proposals.enabled:
        return "harmonized_only"
    return "dual_path"


class ChangeAgent:
    """Discover changes on harmonized evidence and confirm semantics on raw
    evidence. 在一致化证据上发现变化，并在原始证据上确认语义。"""

    name: AgentName = "change_agent"
    supported_tasks: frozenset[str] = frozenset({"change_caption", "change_qa"})

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        semantic_client: DenseSemanticClient | None = None,
        learned_change_client: LearnedChangeClient | None = None,
        prompt: PromptBinding | None = None,
        settings: AgentChangeSettings | None = None,
    ) -> None:
        self._client = client
        self._semantic_client = semantic_client
        self._learned_change_client = learned_change_client
        if prompt is None:
            raise ValueError("ChangeAgent requires an injected PromptBinding")
        self._prompt = prompt
        self._settings = settings or AgentChangeSettings()

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        """Execute the auditable dual-path pipeline for one sample.
        为单条样本执行可审计双路径管线。"""
        if sample.task not in self.supported_tasks:
            raise AgentTaskMismatchError(
                self.name, sample.task, supported=self.supported_tasks
            )

        # The client's own cache identity is the only hash source; it must be
        # a real ModelCacheIdentity so path-like models cannot bypass
        # validation. Fail before preprocessing, consuming budget, or calling
        # the model.
        # 客户端自身的缓存身份是唯一哈希来源；必须是真正的
        # ModelCacheIdentity，使 path-like 模型无法绕过校验。在预处理、
        # 消费 budget、调用模型之前显式失败。
        identity = getattr(self._client, "cache_identity", None)
        if not isinstance(identity, ModelCacheIdentity):
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="model client returned an invalid cache_identity",
            )
        if context.data_root is None:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="data_root is required to resolve relative ImageRef paths",
            )

        settings = self._settings
        preprocess = self._prepare_perception_and_publish(sample, context)
        # Invalid temporal pairs must fail before evidence build, budget
        # reservation, or any model call. 无效时相图对必须在构建证据、消费
        # budget 或调用模型之前稳定失败。
        mode = resolve_input_mode(settings)
        content, image_hashes, image_manifest, evidence_audit = self._build_evidence(
            sample, context, preprocess, mode, stage="initial"
        )
        perception_audit = _perception_audit(
            preprocess,
            settings=settings,
        )
        payload = self._request_payload(
            sample=sample, preprocess=preprocess, mode=mode, perception_audit=perception_audit,
            image_manifest=image_manifest, evidence_audit=evidence_audit, stage="initial",
        )
        content.append({"type": "text", "text": json.dumps(payload, ensure_ascii=False)})
        result, request_hash = await self._call_vlm_json(
            sample=sample, context=context, identity=identity, content=content,
            image_hashes=image_hashes, response_model=ChangeInitialResult, decision_stage="initial",
        )
        if result.agent_name != self.name:
            raise AgentExecutionError(self.name, sample.sample_id, cause=f"model returned agent_name {result.agent_name!r}")

        result = _normalize_positive_caption(result)
        review = review_outcome(result, preprocess.proposals, settings.review, task=sample.task)
        reviewed, initial_warnings = review.result, list(review.warnings)
        adjudication_used = False
        adjudication_request_hash: str | None = None
        adjudication_candidate_ids: list[str] = []
        adjudication_global_verdict: str | None = None
        adjudication_verdicts: dict[str, str] = {}
        adjudication_outcome: str | None = None
        consistency_warnings: list[str] = []
        final_warnings = initial_warnings
        if settings.review.adjudication_enabled and review.route != "accept":
            selected = self._select_proposals(
                meaningful_proposals(preprocess.proposals, settings.review) or preprocess.proposals,
                limit=settings.evidence.adjudication_max_proposals,
            )
            adjudication_candidate_ids = [item.proposal_id for item in selected]
            content, image_hashes, image_manifest, adjudication_audit = self._build_evidence(
                sample, context, preprocess, mode, stage="adjudication", selected_proposals=selected
            )
            adjudication_payload = self._request_payload(
                sample=sample, preprocess=preprocess, mode=mode, perception_audit=perception_audit,
                image_manifest=image_manifest, evidence_audit=adjudication_audit, stage="adjudication",
                first_pass={"answer": result.answer, "review_warnings": initial_warnings, "review_route_reasons": list(review.route_reasons)},
                selected_proposals=selected,
            )
            content.append({"type": "text", "text": json.dumps(adjudication_payload, ensure_ascii=False)})
            adjudication, adjudication_request_hash = await self._call_vlm_json(
                sample=sample, context=context, identity=identity, content=content, image_hashes=image_hashes,
                response_model=ChangeAdjudicationResult, decision_stage="adjudication",
            )
            consistency_warnings = self._validate_adjudication(adjudication, adjudication_candidate_ids)
            reviewed, adjudication_outcome = self._merge_adjudication(adjudication, sample.task, consistency_warnings)
            reviewed, final_warnings = review_result(reviewed, preprocess.proposals, settings.review)
            if adjudication_outcome == "negative":
                final_warnings = [warning for warning in final_warnings if warning != "CHANGE_RESULT_CONFLICT"]
                if not final_warnings:
                    reviewed = reviewed.model_copy(update={"status": "completed"})
            if adjudication_outcome == "unresolved":
                final_warnings = list(dict.fromkeys([*final_warnings, "CHANGE_ADJUDICATION_UNRESOLVED"]))
                reviewed = reviewed.model_copy(update={"status": "partial"})
            adjudication_used = True
            evidence_audit = adjudication_audit
            adjudication_global_verdict = adjudication.global_review.verdict
            adjudication_verdicts = {item.proposal_id: item.verdict for item in adjudication.candidate_reviews}
        metrics = preprocess.decision.metrics
        trace = {
            "prompt_version": self._prompt.version,
            "request_hash": request_hash,
            "image_sha256": image_hashes,
            "model": identity.model,
            "image_roles": [ref.role for ref in sample.images],
            "input_mode": mode,
            **evidence_audit,
            "harmonization_version": preprocess.decision.version,
            "harmonization_status": preprocess.decision.status,
            "harmonization_reason_codes": preprocess.decision.reason_codes,
            **_registration_payload(preprocess),
            "pif_ratio": metrics.pif_ratio if metrics else None,
            "mad_pif_before": metrics.mad_pif_before if metrics else None,
            "mad_pif_after": metrics.mad_pif_after if metrics else None,
            "raw_fallback_used": not preprocess.decision.used_for_proposal,
            "sharpness_adjustment_used": bool(
                preprocess.transform_summary.get("sharpness_adjustment_used", False)
            ),
            "proposal_count": len(preprocess.proposals),
            **perception_audit,
            # Compatibility aliases retained for readers of the Task 08 trace.
            "semantic_reason_code": preprocess.diagnostics.get(
                "semantic_reason_code"
            ),
            "segformer_model": preprocess.diagnostics.get("semantic_model"),
            "review_used": settings.review.enabled,
            "review_warnings": final_warnings,
            "adjudication_enabled": settings.review.adjudication_enabled,
            "adjudication_used": adjudication_used,
            "adjudication_trigger": ("negative_conflict" if review.route == "adjudicate_negative" else "positive_conflict") if adjudication_used else None,
            "adjudication_request_hash": adjudication_request_hash,
            "adjudication_candidate_ids": adjudication_candidate_ids,
            "adjudication_global_verdict": adjudication_global_verdict,
            "adjudication_verdicts": adjudication_verdicts,
            "adjudication_outcome": adjudication_outcome,
            "initial_review_warnings": initial_warnings,
            "final_review_warnings": final_warnings,
            "review_route": review.route,
            "review_route_reasons": list(review.route_reasons),
            "initial_result_normalizations": result.geometry.get("change_input_normalizations", []),
            "adjudication_consistency_warnings": consistency_warnings if adjudication_used else [],
            "final_merge_reason": adjudication_outcome,
            "preprocess_artifacts": preprocess.artifact_files,
            "perception_artifacts": preprocess.artifact_files,
        }
        return AgentExecution(
            agent_name=self.name,
            payload=reviewed,
            result_filename="agent_result.json",
            trace=trace,
        )

    async def _call_vlm_json(
        self, *, sample: UnifiedSample, context: AgentContext, identity: ModelCacheIdentity,
        content: list[dict[str, Any]], image_hashes: list[str], response_model: Any,
        decision_stage: EvidenceStage,
    ) -> tuple[Any, str]:
        suffix = (
            "Decision stage is initial. Return valid JSON matching AgentResult only. "
            "Set agent_name to change_agent and status to completed."
            if decision_stage == "initial" else
            "Decision stage is adjudication. Return valid JSON matching ChangeAdjudicationResult only. "
            "Review the global raw pair and every supplied adjudication candidate exactly once."
        )
        messages = [
            {"role": "system", "content": self._prompt.text + "\n\n" + suffix},
            {"role": "user", "content": content},
        ]
        request_hash = build_request_hash(
            model=identity.model, generation=identity.generation_payload(), prompt_version=self._prompt.version,
            messages=messages, image_sha256="|".join(image_hashes),
            response_schema=response_model.model_json_schema(), client_version=identity.client_version,
            model_revision=identity.revision,
        )
        context.call_budget.reserve_qwen()
        result = await self._client.complete_json(
            messages=messages, response_model=response_model,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:{self.name}" + ("" if decision_stage == "initial" else ":adjudication"),
                request_hash=request_hash, prompt_version=self._prompt.version, sample_id=sample.sample_id,
                artifact_dir=context.artifact_dir / self.name,
            ),
        )
        return result, request_hash

    def _request_payload(
        self, *, sample: UnifiedSample, preprocess: ChangePreprocessResult, mode: InputMode,
        perception_audit: dict[str, object], image_manifest: list[dict[str, str]],
        evidence_audit: dict[str, object], stage: EvidenceStage,
        first_pass: dict[str, object] | None = None,
        selected_proposals: list[Any] | None = None,
    ) -> dict[str, Any]:
        proposals = selected_proposals if stage == "adjudication" and selected_proposals is not None else preprocess.proposals
        payload: dict[str, Any] = {
            "decision_stage": stage, "question": sample.question, "task": sample.task,
            "coordinate_frame": "normalized_0_999_top_left", "input_mode": mode,
            "evidence_audit": evidence_audit, "temporal_roles": [ref.role for ref in sample.images],
            "image_manifest": image_manifest,
            "harmonization": {"status": preprocess.decision.status, "reason_codes": preprocess.decision.reason_codes,
                               "used_for_proposal": preprocess.decision.used_for_proposal},
            "registration": _registration_payload(preprocess), "perception": perception_audit,
            "proposals": [self._proposal_payload(item) for item in proposals],
            "semantic_support_note": "semantic_support is auxiliary attention evidence only; unavailable or uninformative support is neutral.",
            "empty_proposals_instruction": "Inspect raw full T1/T2; do not infer no change from an empty proposal list.",
        }
        if stage == "adjudication":
            payload["first_pass"] = first_pass or {}
            payload["adjudication_trigger"] = "positive_conflict" if any(
                str(item).startswith("POSITIVE_") for item in (first_pass or {}).get("review_route_reasons", [])
            ) else "negative_conflict"
            payload["adjudication_candidates"] = [self._proposal_payload(item) for item in proposals]
        return payload

    def _proposal_payload(self, item: Any) -> dict[str, object]:
        return {
            "proposal_id": item.proposal_id, "box": item.box, "score": round(item.score, 6),
            "source": item.source,
            "component_scores": {name: round(score, 6) for name, score in item.component_scores.items()},
            "semantic_support": _semantic_support_payload(
                item.semantic_transition, confidence_floor=self._settings.semantic.semantic_confidence_floor
            ),
            "effective_weights": item.effective_weights, "reliability": item.reliability,
            "registration_confidence": item.registration_confidence,
        }

    def _validate_adjudication(
        self, result: ChangeAdjudicationResult, selected_ids: list[str]
    ) -> list[str]:
        warnings: list[str] = []
        if result.agent_name != self.name:
            warnings.append("ADJUDICATION_INVALID_AGENT")
        ids = [item.proposal_id for item in result.candidate_reviews]
        if len(ids) != len(set(ids)):
            warnings.append("ADJUDICATION_DUPLICATE_PROPOSAL_ID")
        if set(ids) - set(selected_ids):
            warnings.append("ADJUDICATION_UNKNOWN_PROPOSAL_ID")
        if set(selected_ids) - set(ids):
            warnings.append("ADJUDICATION_MISSING_PROPOSAL_ID")
        positive = result.global_review.verdict == "persistent_change" or any(
            item.verdict == "persistent_change" for item in result.candidate_reviews
        )
        if positive and (result.answer.strip() == "No significant semantic change detected." or result.answer.strip().upper() in {"CHANGE", "NO_CHANGE", "YES", "NO"}):
            warnings.append("ADJUDICATION_POSITIVE_WITH_CANONICAL_NEGATIVE_ANSWER")
        nuisance = ("seasonal", "greener", "brown", "water-filled", "vehicle", "truck", "car", "brightness", "shadow")
        for item in [result.global_review, *result.candidate_reviews]:
            if item.verdict == "persistent_change" and any(token in item.reason.casefold() for token in nuisance):
                warnings.append("ADJUDICATION_PERSISTENT_WITH_NUISANCE_ONLY_REASON")
        return list(dict.fromkeys(warnings))

    def _merge_adjudication(
        self, result: ChangeAdjudicationResult, task: str, consistency_warnings: list[str]
    ) -> tuple[AgentResult, Literal["positive", "negative", "unresolved"]]:
        positive = not consistency_warnings and (result.global_review.verdict == "persistent_change" or any(
            item.verdict == "persistent_change" for item in result.candidate_reviews
        ))
        unresolved = result.global_review.verdict == "insufficient_visual_evidence" or any(
            item.verdict == "insufficient_visual_evidence" for item in result.candidate_reviews
        )
        if positive:
            outcome: Literal["positive", "negative", "unresolved"] = "positive"
            status = "completed"
            answer = result.answer
        elif unresolved:
            outcome = "unresolved"; status = "partial"
            answer = "Unable to confirm a persistent semantic change from the available evidence." if task == "change_caption" else result.answer
        else:
            outcome = "negative"; status = "completed"
            answer = "No significant semantic change detected." if task == "change_caption" else result.answer
        return AgentResult(agent_name=self.name, answer=answer, boxes=result.boxes, evidence=result.evidence,
                           evidence_items=result.evidence_items, geometry=dict(result.geometry), status=status), outcome

    def _select_proposals(self, proposals: list[Any], *, limit: int) -> list[Any]:
        ranked = sorted(proposals, key=_proposal_priority)
        selected: list[Any] = []
        if ranked:
            selected.append(ranked[0])
        edge = next((item for item in ranked if item not in selected and _is_edge_proposal(item, self._settings.evidence.edge_margin_ratio)), None)
        if edge is not None and len(selected) < limit:
            selected.append(edge)
        diverse = next((item for item in ranked if item not in selected and all(_box_iou(item.box, other.box) < 0.5 for other in selected)), None)
        if diverse is not None and len(selected) < limit:
            selected.append(diverse)
        for item in ranked:
            if item not in selected and len(selected) < limit:
                selected.append(item)
        return selected

    def _prepare_perception_and_publish(
        self,
        sample: UnifiedSample,
        context: AgentContext,
    ) -> ChangePreprocessResult:
        """Prepare once, run abstract perception, then publish final evidence."""

        settings = self._settings
        assert context.data_root is not None
        try:
            prepared = prepare_pair(
                sample,
                settings,
                context.artifact_dir,
                data_root=context.data_root,
            )
        except RegistrationError as error:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=error.reason_code,
            ) from error
        if not prepared.validation.valid:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="INVALID_CHANGE_PAIR",
            )
        try:
            perception = ChangePerceptionPipeline(
                self._semantic_client,
                settings,
                learned_change_client=self._learned_change_client,
            ).run(prepared)
        except ChangePerceptionError as error:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=error.reason_code,
            ) from error
        except RegistrationError as error:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=error.reason_code,
            ) from error
        return publish_change_proposals(
            prepared,
            score_map=perception.score_map,
            proposals=perception.proposals,
            artifact_dir=context.artifact_dir,
            settings=settings,
            component_maps=perception.component_maps,
            component_masks=perception.component_masks,
            diagnostics=perception.diagnostics,
        )

    def _build_evidence(
        self,
        sample: UnifiedSample,
        context: AgentContext,
        preprocess: ChangePreprocessResult,
        mode: InputMode,
        *,
        stage: EvidenceStage,
        selected_proposals: list[Any] | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[str],
        list[dict[str, str]],
        dict[str, object],
    ]:
        """Build the mode-specific ordered image content and manifest.
        构建模式专属的有序图像内容与清单。"""

        paths: list[tuple[str, Path, str]] = []
        raw = [
            ("raw_full_t1", self._resolve_raw(sample.images[0].path, context, sample.sample_id), "image"),
            ("raw_full_t2", self._resolve_raw(sample.images[1].path, context, sample.sample_id), "image"),
        ]
        harmonized: list[tuple[str, Path, str]] = []
        for key in ("harmonized_t1", "harmonized_t2"):
            relative = preprocess.artifact_files.get(key)
            if relative:
                harmonized.append((key, context.artifact_dir / relative, "artifact"))
        registered: list[tuple[str, Path, str]] = []
        relative_registered_t2 = preprocess.artifact_files.get("registered_t2")
        if relative_registered_t2:
            registered.append(
                (
                    "registered_t2",
                    context.artifact_dir / relative_registered_t2,
                    "artifact",
                )
            )
        # Select base comparison evidence independently from proposal
        # attention evidence. A rejected transform legitimately degrades a
        # dual path to the authoritative raw pair. / base comparison evidence
        # 与 proposal attention evidence 独立选择。transform rejected 时，
        # dual path 合法降级为 authoritative raw pair。
        # Raw global authority is unconditional.  Derived evidence can be
        # absent, rejected, or stale; the VLM must still see the source pair.
        raw_authority_required = True
        paths.extend(raw)
        if self._settings.evidence.attach_registered_global and mode != "raw_only":
            paths.extend(registered)
        if self._settings.evidence.attach_harmonized_global and mode != "raw_only":
            paths.extend(harmonized)

        proposal_evidence_count = 0
        overlay = preprocess.artifact_files.get("proposal_overlay")
        # Preserve the established empty-overlay dual-path contract while
        # attaching meaningful proposal evidence in every base mode.
        # 保留既有 dual-path 空 overlay 契约，同时在任意 base mode 下独立附加
        # 有意义的 proposal evidence。
        if self._settings.evidence.attach_proposal_overlay and overlay and preprocess.proposals:
            paths.append(
                ("proposal_overlay", context.artifact_dir / overlay, "artifact")
            )
            proposal_evidence_count += 1
        selected = selected_proposals if selected_proposals is not None else self._select_proposals(
            preprocess.proposals,
            limit=self._settings.evidence.initial_max_proposals if stage == "initial" else self._settings.evidence.adjudication_max_proposals,
        )
        skipped_incomplete: list[str] = []
        attached_ids: list[str] = []
        for proposal in selected:
            choices: dict[str, tuple[str, Path, str]] = {}
            for relative in proposal.evidence_filenames:
                role = _proposal_manifest_role(proposal, relative, preprocess)
                if role.endswith(":reference_t1_crop"):
                    choices["t1"] = (role, context.artifact_dir / relative, "artifact")
                elif role.endswith(":t2_registered_crop"):
                    choices["t2"] = (role, context.artifact_dir / relative, "artifact")
                elif role.endswith(":t2_raw_fallback_crop") and "t2" not in choices:
                    choices["t2"] = (role, context.artifact_dir / relative, "artifact")
            if set(choices) != {"t1", "t2"}:
                skipped_incomplete.append(proposal.proposal_id)
                continue
            paths.extend([choices["t1"], choices["t2"]])
            attached_ids.append(proposal.proposal_id)
            proposal_evidence_count += 2
        content: list[dict[str, Any]] = []
        hashes: list[str] = []
        manifest: list[dict[str, str]] = []
        content.append({"type": "text", "text": f"Decision stage: {stage}. Compare the next two authoritative raw images first."})
        for index, (role, path, kind) in enumerate(paths):
            data = self._read_artifact(path, sample.sample_id, kind=kind)
            try:
                mime = detect_image_mime(path)
            except (UnsupportedImageFormatError, OSError) as error:
                raise AgentExecutionError(
                    self.name,
                    sample.sample_id,
                    cause=f"image_format_error:{type(error).__name__}",
                ) from error
            content.append({"type": "text", "text": _evidence_label(role)})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(data, mime)}})
            hashes.append(hashlib.sha256(data).hexdigest())
            manifest.append({"index": str(index), "role": role})
        evidence_audit: dict[str, object] = {
            "raw_authority_required": raw_authority_required,
            "raw_authority_attached": bool(
                raw_authority_required
                and [item[0] for item in paths[:2]] == ["raw_full_t1", "raw_full_t2"]
            ),
            "harmonized_evidence_available": bool(harmonized),
            "harmonized_evidence_count": len(harmonized),
            "registered_evidence_available": bool(registered),
            "registered_evidence_count": len(registered),
            "proposal_evidence_attached": proposal_evidence_count > 0,
            "proposal_evidence_count": proposal_evidence_count,
            "proposal_count_total": len(preprocess.proposals),
            "proposal_count_attached": len(attached_ids),
            "image_manifest_roles": [item["role"] for item in manifest],
            "evidence_stage": stage,
            "image_count_attached": len(manifest),
            "initial_max_proposals": self._settings.evidence.initial_max_proposals,
            "adjudication_max_proposals": self._settings.evidence.adjudication_max_proposals,
            "selected_proposal_ids": attached_ids,
            "skipped_incomplete_proposal_ids": skipped_incomplete,
            "registered_global_attached": self._settings.evidence.attach_registered_global and bool(registered),
            "harmonized_global_attached": self._settings.evidence.attach_harmonized_global and bool(harmonized),
            "proposal_overlay_attached": bool(self._settings.evidence.attach_proposal_overlay and overlay and preprocess.proposals),
        }
        return content, hashes, manifest, evidence_audit

    def _resolve_raw(
        self,
        path: Path,
        context: AgentContext,
        sample_id: str,
    ) -> Path:
        """Resolve an ImageRef path against context.data_root with escape
        guards; returns the absolute candidate. Never reads relative to the
        current working directory when data_root is absent.
        按 context.data_root 防逃逸解析 ImageRef 路径；返回绝对候选路径。
        data_root 缺失时不相对当前工作目录静默读取。"""
        root = context.data_root.resolve()
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise AgentExecutionError(
                self.name, sample_id,
                cause=f"image path escapes data root: {path.as_posix()}",
            )
        if not candidate.is_file():
            raise AgentExecutionError(
                self.name, sample_id,
                cause=f"image file does not exist: {path.as_posix()}",
            )
        return candidate

    def _read_artifact(self, path: Path, sample_id: str, *, kind: str = "artifact") -> bytes:
        """Read one derived artifact with a stable failure code; never leaks
        the absolute machine path into the public message.
        以稳定错误码读取一条派生产物；公共消息绝不泄漏绝对机器路径。"""
        try:
            return path.read_bytes()
        except OSError as error:
            raise AgentExecutionError(
                self.name,
                sample_id,
                cause=f"{kind}_read_failed:{type(error).__name__}",
            ) from error


def _perception_audit(
    preprocess: ChangePreprocessResult,
    *,
    settings: AgentChangeSettings,
) -> dict[str, object]:
    """Flatten compact perception diagnostics into the request/trace contract."""

    diagnostics = preprocess.diagnostics
    fusion = diagnostics.get("fusion")
    fusion_data = fusion if isinstance(fusion, dict) else {}
    reliability = diagnostics.get("reliability")
    reliability_data = reliability if isinstance(reliability, dict) else {}
    feature = diagnostics.get("feature_residual")
    feature_data = feature if isinstance(feature, dict) else {}
    reason_codes = diagnostics.get("semantic_reason_codes")
    if not isinstance(reason_codes, list):
        reason = diagnostics.get("semantic_reason_code")
        reason_codes = [reason] if isinstance(reason, str) and reason else []
    return {
        "mode": diagnostics.get("perception_mode", "legacy"),
        "perception_mode": diagnostics.get("perception_mode", "legacy"),
        "perception_version": diagnostics.get("perception_version"),
        "semantic_enabled": bool(
            diagnostics.get("semantic_enabled", settings.semantic.enabled)
        ),
        "semantic_status": diagnostics.get("semantic_status"),
        "semantic_reason_codes": reason_codes,
        "semantic_model": diagnostics.get("semantic_model"),
        "semantic_client_version": diagnostics.get("semantic_client_version"),
        "semantic_model_revision": diagnostics.get("semantic_model_revision"),
        "semantic_weights_sha256": diagnostics.get("semantic_weights_sha256"),
        "feature_stage": diagnostics.get(
            "feature_stage", settings.semantic.feature_stage
        ),
        "feature_stages": diagnostics.get(
            "feature_stages", list(settings.semantic.feature_stages)
        ),
        "feature_stage_weights": diagnostics.get(
            "feature_stage_weights",
            {
                str(stage): float(weight)
                for stage, weight in settings.semantic.feature_stage_weights.items()
            },
        ),
        "multiscale_enabled": bool(
            diagnostics.get(
                "multiscale_enabled", len(settings.semantic.feature_stages) > 1
            )
        ),
        "tile_size": diagnostics.get("tile_size", settings.semantic.tile_size),
        "tile_overlap": diagnostics.get(
            "tile_overlap", settings.semantic.tile_overlap
        ),
        "local_match_radius": diagnostics.get(
            "local_match_radius", settings.semantic.local_match_radius
        ),
        "semantic_confidence_floor": diagnostics.get(
            "semantic_confidence_floor",
            settings.semantic.semantic_confidence_floor,
        ),
        "js_epsilon": diagnostics.get("js_epsilon", settings.semantic.js_epsilon),
        "min_pif_feature_cells": diagnostics.get(
            "min_pif_feature_cells",
            settings.semantic.min_pif_feature_cells,
        ),
        "feature_scale_epsilon": diagnostics.get(
            "feature_scale_epsilon",
            settings.semantic.feature_scale_epsilon,
        ),
        "feature_residual_version": diagnostics.get("feature_residual_version"),
        "semantic_difference_version": diagnostics.get(
            "semantic_difference_version"
        ),
        "fusion_version": diagnostics.get("fusion_version"),
        "fusion_effective_weights": fusion_data.get("effective_weights", {}),
        "effective_weights": fusion_data.get("effective_weights", {}),
        "available_components": fusion_data.get("available_components", []),
        "missing_components": fusion_data.get("missing_components", []),
        "reliability": reliability_data.get("reliability", {}),
        "reliability_raw": reliability_data.get("raw", {}),
        "registration_confidence": fusion_data.get("registration_confidence"),
        "semantic_transition_note": diagnostics.get("semantic_transition_note"),
        "threshold_mode": fusion_data.get("threshold_mode"),
        "threshold_value": fusion_data.get("threshold"),
        "threshold_floor": diagnostics.get(
            "threshold_floor", settings.proposals.threshold_floor
        ),
        "pif_threshold_k": diagnostics.get(
            "pif_threshold_k", settings.proposals.pif_threshold_k
        ),
        "pif_fallback_quantile": diagnostics.get(
            "pif_fallback_quantile",
            settings.proposals.pif_fallback_quantile,
        ),
        "pif_valid": bool(diagnostics.get("pif_valid", False)),
        "pif_used_for_feature_alignment": bool(
            diagnostics.get("pif_used_for_feature_alignment", False)
        ),
        "pif_used_for_threshold": bool(
            diagnostics.get("pif_used_for_threshold", False)
        ),
        "pif_feature_cells": feature_data.get("pif_feature_cells"),
        "pif_threshold_fallback_used": fusion_data.get(
            "pif_threshold_fallback_used", False
        ),
        "proposal_count": len(preprocess.proposals),
        "proposal_source": diagnostics.get("proposal_source", "difference_map_v1"),
        "proposal_metadata": [
            {
                "proposal_id": proposal.proposal_id,
                "box": proposal.box,
                "score": round(proposal.score, 6),
                "component_scores": {
                    name: round(score, 6)
                    for name, score in proposal.component_scores.items()
                },
                "semantic_transition": (
                    proposal.semantic_transition.model_dump(mode="json")
                    if proposal.semantic_transition is not None
                    else None
                ),
                "effective_weights": proposal.effective_weights,
                "reliability": proposal.reliability,
            }
            for proposal in preprocess.proposals
        ],
        "learned_change": diagnostics.get(
            "learned_change",
            {
                "enabled": settings.learned_change.enabled,
                "status": "unavailable"
                if settings.learned_change.enabled
                else "disabled",
                "available": False,
                "reason_codes": [],
                "fusion_weight": settings.learned_change.fusion_weight,
            },
        ),
        "training_capture": {
            "enabled": False,
            "save_dense_features": False,
            "dense_features_saved": False,
        },
        "score_maps": diagnostics.get("score_maps", {}),
    }


def _proposal_priority(proposal: Any) -> tuple[float, str]:
    """Rank attachment candidates by score multiplied by evidence reliability."""

    reliability = proposal.reliability or {}
    values = [
        float(value)
        for value in reliability.values()
    ]
    confidence = sum(values) / len(values) if values else 1.0
    return (-float(proposal.score) * confidence, str(proposal.proposal_id))


def _is_edge_proposal(proposal: Any, margin_ratio: float) -> bool:
    margin = round(999 * margin_ratio)
    x1, y1, x2, y2 = proposal.box
    return x1 <= margin or y1 <= margin or x2 >= 999 - margin or y2 >= 999 - margin


def _box_iou(first: list[int], second: list[int]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / union if union else 0.0


def _normalize_positive_caption(result: AgentResult) -> AgentResult:
    answer = result.answer.strip()
    lowered = answer.casefold()
    for prefix in ("persistent change detected:", "change detected:", "significant change detected:"):
        if lowered.startswith(prefix):
            caption = answer[len(prefix):].strip()
            if caption:
                geometry = dict(result.geometry)
                values = list(geometry.get("change_input_normalizations") or [])
                values.append("generic_positive_prefix_stripped")
                geometry["change_input_normalizations"] = list(dict.fromkeys(values))
                return result.model_copy(update={"answer": caption, "geometry": geometry})
    return result


def _semantic_support_payload(
    transition: SemanticTransition | None, *, confidence_floor: float
) -> dict[str, object]:
    """Make auxiliary class evidence neutral unless it is a confident transition."""
    if transition is None:
        return {"status": "unavailable"}
    if transition.from_class.casefold() == transition.to_class.casefold():
        return {"status": "uninformative", "reason": "same_auxiliary_class"}
    if transition.from_confidence < confidence_floor or transition.to_confidence < confidence_floor:
        return {"status": "uninformative", "reason": "low_auxiliary_class_confidence"}
    return {
        "status": "informative", "from_class": transition.from_class, "to_class": transition.to_class,
        "from_confidence": transition.from_confidence, "to_confidence": transition.to_confidence,
        "transition_confidence": transition.transition_confidence, "support_ratio": transition.support_ratio,
    }


def _evidence_label(role: str) -> str:
    if role == "raw_full_t1":
        return "AUTHORITATIVE RAW T1 - earlier full scene"
    if role == "raw_full_t2":
        return "AUTHORITATIVE RAW T2 - later full scene"
    if role == "proposal_overlay":
        return "AUXILIARY PROPOSAL OVERLAY - attention guidance only; not proof of change"
    if ":" in role:
        proposal_id, crop_role = role.split(":", 1)
        if crop_role == "reference_t1_crop":
            return f"CANDIDATE {proposal_id} - T1 reference crop - inspect the same location"
        if crop_role in {"t2_registered_crop", "t2_raw_fallback_crop"}:
            return f"CANDIDATE {proposal_id} - T2 comparison crop - inspect the same location"
    return f"AUXILIARY {role} - attention evidence only"


def _proposal_manifest_role(
    proposal: Any,
    relative: str,
    preprocess: ChangePreprocessResult,
) -> str:
    """Map artifact filenames to explicit Level-C evidence roles."""

    stem = Path(relative).stem
    prefix = f"{proposal.proposal_id}_"
    suffix = stem[len(prefix) :] if stem.startswith(prefix) else stem
    if suffix == "raw_t1":
        role = "reference_t1_crop"
    elif suffix == "registered_t2":
        role = "t2_registered_crop"
    elif suffix == "raw_t2":
        report = preprocess.registration
        identity_aligned = bool(
            report is not None
            and report.decision.used_for_comparison
            and report.decision.model == "identity"
        )
        role = "t2_registered_crop" if identity_aligned else "t2_raw_fallback_crop"
    elif suffix == "mask_overlay":
        role = "mask_overlay"
    elif suffix == "mask":
        role = "mask_attention_prior"
    elif suffix == "harmonized_t1":
        role = "harmonized_t1_crop"
    elif suffix == "harmonized_t2":
        role = "harmonized_t2_crop"
    else:
        role = suffix
    return f"{proposal.proposal_id}:{role}"


def _registration_payload(preprocess: ChangePreprocessResult) -> dict[str, object]:
    """Flatten registration decisions into stable payload/trace fields."""

    report = preprocess.registration
    if report is None:
        return {
            "registration_version": None,
            "registration_status": None,
            "registration_model": None,
            "registration_reason_codes": [],
            "registration_match_count": None,
            "registration_inlier_count": None,
            "registration_inlier_ratio": None,
            "registration_median_reprojection_error": None,
            "registration_overlap_ratio": None,
            "registration_used_for_comparison": False,
            "quality": {
                "inlier_ratio": None,
                "median_reprojection_error": None,
                "overlap_ratio": None,
            },
        }
    metrics = report.metrics
    quality = {
        "inlier_ratio": metrics.inlier_ratio if metrics else None,
        "median_reprojection_error": (
            metrics.median_reprojection_error if metrics else None
        ),
        "overlap_ratio": metrics.overlap_ratio if metrics else None,
    }
    return {
        "registration_version": report.decision.version,
        "registration_status": report.decision.status,
        "registration_model": report.decision.model,
        "registration_reason_codes": list(report.decision.reason_codes),
        "registration_match_count": metrics.match_count if metrics else None,
        "registration_inlier_count": metrics.inlier_count if metrics else None,
        "registration_inlier_ratio": metrics.inlier_ratio if metrics else None,
        "registration_median_reprojection_error": (
            metrics.median_reprojection_error if metrics else None
        ),
        "registration_overlap_ratio": metrics.overlap_ratio if metrics else None,
        "registration_used_for_comparison": report.decision.used_for_comparison,
        "quality": quality,
    }
