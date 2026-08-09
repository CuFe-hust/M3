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
from agents.change.reviewer import review_result
from agents.change.schema import ChangePreprocessResult
from agents.change.settings import AgentChangeSettings
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.schema import AgentName, AgentResult
from agents.visual_base import PromptBinding
from data.schema import UnifiedSample
from models.base import (
    DenseSemanticClient,
    ModelCacheIdentity,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
)
from models.images import UnsupportedImageFormatError, detect_image_mime, image_to_data_url

# Neutral dual-path prompt text extended with V2 auxiliary-evidence authority.
# The repository prompt file is intentionally not read by agents.
# 中性双路径提示词加入 V2 辅助证据权威边界；Agent 不读取仓库提示词文件。
_DEFAULT_PROMPT_TEXT = (
    "You analyze bi-temporal remote-sensing imagery with auditable dual-path "
    "evidence. The raw T1/T2 images and raw candidate crops are the "
    "authoritative source for object identity, fine texture, and small "
    "targets. Harmonized images are comparison aids used to suppress sensor, "
    "exposure, color, and resolution-domain differences; they are not a "
    "replacement for raw high-resolution facts. The proposal overlay and "
    "proposal boxes are attention hints, not proof of real change. SegFormer "
    "labels and features are attention hints only, and proposal masks are not "
    "proof. Semantic conclusions must be supported by authoritative raw "
    "T1/T2 evidence. "
    "Describe only changes visibly supported by the supplied full images or "
    "candidate crops. Do not classify brightness, color, shadow, seasonal, or "
    "sharpness differences as land-cover or object changes by themselves. "
    "When evidence is insufficient, answer `uncertain` rather than inventing "
    "a change. If no proposal is present, still inspect the raw full pair and "
    "distinguish `no_visible_change` from `insufficient_evidence`. "
    "For change_caption, give a concise change description. For change_qa, "
    "answer the question directly. Preserve relevant proposal-aligned boxes "
    "in evidence_items and record proposal identifiers and whether raw or "
    "harmonized evidence was used in geometry."
)

_DEFAULT_PROMPT_VERSION = "change_dual_path_v2"

InputMode = Literal["raw_only", "harmonized_only", "dual_path"]


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
        prompt: PromptBinding | None = None,
        settings: AgentChangeSettings | None = None,
    ) -> None:
        self._client = client
        self._semantic_client = semantic_client
        self._prompt = prompt or PromptBinding(
            text=_DEFAULT_PROMPT_TEXT, version=_DEFAULT_PROMPT_VERSION
        )
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
        content, image_hashes, image_manifest = self._build_evidence(
            sample, context, preprocess, mode
        )
        payload: dict[str, Any] = {
            "question": sample.question,
            "task": sample.task,
            "coordinate_frame": "normalized_0_999_top_left",
            "input_mode": mode,
            "temporal_roles": [ref.role for ref in sample.images],
            "image_manifest": image_manifest,
            "harmonization": {
                "status": preprocess.decision.status,
                "reason_codes": preprocess.decision.reason_codes,
                "used_for_proposal": preprocess.decision.used_for_proposal,
            },
            "perception": {
                "perception_version": preprocess.diagnostics.get(
                    "perception_version"
                ),
                "semantic_status": preprocess.diagnostics.get("semantic_status"),
                "semantic_reason_code": preprocess.diagnostics.get(
                    "semantic_reason_code"
                ),
                "segformer_model": preprocess.diagnostics.get("segformer_model"),
                "feature_residual_version": preprocess.diagnostics.get(
                    "feature_residual_version"
                ),
                "semantic_difference_version": preprocess.diagnostics.get(
                    "semantic_difference_version"
                ),
                "fusion_version": preprocess.diagnostics.get("fusion_version"),
            },
            "proposals": [
                {
                    "proposal_id": item.proposal_id,
                    "box": item.box,
                    "score": round(item.score, 6),
                    "source": item.source,
                    "component_scores": {
                        name: round(score, 6)
                        for name, score in item.component_scores.items()
                    },
                }
                for item in preprocess.proposals
            ],
            "empty_proposals_instruction": (
                "Inspect raw full T1/T2; do not infer no change from an empty proposal list."
            ),
        }
        content.append({"type": "text", "text": json.dumps(payload, ensure_ascii=False)})
        structured_prompt = (
            self._prompt.text
            + "\n\nReturn valid JSON only. Set agent_name to 'change_agent'; "
            "put the concise final answer in answer, retain relevant labeled boxes "
            "in evidence_items and boxes, use concise factual evidence strings, "
            "record proposal ids and evidence path types in geometry, and set "
            "status to 'completed'."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": structured_prompt},
            {"role": "user", "content": content},
        ]
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=self._prompt.version,
            messages=messages,
            image_sha256="|".join(image_hashes),
            response_schema=AgentResult.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )

        # Consume exactly one Qwen budget entry before the call.
        # 调用前恰好消费一次 Qwen budget。
        context.call_budget.reserve_qwen()
        result = await self._client.complete_json(
            messages=messages,
            response_model=AgentResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:{self.name}",
                request_hash=request_hash,
                prompt_version=self._prompt.version,
                sample_id=sample.sample_id,
                artifact_dir=context.artifact_dir / self.name,
            ),
        )
        if result.agent_name != self.name:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=f"model returned agent_name {result.agent_name!r}",
            )

        reviewed, review_warnings = review_result(result, preprocess.proposals, settings.review)
        metrics = preprocess.decision.metrics
        trace = {
            "prompt_version": self._prompt.version,
            "request_hash": request_hash,
            "image_sha256": image_hashes,
            "model": identity.model,
            "image_roles": [ref.role for ref in sample.images],
            "input_mode": mode,
            "harmonization_version": preprocess.decision.version,
            "harmonization_status": preprocess.decision.status,
            "harmonization_reason_codes": preprocess.decision.reason_codes,
            "pif_ratio": metrics.pif_ratio if metrics else None,
            "mad_pif_before": metrics.mad_pif_before if metrics else None,
            "mad_pif_after": metrics.mad_pif_after if metrics else None,
            "raw_fallback_used": not preprocess.decision.used_for_proposal,
            "sharpness_adjustment_used": bool(
                preprocess.transform_summary.get("sharpness_adjustment_used", False)
            ),
            "proposal_count": len(preprocess.proposals),
            "proposal_source": preprocess.diagnostics.get(
                "proposal_source", "difference_map_v1"
            ),
            "perception_version": preprocess.diagnostics.get("perception_version"),
            "semantic_status": preprocess.diagnostics.get("semantic_status"),
            "semantic_reason_code": preprocess.diagnostics.get(
                "semantic_reason_code"
            ),
            "segformer_model": preprocess.diagnostics.get("segformer_model"),
            "feature_residual_version": preprocess.diagnostics.get(
                "feature_residual_version"
            ),
            "semantic_difference_version": preprocess.diagnostics.get(
                "semantic_difference_version"
            ),
            "fusion_version": preprocess.diagnostics.get("fusion_version"),
            "review_used": settings.review.enabled,
            "review_warnings": review_warnings,
            "preprocess_artifacts": preprocess.artifact_files,
        }
        return AgentExecution(
            agent_name=self.name,
            payload=reviewed,
            result_filename="agent_result.json",
            trace=trace,
        )

    def _prepare_perception_and_publish(
        self,
        sample: UnifiedSample,
        context: AgentContext,
    ) -> ChangePreprocessResult:
        """Prepare once, run abstract perception, then publish final evidence."""

        settings = self._settings
        assert context.data_root is not None
        prepared = prepare_pair(
            sample,
            settings,
            context.artifact_dir,
            data_root=context.data_root,
        )
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
            ).run(prepared)
        except ChangePerceptionError as error:
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
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
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
        if mode == "raw_only" or not harmonized:
            paths.extend(raw)
        elif mode == "harmonized_only":
            paths.extend(harmonized)
        else:  # dual_path / 双路径
            paths.extend(raw)
            paths.extend(harmonized)
            overlay = preprocess.artifact_files.get("proposal_overlay")
            if overlay:
                paths.append(("proposal_overlay", context.artifact_dir / overlay, "artifact"))
            for proposal in preprocess.proposals[: self._settings.proposals.max_proposals]:
                for relative in proposal.evidence_filenames:
                    paths.append(
                        (
                            f"{proposal.proposal_id}:{Path(relative).stem}",
                            context.artifact_dir / relative,
                            "artifact",
                        )
                    )
        content: list[dict[str, Any]] = []
        hashes: list[str] = []
        manifest: list[dict[str, str]] = []
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
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(data, mime)}})
            hashes.append(hashlib.sha256(data).hexdigest())
            manifest.append({"index": str(index), "role": role})
        return content, hashes, manifest

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
