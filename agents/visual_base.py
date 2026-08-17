"""Dataset-neutral visual agent base — structured Qwen request dispatch.

与数据集无关的视觉 Agent 基类 — 结构化 Qwen 请求分发。由具体视觉 Agent
共用：统一图片输入（data URL + 内容哈希 + 正确 MIME）、中性 PromptBinding、
完整 request hash、Qwen budget 消费与 AgentExecution 输出。本模块不按
数据集名分支、不读取 Prompt 文件、不做评测或几何改答案。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.schema import AgentName, AgentResult
from data.schema import UnifiedSample
from models.base import (
    ModelCacheIdentity,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
)
from models.images import (
    UnsupportedImageFormatError,
    detect_image_mime,
    image_to_data_url,
    read_normalized_image,
)


@dataclass(frozen=True)
class PromptBinding:
    """Neutral prompt text and version selected for one sample.
    为单条样本选择的中性 Prompt 文本与版本。"""

    text: str
    version: str


class VisualAgentBase:
    """Generic visual agent base — constructs Qwen request, calls complete_json.
    通用视觉 Agent 基类 — 构造 Qwen 请求、调用 complete_json。

    Subclasses inject behaviour via hooks; do NOT override ``run``.
    子类通过 hook 注入行为；不要覆盖 ``run``。
    """

    name: AgentName
    supported_tasks: frozenset[str]

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        agent_name: AgentName,
        supported_tasks: frozenset[str],
        prompt: PromptBinding,
    ) -> None:
        if not supported_tasks:
            raise ValueError("supported_tasks must not be empty")
        for task in supported_tasks:
            if not isinstance(task, str) or not task:
                raise ValueError("supported_tasks must contain non-empty strings")
        self.name = agent_name
        self.supported_tasks = frozenset(supported_tasks)
        self._client = client
        self._prompt = prompt

    # ── hooks / hook ─────────────────────────────────────────────────────

    def select_prompt(self, sample: UnifiedSample) -> PromptBinding:
        """Return the prompt for this sample. Override for subtype selection.
        返回此样本的 Prompt。子类可覆盖以实现子类型选择。"""
        return self._prompt

    def build_user_payload(self, sample: UnifiedSample) -> dict[str, Any]:
        """Build the JSON user payload sent alongside images. Never leaks
        ground truth and never branches on the dataset name.
        构造与图像一同发送的 JSON 用户载荷。绝不泄漏 ground truth，
        也绝不按数据集名分支。"""
        payload: dict[str, Any] = {
            "question": sample.question,
            "task": sample.task,
            "coordinate_frame": "normalized_0_999_top_left",
            "box_format": "integer_xyxy_json",
            "answer_constraints": {},
        }
        if sample.normalization is not None:
            payload["semantic_subtype"] = sample.normalization.semantic_subtype
            payload["answer_constraints"] = sample.normalization.answer_constraints
        return payload

    async def postprocess(self, sample: UnifiedSample, result: AgentResult) -> AgentResult:
        """Post-process the raw model result. Override for geometry fixes.
        对原始模型结果进行后处理。子类可覆盖以应用几何修正。"""
        return result

    def result_filename(self, sample: UnifiedSample) -> str:
        """Default result filename; subclasses may override.
        默认结果文件名；子类可覆盖。"""
        return "agent_result.json"

    # ── core execution / 核心执行 ────────────────────────────────────────

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        """Execute the visual agent pipeline and return an AgentExecution.
        执行视觉 Agent 管线并返回 AgentExecution。"""
        if sample.task not in self.supported_tasks:
            raise AgentTaskMismatchError(self.name, sample.task, supported=self.supported_tasks)

        # The client's own cache identity is the only hash source; it must be
        # a real ModelCacheIdentity (never a duck-typed stand-in) so path-like
        # models cannot bypass validation. Fail before reading images,
        # consuming budget, or calling the model.
        # 客户端自身的缓存身份是唯一哈希来源；它必须是真正的
        # ModelCacheIdentity（不接受任意鸭子类型替代品），使 path-like 模型
        # 无法绕过校验。在读图、消费 budget、调用模型之前显式失败。
        identity = getattr(self._client, "cache_identity", None)
        if not isinstance(identity, ModelCacheIdentity):
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="model client returned an invalid cache_identity",
            )

        # Read images and build content / 读取图像并构建内容
        content: list[dict[str, Any]] = []
        image_hashes: list[str] = []
        for image_ref in sample.images:
            candidate_path, data = self._read_image(
                image_ref.path, context, sample_id=sample.sample_id
            )
            try:
                mime = detect_image_mime(candidate_path)
            except (UnsupportedImageFormatError, OSError) as error:
                raise AgentExecutionError(
                    self.name,
                    sample.sample_id,
                    cause=f"image_format_error:{type(error).__name__}",
                ) from error
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(data, mime)}}
            )
            image_hashes.append(hashlib.sha256(data).hexdigest())

        user_payload = self.build_user_payload(sample)
        content.append({"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)})

        prompt_sel = self.select_prompt(sample)
        structured_prompt = (
            prompt_sel.text
            + f"\n\nReturn valid JSON only. Set agent_name to {self.name!r}; "
            "put the concise final answer in answer, retain relevant labeled boxes or points "
            "in evidence_items, copy evidence boxes into boxes, use concise factual evidence "
            "strings, and set status to 'completed'. When boxes are returned, use integer "
            "0..999 whole-image xyxy coordinates in JSON, with each box as a flat "
            "[x1,y1,x2,y2] array."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": structured_prompt},
            {"role": "user", "content": content},
        ]
        request_hash = self._build_hash(messages, prompt_sel.version, image_hashes, identity)

        # Consume exactly one Qwen budget entry before the call.
        # 调用前恰好消费一次 Qwen budget。
        context.call_budget.reserve_qwen()

        result = await self._client.complete_json(
            messages=messages,
            response_model=AgentResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:{self.name}",
                request_hash=request_hash,
                prompt_version=prompt_sel.version,
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

        result = await self.postprocess(sample, result)

        return AgentExecution(
            agent_name=self.name,
            payload=result,
            result_filename=self.result_filename(sample),
            trace={
                "prompt_version": prompt_sel.version,
                "request_hash": request_hash,
                "image_sha256": image_hashes,
                "model": identity.model,
            },
        )

    def _build_hash(
        self,
        messages: list[dict[str, Any]],
        prompt_version: str,
        image_hashes: list[str],
        identity: Any,
    ) -> str:
        """Build a cache hash covering the full inference semantics. Model
        name, generation config, client version, and revision all come from
        the client's own cache identity so the hash and the actual call can
        never drift.
        构建覆盖完整推理语义的缓存哈希。模型名、生成配置、客户端版本与
        revision 全部取自客户端自身的缓存身份，使哈希与实际调用不会漂移。"""
        return build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=prompt_version,
            messages=messages,
            image_sha256="|".join(image_hashes),
            response_schema=AgentResult.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )

    def _read_evidence_images(
        self,
        sample: UnifiedSample,
        context: AgentContext,
    ) -> dict[str, Image.Image]:
        """Decode every sample image through the escape-guarded seam, keyed by
        image_id; I/O failures map to stable codes and never leak machine
        paths. Shared by the object-evidence protocol owners (14A2 §4.3).
        通过防逃逸 seam 解码每条样本图像，按 image_id 索引；I/O 失败映射为
        稳定 code，绝不泄漏机器路径。由对象证据协议 owner 共享（14A2 §4.3）。"""
        images: dict[str, Image.Image] = {}
        for image_ref in sample.images:
            candidate_path, _ = self._read_image(
                image_ref.path, context, sample_id=sample.sample_id
            )
            try:
                images[image_ref.image_id] = read_normalized_image(candidate_path)
            except (OSError, ValueError) as exc:
                raise AgentExecutionError(
                    self.name,
                    sample.sample_id,
                    cause=f"image_decode_failed:{type(exc).__name__}",
                ) from exc
        return images

    def _read_image(
        self,
        path: Path,
        context: AgentContext,
        *,
        sample_id: str,
    ) -> tuple[Path, bytes]:
        """Resolve an ImageRef path against context.data_root, guarding against
        escape; returns (candidate_path, bytes). Never reads relative to the
        current working directory when data_root is absent. I/O failures are
        converted to AgentExecutionError without leaking machine paths.
        按 context.data_root 解析 ImageRef 路径并防逃逸；返回（路径、字节）。
        data_root 缺失时不相对当前工作目录静默读取。I/O 失败统一转换为
        AgentExecutionError，且不泄漏机器路径。"""
        if context.data_root is None:
            raise AgentExecutionError(
                self.name, sample_id,
                cause="data_root is required to resolve relative ImageRef paths",
            )
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
        try:
            data = candidate.read_bytes()
        except OSError as error:
            raise AgentExecutionError(
                self.name,
                sample_id,
                cause=f"image_read_failed:{type(error).__name__}",
            ) from error
        return candidate, data
