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

from agents.base import AgentContext, AgentExecution
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.schema import AgentName, AgentResult
from data.schema import UnifiedSample
from models.base import RequestMeta, VisionLanguageClient, build_request_hash
from models.images import guess_image_mime, image_to_data_url


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
        model: str,
        *,
        agent_name: AgentName,
        supported_tasks: frozenset[str],
        prompt: PromptBinding,
        model_revision: str | None = None,
    ) -> None:
        if not supported_tasks:
            raise ValueError("supported_tasks must not be empty")
        for task in supported_tasks:
            if not isinstance(task, str) or not task:
                raise ValueError("supported_tasks must contain non-empty strings")
        self.name = agent_name
        self.supported_tasks = frozenset(supported_tasks)
        self._client = client
        self._model = model
        self._prompt = prompt
        self._model_revision = model_revision

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

        # Read images and build content / 读取图像并构建内容
        content: list[dict[str, Any]] = []
        image_hashes: list[str] = []
        for image_ref in sample.images:
            candidate_path, data = self._read_image(image_ref.path, context)
            mime = guess_image_mime(candidate_path)
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
            "strings, and set status to 'completed'."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": structured_prompt},
            {"role": "user", "content": content},
        ]
        request_hash = self._build_hash(messages, prompt_sel.version, image_hashes)

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
                "model": self._model,
            },
        )

    def _build_hash(
        self,
        messages: list[dict[str, Any]],
        prompt_version: str,
        image_hashes: list[str],
    ) -> str:
        """Build a cache hash covering the full inference semantics. The
        generation config comes from the client's own stable attribute so the
        hash and the actual call cannot drift.
        构建覆盖完整推理语义的缓存哈希。生成配置取自客户端自身的稳定属性，
        使哈希与实际调用不会漂移。"""
        from models.qwen_transformers import QWEN_CLIENT_VERSION

        generation = getattr(self._client, "cache_generation_config", None)
        if generation is None:
            generation = {"temperature": 0.0, "do_sample": False, "max_tokens": 0}
        return build_request_hash(
            model=self._model,
            generation=generation,
            prompt_version=prompt_version,
            messages=messages,
            image_sha256="|".join(image_hashes),
            response_schema=AgentResult.model_json_schema(),
            client_version=QWEN_CLIENT_VERSION,
            model_revision=self._model_revision,
        )

    @staticmethod
    def _read_image(path: Path, context: AgentContext) -> tuple[Path, bytes]:
        """Resolve an ImageRef path against context.data_root, guarding against
        escape; returns (candidate_path, bytes). Never reads relative to the
        current working directory when data_root is absent.
        按 context.data_root 解析 ImageRef 路径并防逃逸；返回（路径、字节）。
        data_root 缺失时不相对当前工作目录静默读取。"""
        if context.data_root is None:
            raise AgentExecutionError(
                "visual_base", "image",
                cause="data_root is required to resolve relative ImageRef paths",
            )
        root = context.data_root.resolve()
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise AgentExecutionError(
                "visual_base", "image",
                cause=f"image path escapes data root: {path.as_posix()}",
            )
        if not candidate.is_file():
            raise AgentExecutionError(
                "visual_base", "image",
                cause=f"image file does not exist: {path.as_posix()}",
            )
        return candidate, candidate.read_bytes()
