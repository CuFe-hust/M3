"""Dataset-neutral visual agent base — structured Qwen request dispatch.

与数据集无关的视觉 Agent 基类 — 结构化 Qwen 请求分发。由具体视觉 Agent
共用：统一图片输入（data URL + 内容哈希）、中性 PromptBinding、request
hash、Qwen budget 消费与结构化输出（AgentResult）。本模块不按数据集名
分支、不读取 Prompt 文件、不做评测或几何改答案。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.base import AgentContext
from agents.schema import AgentResult
from data.schema import UnifiedSample
from models.base import RequestMeta, VisionLanguageClient, build_request_hash
from models.images import image_to_data_url


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

    def __init__(
        self,
        client: VisionLanguageClient,
        model: str,
        *,
        agent_name: str,
        prompt: PromptBinding,
    ) -> None:
        self._client = client
        self._model = model
        self._agent_name = agent_name
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

    # ── core execution / 核心执行 ────────────────────────────────────────

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentResult:
        """Execute the visual agent pipeline. / 执行视觉 Agent 管线。"""

        # Read images and build content / 读取图像并构建内容
        content: list[dict[str, Any]] = []
        image_hashes: list[str] = []
        for image_ref in sample.images:
            data = self._read_image_bytes(image_ref.path, context)
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(data, "image/png")}}
            )
            image_hashes.append(hashlib.sha256(data).hexdigest())

        user_payload = self.build_user_payload(sample)
        content.append({"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)})

        prompt_sel = self.select_prompt(sample)
        structured_prompt = (
            prompt_sel.text
            + f"\n\nReturn valid JSON only. Set agent_name to {self._agent_name!r}; "
            "put the concise final answer in answer, retain relevant labeled boxes or points "
            "in evidence_items, copy evidence boxes into boxes, use concise factual evidence "
            "strings, and set status to 'completed'."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": structured_prompt},
            {"role": "user", "content": content},
        ]
        request_hash = build_request_hash(
            model=self._model,
            generation={"temperature": 0.0},
            prompt_version=prompt_sel.version,
            messages=messages,
            image_sha256="|".join(image_hashes),
        )

        # Consume exactly one Qwen budget entry before the call.
        # 调用前恰好消费一次 Qwen budget。
        context.call_budget.reserve_qwen()

        result = await self._client.complete_json(
            messages=messages,
            response_model=AgentResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:{self._agent_name}",
                request_hash=request_hash,
                prompt_version=prompt_sel.version,
                sample_id=sample.sample_id,
                artifact_dir=context.artifact_dir / self._agent_name,
            ),
        )

        return await self.postprocess(sample, result)

    @staticmethod
    def _read_image_bytes(path: Path, context: AgentContext) -> bytes:
        """Resolve an ImageRef path against the injected data root when
        available; otherwise read the path directly.
        有注入 data root 时按它解析 ImageRef 路径；否则直接读取路径。"""
        data_root = context.request_context.get("data_root")
        candidate = Path(data_root) / path if data_root is not None else Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"image file does not exist: {candidate}")
        return candidate.read_bytes()
