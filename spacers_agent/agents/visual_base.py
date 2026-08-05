"""Shared visual agent base — structured Qwen request dispatch.
共享视觉 Agent 基类 — 结构化 Qwen 请求分发。

Shared by concrete visual Agents. All agent-specific behaviour
is injected via hook methods rather than subclass overrides.
由具体视觉 Agent 共用。所有特定 Agent 行为通过 hook 方法注入，
而非子类覆盖。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash, image_to_data_url
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.schemas import AgentResult, UnifiedSample
from spacers_agent.vqa_geometry import vrsbench_answer_vocabulary, vrsbench_question_subtype


@dataclass(frozen=True)
class PromptSelection:
    """Prompt text and version chosen for one sample. / 为单条样本选择的 Prompt 文本与版本。"""

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
        default_prompt: PromptAsset,
    ) -> None:
        self._client = client
        self._model = model
        self._agent_name = agent_name
        self._default_prompt = default_prompt

    # ── hooks / hook ─────────────────────────────────────────────────────

    def select_prompt(self, sample: UnifiedSample) -> PromptSelection:
        """Return the prompt for this sample. Override for subtype selection.
        返回此样本的 Prompt。子类可覆盖以实现子类型选择。
        """
        return PromptSelection(text=self._default_prompt.text, version=self._default_prompt.version)

    def build_user_payload(self, sample: UnifiedSample) -> dict[str, Any]:
        """Build the JSON user payload sent alongside images. / 构造与图像一同发送的 JSON 用户载荷。"""
        payload: dict[str, Any] = {
            "question": sample.question,
            "dataset_question_type": sample.metadata.get("question_type"),
            "coordinate_frame": "normalized_0_999_top_left",
        }
        if sample.dataset == "VRSBench":
            subtype = vrsbench_question_subtype(
                sample.question,
                str(sample.metadata.get("question_type", "")),
            )
            answer_vocabulary = (
                [] if subtype == "grid_position" else vrsbench_answer_vocabulary(subtype, sample.question)
            )
            payload.update({"semantic_subtype": subtype, "answer_vocabulary": answer_vocabulary})
        return payload

    async def postprocess(self, sample: UnifiedSample, result: AgentResult) -> AgentResult:
        """Post-process the raw model result. Override for geometry fixes.
        对原始模型结果进行后处理。子类可覆盖以应用几何修正。
        """
        return result

    # ── core execution / 核心执行 ────────────────────────────────────────

    async def run(self, sample: UnifiedSample, *, artifact_dir: Path) -> AgentResult:
        """Execute the visual agent pipeline. / 执行视觉 Agent 管线。"""

        # Read images and build content / 读取图像并构建内容
        content: list[dict[str, Any]] = []
        image_hashes: list[str] = []
        for image_ref in sample.images:
            data = image_ref.path.read_bytes()
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(data, "image/png")}})
            image_hashes.append(hashlib.sha256(data).hexdigest())

        user_payload = self.build_user_payload(sample)
        content.append({"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)})

        # Select prompt / 选择 Prompt
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

        result = await self._client.complete_json(
            messages=messages,
            response_model=AgentResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:{self._agent_name}",
                request_hash=request_hash,
                prompt_version=prompt_sel.version,
                sample_id=sample.sample_id,
                artifact_dir=artifact_dir / self._agent_name,
            ),
        )

        return await self.postprocess(sample, result)
