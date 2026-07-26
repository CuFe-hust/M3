"""Structured counting-target parsing for non-VRSBench count tasks.
用于非 VRSBench 计数任务的结构化计数目标解析。
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash
from spacers_agent.schemas import CountTargetSpec


class CountTargetParser:
    """Parse one stable target with the frozen text-only Qwen contract.
    使用冻结的纯文本 Qwen 契约解析一个稳定目标。
    """

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        self.client = client
        self.prompt = prompt
        self.model = model

    async def parse(
        self,
        question: str,
        *,
        sample_id: str,
        artifact_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> CountTargetSpec:
        """Issue the same request used by the frozen legacy dataset path.
        发出与冻结旧数据集路径相同的请求。
        """

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": question},
        ]
        request_hash = build_request_hash(
            model=self.model,
            generation={"temperature": 0.0},
            prompt_version="target-parse-v1",
            messages=messages,
            image_sha256=None,
        )
        return await self.client.complete_json(
            messages=messages,
            response_model=CountTargetSpec,
            request_meta=RequestMeta(
                request_id=f"{sample_id}:target",
                request_hash=request_hash,
                prompt_version="target-parse-v1",
                sample_id=sample_id,
                artifact_dir=artifact_dir / "target_parse",
            ),
        )


TargetParser = CountTargetParser


def _rule_target(question: str) -> CountTargetSpec | None:
    """Retain the former deterministic parser for explicit compatibility callers.
    为显式兼容调用方保留原确定性解析器。
    """

    text = question.strip()
    patterns = (
        r"(?:有多少|多少个)\s*(.+?)(?:[？?。.!！]|$)",
        r"(?:how many|count the number of)\s+(.+?)(?:[?!.]|$)",
    )
    label = next(
        (
            match.group(1).strip(" ，,。.")
            for pattern in patterns
            if (match := re.search(pattern, text, re.IGNORECASE))
        ),
        None,
    )
    if not label or len(label) > 40 or re.search(r"\d", label):
        return None
    singular = label.rstrip("s").strip() or label
    return CountTargetSpec(
        canonical_label=singular,
        aliases=[label] if label.casefold() != singular.casefold() else [],
        required_attributes=["independent visible instance"],
        excluded_attributes=["tiny ambiguous fragment"],
        inclusion_rule="Count each distinct visible instance whose centre lies in the owner core once.",
        exclusion_rule="Do not count duplicate halo views, shadows, or ambiguous fragments.",
    )
