"""Structured counting-target parsing with hint priority.

结构化计数目标解析。优先使用 metadata.count_target_hint（dict 或字符串）；
缺失或无效时才调用 Qwen 纯文本契约解析。不按数据集分支。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from models.base import RequestMeta, VisionLanguageClient, build_request_hash
from agents.counting.schema import CountTargetSpec


class CountTargetParser:
    """Parse one stable target: hint first, frozen text-only Qwen contract
    second. 解析一个稳定目标：优先 hint，其次冻结的纯文本 Qwen 契约。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        prompt: str,
        model: str,
        *,
        prompt_version: str = "target-parse-v1",
    ) -> None:
        self.client = client
        self.prompt = prompt
        self.model = model
        self.prompt_version = prompt_version

    async def parse(
        self,
        question: str,
        *,
        sample_id: str,
        artifact_dir: Path,
        metadata: dict[str, Any] | None = None,
        budget: Any = None,
    ) -> CountTargetSpec:
        """Resolve the target from metadata.count_target_hint when present,
        otherwise issue the frozen text-only Qwen request.
        存在 metadata.count_target_hint 时据此解析目标，否则发出冻结的
        纯文本 Qwen 请求。"""
        hinted = _target_from_hint(metadata)
        if hinted is not None:
            return hinted
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": question},
        ]
        request_hash = build_request_hash(
            model=self.model,
            generation={"temperature": 0.0},
            prompt_version=self.prompt_version,
            messages=messages,
            image_sha256=None,
        )
        if budget is not None:
            budget.reserve_qwen()
        return await self.client.complete_json(
            messages=messages,
            response_model=CountTargetSpec,
            request_meta=RequestMeta(
                request_id=f"{sample_id}:target",
                request_hash=request_hash,
                prompt_version=self.prompt_version,
                sample_id=sample_id,
                artifact_dir=artifact_dir / "target_parse",
            ),
        )


TargetParser = CountTargetParser


def _target_from_hint(metadata: dict[str, Any] | None) -> CountTargetSpec | None:
    """Resolve a CountTargetSpec from a structured hint; invalid hints are
    ignored (never guessed). 从结构化 hint 解析 CountTargetSpec；无效 hint
    被忽略（绝不猜测）。"""
    if not isinstance(metadata, dict):
        return None
    hint = metadata.get("count_target_hint")
    if isinstance(hint, dict):
        try:
            return CountTargetSpec.model_validate(hint)
        except Exception:
            return None
    if isinstance(hint, str) and hint.strip():
        return _rule_target(hint)
    return None


def _rule_target(question: str) -> CountTargetSpec | None:
    """Deterministic label extraction for a string hint; returns None when no
    clear label can be derived. 字符串 hint 的确定性标签提取；无法导出明确
    标签时返回 None。"""

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
        inclusion_rule=(
            "Count each distinct visible instance whose centre lies in the "
            "owner core once."
        ),
        exclusion_rule="Do not count duplicate halo views, shadows, or ambiguous fragments.",
    )
