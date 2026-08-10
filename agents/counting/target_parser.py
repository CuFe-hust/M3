"""Structured counting-target parsing with hint priority.

结构化计数目标解析。优先级固定：normalization.count_target_hint →
legacy metadata["count_target_hint"] → Qwen 纯文本契约。无效 hint 抛出
稳定错误，绝不静默吞掉。所有模型调用使用完整 ModelCacheIdentity 并携带
response schema。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agents.base import CallBudget
from agents.counting.backends.base import require_model_cache_identity
from agents.counting.schema import CountTargetSpec
from models.base import RequestMeta, VisionLanguageClient, build_request_hash


class InvalidCountTargetHintError(ValueError):
    """Raised when a provided count_target_hint cannot be validated.
    提供的 count_target_hint 无法校验时抛出。"""


class CountTargetParser:
    """Parse one stable target: normalization hint first, legacy metadata
    second, frozen text-only Qwen contract last.
    解析一个稳定目标：优先 normalization hint，其次 legacy metadata，最后
    冻结的纯文本 Qwen 契约。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        prompt: str,
        model: str | None = None,
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
        count_target_hint: dict[str, Any] | str | None = None,
        legacy_metadata: dict[str, Any] | None = None,
        budget: CallBudget | None = None,
    ) -> CountTargetSpec:
        """Resolve the target; hint hits never call Qwen.
        解析目标；hint 命中绝不调用 Qwen。"""
        hint = count_target_hint
        if hint is None and isinstance(legacy_metadata, dict):
            hint = legacy_metadata.get("count_target_hint")
        if hint is not None:
            return _target_from_hint(hint)
        return await self._parse_via_qwen(question, sample_id=sample_id, artifact_dir=artifact_dir, budget=budget)

    async def _parse_via_qwen(
        self,
        question: str,
        *,
        sample_id: str,
        artifact_dir: Path,
        budget: CallBudget | None,
    ) -> CountTargetSpec:
        identity = require_model_cache_identity(self.client, component="target_parser")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": question},
        ]
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=self.prompt_version,
            messages=messages,
            image_sha256=None,
            response_schema=CountTargetSpec.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
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


def _target_from_hint(hint: Any) -> CountTargetSpec:
    """Resolve a CountTargetSpec from a structured hint; invalid hints raise a
    stable error instead of being silently swallowed.
    从结构化 hint 解析 CountTargetSpec；无效 hint 抛出稳定错误而非静默吞掉。"""
    if isinstance(hint, dict):
        try:
            return CountTargetSpec.model_validate(hint)
        except ValidationError as error:
            raise InvalidCountTargetHintError(
                f"invalid count_target_hint dict: {type(error).__name__}"
            ) from error
    if isinstance(hint, str) and hint.strip():
        rule = _rule_target(hint)
        if rule is not None:
            return rule
        raise InvalidCountTargetHintError(
            f"count_target_hint string could not be parsed: {hint[:40]!r}"
        )
    raise InvalidCountTargetHintError(
        f"unsupported count_target_hint value type: {type(hint).__name__}"
    )


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


