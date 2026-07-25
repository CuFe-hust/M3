"""Counting target parser — rule-first local parse, then text-only Qwen.
计数目标解析 — 规则优先本地解析，然后纯文本 Qwen。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash
from spacers_agent.schemas import CountTargetSpec


class CountTargetParser:
    """Parse counting targets from question text. / 从问题文本解析计数目标。"""

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        self.client, self.prompt, self.model = client, prompt, model

    async def parse(self, question: str, *, sample_id: str, artifact_dir: Path,
                    metadata: dict[str, Any] | None = None) -> CountTargetSpec:
        rule = _rule_target(question)
        if rule is not None:
            _validate_target(rule)
            return rule
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": {"question": question, "metadata": metadata or {}}},
        ]
        request_hash = build_request_hash(
            model=self.model, generation={"temperature": 0.0}, prompt_version="target-parse-v1",
            messages=messages, image_sha256=None,
            target_spec={"prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(), "metadata": metadata or {}},
        )
        result = await self.client.complete_json(
            messages=messages, response_model=CountTargetSpec,
            request_meta=RequestMeta(
                request_id=f"{sample_id}:target", request_hash=request_hash,
                prompt_version="target-parse-v1", sample_id=sample_id,
                artifact_dir=artifact_dir / "target_parse",
            ),
        )
        _validate_target(result)
        return result


def _rule_target(question: str) -> CountTargetSpec | None:
    text = question.strip()
    patterns = (
        r"(?:有多少|多少个)\s*(.+?)(?:[？?。.!！]|$)",
        r"(?:how many|count the number of)\s+(.+?)(?:[?!.]|$)",
    )
    label = next(
        (match.group(1).strip(" ，,。.") for pattern in patterns
         if (match := re.search(pattern, text, re.IGNORECASE))), None,
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


def _validate_target(target: CountTargetSpec) -> None:
    if not target.canonical_label.strip() or not target.inclusion_rule.strip() or not target.exclusion_rule.strip():
        raise ValueError("invalid target specification")
    if any(re.search(r"\d", value) for value in (target.canonical_label, target.inclusion_rule, target.exclusion_rule)):
        raise ValueError("target specification must not contain an answer number")


# Legacy alias / 旧版别名
TargetParser = CountTargetParser
