"""Contract tests for the counting target parser.

计数目标解析器契约测试：count_target_hint 优先（dict/字符串）、缺失时调用
Qwen、无效 hint 忽略、budget 消费时机。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agents.counting.schema import CountTargetSpec
from agents.counting.target_parser import CountTargetParser, TargetParser
from models.base import ModelCacheIdentity


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 64},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(request_meta.request_id)
        return response_model.model_validate(
            {
                "canonical_label": "car",
                "inclusion_rule": "visible vehicle",
                "exclusion_rule": "occluded more than half",
            }
        )


def _parser(client: _RecordingClient | None = None) -> CountTargetParser:
    return CountTargetParser(
        client or _RecordingClient(),
        "Parse the target.",
        "fake-model",
    )


def _parse(parser: CountTargetParser, *, metadata=None, budget=None):
    return asyncio.run(
        parser.parse(
            "How many cars are there?",
            sample_id="s1",
            artifact_dir=Path("/tmp/run"),
            metadata=metadata,
            budget=budget,
        )
    )


def test_hint_dict_takes_priority_over_qwen() -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    hinted = CountTargetSpec(
        canonical_label="ship",
        inclusion_rule="visible ship",
        exclusion_rule="occluded",
    )
    target = _parse(_parser(client), metadata={"count_target_hint": hinted.model_dump(mode="json")}, budget=budget)
    assert target.canonical_label == "ship"
    assert client.calls == []  # Qwen never called / Qwen 未被调用
    assert budget.qwen_calls == 0


def test_hint_string_uses_rule_parser() -> None:
    target = _parse(_parser(), metadata={"count_target_hint": "how many trucks"})
    assert target.canonical_label == "truck"
    assert target.aliases == ["trucks"]
    assert target.inclusion_rule


def test_missing_hint_calls_qwen_and_consumes_budget() -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    target = _parse(_parser(client), metadata={}, budget=budget)
    assert target.canonical_label == "car"
    assert client.calls == ["s1:target"]
    assert budget.qwen_calls == 1


def test_none_metadata_calls_qwen() -> None:
    client = _RecordingClient()
    target = _parse(_parser(client), metadata=None)
    assert target.canonical_label == "car"
    assert len(client.calls) == 1


def test_invalid_hint_dict_is_ignored() -> None:
    """Malformed hint dicts must not crash and must fall back to Qwen.
    畸形 hint dict 不得崩溃，必须回退到 Qwen。"""
    client = _RecordingClient()
    target = _parse(_parser(client), metadata={"count_target_hint": {"canonical_label": 42}})
    assert target.canonical_label == "car"
    assert len(client.calls) == 1


def test_empty_string_hint_calls_qwen() -> None:
    client = _RecordingClient()
    target = _parse(_parser(client), metadata={"count_target_hint": "   "})
    assert target.canonical_label == "car"
    assert len(client.calls) == 1


def test_target_parser_alias() -> None:
    assert TargetParser is CountTargetParser


def test_parser_has_no_dataset_branch() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "counting" / "target_parser.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source
