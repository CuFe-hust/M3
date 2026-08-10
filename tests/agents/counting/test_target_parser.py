"""Contract tests for the counting target parser.

计数目标解析器契约测试：normalization hint 优先、legacy metadata 兼容、
无效 hint 稳定失败、缺失时调用 Qwen、budget 消费与解析路径一致、完整
cache identity。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agents.counting.backends.base import MissingModelCacheIdentityError
from agents.counting.schema import CountTargetSpec
from agents.counting.target_parser import (
    CountTargetParser,
    InvalidCountTargetHintError,
    TargetParser,
)
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


class _NoIdentityClient(_RecordingClient):
    @property
    def cache_identity(self):
        return None


def _parser(client: _RecordingClient | None = None) -> CountTargetParser:
    return CountTargetParser(
        client or _RecordingClient(),
        "Parse the target.",
        "fake-model",
    )


def _parse(parser: CountTargetParser, *, hint=None, legacy=None, budget=None):
    return asyncio.run(
        parser.parse(
            "How many cars are there?",
            sample_id="s1",
            artifact_dir=Path("/tmp/run"),
            count_target_hint=hint,
            legacy_metadata=legacy,
            budget=budget,
        )
    )


# ── 优先级 / priority ─────────────────────────────────────────────────────


def test_normalization_hint_takes_priority_over_qwen() -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    hinted = CountTargetSpec(
        canonical_label="ship",
        inclusion_rule="visible ship",
        exclusion_rule="occluded",
    )
    target = _parse(
        _parser(client),
        hint=hinted.model_dump(mode="json"),
        legacy={"count_target_hint": {"canonical_label": "old"}},
        budget=budget,
    )
    assert target.canonical_label == "ship"
    assert client.calls == []  # Qwen never called / Qwen 未被调用
    assert budget.qwen_calls == 0


def test_legacy_metadata_hint_is_compatible() -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    target = _parse(
        _parser(client),
        legacy={"count_target_hint": {"canonical_label": "plane",
                                      "inclusion_rule": "r", "exclusion_rule": "e"}},
        budget=budget,
    )
    assert target.canonical_label == "plane"
    assert client.calls == []
    assert budget.qwen_calls == 0


def test_hint_string_uses_rule_parser() -> None:
    target = _parse(_parser(), hint="how many trucks")
    assert target.canonical_label == "truck"
    assert target.aliases == ["trucks"]
    assert target.inclusion_rule


def test_missing_hint_calls_qwen_and_consumes_budget() -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    target = _parse(_parser(client), legacy={}, budget=budget)
    assert target.canonical_label == "car"
    assert client.calls == ["s1:target"]
    assert budget.qwen_calls == 1


def test_none_hint_and_none_metadata_calls_qwen() -> None:
    client = _RecordingClient()
    target = _parse(_parser(client), hint=None, legacy=None)
    assert target.canonical_label == "car"
    assert len(client.calls) == 1


# ── 无效 hint / invalid hints ─────────────────────────────────────────────


def test_invalid_hint_dict_raises_stable_error() -> None:
    """Malformed hint dicts raise InvalidCountTargetHintError — never a silent
    Qwen fallback. 畸形 hint dict 抛出 InvalidCountTargetHintError——绝不静默
    回退 Qwen。"""
    client = _RecordingClient()
    with pytest.raises(InvalidCountTargetHintError, match="invalid count_target_hint"):
        _parse(_parser(client), hint={"canonical_label": 42})
    assert client.calls == []


def test_unparseable_hint_string_raises_stable_error() -> None:
    client = _RecordingClient()
    with pytest.raises(InvalidCountTargetHintError, match="could not be parsed"):
        _parse(_parser(client), hint="no count wording here at all")
    assert client.calls == []


def test_empty_string_hint_raises_stable_error() -> None:
    client = _RecordingClient()
    with pytest.raises(InvalidCountTargetHintError, match="unsupported"):
        _parse(_parser(client), hint="   ")
    assert client.calls == []


# ── 缓存身份 / cache identity ─────────────────────────────────────────────


def test_missing_cache_identity_fails_before_model_call() -> None:
    client = _NoIdentityClient()
    with pytest.raises(MissingModelCacheIdentityError, match="ModelCacheIdentity"):
        _parse(_parser(client), hint=None)
    assert client.calls == []


def test_target_parser_alias() -> None:
    assert TargetParser is CountTargetParser


def test_parser_has_no_dataset_branch() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "counting" / "target_parser.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source


def test_duck_typed_identity_is_rejected() -> None:
    class _DuckIdentity:
        model = "fake-model"
        client_version = "1"
        revision = None

        def generation_payload(self):
            return {"temperature": 0.0}

    class _DuckClient(_RecordingClient):
        cache_identity = _DuckIdentity()

    with pytest.raises(MissingModelCacheIdentityError, match="ModelCacheIdentity"):
        _parse(_parser(_DuckClient()), hint=None)
    assert _DuckClient().calls == []


def test_parse_budget_signature_uses_call_budget() -> None:
    """The parser budget parameter is typed CallBudget | None, never Any.
    解析器 budget 参数类型为 CallBudget | None 而非 Any。"""
    import typing

    from agents.base import CallBudget

    hints = typing.get_type_hints(CountTargetParser.parse)
    annotation = hints["budget"]
    args = typing.get_args(annotation)
    assert CallBudget in args
    assert type(None) in args
    assert annotation is not Any
