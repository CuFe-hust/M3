"""Test counting backend registry. / 测试计数后端注册表。"""

from __future__ import annotations

import pytest

from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.agents.counting.backends.selector import BackendSelector
from spacers_agent.schemas import CountTargetSpec


class _FakeBackend:
    name = "fake"
    priority = 5

    def is_available(self): return True
    def supports(self, target): return target.canonical_label == "building"
    async def count(self, *args, **kwargs): raise NotImplementedError


class _FakeBackendHighPri:
    name = "high_pri"
    priority = 10

    def is_available(self): return True
    def supports(self, target): return True
    async def count(self, *args, **kwargs): raise NotImplementedError


class _FakeBackendNoMatch:
    name = "no_match"
    priority = 20

    def is_available(self): return True
    def supports(self, target): return False
    async def count(self, *args, **kwargs): raise NotImplementedError


def _target(label: str = "building") -> CountTargetSpec:
    return CountTargetSpec(canonical_label=label, inclusion_rule="count", exclusion_rule="none")


def test_empty_registry_selects_none():
    reg = BackendRegistry()
    selector = BackendSelector(reg)
    assert selector.select(_target()) is None


def test_registry_selects_matching_backend():
    reg = BackendRegistry()
    reg.register(_FakeBackend())
    selector = BackendSelector(reg)
    sel = selector.select(_target("building"))
    assert sel is not None
    assert sel.backend_name == "fake"


def test_registry_selects_highest_priority():
    reg = BackendRegistry()
    reg.register(_FakeBackend())
    reg.register(_FakeBackendHighPri())
    selector = BackendSelector(reg)
    sel = selector.select(_target("building"))
    assert sel.backend_name == "high_pri"  # highest priority wins, not registration order / 最高优先级胜出，非注册顺序


def test_registry_skips_non_matching():
    reg = BackendRegistry()
    reg.register(_FakeBackendNoMatch())
    reg.register(_FakeBackend())
    selector = BackendSelector(reg)
    sel = selector.select(_target("building"))
    assert sel.backend_name == "fake"


def test_registry_no_match_returns_none():
    reg = BackendRegistry()
    reg.register(_FakeBackendNoMatch())
    selector = BackendSelector(reg)
    assert selector.select(_target("something_else")) is None
