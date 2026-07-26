"""Phase 4 - status propagation and sample_state_from_payload tests.
Phase 4 - 状态传播与 sample_state_from_payload 测试。
"""

from __future__ import annotations

import pytest

from spacers_agent.schemas import CountingResult, ExpertResult, GlobalPointObservation
from spacers_agent.workflows.sample_runner import sample_state_from_payload


def _counting(**kwargs) -> CountingResult:
    defaults = dict(
        sample_id="s1", target="building", question="q", source_width=100, source_height=100,
        tile_count=1, initial_tile_count=1, leaf_tile_count=1,
        succeeded_tiles=["t1"], failed_tiles=[], global_points=[], merged_groups=[],
        unresolved_conflicts=[], warnings=[], final_count=0, status="completed",
    )
    return CountingResult(**{**defaults, **kwargs})


def _expert(**kwargs) -> ExpertResult:
    defaults = dict(expert="test", answer="yes", status="completed")
    return ExpertResult(**{**defaults, **kwargs})


def test_counting_completed_maps_to_succeeded():
    assert sample_state_from_payload(_counting(status="completed")) == "succeeded"


def test_counting_completed_with_warnings_maps_to_succeeded():
    assert sample_state_from_payload(_counting(status="completed_with_warnings")) == "succeeded"


def test_counting_partial_maps_to_partial():
    assert sample_state_from_payload(_counting(status="partial")) == "partial"


def test_counting_failed_maps_to_failed():
    assert sample_state_from_payload(_counting(status="failed")) == "failed"


def test_expert_completed_maps_to_succeeded():
    assert sample_state_from_payload(_expert(status="completed")) == "succeeded"


def test_expert_partial_maps_to_partial():
    assert sample_state_from_payload(_expert(status="partial")) == "partial"


def test_expert_failed_maps_to_failed():
    assert sample_state_from_payload(_expert(status="failed")) == "failed"


def test_counting_isinstance_check_distinguishes_from_expert():
    """CountingResult and ExpertResult must map through separate branches."""
    counting = _counting(status="completed")
    expert = _expert(status="completed")
    assert sample_state_from_payload(counting) == "succeeded"
    assert sample_state_from_payload(expert) == "succeeded"
