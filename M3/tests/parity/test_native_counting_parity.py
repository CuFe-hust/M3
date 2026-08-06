"""Phase 4 - parity of new CountingAgent with frozen native-counting fixtures.
Phase 4 - 新 CountingAgent 与冻结原生计数 fixture 的等价性。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacers_agent.agents.base import AgentContext
from spacers_agent.bootstrap import build_agent_registry
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing import CallBudgetFactory
from spacers_agent.workflows.artifact_writer import ArtifactWriter
from tests.parity.canonicalize import canonicalize_artifact
from tests.parity.fake_clients import RecordingFakeQwen
from tests.parity.fixture_harness import (
    PROJECT_ROOT,
    build_sample,
    harness_settings,
)

FIXTURE_ROOT = Path(__file__).with_name("fixtures")
PROMPT_ROOT = PROJECT_ROOT / "prompts"
NATIVE_COUNTING_CASES = ("native_counting", "fine_grained_counting", "counting_one_failed_tile")


@pytest.mark.asyncio
@pytest.mark.parametrize("case_name", NATIVE_COUNTING_CASES)
async def test_new_counting_agent_matches_frozen_native_count_fixtures(
    case_name: str,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    sample = build_sample(case_name)
    settings = harness_settings(tmp_path)
    catalog = PromptCatalog(PROMPT_ROOT)
    client = RecordingFakeQwen(run_root, scenario=case_name)
    registry = build_agent_registry(
        settings=settings,
        qwen_client=client,
        prompt_catalog=catalog,
    )
    sample_dir = run_root / "samples" / sample.sample_id
    context = AgentContext(
        artifact_dir=sample_dir,
        settings=settings,
        qwen_client=client,
        call_budget=CallBudgetFactory().create_for_sample(sample.task),
        prompt_catalog=catalog,
    )

    execution = await registry.get("counting_agent").run(sample, context)
    result_path = ArtifactWriter().write_execution(sample_dir, execution)

    fixture_dir = FIXTURE_ROOT / case_name
    expected_calls = json.loads((fixture_dir / "expected_calls.json").read_text(encoding="utf-8"))["qwen"]
    expected_result = json.loads((fixture_dir / "expected_result.json").read_text(encoding="utf-8"))

    actual_calls = canonicalize_artifact(client.calls, run_root=run_root, project_root=PROJECT_ROOT)
    assert actual_calls == expected_calls, f"call mismatch for {case_name}"

    assert result_path.name == "counting_result.json"
    actual_result = canonicalize_artifact(execution.payload.model_dump(mode="json"))
    assert actual_result == expected_result, f"result mismatch for {case_name}"

    assert len(client.calls) == len(expected_calls)
    assert execution.result_filename == "counting_result.json"
    assert execution.agent_name == "counting_agent"
    assert execution.trace.get("status") == expected_result.get("status")
