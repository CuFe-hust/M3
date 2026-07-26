"""Phase 4 - parity of new CountingAgent with frozen VRSBench counting fixtures.
Phase 4 - 新 CountingAgent 与冻结 VRSBench 计数 fixture 的等价性。
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


@pytest.mark.asyncio
async def test_new_counting_agent_matches_frozen_vrsbench_count_fixture(
    tmp_path: Path,
) -> None:
    """VRSBench quantity must produce exactly a proposal + localizer call pair.
    VRSBench 数量题必须恰好产生 proposal + localizer 两次调用。
    """

    case_name = "vrsbench_quantity_count"
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

    # VRSBench quantity must NOT call target parser / VRSBench 数量不能调用 target parser
    assert len(client.calls) == 2, f"expected 2 calls (proposal+localizer), got {len(client.calls)}"
    request_ids = [call["request_id"] for call in client.calls]
    assert not any(":target" in rid for rid in request_ids), "VRSBench count must not call target parser"

    actual_calls = canonicalize_artifact(client.calls, run_root=run_root, project_root=PROJECT_ROOT)
    assert actual_calls == expected_calls, f"call mismatch for {case_name}"

    # result_filename is expert_result.json for VRSBench quantity
    # VRSBench 数量的 result_filename 为 expert_result.json
    assert execution.result_filename == "expert_result.json"
    assert result_path.name == "expert_result.json"
    assert "counting_result.json" in execution.additional_results

    # Primary: ExpertResult, additional: CountingResult
    # 主结果：ExpertResult，补充：CountingResult
    assert execution.payload.status == "completed"
    assert execution.additional_results["counting_result.json"].final_count == 2

    # Compare the counting result with the frozen fixture (legacy harness reads counting_result.json first)
    # 将计数结果与冻结 fixture 对比（旧 harness 优先读取 counting_result.json）
    actual_counting = canonicalize_artifact(
        execution.additional_results["counting_result.json"].model_dump(mode="json")
    )
    assert actual_counting == expected_result, f"counting result mismatch for {case_name}"

    assert execution.agent_name == "counting_agent"
