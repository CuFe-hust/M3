"""Request and result parity for the standalone non-counting Agents.
独立非计数 Agent 的请求与结果等价测试。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacers_agent.agents.base import AgentContext
from spacers_agent.bootstrap import assemble_runtime, build_agent_registry
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing import CallBudgetFactory
from spacers_agent.schemas import AgentResult
from spacers_agent.workflows.artifact_writer import ArtifactWriter
from tests.parity.canonicalize import canonicalize_artifact
from tests.parity.fake_clients import RecordingFakeQwen
from tests.parity.fixture_harness import PROJECT_ROOT, build_sample, harness_settings


FIXTURE_ROOT = Path(__file__).with_name("fixtures")
PROMPT_ROOT = PROJECT_ROOT / "prompts"
AGENT_BY_CASE = {
    "change_caption": "change_agent",
    "change_qa": "change_agent",
    "grounding": "grounding_agent",
    "general_vqa": "general_vqa_agent",
    "multiple_choice_vqa": "general_vqa_agent",
    "scene_classification": "general_vqa_agent",
    "partial_expert": "general_vqa_agent",
    "failed_expert": "general_vqa_agent",
    "spatial_relation": "spatial_agent",
    "vrsbench_grid_position": "spatial_agent",
    "vrsbench_extreme_category": "spatial_agent",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case_name", tuple(AGENT_BY_CASE))
async def test_new_visual_agent_matches_frozen_request_and_result(
    case_name: str,
    tmp_path: Path,
) -> None:
    """Match the frozen messages, hashes, metadata, result, and request count.
    对照冻结的消息、哈希、元数据、结果与请求次数。
    """

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

    execution = await registry.get(AGENT_BY_CASE[case_name]).run(sample, context)
    result_path = ArtifactWriter().write_execution(sample_dir, execution)

    fixture_dir = FIXTURE_ROOT / case_name
    expected_calls = json.loads((fixture_dir / "expected_calls.json").read_text(encoding="utf-8"))["qwen"]
    expected_result = json.loads((fixture_dir / "expected_result.json").read_text(encoding="utf-8"))
    assert canonicalize_artifact(client.calls, run_root=run_root, project_root=PROJECT_ROOT) == expected_calls
    assert canonicalize_artifact(execution.payload.model_dump(mode="json")) == expected_result
    assert execution.result_filename == "agent_result.json"
    assert result_path == sample_dir / "agent_result.json"
    assert json.loads(result_path.read_text(encoding="utf-8")) == execution.payload.model_dump(mode="json")
    assert execution.trace["prompt_version"] == expected_calls[0]["prompt_version"]


@pytest.mark.asyncio
async def test_caption_agent_uses_its_dedicated_contract(tmp_path: Path) -> None:
    """Close the frozen caption gap with one dedicated, auditable request.
    通过一次专用且可审计的请求修复冻结的 caption 缺口。
    """

    run_root = tmp_path / "run"
    sample = build_sample("caption")
    settings = harness_settings(tmp_path)
    catalog = PromptCatalog(PROMPT_ROOT)
    client = RecordingFakeQwen(run_root, scenario="caption")
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

    execution = await registry.get("caption_agent").run(sample, context)
    result_path = ArtifactWriter().write_execution(sample_dir, execution)

    assert len(client.calls) == 1
    assert client.calls[0]["response_model"] == "AgentResult"
    assert client.calls[0]["request_id"] == "caption:caption_agent"
    assert client.calls[0]["prompt_version"] == "caption-v1"
    assert client.calls[0]["artifact_dir"] == "samples/caption/caption_agent"
    assert len(client.calls[0]["image_inputs"]) == 1
    assert execution.agent_name == "caption_agent"
    assert isinstance(execution.payload, AgentResult)
    assert execution.payload.agent_name == "caption_agent"
    assert execution.payload.status == "completed"
    assert result_path.name == "agent_result.json"


class _ChangeFallbackClient:
    """Deterministic client for testing primary-only fallback semantics.
    用于测试仅主请求失败时兜底语义的确定性客户端。
    """

    def __init__(self, *, primary_status: str | None) -> None:
        self.primary_status = primary_status
        self.request_ids: list[str] = []

    async def complete_json(self, *, messages, response_model, request_meta):
        self.request_ids.append(request_meta.request_id)
        if request_meta.request_id.endswith(":change_agent"):
            if self.primary_status is None:
                raise RuntimeError("deterministic change primary failure")
            return response_model.model_validate(
                {
                    "agent_name": "change_agent",
                    "answer": "primary answer",
                    "status": self.primary_status,
                }
            )
        return response_model.model_validate(
            {
                "agent_name": "general_vqa_agent",
                "answer": "fallback answer",
                "status": "completed",
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary_status", "expected_calls", "fallback_used", "expected_state"),
    [
        (None, 2, True, "succeeded"),
        ("partial", 1, False, "partial"),
    ],
)
async def test_change_qa_falls_back_only_after_primary_failure(
    primary_status: str | None,
    expected_calls: int,
    fallback_used: bool,
    expected_state: str,
    tmp_path: Path,
) -> None:
    """Do not turn a visible partial result into an implicit fallback request.
    不将可见的 partial 结果隐式转换为兜底请求。
    """

    sample = build_sample("change_qa")
    settings = harness_settings(tmp_path)
    client = _ChangeFallbackClient(primary_status=primary_status)
    runtime = assemble_runtime(settings, qwen_client=client)

    outcome = await runtime.sample_runner.run_one(
        sample,
        tmp_path / "samples" / sample.sample_id,
    )

    assert len(client.request_ids) == expected_calls
    assert outcome.fallback_used is fallback_used
    assert outcome.status.state == expected_state
