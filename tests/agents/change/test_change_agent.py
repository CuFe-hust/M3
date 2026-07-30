"""Test ChangeAgent — request schema, prompt version, supported tasks. / 测试 ChangeAgent。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacers_agent.agents.change.agent import (
    ChangeAgent,
    _is_no_change_answer,
    _verification_reasons,
)
from spacers_agent.agents.base import AgentContext
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.routing import CallBudget
from spacers_agent.schemas import ExpertResult, ImageRef, UnifiedSample

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"
CHANGE_PROMPT = PromptAsset("change", Path("change_v1.md"), "change-expert-v1", "prompt")
ANALYSIS_PROMPT = PromptAsset(
    "change_analysis",
    Path("change_analysis_v2.md"),
    "change-analysis-v2",
    "analysis prompt",
)
VERIFY_PROMPT = PromptAsset(
    "change_verification",
    Path("change_verification_v4.md"),
    "change-verification-v4",
    "verification prompt",
)


class _FakeClient:
    def __init__(self, results: list[ExpertResult] | None = None) -> None:
        self.calls: list[dict] = []
        self.results = list(results or [])

    async def complete_json(self, *, messages, response_model, request_meta):
        self.calls.append({"messages": messages, "request_meta": request_meta})
        if self.results:
            return self.results.pop(0)
        return ExpertResult(expert="change_expert", answer="A building appeared.", status="completed")


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="LEVIR-CC", split="test", task="change_caption",
        images=[
            ImageRef(image_id="t1", path=TEST_IMAGE, role="t1"),
            ImageRef(image_id="t2", path=TEST_IMAGE, role="t2"),
        ],
        question="Describe the change.",
    )


@pytest.mark.asyncio
async def test_change_agent_supported_tasks():
    agent = ChangeAgent(None, CHANGE_PROMPT, "model")
    assert "change_caption" in agent.supported_tasks
    assert "change_qa" in agent.supported_tasks


@pytest.mark.asyncio
async def test_change_agent_result_filename():
    client = _FakeClient()
    agent = ChangeAgent(client, CHANGE_PROMPT, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.result_filename == "expert_result.json"


@pytest.mark.asyncio
async def test_change_agent_prompt_version_in_trace():
    client = _FakeClient()
    agent = ChangeAgent(client, CHANGE_PROMPT, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.trace["prompt_version"] == "change-expert-v1"


@pytest.mark.asyncio
async def test_change_agent_coordinate_frame():
    client = _FakeClient()
    agent = ChangeAgent(client, CHANGE_PROMPT, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    await agent.run(_sample(), ctx)
    user_content = json.loads(client.calls[0]["messages"][1]["content"][2]["text"])
    assert user_content["coordinate_frame"] == "normalized_0_999_top_left"


@pytest.mark.asyncio
async def test_change_caption_runs_analysis_then_reference_free_verification(tmp_path: Path):
    client = _FakeClient(
        [
            ExpertResult(
                expert="change_expert",
                answer="Possible vegetation-color difference.",
                evidence=["The structures remain stable."],
                status="completed",
            ),
            ExpertResult(
                expert="change_expert",
                answer="There is no significant land-cover change.",
                status="completed",
            ),
        ]
    )
    budget = CallBudget(max_qwen_calls=2)
    agent = ChangeAgent(
        client,
        CHANGE_PROMPT,
        "model",
        analysis_prompt=ANALYSIS_PROMPT,
        verification_prompt=VERIFY_PROMPT,
    )
    ctx = AgentContext(
        artifact_dir=tmp_path,
        settings=None,
        qwen_client=None,
        call_budget=budget,
    )

    execution = await agent.run(_sample(), ctx)

    assert execution.payload.answer == "There is no significant land-cover change."
    assert len(client.calls) == 2
    assert budget.qwen_calls_used == 2
    assert client.calls[0]["request_meta"].artifact_dir == tmp_path / "change_expert" / "analysis"
    assert client.calls[1]["request_meta"].artifact_dir == tmp_path / "change_expert"
    verification_payload = json.loads(
        client.calls[1]["messages"][1]["content"][2]["text"]
    )
    assert verification_payload["first_pass_analysis"]["answer"] == (
        "Possible vegetation-color difference."
    )
    assert "ground_truth" not in verification_payload
    assert execution.trace["model_call_count"] == 2
    assert [stage["name"] for stage in execution.trace["stages"]] == [
        "analysis",
        "verification",
    ]
    assert execution.trace["selected_stage"] == "verification"
    assert execution.trace["verification_guard"] is None


@pytest.mark.asyncio
async def test_change_caption_rejects_positive_override_without_evidence(
    tmp_path: Path,
):
    analysis = ExpertResult(
        expert="change_expert",
        answer="No verifiable land-cover changes detected; evidence is uncertain.",
        status="completed",
    )
    unsupported_override = ExpertResult(
        expert="change_expert",
        answer="The vegetation area increased.",
        status="completed",
    )
    client = _FakeClient([analysis, unsupported_override])
    agent = ChangeAgent(
        client,
        CHANGE_PROMPT,
        "model",
        analysis_prompt=ANALYSIS_PROMPT,
        verification_prompt=VERIFY_PROMPT,
    )
    ctx = AgentContext(
        artifact_dir=tmp_path,
        settings=None,
        qwen_client=None,
        call_budget=CallBudget(max_qwen_calls=2),
    )

    execution = await agent.run(_sample(), ctx)

    assert len(client.calls) == 2
    assert execution.payload == analysis
    assert execution.trace["selected_stage"] == "analysis"
    assert execution.trace["verification_guard"] == (
        "rejected_non_contrastive_positive_override"
    )


@pytest.mark.asyncio
async def test_change_caption_rejects_generic_positive_evidence(
    tmp_path: Path,
):
    analysis = ExpertResult(
        expert="change_expert",
        answer="No verifiable land-cover changes detected; evidence is uncertain.",
        status="completed",
    )
    generic_override = ExpertResult(
        expert="change_expert",
        answer="The vegetation area increased.",
        evidence=["Green vegetation is more extensive in the second image."],
        status="completed",
    )
    client = _FakeClient([analysis, generic_override])
    agent = ChangeAgent(
        client,
        CHANGE_PROMPT,
        "model",
        analysis_prompt=ANALYSIS_PROMPT,
        verification_prompt=VERIFY_PROMPT,
    )
    ctx = AgentContext(
        artifact_dir=tmp_path,
        settings=None,
        qwen_client=None,
        call_budget=CallBudget(max_qwen_calls=2),
    )

    execution = await agent.run(_sample(), ctx)

    assert execution.payload == analysis
    assert execution.trace["selected_stage"] == "analysis"
    assert execution.trace["verification_guard"] == (
        "rejected_non_contrastive_positive_override"
    )


@pytest.mark.asyncio
async def test_change_caption_accepts_positive_override_with_evidence(
    tmp_path: Path,
):
    analysis = ExpertResult(
        expert="change_expert",
        answer="No verifiable land-cover changes detected; evidence is uncertain.",
        status="completed",
    )
    supported_override = ExpertResult(
        expert="change_expert",
        answer="A new building appeared.",
        evidence=[
            "In the upper-left region, a rectangular roof is absent in T1 "
            "and present in T2."
        ],
        status="completed",
    )
    client = _FakeClient([analysis, supported_override])
    agent = ChangeAgent(
        client,
        CHANGE_PROMPT,
        "model",
        analysis_prompt=ANALYSIS_PROMPT,
        verification_prompt=VERIFY_PROMPT,
    )
    ctx = AgentContext(
        artifact_dir=tmp_path,
        settings=None,
        qwen_client=None,
        call_budget=CallBudget(max_qwen_calls=2),
    )

    execution = await agent.run(_sample(), ctx)

    assert execution.payload == supported_override
    assert execution.trace["selected_stage"] == "verification"
    assert execution.trace["verification_guard"] is None


@pytest.mark.asyncio
async def test_change_qa_remains_single_pass_with_verification_configured(tmp_path: Path):
    sample = _sample().model_copy(update={"task": "change_qa"})
    client = _FakeClient()
    agent = ChangeAgent(
        client,
        CHANGE_PROMPT,
        "model",
        analysis_prompt=ANALYSIS_PROMPT,
        verification_prompt=VERIFY_PROMPT,
    )
    ctx = AgentContext(
        artifact_dir=tmp_path,
        settings=None,
        qwen_client=None,
        call_budget=CallBudget(max_qwen_calls=2),
    )

    execution = await agent.run(sample, ctx)

    assert len(client.calls) == 1
    assert execution.trace["model_call_count"] == 1


@pytest.mark.asyncio
async def test_change_caption_fails_before_conditional_verification_when_budget_is_exhausted(
    tmp_path: Path,
):
    client = _FakeClient()
    agent = ChangeAgent(
        client,
        CHANGE_PROMPT,
        "model",
        analysis_prompt=ANALYSIS_PROMPT,
        verification_prompt=VERIFY_PROMPT,
    )
    budget = CallBudget(max_qwen_calls=1)
    ctx = AgentContext(
        artifact_dir=tmp_path,
        settings=None,
        qwen_client=None,
        call_budget=budget,
    )

    with pytest.raises(
        RuntimeError,
        match="requires one additional Qwen call",
    ):
        await agent.run(_sample(), ctx)

    assert len(client.calls) == 1
    assert budget.qwen_calls_used == 1


@pytest.mark.asyncio
async def test_change_caption_skips_verification_for_clean_no_change(
    tmp_path: Path,
):
    analysis = ExpertResult(
        expert="change_expert",
        answer="There is no significant land-cover change.",
        status="completed",
    )
    client = _FakeClient([analysis])
    budget = CallBudget(max_qwen_calls=1)
    agent = ChangeAgent(
        client,
        CHANGE_PROMPT,
        "model",
        analysis_prompt=ANALYSIS_PROMPT,
        verification_prompt=VERIFY_PROMPT,
    )
    ctx = AgentContext(
        artifact_dir=tmp_path,
        settings=None,
        qwen_client=None,
        call_budget=budget,
    )

    execution = await agent.run(_sample(), ctx)

    assert execution.payload == analysis
    assert len(client.calls) == 1
    assert budget.qwen_calls_used == 1
    assert execution.trace["model_call_count"] == 1
    assert execution.trace["selected_stage"] == "analysis"
    assert execution.trace["verification_triggered"] is False
    assert execution.trace["verification_reasons"] == []
    assert execution.trace["stages"] == [
        {
            "name": "analysis",
            "prompt_version": "change-analysis-v2",
            "artifact_path": "change_expert/analysis",
        }
    ]


@pytest.mark.asyncio
async def test_change_caption_skips_verification_for_supported_positive_change(
    tmp_path: Path,
):
    analysis = ExpertResult(
        expert="change_expert",
        answer="A new building appeared.",
        evidence=[
            "In the upper-left region, bare ground in T1 contains a roof in T2."
        ],
        status="completed",
    )
    client = _FakeClient([analysis])
    agent = ChangeAgent(
        client,
        CHANGE_PROMPT,
        "model",
        analysis_prompt=ANALYSIS_PROMPT,
        verification_prompt=VERIFY_PROMPT,
    )
    ctx = AgentContext(
        artifact_dir=tmp_path,
        settings=None,
        qwen_client=None,
        call_budget=CallBudget(max_qwen_calls=2),
    )

    execution = await agent.run(_sample(), ctx)

    assert len(client.calls) == 1
    assert execution.trace["verification_triggered"] is False
    assert execution.trace["verification_reasons"] == []


def test_change_risk_rules_keep_local_unchanged_clause_positive() -> None:
    answer = (
        "A new building appeared in the lower-left region. "
        "The rest of the landscape remains unchanged."
    )
    result = ExpertResult(
        expert="change_expert",
        answer=answer,
        status="completed",
    )

    assert _is_no_change_answer(answer) is False
    assert _verification_reasons(result) == [
        "positive_without_contrastive_evidence"
    ]
