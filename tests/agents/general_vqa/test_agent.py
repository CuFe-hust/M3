"""Offline unit tests for the General VQA agent.

通用 VQA Agent 离线单测：注入带 cache_identity 的 fake client，覆盖三个
受支持 task 的完整 run、选择题载荷（choices/allow_multiple）、task mismatch
前置失败、无 VRSBench geometry、无 spacers_agent 依赖。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.errors import AgentTaskMismatchError
from agents.general_vqa import GeneralVQAAgent
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, TaskNormalization, UnifiedSample
from models.base import ModelCacheIdentity


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _RecordingClient:
    """Records messages and request meta; returns a stable AgentResult.
    记录消息与请求元数据；返回稳定的 AgentResult。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append({"messages": messages, "request_hash": request_meta.request_hash})
        return response_model.model_validate(
            {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"}
        )


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format="PNG")


def _sample(
    root: Path,
    *,
    task: str = "general_vqa",
    constraints: dict[str, Any] | None = None,
) -> UnifiedSample:
    _make_image(root / "img.png")
    normalization = None
    if constraints is not None:
        normalization = TaskNormalization(
            source_task=task,
            normalized_task=task,  # type: ignore[arg-type]
            normalizer="test", version="1",
            answer_constraints=constraints,
        )
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="What is in the image?",
        ground_truth=GroundTruth(answers=["yes"]),
        normalization=normalization,
    )


def _agent(client: _RecordingClient | None = None) -> GeneralVQAAgent:
    return GeneralVQAAgent(client or _RecordingClient())


def _context(root: Path, budget: _FakeBudget | None = None) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=budget or _FakeBudget(),
        data_root=root,
    )


def _last_user_payload(client: _RecordingClient) -> dict[str, Any]:
    """Parse the JSON payload embedded in the last recorded user message.
    解析最后一条已记录 user 消息中的 JSON 载荷。"""
    messages = client.calls[-1]["messages"]
    user_content = messages[1]["content"]
    text = user_content[-1]["text"]
    return json.loads(text)


# ── 协议 / protocol ────────────────────────────────────────────────────────


def test_agent_identity_and_tasks() -> None:
    agent = _agent()
    assert agent.name == "general_vqa_agent"
    assert agent.supported_tasks == frozenset(
        {"general_vqa", "scene_classification", "multiple_choice_vqa"}
    )


def test_run_returns_agent_execution_with_default_filename(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(client)
    execution = asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path)))
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, AgentResult)
    assert execution.agent_name == "general_vqa_agent"
    assert execution.result_filename == "agent_result.json"
    assert execution.trace["model"] == "fake-model"
    assert len(client.calls) == 1


@pytest.mark.parametrize("task", ["general_vqa", "scene_classification", "multiple_choice_vqa"])
def test_all_supported_tasks_run(task: str, tmp_path: Path) -> None:
    client = _RecordingClient()
    execution = asyncio.run(_agent(client).run(_sample(tmp_path, task=task), _context(tmp_path)))
    assert execution.payload.answer == "yes"
    assert len(client.calls) == 1


def test_unsupported_task_fails_before_model_call(tmp_path: Path) -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    with pytest.raises(AgentTaskMismatchError):
        asyncio.run(
            _agent(client).run(_sample(tmp_path, task="counting"), _context(tmp_path, budget))
        )
    assert budget.qwen_calls == 0
    assert client.calls == []


# ── 选择题载荷 / multiple-choice payload ───────────────────────────────────


def test_multiple_choice_payload_contains_choices_and_constraint(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(client)
    sample = _sample(
        tmp_path,
        task="multiple_choice_vqa",
        constraints={"type": "closed_vocabulary", "values": ["A", "B", "C", "D"]},
    )
    asyncio.run(agent.run(sample, _context(tmp_path)))
    payload = _last_user_payload(client)
    assert payload["choices"] == ["A", "B", "C", "D"]
    assert payload["allow_multiple"] is False
    # Ground truth is never leaked. / ground truth 绝不泄漏。
    assert "ground_truth" not in payload


def test_multiple_choice_payload_allow_multiple(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(client)
    sample = _sample(
        tmp_path,
        task="multiple_choice_vqa",
        constraints={"type": "closed_vocabulary", "values": ["A", "B"], "allow_multiple": True},
    )
    asyncio.run(agent.run(sample, _context(tmp_path)))
    payload = _last_user_payload(client)
    assert payload["choices"] == ["A", "B"]
    assert payload["allow_multiple"] is True


def test_multiple_choice_payload_without_constraints(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(client)
    sample = _sample(tmp_path, task="multiple_choice_vqa")
    asyncio.run(agent.run(sample, _context(tmp_path)))
    payload = _last_user_payload(client)
    assert payload["choices"] == []
    assert payload["allow_multiple"] is False


def test_non_choice_payload_has_no_choices_key(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(client)
    asyncio.run(agent.run(_sample(tmp_path, task="scene_classification"), _context(tmp_path)))
    payload = _last_user_payload(client)
    assert "choices" not in payload
    assert "allow_multiple" not in payload


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_agent_has_no_vrsbench_geometry() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "general_vqa" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "geometry" not in source


def test_agent_has_no_spacers_agent_import() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "general_vqa" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "spacers_agent" not in source
    assert "eval" not in source


def test_import_agent_does_not_load_legacy_packages() -> None:
    import agents.general_vqa  # noqa: F401

    for legacy in ("spacers_agent", "eval"):
        assert legacy not in sys.modules
