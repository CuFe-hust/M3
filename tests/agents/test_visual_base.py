"""Contract tests for the dataset-neutral visual agent base.

数据集无关视觉 Agent 基类测试：稳定 request hash、用户载荷不泄漏 ground
truth、budget 消费、data root 图片解析、无 VRSBench 分支。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from agents.base import AgentContext
from agents.schema import AgentName, AgentResult
from agents.visual_base import PromptBinding, VisualAgentBase
from data.schema import GroundTruth, ImageRef, TaskNormalization, UnifiedSample


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0
        self.deepseek_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        self.deepseek_calls += 1


class _RecordingClient:
    """Records messages and request meta; returns a stable AgentResult.
    记录消息与请求元数据；返回稳定的 AgentResult。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(
            {"messages": messages, "request_meta": request_meta, "request_hash": request_meta.request_hash}
        )
        return response_model.model_validate(
            {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"}
        )


def _make_image(path: Path, seed: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (seed, seed, seed)).save(path)


def _sample(root: Path, *, task: str = "general_vqa", normalization: TaskNormalization | None = None) -> UnifiedSample:
    _make_image(root / "img.png", seed=3)
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task=task,
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="Is the statement correct?",
        ground_truth=GroundTruth(answers=["yes"]),
        normalization=normalization,
    )


def _base(client, agent_name: str = "general_vqa_agent") -> VisualAgentBase:
    return VisualAgentBase(
        client,
        "fake-model",
        agent_name=agent_name,
        prompt=PromptBinding(text="Answer the question.", version="test-v1"),
    )


def _context(root: Path, budget: _FakeBudget) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=budget,
        request_context={"data_root": str(root)},
    )


def test_run_produces_agent_result_and_stable_hash(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    base = _base(client)
    sample = _sample(root)
    first = asyncio.run(base.run(sample, _context(root, _FakeBudget())))
    second_hash = client.calls[0]["request_hash"]
    client.calls.clear()
    asyncio.run(base.run(sample, _context(root, _FakeBudget())))
    assert isinstance(first, AgentResult)
    assert first.answer == "yes"
    assert client.calls[0]["request_hash"] == second_hash


def test_request_hash_changes_with_image_content(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    base = _base(client)
    sample = _sample(root)
    asyncio.run(base.run(sample, _context(root, _FakeBudget())))
    first = client.calls[0]["request_hash"]
    client.calls.clear()
    _make_image(root / "img.png", seed=99)  # overwrite with different bytes / 覆盖为不同字节
    asyncio.run(base.run(sample, _context(root, _FakeBudget())))
    assert client.calls[0]["request_hash"] != first


def test_user_payload_never_leaks_ground_truth(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    base = _base(client)
    sample = _sample(root)
    asyncio.run(base.run(sample, _context(root, _FakeBudget())))
    user_content = client.calls[0]["messages"][1]["content"]
    payload_text = user_content[-1]["text"]
    assert "yes" not in payload_text
    assert "ground_truth" not in payload_text
    assert "answers" not in payload_text


def test_user_payload_contains_task_and_constraints(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    norm = TaskNormalization(
        source_task="vrsbench_vqa", normalized_task="spatial_relation",
        semantic_subtype="extreme_category",
        normalizer="vrsbench_task_normalizer", version="1",
        answer_constraints={"type": "closed_vocabulary", "values": ["small-vehicle", "large-vehicle"]},
    )
    sample = _sample(root, task="spatial_relation", normalization=norm)
    asyncio.run(_base(client).run(sample, _context(root, _FakeBudget())))
    payload_text = client.calls[0]["messages"][1]["content"][-1]["text"]
    assert '"task": "spatial_relation"' in payload_text
    assert '"semantic_subtype": "extreme_category"' in payload_text
    assert "closed_vocabulary" in payload_text
    assert '"coordinate_frame": "normalized_0_999_top_left"' in payload_text


def test_budget_consumed_exactly_once_before_call(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    budget = _FakeBudget()
    asyncio.run(_base(client).run(_sample(root), _context(root, budget)))
    assert budget.qwen_calls == 1
    assert budget.deepseek_calls == 0


def test_images_encoded_as_data_urls(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    asyncio.run(_base(client).run(_sample(root), _context(root, _FakeBudget())))
    image_item = client.calls[0]["messages"][1]["content"][0]
    assert image_item["type"] == "image_url"
    assert image_item["image_url"]["url"].startswith("data:image/png;base64,")


def test_missing_image_fails(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = UnifiedSample(
        sample_id="s1", dataset="parity", split="test", task="general_vqa",
        images=[ImageRef(image_id="i1", path="missing.png", role="image")],
        question="Q",
    )
    with __import__("pytest").raises(FileNotFoundError):
        asyncio.run(_base(_RecordingClient()).run(sample, _context(root, _FakeBudget())))


def test_visual_base_has_no_dataset_branch() -> None:
    source = (Path(__file__).resolve().parents[2] / "agents" / "visual_base.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "sample.dataset" not in source
    assert "dataset ==" not in source
