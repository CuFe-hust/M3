"""Contract tests for the dataset-neutral visual agent base.

数据集无关视觉 Agent 基类测试：稳定 request hash、用户载荷不泄漏 ground
truth、budget 消费、data_root 显式解析与逃逸防护、正确 MIME、AgentExecution
协议一致性（可注册、task mismatch 前置、payload 名称一致）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from agents.base import AgentContext, AgentExecution
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.registry import AgentRegistry
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

    def __init__(self, agent_name: str = "general_vqa_agent") -> None:
        self.agent_name = agent_name
        self.calls: list[dict[str, Any]] = []

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(
            {"messages": messages, "request_meta": request_meta, "request_hash": request_meta.request_hash}
        )
        return response_model.model_validate(
            {"agent_name": self.agent_name, "answer": "yes", "status": "completed"}
        )


def _make_image(path: Path, seed: int = 1, suffix: str = ".png") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (seed, seed, seed)).save(path, format="PNG" if suffix == ".png" else "JPEG")


def _sample(root: Path, *, task: str = "general_vqa", images=None, normalization: TaskNormalization | None = None) -> UnifiedSample:
    if images is None:
        _make_image(root / "img.png")
        images = [ImageRef(image_id="i1", path="img.png", role="image")]
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task=task,
        images=images,
        question="Is the statement correct?",
        ground_truth=GroundTruth(answers=["yes"]),
        normalization=normalization,
    )


def _base(client, agent_name: str = "general_vqa_agent") -> VisualAgentBase:
    return VisualAgentBase(
        client,
        "fake-model",
        agent_name=agent_name,
        supported_tasks=frozenset({"general_vqa", "spatial_relation"}),
        prompt=PromptBinding(text="Answer the question.", version="test-v1"),
    )


def _context(root: Path, budget: _FakeBudget | None = None) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=budget or _FakeBudget(),
        data_root=root,
    )


# ── 协议一致性 / protocol consistency ───────────────────────────────────────


def test_run_returns_agent_execution_with_agent_result_payload(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    base = _base(client)
    execution = asyncio.run(base.run(_sample(root), _context(root)))
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, AgentResult)
    assert execution.agent_name == "general_vqa_agent"
    assert execution.payload.agent_name == "general_vqa_agent"
    assert execution.result_filename == "agent_result.json"
    assert execution.trace["prompt_version"] == "test-v1"
    assert execution.trace["request_hash"]


def test_visual_agent_registers_into_registry(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    agent = _base(client)
    registry = AgentRegistry()
    registry.register(agent)
    assert registry.get(agent.name) is agent
    assert registry.supports("general_vqa") == ["general_vqa_agent"]
    execution = asyncio.run(agent.run(_sample(root), _context(root)))
    assert isinstance(execution, AgentExecution)


def test_task_mismatch_fails_before_model_call(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    sample = _sample(root, task="counting")
    with __import__("pytest").raises(AgentTaskMismatchError):
        asyncio.run(_base(client).run(sample, _context(root)))
    assert client.calls == []


def test_payload_agent_name_mismatch_raises(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient(agent_name="change_agent")
    with __import__("pytest").raises(AgentExecutionError):
        asyncio.run(_base(client).run(_sample(root), _context(root)))


# ── request hash / 请求哈希 ─────────────────────────────────────────────────


def test_run_produces_stable_hash(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    base = _base(client)
    sample = _sample(root)
    asyncio.run(base.run(sample, _context(root)))
    first = client.calls[0]["request_hash"]
    client.calls.clear()
    asyncio.run(base.run(sample, _context(root)))
    assert client.calls[0]["request_hash"] == first


def test_request_hash_changes_with_image_content(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    base = _base(client)
    sample = _sample(root)
    asyncio.run(base.run(sample, _context(root)))
    first = client.calls[0]["request_hash"]
    client.calls.clear()
    _make_image(root / "img.png", seed=99)
    asyncio.run(base.run(sample, _context(root)))
    assert client.calls[0]["request_hash"] != first


def test_request_hash_changes_with_image_order(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _make_image(root / "a.png", seed=1)
    _make_image(root / "b.png", seed=2)
    client = _RecordingClient()
    base = _base(client)
    forward = [
        ImageRef(image_id="a", path="a.png", role="image"),
        ImageRef(image_id="b", path="b.png", role="context"),
    ]
    backward = [
        ImageRef(image_id="b", path="b.png", role="image"),
        ImageRef(image_id="a", path="a.png", role="context"),
    ]
    asyncio.run(base.run(_sample(root, images=forward), _context(root)))
    first = client.calls[0]["request_hash"]
    client.calls.clear()
    asyncio.run(base.run(_sample(root, images=backward), _context(root)))
    assert client.calls[0]["request_hash"] != first


# ── payload / 载荷 ──────────────────────────────────────────────────────────


def test_user_payload_never_leaks_ground_truth(tmp_path: Path) -> None:
    root = tmp_path / "data"
    client = _RecordingClient()
    asyncio.run(_base(client).run(_sample(root), _context(root)))
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
    asyncio.run(_base(client).run(sample, _context(root)))
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


# ── images / 图片 ───────────────────────────────────────────────────────────


def test_images_encoded_as_data_urls_with_correct_mime(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _make_image(root / "img.png")
    _make_image(root / "img.jpg", seed=5, suffix=".jpg")
    client = _RecordingClient()
    sample = _sample(
        root,
        images=[
            ImageRef(image_id="a", path="img.png", role="image"),
            ImageRef(image_id="b", path="img.jpg", role="context"),
        ],
    )
    asyncio.run(_base(client).run(sample, _context(root)))
    content = client.calls[0]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[2]["type"] == "text"


def test_data_root_required_for_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _sample(root)
    context = AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=_FakeBudget(),
        data_root=None,
    )
    with __import__("pytest").raises(AgentExecutionError, match="data_root"):
        asyncio.run(_base(_RecordingClient()).run(sample, context))


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "data"
    context = _context(root)
    # The ImageRef schema already rejects '..'; the base guard is defensive.
    # ImageRef schema 已拒绝 '..'；基类防护为防御层。
    with __import__("pytest").raises(AgentExecutionError, match="escape"):
        VisualAgentBase._read_image(Path("../outside.png"), context)


def test_missing_image_raises_agent_error(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = UnifiedSample(
        sample_id="s1", dataset="parity", split="test", task="general_vqa",
        images=[ImageRef(image_id="i1", path="missing.png", role="image")],
        question="Q",
    )
    with __import__("pytest").raises(AgentExecutionError, match="does not exist"):
        asyncio.run(_base(_RecordingClient()).run(sample, _context(root)))


def test_visual_base_has_no_dataset_branch() -> None:
    source = (Path(__file__).resolve().parents[2] / "agents" / "visual_base.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "sample.dataset" not in source
