"""Contract tests for the dataset-neutral visual agent base.

数据集无关视觉 Agent 基类测试：稳定 request hash、用户载荷不泄漏 ground
truth、budget 消费、data_root 显式解析与逃逸防护、正确 MIME、AgentExecution
协议一致性（可注册、task mismatch 前置、payload 名称一致）。
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from agents.base import AgentContext, AgentExecution
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.registry import AgentRegistry
from agents.schema import AgentName, AgentResult, MaterializedVisualView, VisualTaskPlan
from agents.visual_base import PromptBinding, VisualAgentBase
from data.schema import GroundTruth, ImageRef, TaskNormalization, UnifiedSample
from models.base import ModelCacheIdentity
from models.images import materialize_quantized_roi


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

    def __init__(
        self,
        agent_name: str = "general_vqa_agent",
        *,
        revision: str | None = None,
        max_tokens: int = 128,
    ) -> None:
        self.agent_name = agent_name
        self._revision = revision
        self._max_tokens = max_tokens
        self.calls: list[dict[str, Any]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": self._max_tokens},
            client_version="1",
            revision=self._revision,
        )

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


def test_shared_user_payload_contains_only_universal_facts(tmp_path: Path) -> None:
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
    assert '"question": "Is the statement correct?"' in payload_text
    assert "semantic_subtype" not in payload_text
    assert "answer_constraints" not in payload_text
    assert "coordinate_frame" not in payload_text


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


def test_direct_agent_receives_the_exact_rectangular_quantized_crop(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _sample(root)
    source = Image.new("RGB", (2048, 1536), (17, 18, 19))
    source.save(root / "img.png", format="PNG")
    geometry = materialize_quantized_roi((2048, 1536), (500, 500, 999, 999))
    view = MaterializedVisualView(
        image_id="i1",
        view_mode="quantized_roi",
        source_size=(2048, 1536),
        crop_xyxy=geometry.crop_xyxy,
        crop_size=geometry.crop_size,
        requested_roi_xyxy_0_999=geometry.requested_roi_xyxy_0_999,
        requested_pixel_xyxy=geometry.requested_pixel_xyxy,
        roi_quantum=geometry.roi_quantum,
        quantized_side=geometry.quantized_side,
        ideal_square_xyxy=geometry.ideal_square_xyxy,
        was_clipped=geometry.was_clipped,
    )
    plan = VisualTaskPlan(version="visual-task-plan-v5", task="general_vqa")
    client = _RecordingClient()
    execution = asyncio.run(
        _base(client).run(
            sample,
            AgentContext(
                artifact_dir=root / "artifacts",
                qwen_client=None,
                call_budget=_FakeBudget(),
                data_root=root,
                visual_task_plan=plan,
                visual_views=(view,),
            ),
        )
    )
    assert execution.payload.answer == "yes"
    encoded = client.calls[0]["messages"][1]["content"][0]["image_url"]["url"]
    received = Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1])))
    assert received.size == geometry.crop_size == (1024, 896)
    assert received.tobytes() == source.crop(geometry.crop_xyxy).tobytes()


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
        _base(_RecordingClient())._read_image(
            Path("../outside.png"), context, sample_id="s1"
        )


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


# ── cache identity 集成 / identity integration (A) ─────────────────────────


def test_request_hash_changes_with_client_revision(tmp_path: Path) -> None:
    """Different client revision must change the hash end-to-end through run().
    不同客户端 revision 必须端到端（经 run()）改变哈希。"""
    root = tmp_path / "data"
    sample = _sample(root)
    client_a = _RecordingClient(revision="rev-a")
    client_b = _RecordingClient(revision="rev-b")
    asyncio.run(_base(client_a).run(sample, _context(root)))
    asyncio.run(_base(client_b).run(sample, _context(root)))
    assert client_a.calls[0]["request_hash"] != client_b.calls[0]["request_hash"]


def test_request_hash_changes_with_client_max_tokens(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _sample(root)
    client_a = _RecordingClient(max_tokens=64)
    client_b = _RecordingClient(max_tokens=128)
    asyncio.run(_base(client_a).run(sample, _context(root)))
    asyncio.run(_base(client_b).run(sample, _context(root)))
    assert client_a.calls[0]["request_hash"] != client_b.calls[0]["request_hash"]


def test_missing_cache_identity_fails_before_budget_and_model(tmp_path: Path) -> None:
    root = tmp_path / "data"
    budget = _FakeBudget()

    class _NoIdentityClient:
        """Client without the cache_identity attribute.
        无 cache_identity 属性的客户端。"""

        def __init__(self) -> None:
            self.agent_name = "general_vqa_agent"
            self.calls: list[Any] = []

        async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
            self.calls.append(1)
            return response_model.model_validate(
                {"agent_name": self.agent_name, "answer": "yes", "status": "completed"}
            )

    client = _NoIdentityClient()
    with __import__("pytest").raises(AgentExecutionError, match="cache_identity"):
        asyncio.run(_base(client).run(_sample(root), _context(root, budget)))
    assert budget.qwen_calls == 0
    assert client.calls == []


def test_invalid_cache_identity_fails_before_budget_and_model(tmp_path: Path) -> None:
    """A duck-typed fake identity (not a ModelCacheIdentity) must be rejected
    before images are read, budget is consumed, or the model is called.
    鸭子类型的假身份（非 ModelCacheIdentity）必须在读图、消费 budget、调用
    模型之前被拒绝。"""
    root = tmp_path / "data"
    budget = _FakeBudget()

    class _FakeIdentityClient(_RecordingClient):
        @property
        def cache_identity(self):
            return {"model": "/models/Qwen", "generation": {}, "client_version": "1"}

    client = _FakeIdentityClient()
    with __import__("pytest").raises(AgentExecutionError, match="cache_identity"):
        asyncio.run(_base(client).run(_sample(root), _context(root, budget)))
    assert budget.qwen_calls == 0
    assert client.calls == []


# ── 图片读取错误 / image read errors (F) ───────────────────────────────────


def test_image_read_oserror_becomes_agent_error(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data"
    _make_image(root / "img.png")
    sample = _sample(root)

    def _broken_read_bytes(self):
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "read_bytes", _broken_read_bytes)
    with __import__("pytest").raises(AgentExecutionError, match="image_read_failed") as info:
        asyncio.run(_base(_RecordingClient()).run(sample, _context(root)))
    # The machine absolute path must not leak into the error message.
    # 机器绝对路径不得泄漏进错误消息。
    assert str(tmp_path) not in str(info.value)


def test_image_read_generic_oserror_becomes_agent_error(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data"
    _make_image(root / "img.png")
    sample = _sample(root)

    def _broken_read_bytes(self):
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "read_bytes", _broken_read_bytes)
    with __import__("pytest").raises(AgentExecutionError, match="OSError"):
        asyncio.run(_base(_RecordingClient()).run(sample, _context(root)))


# ── MIME 覆盖 / MIME coverage (G) ──────────────────────────────────────────


def test_images_mime_covers_png_jpeg_webp_tiff(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _make_image(root / "img.png")
    _make_image(root / "img.jpg", seed=5, suffix=".jpg")
    (root / "img.webp").write_bytes(_real_webp_bytes())
    (root / "img.tif").write_bytes(_real_tiff_bytes())
    client = _RecordingClient()
    sample = _sample(
        root,
        images=[
            ImageRef(image_id="a", path="img.png", role="image"),
            ImageRef(image_id="b", path="img.jpg", role="context"),
            ImageRef(image_id="c", path="img.webp", role="context"),
            ImageRef(image_id="d", path="img.tif", role="context"),
        ],
    )
    asyncio.run(_base(client).run(sample, _context(root)))
    content = client.calls[0]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[2]["image_url"]["url"].startswith("data:image/webp;base64,")
    assert content[3]["image_url"]["url"].startswith("data:image/tiff;base64,")


def _real_webp_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format="WEBP")
    return buffer.getvalue()


def _real_tiff_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (40, 50, 60)).save(buffer, format="TIFF")
    return buffer.getvalue()


def _real_png_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (70, 80, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


# ── MIME 内容检测 / content-based MIME (C) ─────────────────────────────────


def test_image_mime_follows_real_content_not_suffix(tmp_path: Path) -> None:
    """Suffix/content mismatch must resolve to the real content format.
    后缀与内容不一致时必须使用真实内容格式。"""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "fake.jpg", format="PNG")
    (root / "noext").write_bytes(_real_png_bytes())
    client = _RecordingClient()
    sample = _sample(
        root,
        images=[
            ImageRef(image_id="a", path="fake.jpg", role="image"),
            ImageRef(image_id="b", path="noext", role="context"),
        ],
    )
    asyncio.run(_base(client).run(sample, _context(root)))
    content = client.calls[0]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_unknown_image_format_raises_agent_error(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "img.bin").write_bytes(b"not an image at all")
    sample = _sample(
        root,
        images=[ImageRef(image_id="i1", path="img.bin", role="image")],
    )
    with __import__("pytest").raises(AgentExecutionError, match="image_format_error") as info:
        asyncio.run(_base(_RecordingClient()).run(sample, _context(root)))
    assert str(tmp_path) not in str(info.value)


def test_corrupted_image_raises_agent_error(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 8)
    sample = _sample(
        root,
        images=[ImageRef(image_id="i1", path="broken.jpg", role="image")],
    )
    with __import__("pytest").raises(AgentExecutionError, match="image_format_error"):
        asyncio.run(_base(_RecordingClient()).run(sample, _context(root)))


# ── 模型逻辑身份 / logical model identity (E) ──────────────────────────────


class _PathIdentityClient(_RecordingClient):
    """Client whose identity model id differs from the physical path.
    身份模型 ID 与物理路径不同的客户端。"""

    def __init__(self, *, model: str, cache_model_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._cache_model_id = cache_model_id

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model=self._cache_model_id,
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": self._max_tokens},
            client_version="1",
            revision=self._revision,
        )


def test_same_logical_id_different_paths_same_hash(tmp_path: Path) -> None:
    """Different checkpoint paths with the same logical id must hash equally
    and traces must never contain the machine path.
    不同 checkpoint 路径 + 相同逻辑 ID 必须产生相同哈希，trace 绝不包含
    机器路径。"""
    root = tmp_path / "data"
    sample = _sample(root)
    client_a = _PathIdentityClient(model="/mnt/a/Qwen3-VL-4B", cache_model_id="qwen3-vl-4b-local")
    client_b = _PathIdentityClient(model="/mnt/b/Qwen3-VL-4B", cache_model_id="qwen3-vl-4b-local")
    execution_a = asyncio.run(_base(client_a).run(sample, _context(root)))
    execution_b = asyncio.run(_base(client_b).run(sample, _context(root)))
    assert execution_a.trace["request_hash"] == execution_b.trace["request_hash"]
    assert execution_a.trace["model"] == "qwen3-vl-4b-local"
    assert "/mnt/a" not in json.dumps(execution_a.trace, ensure_ascii=False)
    assert "/mnt/b" not in json.dumps(execution_b.trace, ensure_ascii=False)


def test_same_path_different_logical_ids_different_hash(tmp_path: Path) -> None:
    """The same checkpoint path with different logical ids must hash differently.
    同一 checkpoint 路径 + 不同逻辑 ID 必须产生不同哈希。"""
    root = tmp_path / "data"
    sample = _sample(root)
    client_a = _PathIdentityClient(model="/mnt/a/Qwen3-VL-4B", cache_model_id="id-one")
    client_b = _PathIdentityClient(model="/mnt/a/Qwen3-VL-4B", cache_model_id="id-two")
    execution_a = asyncio.run(_base(client_a).run(sample, _context(root)))
    execution_b = asyncio.run(_base(client_b).run(sample, _context(root)))
    assert execution_a.trace["request_hash"] != execution_b.trace["request_hash"]
