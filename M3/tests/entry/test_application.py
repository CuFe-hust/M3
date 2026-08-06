"""RuntimeApplication ask flow, task rules, artifacts, and no-fallback contract.
RuntimeApplication 的 ask 流程、任务规则、产物与无 fallback 契约。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from models.base import JsonResponseCache
from spacers_agent.agents.base import AgentExecution
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.application import (
    PublicAnswer,
    RuntimeApplication,
    build_image_refs,
    collect_images,
    resolve_task_rules,
    to_public_answer,
    validate_image_count,
)
from spacers_agent.routing import CallBudgetFactory, ROUTES, TaskRouter
from spacers_agent.routing.schemas import RoutingDecision
from spacers_agent.schemas import (
    AgentResult,
    CountingResult,
    GlobalPointObservation,
    GroundTruth,
    UnifiedSample,
)
from spacers_agent.settings import AppSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── helpers / 辅助 ────────────────────────────────────────────────────────


def _make_image(path: Path, size: tuple[int, int] = (16, 12)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(10, 20, 30))
    image.save(path)
    return path


class _FakeClient:
    def __init__(self) -> None:
        self.load_seconds = 0.0


class _RecordingAgent:
    """Registry-compatible agent that records every call. / 记录每次调用的注册表兼容 Agent。"""

    def __init__(
        self,
        name: str,
        *,
        payload: AgentResult | CountingResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.supported_tasks = frozenset()
        self.payload = payload
        self.error = error
        self.run_calls = 0
        self.last_sample: UnifiedSample | None = None

    async def run(self, sample, context) -> AgentExecution:
        self.run_calls += 1
        self.last_sample = sample
        if self.error is not None:
            raise self.error
        payload = self.payload or AgentResult(
            agent_name=self.name, answer="ok", status="completed"
        )
        return AgentExecution(
            agent_name=self.name,
            payload=payload,
            result_filename="agent_result.json",
            trace={"route": "test"},
        )


class _StubRouter(TaskRouter):
    """Real TaskRouter plus a scriptable route_unknown stub.
    真实 TaskRouter 加上可脚本化的 route_unknown 桩。
    """

    def __init__(self, *, decision: RoutingDecision | None = None, error: Exception | None = None):
        super().__init__(router_client=None)
        self.decision = decision
        self.error = error
        self.unknown_calls = 0

    async def route_unknown(self, question, *, budget, sample_id, artifact_dir=None):
        self.unknown_calls += 1
        if self.error is not None:
            raise self.error
        if self.decision is None:
            raise RuntimeError("stub router has no decision")
        return self.decision


def _registry(*agents) -> AgentRegistry:
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    return registry


def _app(
    tmp_path: Path,
    *,
    agents: list[_RecordingAgent] | None = None,
    router: TaskRouter | None = None,
) -> RuntimeApplication:
    settings = AppSettings()
    settings.runs.root = tmp_path / "runs"
    registry = _registry(*(agents or []))
    runtime = SimpleNamespace(
        router=router or _StubRouter(),
        agent_registry=registry,
        call_budget_factory=CallBudgetFactory(),
        prompt_catalog=None,
    )
    return RuntimeApplication(
        settings=settings,
        project_root=PROJECT_ROOT,
        qwen_client=_FakeClient(),
        runtime=runtime,
        service_root=settings.runs.root / "service",
    )


def _known_decision(task: str) -> RoutingDecision:
    """Mirror TaskRouter.route_known for stub assertions. / 桩断言用的 route_known 镜像。"""

    agents = ROUTES[task]
    primary = agents[0]
    fallbacks = list(agents[1:])
    return RoutingDecision(
        task=task,
        primary_agent=primary,
        fallback_agents=fallbacks,
        execution_mode="fallback" if fallbacks else "single",
        requires_tiling=False,
        reason_codes=[f"task_{task}"],
        router_source="dataset_task",
    )


# ── auto task rules / 自动任务规则 ───────────────────────────────────────


def test_resolve_task_rules_auto():
    assert resolve_task_rules("", 2) == "change_caption"
    assert resolve_task_rules("question", 2) == "change_qa"
    assert resolve_task_rules("", 1) == "caption"
    assert resolve_task_rules("question", 1) == "__router__"
    assert resolve_task_rules("question", 3) == "__router__"


def test_resolve_task_rules_empty_question_three_images_fails():
    with pytest.raises(ValueError, match="empty question"):
        resolve_task_rules("", 3)


# ── image count validation / 图片数量校验 ─────────────────────────────────


def test_validate_image_count_change_requires_exactly_two():
    one = [SimpleNamespace()] * 1
    with pytest.raises(ValueError, match="exactly two"):
        validate_image_count("change_caption", one)
    with pytest.raises(ValueError, match="exactly two"):
        validate_image_count("change_qa", one)
    validate_image_count("change_caption", [SimpleNamespace()] * 2)


def test_validate_image_count_other_tasks_require_at_least_one():
    with pytest.raises(ValueError, match="at least one"):
        validate_image_count("caption", [])
    validate_image_count("caption", [SimpleNamespace()] * 3)


# ── image roles / 图片角色 ───────────────────────────────────────────────


def test_build_image_refs_single_image():
    refs = build_image_refs("caption", [SimpleNamespace(path=Path("a.png"), width=10, height=20)])
    assert [(ref.image_id, ref.role) for ref in refs] == [("image-0", "image")]
    assert refs[0].width == 10 and refs[0].height == 20


def test_build_image_refs_image_plus_context():
    collected = [
        SimpleNamespace(path=Path("a.png"), width=1, height=2),
        SimpleNamespace(path=Path("b.png"), width=3, height=4),
        SimpleNamespace(path=Path("c.png"), width=5, height=6),
    ]
    refs = build_image_refs("general_vqa", collected)
    assert [(ref.image_id, ref.role) for ref in refs] == [
        ("image-0", "image"),
        ("context-1", "context"),
        ("context-2", "context"),
    ]


def test_build_image_refs_change_tasks():
    collected = [
        SimpleNamespace(path=Path("01_t1.png"), width=1, height=2),
        SimpleNamespace(path=Path("02_t2.png"), width=3, height=4),
    ]
    refs = build_image_refs("change_caption", collected)
    assert [(ref.image_id, ref.role, ref.path.name) for ref in refs] == [
        ("t1", "t1", "01_t1.png"),
        ("t2", "t2", "02_t2.png"),
    ]


def test_change_sample_roles_t1_t2():
    """A change sample built from refs must pass UnifiedSample validation.
    由引用构建的变化样本必须通过 UnifiedSample 校验。
    """

    image_dir = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "legacy"
    image = image_dir / "test_image.png"
    collected = [
        SimpleNamespace(path=image, width=16, height=16),
        SimpleNamespace(path=image, width=16, height=16),
    ]
    refs = build_image_refs("change_caption", collected)
    sample = UnifiedSample(
        sample_id="s",
        dataset="manual",
        split="user",
        task="change_caption",
        images=refs,
        question="",
        ground_truth=GroundTruth(),
        metadata={},
    )
    assert [ref.role for ref in sample.images] == ["t1", "t2"]


# ── ask flow / ask 流程 ──────────────────────────────────────────────────


def test_ask_explicit_task_runs_only_primary_agent(tmp_path):
    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    router = _StubRouter()
    general = _RecordingAgent("general_vqa_agent")
    caption = _RecordingAgent("caption_agent")
    app = _app(tmp_path, agents=[general, caption], router=router)

    answer = asyncio.run(
        app.ask(image_dir=image_dir, question="q", task="general_vqa")
    )
    assert answer.task == "general_vqa"
    assert answer.agent == "general_vqa_agent"
    assert answer.answer == "ok"
    assert general.run_calls == 1
    assert caption.run_calls == 0
    assert router.unknown_calls == 0


def test_ask_explicit_task_with_fallback_ignores_fallback(tmp_path):
    """change_qa has a declared fallback; the manual path must ignore it.
    change_qa 声明了 fallback；手动路径必须忽略它。
    """

    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    _make_image(image_dir / "b.png")
    change = _RecordingAgent("change_agent")
    general = _RecordingAgent("general_vqa_agent")
    app = _app(tmp_path, agents=[change, general])

    answer = asyncio.run(
        app.ask(image_dir=image_dir, question="q", task="change_qa")
    )
    assert answer.agent == "change_agent"
    assert change.run_calls == 1
    assert general.run_calls == 0


def test_ask_auto_two_images_empty_question_is_change_caption(tmp_path):
    image_dir = tmp_path / "img"
    _make_image(image_dir / "01_t1.png")
    _make_image(image_dir / "02_t2.png")
    router = _StubRouter()
    change = _RecordingAgent("change_agent")
    app = _app(tmp_path, agents=[change], router=router)

    answer = asyncio.run(app.ask(image_dir=image_dir, question=""))
    assert answer.task == "change_caption"
    assert answer.agent == "change_agent"
    assert change.last_sample is not None
    assert [ref.role for ref in change.last_sample.images] == ["t1", "t2"]
    assert router.unknown_calls == 0


def test_ask_auto_two_images_question_is_change_qa(tmp_path):
    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    _make_image(image_dir / "b.png")
    router = _StubRouter()
    change = _RecordingAgent("change_agent")
    app = _app(tmp_path, agents=[change], router=router)

    answer = asyncio.run(app.ask(image_dir=image_dir, question="what changed?"))
    assert answer.task == "change_qa"
    assert router.unknown_calls == 0


def test_ask_auto_one_image_empty_question_is_caption(tmp_path):
    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    router = _StubRouter()
    caption = _RecordingAgent("caption_agent")
    app = _app(tmp_path, agents=[caption], router=router)

    answer = asyncio.run(app.ask(image_dir=image_dir, question=""))
    assert answer.task == "caption"
    assert answer.agent == "caption_agent"
    assert router.unknown_calls == 0


def test_ask_auto_one_image_question_calls_router_once(tmp_path):
    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    decision = _known_decision("general_vqa")
    router = _StubRouter(decision=decision)
    general = _RecordingAgent("general_vqa_agent")
    app = _app(tmp_path, agents=[general], router=router)

    answer = asyncio.run(app.ask(image_dir=image_dir, question="what is this?"))
    assert answer.task == "general_vqa"
    assert router.unknown_calls == 1
    assert general.run_calls == 1


def test_ask_router_failure_fails_request_without_fallback(tmp_path):
    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    router = _StubRouter(error=RuntimeError("router exploded"))
    general = _RecordingAgent("general_vqa_agent")
    app = _app(tmp_path, agents=[general], router=router)

    with pytest.raises(RuntimeError, match="router exploded"):
        asyncio.run(app.ask(image_dir=image_dir, question="what is this?"))
    assert general.run_calls == 0


def test_ask_router_incompatible_task_fails(tmp_path):
    """Router returning a change task for one image must fail the request.
    Router 对单张图片返回变化任务时请求必须失败。
    """

    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    decision = _known_decision("change_qa")
    router = _StubRouter(decision=decision)
    change = _RecordingAgent("change_agent")
    app = _app(tmp_path, agents=[change], router=router)

    with pytest.raises(ValueError, match="exactly two"):
        asyncio.run(app.ask(image_dir=image_dir, question="what changed?"))


def test_ask_unknown_explicit_task_fails(tmp_path):
    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    app = _app(tmp_path, agents=[_RecordingAgent("general_vqa_agent")])
    with pytest.raises(ValueError, match="unknown task"):
        asyncio.run(app.ask(image_dir=image_dir, question="q", task="bogus"))


def test_ask_missing_directory_fails(tmp_path):
    app = _app(tmp_path, agents=[_RecordingAgent("general_vqa_agent")])
    with pytest.raises(ValueError, match="does not exist"):
        asyncio.run(app.ask(image_dir=tmp_path / "missing", question="q"))


def test_ask_primary_agent_failure_fails_without_fallback(tmp_path):
    """A failing primary Agent must fail the request; fallback stays untouched.
    主 Agent 失败时请求必须失败；fallback 保持不被调用。
    """

    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    _make_image(image_dir / "b.png")
    change = _RecordingAgent("change_agent", error=RuntimeError("primary exploded"))
    general = _RecordingAgent("general_vqa_agent")
    app = _app(tmp_path, agents=[change, general])

    with pytest.raises(RuntimeError, match="primary exploded"):
        asyncio.run(app.ask(image_dir=image_dir, question="q", task="change_qa"))
    assert change.run_calls == 1
    assert general.run_calls == 0


# ── artifacts / 产物 ─────────────────────────────────────────────────────


def test_ask_writes_request_and_result_artifacts(tmp_path):
    image_dir = tmp_path / "img"
    image_path = _make_image(image_dir / "a.png", size=(64, 32))
    general = _RecordingAgent("general_vqa_agent")
    app = _app(tmp_path, agents=[general])

    answer = asyncio.run(app.ask(image_dir=image_dir, question="q", task="general_vqa"))
    request_dir = Path(answer.artifact_dir)
    request_json = request_dir / "request.json"
    result_json = request_dir / "result.json"
    assert request_json.is_file()
    assert result_json.is_file()

    request = json.loads(request_json.read_text(encoding="utf-8"))
    assert request["request_id"] == answer.request_id
    assert request["source"] == "main_cli"
    assert request["image_dir"] == str(image_dir.resolve())
    assert request["images"] == [
        {
            "path": str(image_path.resolve()),
            "role": "image",
            "width": 64,
            "height": 32,
        }
    ]
    assert request["question"] == "q"
    assert request["requested_task"] == "general_vqa"
    assert request["resolved_task"] == "general_vqa"
    assert request["created_at"].endswith("Z")

    result = json.loads(result_json.read_text(encoding="utf-8"))
    assert result["request_id"] == answer.request_id
    assert result["task"] == "general_vqa"
    assert result["status"] == "completed"


def test_artifacts_contain_no_sensitive_content(tmp_path):
    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    general = _RecordingAgent("general_vqa_agent")
    app = _app(tmp_path, agents=[general])

    answer = asyncio.run(app.ask(image_dir=image_dir, question="q", task="general_vqa"))
    request_dir = Path(answer.artifact_dir)
    for artifact in request_dir.rglob("*.json"):
        text = artifact.read_text(encoding="utf-8")
        assert "base64" not in text.casefold()
        assert "api_key" not in text.casefold()
        assert "authorization" not in text.casefold()


def test_ask_request_ids_unique_within_same_second(tmp_path):
    image_dir = tmp_path / "img"
    _make_image(image_dir / "a.png")
    general = _RecordingAgent("general_vqa_agent")
    app = _app(tmp_path, agents=[general])

    first = asyncio.run(app.ask(image_dir=image_dir, question="q1", task="general_vqa"))
    second = asyncio.run(app.ask(image_dir=image_dir, question="q2", task="general_vqa"))
    assert first.request_id != second.request_id
    assert first.request_id.startswith("manual-")
    assert len(first.request_id.split("-")[-1]) == 6


# ── PublicAnswer mapping / 统一结果映射 ──────────────────────────────────


def _counting_payload() -> CountingResult:
    accepted = GlobalPointObservation(
        global_id="g1", target="plane", source_tile_id="tile-0", local_id="l1",
        local_x_norm=10, local_y_norm=20, local_radius_norm=5,
        global_x_px=10, global_y_px=20, global_x_norm=100, global_y_norm=200,
        radius_px=5.0, confidence=0.9, ownership_valid=True,
        near_core_boundary=False, accepted=True, short_evidence="seen",
    )
    rejected = accepted.model_copy(update={"global_id": "g2", "accepted": False,
                                           "rejection_reason": "boundary"})
    return CountingResult(
        sample_id="s", target="plane", question="how many?",
        source_width=100, source_height=100, tile_count=1,
        global_points=[accepted, rejected],
        final_count=1, status="completed_with_warnings",
        warnings=[
            {"code": "w1", "message": "tile retried", "tile_ids": ["tile-0"], "point_ids": []}
        ],
    )


def test_to_public_answer_counts_accepted_points():
    payload = _counting_payload()
    execution = AgentExecution(
        agent_name="counting_agent",
        payload=payload,
        result_filename="counting_result.json",
        trace={},
    )
    answer = to_public_answer(
        request_id="manual-1",
        resolved_task="counting",
        execution=execution,
        request_dir=Path("/tmp/req"),
        elapsed_seconds=1.25,
    )
    assert answer.count == 1
    assert answer.answer == "1"
    assert answer.target == "plane"
    assert answer.status == "completed_with_warnings"
    assert answer.evidence == [
        {
            "point": [100, 200],
            "confidence": 0.9,
            "image_id": "image-0",
            "source_tile_id": "tile-0",
        }
    ]
    assert answer.warnings[0]["code"] == "w1"
    assert answer.elapsed_seconds == 1.25


def test_to_public_answer_agent_result_evidence():
    payload = AgentResult(
        agent_name="general_vqa_agent",
        answer="a city",
        status="completed",
        evidence_items=[
            {"label": "building", "box": [1, 2, 3, 4], "confidence": 0.8}
        ],
    )
    execution = AgentExecution(
        agent_name="general_vqa_agent",
        payload=payload,
        result_filename="agent_result.json",
        trace={},
    )
    answer = to_public_answer(
        request_id="manual-2",
        resolved_task="general_vqa",
        execution=execution,
        request_dir=Path("/tmp/req2"),
        elapsed_seconds=0.5,
    )
    assert answer.answer == "a city"
    assert answer.count is None
    assert answer.evidence[0]["label"] == "building"
    assert answer.status == "completed"


def test_public_answer_forbids_extra_fields():
    with pytest.raises(Exception):
        PublicAnswer(
            request_id="x", task="caption", agent="caption_agent", status="completed",
            answer="a", elapsed_seconds=0.1, artifact_dir="/tmp/x",
            secret_field="leak",
        )


# ── create / 创建 ────────────────────────────────────────────────────────


def test_create_uses_models_entry_and_assemble_runtime(monkeypatch, tmp_path):
    from spacers_agent import application as app_module

    created: list[str] = []
    assembled: list[str] = []

    def fake_create_model(name, **kwargs):
        created.append(name)
        return _FakeClient()

    def fake_assemble_runtime(settings, *, qwen_client, judge_client=None,
                              prompt_root=None, router_prompt=""):
        assembled.append("assemble")
        general = _RecordingAgent("general_vqa_agent")
        return SimpleNamespace(
            router=_StubRouter(decision=_known_decision("general_vqa")),
            agent_registry=_registry(general),
            call_budget_factory=CallBudgetFactory(),
            prompt_catalog=None,
        )

    monkeypatch.setattr(app_module, "create_model", fake_create_model)
    monkeypatch.setattr(app_module, "assemble_runtime", fake_assemble_runtime)

    settings = AppSettings()
    settings.runs.root = tmp_path / "runs"
    app = RuntimeApplication.create(settings=settings, project_root=PROJECT_ROOT)
    assert created == ["qwen_transformers"]
    assert len(assembled) == 1
    assert (tmp_path / "runs" / "service").is_dir()
    assert isinstance(app.runtime.agent_registry, AgentRegistry)


def test_create_judge_client_is_none(monkeypatch, tmp_path):
    from spacers_agent import application as app_module

    seen = {}

    def fake_create_model(name, **kwargs):
        return _FakeClient()

    def fake_assemble_runtime(settings, *, qwen_client, judge_client=None,
                              prompt_root=None, router_prompt=""):
        seen["judge_client"] = judge_client
        return SimpleNamespace(
            router=_StubRouter(),
            agent_registry=_registry(_RecordingAgent("general_vqa_agent")),
            call_budget_factory=CallBudgetFactory(),
            prompt_catalog=None,
        )

    monkeypatch.setattr(app_module, "create_model", fake_create_model)
    monkeypatch.setattr(app_module, "assemble_runtime", fake_assemble_runtime)

    settings = AppSettings()
    settings.runs.root = tmp_path / "runs"
    RuntimeApplication.create(settings=settings, project_root=PROJECT_ROOT)
    assert seen["judge_client"] is None
