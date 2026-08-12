"""Contract tests for the high-level runtime: dataset delegation, report
building, and run manifest creation.

高层运行时契约测试：数据集委托、报告构建与 run manifest 创建。使用注入的
fake Qwen 客户端与自定义 registry 的 manifest 数据集。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from PIL import Image

from agents.base import AgentExecution
from agents.registry import AgentRegistry
from agents.schema import AgentResult, VisualEvidence
from application.bootstrap import assemble_runtime
from application.runtime import (
    Runtime,
    collect_images,
    to_public_answer,
)
from application.settings import AppSettings, RunSettings
from data.adapters.manifest import ManifestDraftAdapter
from data.registry import DatasetRegistry
from models.base import ModelCacheIdentity
from workflows.run_store import RunManifest
from workflows.schema import DatasetRunOptions

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeQwenClient:
    """Branches on the response model: task resolutions for the resolver,
    agent results for the visual agents. 按 response model 分支：解析器收到
    任务解析、视觉 Agent 收到 Agent 结果。"""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake",
            generation={"temperature": 0.0},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls += 1
        name = response_model.__name__
        if name == "_ModelTaskResolution":
            return response_model.model_validate(
                {
                    "task": "general_vqa",
                    "confidence": 0.95,
                    "candidate_tasks": ["general_vqa"],
                    "reason_codes": ["model_high_confidence"],
                }
            )
        return response_model.model_validate(
            {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"}
        )


def _make_dataset(root: Path, *, with_task: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "img.png", format="PNG")
    fields = {
        "id": "id",
        "split": "split",
        "question": "question",
        "images": "images",
    }
    if with_task:
        fields["task"] = "task"
    (root / "spacers_adapter.json").write_text(
        json.dumps(
            {
                "dataset": "auto-demo",
                "version": "1",
                "samples_file": "samples.jsonl",
                "fields": fields,
            }
        ),
        encoding="utf-8",
    )
    row = {
        "id": "a1",
        "split": "test",
        "question": "Is there a road?",
        "images": ["img.png"],
    }
    if with_task:
        row["task"] = "general_vqa"
    (root / "samples.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )


def _runtime(tmp_path: Path, *, api_key: str | None = None) -> Runtime:
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=api_key,
    )
    registry = DatasetRegistry()
    registry.register("auto-demo", lambda: ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"}))
    return Runtime(settings=settings, components=components, registry=registry)


def test_runtime_run_dataset_delegates_to_dataset_runner(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    runtime = _runtime(tmp_path)
    options = DatasetRunOptions(
        dataset="auto-demo",
        root=data_root,
        split="test",
        tasks=(),
        auto_task=True,
    )
    results = _run(runtime, options)
    assert "auto" in results
    assert results["auto"].succeeded == 1
    assert results["auto"].total == 1
    run_id = results["auto"].run_id
    assert run_id != "auto-demo-test"  # unique run, never dataset-split default
    run_dir = tmp_path / "runs" / run_id
    assert run_dir.is_dir()
    RunManifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((run_dir / "config.snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["runs"]["root"] == (tmp_path / "runs").as_posix()
    assert (run_dir / "prompts.snapshot" / "task_resolver_v1.md").is_file()
    assert (run_dir / "tasks" / "auto" / "dataset_probe.json").is_file()
    assert (run_dir / "predictions.jsonl").is_file()


def test_runtime_build_report(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    runtime = _runtime(tmp_path)
    results = _run(
        runtime,
        DatasetRunOptions(
            dataset="auto-demo",
            root=data_root,
            split="test",
            tasks=(),
            auto_task=True,
        ),
    )
    run_id = results["auto"].run_id
    report = runtime.build_report(run_id)
    assert report.run_id == run_id
    assert report.total == 1
    assert report.succeeded == 1
    assert report.samples[0].task == "general_vqa"
    assert report.samples[0].prediction == "yes"


def test_runtime_resume_does_not_rerun_agents(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    runtime = _runtime(tmp_path)
    options = DatasetRunOptions(
        dataset="auto-demo",
        root=data_root,
        split="test",
        tasks=(),
        auto_task=True,
        run_id="fixed-run",
    )
    first = _run(runtime, options)
    assert first["auto"].succeeded == 1
    calls_after_first = runtime.components.qwen_client.calls
    second = _run(runtime, options, resume=True)
    assert second["auto"].succeeded == 1
    assert runtime.components.qwen_client.calls == calls_after_first  # no re-run


def test_runtime_auto_task_mode(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root, with_task=False)
    runtime = _runtime(tmp_path)
    options = DatasetRunOptions(
        dataset="auto-demo",
        root=data_root,
        split="test",
        tasks=(),
        auto_task=True,
        run_id="auto-run",
    )
    results = _run(runtime, options)
    assert "auto" in results
    assert results["auto"].succeeded == 1
    run_dir = tmp_path / "runs" / "auto-run"
    assert (run_dir / "tasks" / "auto" / "dataset_probe.json").is_file()
    report = runtime.build_report("auto-run")
    assert report.samples[0].task == "general_vqa"  # resolver picked the task


def test_runtime_with_api_key_creates_judge_client(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, api_key="test-api-key")
    assert runtime.components.judge_client is not None


# ── run identity contract (Fix C) / run identity 契约 ───────────────────────


def _options(**overrides: Any) -> DatasetRunOptions:
    values = dict(
        dataset="auto-demo",
        root=Path("data"),
        split="test",
        tasks=(),
        auto_task=True,
    )
    values.update(overrides)
    return DatasetRunOptions(**values)  # type: ignore[arg-type]


def test_fresh_without_run_id_twice_creates_unique_runs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    runtime = _runtime(tmp_path)
    first = _run(runtime, _options(root=data_root))
    second = _run(runtime, _options(root=data_root))
    first_id = first["auto"].run_id
    second_id = second["auto"].run_id
    assert first_id != second_id
    assert first_id != "auto-demo-test"  # never the dataset-split default
    assert (tmp_path / "runs" / first_id / "manifest.json").is_file()
    assert (tmp_path / "runs" / second_id / "manifest.json").is_file()


def test_fresh_explicit_duplicate_run_id_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    runtime = _runtime(tmp_path)
    options = _options(root=data_root, run_id="fixed")
    first = _run(runtime, options)
    assert first["auto"].succeeded == 1
    manifest_before = (tmp_path / "runs" / "fixed" / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        _run(runtime, options)
    # The old manifest is untouched and no second predictions file exists.
    # 旧 manifest 未被修改，也不会产生第二个 predictions。
    assert (tmp_path / "runs" / "fixed" / "manifest.json").read_bytes() == manifest_before


def test_resume_requires_explicit_run_id(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(ValueError, match="resume requires"):
        _run(runtime, _options(resume=True))


def test_resume_missing_run_fails(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(ValueError, match="resume run does not exist"):
        _run(runtime, _options(run_id="does-not-exist", resume=True))


def test_resume_wrong_dataset_or_split_fails(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    runtime = _runtime(tmp_path)
    _run(runtime, _options(root=data_root, run_id="fixed"))
    with pytest.raises(ValueError, match="dataset mismatch"):
        _run(runtime, _options(root=data_root, run_id="fixed", dataset="other", resume=True))
    with pytest.raises(ValueError, match="split mismatch"):
        _run(runtime, _options(root=data_root, run_id="fixed", split="train", resume=True))


def test_normal_resume_preserves_manifest_identity(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    runtime = _runtime(tmp_path)
    options = _options(root=data_root, run_id="fixed")
    first = _run(runtime, options)
    assert first["auto"].succeeded == 1
    calls_after_first = runtime.components.qwen_client.calls
    second = _run(runtime, options, resume=True)
    assert second["auto"].run_id == "fixed"
    assert runtime.components.qwen_client.calls == calls_after_first
    manifest = json.loads(
        (tmp_path / "runs" / "fixed" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_id"] == "fixed"
    assert manifest["dataset"] == "auto-demo"
    assert manifest["split"] == "test"


class _SampleAdapter:
    """Minimal DatasetAdapter yielding samples with an explicit task.
    产出显式 task 样本的最小 DatasetAdapter。"""

    name = "sample-demo"
    supported_tasks = frozenset({"general_vqa"})

    def __init__(self, sample: Any) -> None:
        self._sample = sample

    def probe(self, root: Path, task: str | None = None):
        from data.adapters.base import AdapterProbe

        return AdapterProbe(
            dataset="sample-demo",
            version="1",
            sample_file=root / "samples.jsonl",
            observed_fields=("id",),
            sample_count=1,
            task=task,
            available_tasks=("general_vqa",),
        )

    def iter_samples(self, root: Path, split: str, task: str):
        yield self._sample


def test_runtime_sample_mode_with_explicit_task(tmp_path: Path) -> None:
    """A task-known dataset runs through iter_samples with a fixed task
    namespace. 已知 task 数据集经 iter_samples 在固定 task 命名空间运行。"""
    from data.schema import GroundTruth, ImageRef, UnifiedSample

    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(data_root / "img.png", format="PNG")
    sample = UnifiedSample(
        sample_id="b1",
        dataset="sample-demo",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="Is there a road?",
        ground_truth=GroundTruth(answers=["yes"]),
    )
    runtime = _runtime(tmp_path)
    registry = DatasetRegistry()
    registry.register("sample-demo", lambda: _SampleAdapter(sample))
    runtime = Runtime(settings=runtime.settings, components=runtime.components, registry=registry)
    options = DatasetRunOptions(
        dataset="sample-demo",
        root=data_root,
        split="test",
        tasks=("general_vqa",),
        run_id="sample-run",
    )
    results = _run(runtime, options)
    assert "general_vqa" in results
    assert results["general_vqa"].succeeded == 1
    assert (tmp_path / "runs" / "sample-run" / "tasks" / "general_vqa" / "dataset_probe.json").is_file()
    report = runtime.build_report("sample-run")
    assert report.samples[0].task == "general_vqa"


def _run(runtime: Runtime, options: DatasetRunOptions, *, resume: bool = False):
    return asyncio.run(
        runtime.run_dataset(
            dataclasses.replace(options, resume=resume) if resume else options
        )
    )



# ── manual ask path (Task 11A) / 手动 ask 路径 ─────────────────────────────


def _make_images(directory: Path, names: list[str]) -> None:
    """Create one small PNG per name. / 每个名字创建一张小 PNG。"""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.new("RGB", (4, 4), (1, 2, 3)).save(directory / name, format="PNG")


class _RecordingAgent:
    """Minimal Agent that records sample/context and returns a fixed
    AgentResult. 记录 sample/context 并返回固定 AgentResult 的最小 Agent。"""

    def __init__(self, name: str, task: str, answer: str = "ok") -> None:
        self.name = name
        self.supported_tasks = frozenset({task})
        self.runs: list[tuple[Any, Any]] = []
        self.answer = answer

    async def run(self, sample, context):
        self.runs.append((sample, context))
        payload = AgentResult(
            agent_name=self.name, answer=self.answer, status="completed"
        )
        return AgentExecution(
            agent_name=self.name, payload=payload, result_filename="agent_result.json"
        )


def _ask_runtime(
    tmp_path: Path,
    client: _FakeQwenClient | None = None,
    agents: dict[str, _RecordingAgent] | None = None,
) -> Runtime:
    """A Runtime with the real resolver/router but recording stub agents.
    使用真实 resolver/router 与记录型 stub Agent 的 Runtime。"""
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = client or _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = AgentRegistry()
    for agent in (agents or {}).values():
        registry.register(agent)
    components = dataclasses.replace(components, agent_registry=registry)
    return Runtime(settings=settings, components=components)


def _ask(runtime: Runtime, *, image_dir: Path, question: str = "", task: str = "auto"):
    return asyncio.run(
        runtime.ask(image_dir=image_dir, question=question, task=task)
    )


# ── image collection / 图片收集 ─────────────────────────────────────────────


def test_collect_images_natural_sort_and_filters(tmp_path: Path) -> None:
    _make_images(
        tmp_path / "imgs",
        ["img10.png", "img2.png", "img1.png", ".hidden.png", "note.txt"],
    )
    collected = collect_images(tmp_path / "imgs")
    assert [item.path.name for item in collected] == ["img1.png", "img2.png", "img10.png"]
    assert all(item.width == 4 and item.height == 4 for item in collected)


def test_collect_images_missing_or_not_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        collect_images(tmp_path / "missing")
    (tmp_path / "plain.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        collect_images(tmp_path / "plain.txt")


def test_collect_images_empty_directory_fails(tmp_path: Path) -> None:
    (tmp_path / "imgs").mkdir()
    with pytest.raises(ValueError, match="no supported images"):
        collect_images(tmp_path / "imgs")


def test_collect_images_corrupt_image_fails(tmp_path: Path) -> None:
    directory = tmp_path / "imgs"
    directory.mkdir()
    (directory / "broken.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="cannot open image"):
        collect_images(directory)


def test_collect_images_too_many_fails(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", [f"img{i}.png" for i in range(9)])
    with pytest.raises(ValueError, match="too many images"):
        collect_images(tmp_path / "imgs")


# ── ask orchestration / ask 编排 ────────────────────────────────────────────


def test_ask_auto_one_image_empty_question_caption(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    agent = _RecordingAgent("caption_agent", "caption", answer="a street scene")
    runtime = _ask_runtime(tmp_path, agents={"caption_agent": agent})
    answer = _ask(runtime, image_dir=tmp_path / "imgs")
    assert answer.task == "caption"
    assert answer.agent == "caption_agent"
    assert answer.answer == "a street scene"
    assert answer.request_id.startswith("manual-")
    assert len(agent.runs) == 1
    sample, context = agent.runs[0]
    assert sample.task == "caption"
    assert sample.question == ""
    assert sample.metadata["image_dir"] == "manual://input"
    assert [image.role for image in sample.images] == ["image"]
    assert sample.images[0].path.as_posix() == "img.png"
    assert context.data_root == (tmp_path / "imgs").resolve()
    # The deterministic rule path never calls the resolver model.
    # 确定性规则路径绝不调用 resolver 模型。
    assert runtime.components.qwen_client.calls == 0


def test_ask_auto_two_images_empty_question_change_caption(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["t1.png", "t2.png"])
    agent = _RecordingAgent("change_agent", "change_caption")
    runtime = _ask_runtime(tmp_path, agents={"change_agent": agent})
    answer = _ask(runtime, image_dir=tmp_path / "imgs")
    assert answer.task == "change_caption"
    assert answer.agent == "change_agent"
    sample, _ = agent.runs[0]
    assert [image.role for image in sample.images] == ["t1", "t2"]
    assert [image.image_id for image in sample.images] == ["t1", "t2"]


def test_ask_auto_question_resolves_via_resolver_once(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeQwenClient()
    agent = _RecordingAgent("general_vqa_agent", "general_vqa")
    runtime = _ask_runtime(
        tmp_path, client=client, agents={"general_vqa_agent": agent}
    )
    answer = _ask(
        runtime, image_dir=tmp_path / "imgs", question="Is there a road?"
    )
    assert client.calls == 1  # exactly one resolver model call
    assert answer.task == "general_vqa"
    assert answer.agent == "general_vqa_agent"


def test_ask_explicit_task_skips_resolver(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeQwenClient()
    agent = _RecordingAgent("general_vqa_agent", "general_vqa")
    runtime = _ask_runtime(
        tmp_path, client=client, agents={"general_vqa_agent": agent}
    )
    answer = _ask(
        runtime,
        image_dir=tmp_path / "imgs",
        question="Any buildings?",
        task="general_vqa",
    )
    assert client.calls == 0  # explicit task never calls the resolver
    assert answer.task == "general_vqa"
    assert answer.agent == "general_vqa_agent"


def test_ask_explicit_unknown_task_fails(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    runtime = _ask_runtime(tmp_path)
    with pytest.raises(ValueError, match="unknown task"):
        _ask(runtime, image_dir=tmp_path / "imgs", task="bogus")


def test_ask_change_task_count_validation(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    runtime = _ask_runtime(tmp_path)
    with pytest.raises(ValueError, match="exactly two images"):
        _ask(runtime, image_dir=tmp_path / "imgs", task="change_caption")
    _make_images(tmp_path / "imgs2", ["a.png", "b.png", "c.png"])
    with pytest.raises(ValueError, match="exactly two images"):
        _ask(
            runtime,
            image_dir=tmp_path / "imgs2",
            question="what changed?",
            task="change_qa",
        )


def test_ask_runs_single_primary_agent_no_fallback(tmp_path: Path) -> None:
    """change_qa declares a fallback agent; the manual path must never use it.
    change_qa 声明了 fallback Agent；手动路径绝不使用它。"""
    _make_images(tmp_path / "imgs", ["a.png", "b.png"])
    change = _RecordingAgent("change_agent", "change_qa")
    general = _RecordingAgent("general_vqa_agent", "general_vqa")
    runtime = _ask_runtime(
        tmp_path,
        agents={"change_agent": change, "general_vqa_agent": general},
    )
    answer = _ask(
        runtime,
        image_dir=tmp_path / "imgs",
        question="what changed?",
        task="change_qa",
    )
    assert answer.agent == "change_agent"
    assert len(change.runs) == 1
    assert not general.runs  # fallback never executed / 兜底绝不执行


def test_ask_artifacts_relative_and_judge_free(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    agent = _RecordingAgent("caption_agent", "caption")
    runtime = _ask_runtime(tmp_path, agents={"caption_agent": agent})
    answer = _ask(runtime, image_dir=tmp_path / "imgs")
    request_dir = tmp_path / "runs" / "service" / "requests" / answer.request_id
    request = json.loads((request_dir / "request.json").read_text(encoding="utf-8"))
    assert request["image_dir"] == "manual://input"
    assert request["images"][0]["path"] == "img.png"
    assert request["images"][0]["role"] == "image"
    result = json.loads((request_dir / "result.json").read_text(encoding="utf-8"))
    absolute = (tmp_path / "imgs").resolve().as_posix()
    assert absolute not in json.dumps(request)
    assert absolute not in json.dumps(result)
    assert answer.artifact_dir == f"service/requests/{answer.request_id}"
    assert not Path(answer.artifact_dir).is_absolute()
    # No Judge/evaluation/report artifacts on the manual path.
    # 手动路径无 Judge/评测/报告产物。
    names = {entry.name for entry in request_dir.iterdir()}
    assert names == {"request.json", "result.json"}


def test_ask_reuses_single_qwen_client(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeQwenClient()
    agent = _RecordingAgent("caption_agent", "caption")
    runtime = _ask_runtime(tmp_path, client=client, agents={"caption_agent": agent})
    first = _ask(runtime, image_dir=tmp_path / "imgs")
    second = _ask(runtime, image_dir=tmp_path / "imgs")
    assert runtime.components.qwen_client is client  # created once, reused
    assert first.request_id != second.request_id


# ── public answer mapping / 公开结果映射 ────────────────────────────────────


def test_to_public_answer_agent_result_mapping() -> None:
    execution = AgentExecution(
        agent_name="grounding_agent",
        payload=AgentResult(
            agent_name="grounding_agent",
            answer="road",
            status="completed",
            evidence_items=[
                VisualEvidence(
                    label="road",
                    box=[1, 2, 3, 4],
                    confidence=0.9,
                    image_id="image-0",
                )
            ],
        ),
        result_filename="agent_result.json",
    )
    answer = to_public_answer(
        request_id="manual-x",
        resolved_task="grounding",
        execution=execution,
        artifact_dir="service/requests/manual-x",
        elapsed_seconds=1.0,
    )
    assert answer.task == "grounding"
    assert answer.agent == "grounding_agent"
    assert answer.answer == "road"
    assert answer.count is None
    assert answer.target is None
    assert answer.evidence[0]["box"] == [1, 2, 3, 4]
    assert answer.artifact_dir == "service/requests/manual-x"


def test_to_public_answer_counting_result_mapping() -> None:
    from agents.counting.schema import (
        CountingResult,
        GlobalPointObservation,
        IssueRecord,
    )

    def point(point_id: str, *, accepted: bool, reason: str | None = None):
        return GlobalPointObservation(
            global_id=point_id,
            target="vehicles",
            source_tile_id="t0",
            local_id="l0",
            local_x_norm=10,
            local_y_norm=10,
            local_radius_norm=5,
            global_x_px=10,
            global_y_px=10,
            global_x_norm=100,
            global_y_norm=200,
            radius_px=5.0,
            confidence=0.9,
            ownership_valid=True,
            near_core_boundary=False,
            accepted=accepted,
            rejection_reason=reason,
            short_evidence="e",
        )

    payload = CountingResult(
        sample_id="manual-x",
        target="vehicles",
        question="how many?",
        source_width=100,
        source_height=100,
        tile_count=1,
        global_points=[
            point("p1", accepted=True),
            point("p2", accepted=False, reason="low_confidence"),
        ],
        warnings=[IssueRecord(code="w1", message="note")],
        final_count=1,
        status="completed_with_warnings",
    )
    execution = AgentExecution(
        agent_name="counting_agent",
        payload=payload,
        result_filename="counting_result.json",
    )
    answer = to_public_answer(
        request_id="manual-x",
        resolved_task="counting",
        execution=execution,
        artifact_dir="service/requests/manual-x",
        elapsed_seconds=0.5,
    )
    assert answer.count == 1
    assert answer.target == "vehicles"
    assert answer.answer == "1"
    assert answer.evidence == [
        {
            "point": [100, 200],
            "confidence": 0.9,
            "image_id": "image-0",
            "source_tile_id": "t0",
        }
    ]
    assert len(answer.warnings) == 1
    assert answer.warnings[0]["code"] == "w1"


# ── manual HTTP serve (Task 11B) / 手动 HTTP 服务 ──────────────────────────


class _ServeHarness:
    """A bound handler server on an ephemeral local port; each
    ``request`` call processes exactly one connection via handle_request.
    绑定 handler 的临时端口服务器；每次 ``request`` 经 handle_request
    处理恰好一个连接。"""

    def __init__(self, runtime: Runtime) -> None:
        import http.server

        from application.commands.serve import RuntimeRequestHandler

        handler = type(
            "BoundRuntimeRequestHandler",
            (RuntimeRequestHandler,),
            {"application": runtime},
        )
        self.server = http.server.HTTPServer(("127.0.0.1", 0), handler)
        self.server.timeout = 5
        self.port = self.server.server_address[1]

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            self.server.handle_request()
            response = conn.getresponse()
            raw = response.read()
            payload = json.loads(raw.decode("utf-8")) if raw else None
            return response.status, payload
        finally:
            conn.close()

    def close(self) -> None:
        self.server.server_close()


def test_serve_health_returns_readiness(tmp_path: Path) -> None:
    client = _FakeQwenClient()
    runtime = _ask_runtime(tmp_path, client=client)
    harness = _ServeHarness(runtime)
    try:
        status, payload = harness.request("GET", "/health")
    finally:
        harness.close()
    assert status == 200
    assert payload["status"] == "ready"
    assert "model" in payload
    assert "model_load_seconds" in payload
    assert "agents" in payload
    assert client.calls == 0  # health never calls a model / health 绝不调模型


def test_serve_unknown_paths_return_404(tmp_path: Path) -> None:
    runtime = _ask_runtime(tmp_path)
    harness = _ServeHarness(runtime)
    try:
        status, payload = harness.request("GET", "/nope")
        assert status == 404
        assert payload["error"] == "not found"
        status, payload = harness.request("POST", "/nope", body=b"{}")
        assert status == 404
        assert payload["error"] == "not found"
    finally:
        harness.close()


def test_serve_ask_success(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    agent = _RecordingAgent("general_vqa_agent", "general_vqa", answer="yes")
    runtime = _ask_runtime(tmp_path, agents={"general_vqa_agent": agent})
    harness = _ServeHarness(runtime)
    try:
        status, payload = harness.request(
            "POST",
            "/ask",
            body=json.dumps(
                {"image_dir": str(tmp_path / "imgs"), "question": "q", "task": "general_vqa"}
            ).encode("utf-8"),
        )
    finally:
        harness.close()
    assert status == 200
    assert payload["task"] == "general_vqa"
    assert payload["agent"] == "general_vqa_agent"
    assert payload["status"] == "completed"
    assert payload["answer"] == "yes"
    assert payload["request_id"].startswith("http-")


def test_serve_ask_bad_bodies_return_400(tmp_path: Path) -> None:
    runtime = _ask_runtime(tmp_path)
    harness = _ServeHarness(runtime)
    try:
        status, payload = harness.request("POST", "/ask", body=b"{not json")
        assert status == 400
        assert payload["error"] == "invalid JSON body"
        status, payload = harness.request("POST", "/ask", body=b"[]")
        assert status == 400
        assert payload["error"] == "request body must be a JSON object"
        status, payload = harness.request("POST", "/ask", body=b"{}")
        assert status == 400
        assert payload["error"] == "image_dir is required"
        status, payload = harness.request(
            "POST", "/ask", body=b'{"image_dir": ""}'
        )
        assert status == 400
        assert payload["error"] == "image_dir is required"
    finally:
        harness.close()


def test_serve_ask_invalid_request_returns_stable_400(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    runtime = _ask_runtime(tmp_path)
    harness = _ServeHarness(runtime)
    try:
        status, payload = harness.request(
            "POST",
            "/ask",
            body=json.dumps(
                {"image_dir": str(tmp_path / "imgs"), "task": "bogus"}
            ).encode("utf-8"),
        )
    finally:
        harness.close()
    assert status == 400
    assert payload["error"] == "invalid request"  # stable, no raw text


def test_serve_ask_oversized_body_returns_413(tmp_path: Path) -> None:
    runtime = _ask_runtime(tmp_path)
    harness = _ServeHarness(runtime)
    try:
        status, payload = harness.request(
            "POST", "/ask", body=b"x" * ((1 << 20) + 1)
        )
    finally:
        harness.close()
    assert status == 413
    assert payload["error"] == "request body too large"


def test_serve_asks_reuse_single_runtime(tmp_path: Path) -> None:
    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeQwenClient()
    agent = _RecordingAgent("general_vqa_agent", "general_vqa")
    runtime = _ask_runtime(tmp_path, client=client, agents={"general_vqa_agent": agent})
    assert runtime.components.judge_client is None  # no DeepSeek on manual service
    harness = _ServeHarness(runtime)
    try:
        body = json.dumps(
            {"image_dir": str(tmp_path / "imgs"), "question": "q"}
        ).encode("utf-8")
        first_status, first = harness.request("POST", "/ask", body=body)
        second_status, second = harness.request("POST", "/ask", body=body)
    finally:
        harness.close()
    assert first_status == 200 and second_status == 200
    assert runtime.components.qwen_client is client  # one client, both requests
    assert client.calls == 2  # one resolver model call per ask
    assert first["request_id"] != second["request_id"]


def test_serve_invalid_port_rejected(capsys) -> None:
    import argparse

    from application.commands.serve import run_serve

    code = run_serve(
        argparse.Namespace(config=None, host="127.0.0.1", port=0)
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "failed"
    assert error["error"] == "ValueError"


def test_serve_server_close_is_clean(tmp_path: Path) -> None:
    import socket

    runtime = _ask_runtime(tmp_path)
    harness = _ServeHarness(runtime)
    status, _ = harness.request("GET", "/health")
    assert status == 200
    port = harness.port
    harness.close()
    # The port is released: a new server can bind it again.
    # 端口已释放：新服务器可再次绑定同一端口。
    import http.server

    probe = http.server.HTTPServer(("127.0.0.1", 0), type(
        "ProbeHandler",
        (http.server.BaseHTTPRequestHandler,),
        {"application": runtime},
    ))
    try:
        assert probe.server_address[1] != 0
    finally:
        probe.server_close()
    assert port != 0  # tests always use an ephemeral local port only

# ── operational commands (Task 11C) / 运维命令 ─────────────────────────────


def _command_namespace(**overrides: Any):
    import argparse

    values = dict(config=None)
    values.update(overrides)
    return argparse.Namespace(**values)


class _DiagnosticFakeClient:
    """Fake Qwen client for health/smoke probes: validates against the
    request schema and records exactly one call per invocation.
    health/smoke 探测的 fake Qwen 客户端：按请求 schema 校验并记录调用。"""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake", generation={"temperature": 0.0}, client_version="1"
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls += 1
        if response_model.__name__ == "_SmokeResponse":
            return response_model.model_validate({"message": "smoke ok"})
        return response_model.model_validate({"status": "ok"})


class _FakeJudgeClient:
    """Duck-typed DeepSeek judge for health --live deepseek tests.
    health --live deepseek 测试的鸭子类型 judge。"""

    def __init__(self) -> None:
        self.calls = 0

    def judge_json(self, payload, *, response_model, request_meta, system_prompt=None):
        self.calls += 1
        return response_model.model_validate({"status": "ok"})


def test_run_init_creates_unique_run_and_duplicate_fails(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands.run_init import run_run_init

    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_run_init(
        _command_namespace(run_id=None, dataset="d", split="s", sample_filter=None)
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    run_id = out["run_id"]
    run_dir = tmp_path / "runs" / run_id
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "config.snapshot.json").is_file()
    assert (run_dir / "prompts.snapshot").is_dir()
    assert (run_dir / "events.jsonl").is_file()
    # explicit duplicate run id fails stably / 显式重复 run id 稳定失败
    code = run_run_init(
        _command_namespace(run_id=run_id, dataset=None, split=None, sample_filter=None)
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "FileExistsError"


def test_run_init_explicit_run_id_and_manifest_fields(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands.run_init import run_run_init

    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_run_init(
        _command_namespace(
            run_id="fixed-run", dataset="d", split="test", sample_filter="a,b"
        )
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"] == "fixed-run"
    manifest = json.loads(
        (tmp_path / "runs" / "fixed-run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_id"] == "fixed-run"
    assert manifest["dataset"] == "d"
    assert manifest["split"] == "test"
    assert manifest["sample_filter"] == "a,b"
    snapshot = json.loads(
        (tmp_path / "runs" / "fixed-run" / "config.snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["runs"]["root"] == (tmp_path / "runs").as_posix()


def test_health_qwen_metadata_ready(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.health import run_health

    monkeypatch.setenv("QWEN_MODEL", "smoke-model")
    code = run_health(_command_namespace(component="qwen", live=False))
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ready"
    assert out["component"] == "qwen"
    assert out["model"] == "smoke-model"
    assert "cache_model_id" in out
    assert "allow_download" in out
    assert "api_key" not in json.dumps(out).lower()


def test_health_deepseek_prints_env_name_not_value(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands.health import run_health

    secret_marker = "SUPER_SECRET_TEST_VALUE_123"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret_marker)
    code = run_health(_command_namespace(component="deepseek", live=False))
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ready"
    assert out["component"] == "deepseek"
    assert out["api_key_env"] == "DEEPSEEK_API_KEY"
    assert secret_marker not in json.dumps(out)
    assert "sk-" not in json.dumps(out)


def test_health_qwen_live_probes_once(tmp_path, capsys) -> None:
    from application.commands.health import run_health

    client = _DiagnosticFakeClient()
    code = run_health(
        _command_namespace(component="qwen", live=True), qwen_client=client
    )
    assert code == 0
    assert client.calls == 1  # exactly one probe / 恰好一次探测
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["probe_status"] == "ok"


def test_health_deepseek_live_probes_once(tmp_path, capsys) -> None:
    from application.commands.health import run_health

    judge = _FakeJudgeClient()
    code = run_health(
        _command_namespace(component="deepseek", live=True), judge_client=judge
    )
    assert code == 0
    assert judge.calls == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["probe_status"] == "ok"


def test_list_datasets_lists_builtins(tmp_path, capsys) -> None:
    from application.commands.list_datasets import run_list_datasets

    code = run_list_datasets(_command_namespace())
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert set(out["datasets"]) >= {
        "VRSBench",
        "LEVIR-CC",
        "MME-RealWorld",
        "XLRS-Bench",
        "XLRS-Bench-lite",
    }


def test_smoke_qwen_one_fake_request(tmp_path, capsys) -> None:
    from application.commands.smoke_qwen import run_smoke_qwen

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _DiagnosticFakeClient()
    code = run_smoke_qwen(
        _command_namespace(image=str(tmp_path / "imgs" / "img.png"), question="q?"),
        qwen_client=client,
    )
    assert code == 0
    assert client.calls == 1  # exactly one request / 恰好一次请求
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["message"] == "smoke ok"


def test_smoke_qwen_missing_image_fails_stably(tmp_path, capsys) -> None:
    from application.commands.smoke_qwen import run_smoke_qwen

    code = run_smoke_qwen(
        _command_namespace(image=str(tmp_path / "missing.png"), question="q?"),
        qwen_client=_DiagnosticFakeClient(),
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "failed"
    assert error["error"] == "FileNotFoundError"


def _make_resumable_run(
    tmp_path: Path,
    *,
    run_id: str = "fixed-run",
    dataset: str = "d",
    split: str = "test",
    sample_filter: str | None = "a,b",
    tasks: tuple[str, ...] = ("general_vqa",),
) -> Path:
    """Create a run directory that resume-run can read (manifest + snapshot +
    task namespaces). 创建 resume-run 可读的 run 目录（manifest + 快照 +
    task 命名空间）。"""
    run_dir = tmp_path / "runs" / run_id
    for task in tasks:
        (run_dir / "tasks" / task).mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "created_at": "2026-08-08T00:00:00Z",
        "git_commit": None,
        "git_dirty": None,
        "config_hash": "hash",
        "prompt_hashes": {},
        "model_ids": {"qwen": "fake"},
        "dataset": dataset,
        "split": split,
        "sample_filter": sample_filter,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run_dir / "config.snapshot.json").write_text(
        json.dumps({"paths": {"dataset_root": str(tmp_path / "data")}}),
        encoding="utf-8",
    )
    auto = tasks == ("auto",)
    request = {
        "dataset": dataset,
        "dataset_root": str(tmp_path / "data"),
        "split": split,
        "task_mode": "auto" if auto else "explicit",
        "tasks": [] if auto else list(tasks),
        "auto_task": auto,
        "sample_ids": sample_filter.split(",") if sample_filter else None,
        "limit": None,
        "start_index": 0,
        "shard_index": 0,
        "shard_count": 1,
        "sample_concurrency": 1,
        "evaluate": True,
        "judge_policy": "all",
        "judge_sample_rate": 0.5,
        "render_errors": False,
        "fail_fast": False,
    }
    (run_dir / "run_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    return run_dir


def test_resume_run_invalid_manifest_fails(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.resume_run import run_resume_run

    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_resume_run(_command_namespace(run_id="missing-run"))
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "ValueError"
    # manifest without dataset/split is insufficient / 缺 dataset/split 不足
    _make_resumable_run(tmp_path, run_id="no-dataset", dataset=None, split=None)
    code = run_resume_run(_command_namespace(run_id="no-dataset"))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
    # missing run_request.json is a stable failure / 缺失 run_request 稳定失败
    _make_resumable_run(tmp_path, run_id="no-request", dataset="d", split="test")
    (tmp_path / "runs" / "no-request" / "run_request.json").unlink()
    code = run_resume_run(_command_namespace(run_id="no-request"))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
    # corrupt run_request.json is a stable failure / 损坏 run_request 稳定失败
    bad = tmp_path / "runs" / "no-request" / "run_request.json"
    bad.write_text("{broken", encoding="utf-8")
    code = run_resume_run(_command_namespace(run_id="no-request"))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
    # run directory name must match manifest identity / 目录名必须匹配 manifest
    _make_resumable_run(tmp_path, run_id="dir-name", dataset="d", split="test")
    (tmp_path / "runs" / "dir-name" / "manifest.json").write_text(
        (tmp_path / "runs" / "dir-name" / "manifest.json")
        .read_text(encoding="utf-8")
        .replace('"run_id": "dir-name"', '"run_id": "other-id"'),
        encoding="utf-8",
    )
    code = run_resume_run(_command_namespace(run_id="dir-name"))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


def test_resume_run_delegates_reconstructed_options(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import resume_run as resume_run_module
    from application.commands.resume_run import run_resume_run

    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    _make_resumable_run(
        tmp_path,
        run_id="fixed-run",
        dataset="d",
        split="test",
        sample_filter="a,b",
        tasks=("general_vqa", "caption"),
    )
    captured: dict[str, Any] = {}

    class _FakeRuntime:
        async def run_dataset(self, options):
            captured["options"] = options
            return {}

    class _FakeRuntimeClass:
        @classmethod
        def create(cls, **kwargs):
            captured["created"] = kwargs
            return _FakeRuntime()

    monkeypatch.setattr(resume_run_module, "Runtime", _FakeRuntimeClass)
    code = run_resume_run(_command_namespace(run_id="fixed-run"))
    assert code == 0
    options = captured["options"]
    assert options.dataset == "d"
    assert options.split == "test"
    assert options.run_id == "fixed-run"
    assert options.resume is True
    assert options.auto_task is False
    assert options.tasks == ("general_vqa", "caption")  # persisted order
    assert options.sample_ids == {"a", "b"}
    assert options.root == (tmp_path / "data").resolve()
    assert options.evaluate is True
    assert options.judge_policy == "all"
    assert options.judge_sample_rate == 0.5
    assert captured["created"]["api_key"] is None


def test_resume_run_auto_namespace_reconstructed(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import resume_run as resume_run_module
    from application.commands.resume_run import run_resume_run

    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    _make_resumable_run(tmp_path, run_id="auto-run", tasks=("auto",))
    captured: dict[str, Any] = {}

    class _FakeRuntime:
        async def run_dataset(self, options):
            captured["options"] = options
            return {}

    class _FakeRuntimeClass:
        @classmethod
        def create(cls, **kwargs):
            return _FakeRuntime()

    monkeypatch.setattr(resume_run_module, "Runtime", _FakeRuntimeClass)
    code = run_resume_run(_command_namespace(run_id="auto-run"))
    assert code == 0
    options = captured["options"]
    assert options.auto_task is True
    assert options.tasks == ()


def test_inspect_data_quick_and_full(tmp_path, capsys) -> None:
    from application.commands.inspect_data import run_inspect_data

    data_root = tmp_path / "data"
    (data_root / "images").mkdir(parents=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(data_root / "images" / "a.png", format="PNG")
    code = run_inspect_data(
        _command_namespace(root=str(data_root), output=None, scan_mode="quick")
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["report"]["scan_mode"] == "quick"
    assert out["report"]["image_count"] == 1
    assert out["report"]["damaged_images"] == []
    code = run_inspect_data(
        _command_namespace(
            root=str(data_root),
            output=str(tmp_path / "audit.json"),
            scan_mode="full",
        )
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["report"]["scan_mode"] == "full"
    written = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert written["status"] == "ok"
    assert written["report"]["scan_mode"] == "full"
    # missing root fails stably / 缺失根目录稳定失败
    code = run_inspect_data(
        _command_namespace(root=str(tmp_path / "missing"), output=None, scan_mode="quick")
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "FileNotFoundError"

# ── run-dataset operational surface (Task 11C2) / run-dataset 运维面 ────────


class _TaskAwareAdapter:
    """Adapter exposing supported_tasks; records iter_samples calls.
    暴露 supported_tasks 的适配器；记录 iter_samples 调用。"""

    name = "task-demo"
    supported_tasks = frozenset({"caption", "general_vqa"})

    def __init__(self) -> None:
        self.iter_tasks: list[str] = []

    def probe(self, root, task=None):
        from data.adapters.base import AdapterProbe

        return AdapterProbe(
            dataset="task-demo",
            version="1",
            sample_file=root / "samples.jsonl",
            observed_fields=("id",),
            sample_count=0,
            task=task,
            available_tasks=("caption", "general_vqa"),
        )

    def iter_samples(self, root, split, task):
        self.iter_tasks.append(task)
        return iter([])


def test_runtime_neither_task_mode_uses_adapter_supported_tasks(tmp_path: Path) -> None:
    """tasks=None runs every adapter.supported_tasks and never touches the
    TaskResolver. tasks=None 运行全部 adapter.supported_tasks，绝不触碰
    TaskResolver。"""
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    adapter = _TaskAwareAdapter()
    registry = DatasetRegistry()
    registry.register("task-demo", lambda: adapter)
    runtime = Runtime(settings=settings, components=components, registry=registry)
    results = asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="task-demo", root=tmp_path / "data", split="test", tasks=None
            )
        )
    )
    assert set(results) == {"caption", "general_vqa"}
    assert sorted(adapter.iter_tasks) == ["caption", "general_vqa"]
    assert client.calls == 0  # adapter-default mode never calls the resolver


def _render_failed_point(point_id: str) -> dict:
    from agents.counting.schema import GlobalPointObservation

    return GlobalPointObservation(
        global_id=point_id,
        target="vehicles",
        source_tile_id="t0",
        local_id="l0",
        local_x_norm=10,
        local_y_norm=10,
        local_radius_norm=5,
        global_x_px=5,
        global_y_px=5,
        global_x_norm=100,
        global_y_norm=100,
        radius_px=5.0,
        confidence=0.9,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=True,
        short_evidence="e",
    ).model_dump(mode="json")


def test_runtime_render_errors_after_execution(tmp_path: Path) -> None:
    """render_errors renders counting overlays for failed samples and turns
    unsupported samples into stable notes; never calls a model.
    render_errors 为 failed 样本渲染计数标注图，不支持的样本转为稳定 note；
    绝不调用模型。"""
    from agents.counting.schema import CountingResult
    from workflows.dataset_runner import storage_key

    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    runtime = Runtime(settings=settings, components=components)
    run_dir = tmp_path / "runs" / "render-run"
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(data_root / "img.png", format="PNG")
    # failed counting sample with a persisted CountingResult / failed 计数样本
    count_dir = run_dir / "tasks" / "counting" / "samples" / storage_key("c1")
    count_dir.mkdir(parents=True)
    (count_dir / "sample.json").write_text(
        json.dumps(
            {
                "sample_id": "c1",
                "dataset": "d",
                "split": "t",
                "task": "counting",
                "images": [{"image_id": "i0", "path": "img.png", "role": "image"}],
                "question": "q",
                "ground_truth": None,
            }
        ),
        encoding="utf-8",
    )
    result = CountingResult(
        sample_id="c1",
        target="vehicles",
        question="q",
        source_width=10,
        source_height=10,
        tile_count=1,
        global_points=[
            {
                **{
                    key: value
                    for key, value in _render_failed_point("p1").items()
                    if key != "accepted"
                },
                "accepted": True,
            }
        ],
        warnings=[],
        final_count=1,
        status="completed_with_warnings",
    )
    (count_dir / "counting_result.json").write_text(
        json.dumps(result.model_dump(mode="json")), encoding="utf-8"
    )
    # failed caption sample without a counting result / 无计数结果的 failed 样本
    cap_dir = run_dir / "tasks" / "caption" / "samples" / storage_key("x1")
    cap_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "c1",
                        "run_task": "counting",
                        "task": "counting",
                        "status": "failed",
                        "result_path": "counting_result.json",
                        "updated_at": "now",
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "x1",
                        "run_task": "caption",
                        "task": "caption",
                        "status": "failed",
                        "result_path": "agent_result.json",
                        "updated_at": "now",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    notes = runtime.render_error_overlays(run_dir, data_root=data_root)
    assert (count_dir / "error_overlay.png").is_file()  # rendered
    by_sample = {note["sample_id"]: note for note in notes}
    assert by_sample["x1"]["note"] == "render_errors_skipped:unsupported_task"
    assert "c1" not in by_sample  # rendered successfully, no note
    assert client.calls == 0  # rendering is model-free / 渲染无模型调用
    persisted = json.loads(
        (run_dir / "render_errors_notes.json").read_text(encoding="utf-8")
    )
    assert len(persisted) == 1
    assert persisted[0]["note"] == "render_errors_skipped:unsupported_task"

# ── counting maintenance tools (Task 11D) / 计数维护工具 ────────────────────


class _FakeCountClient:
    """Fake Qwen client for the counting pipeline: target spec, tile points.
    计数管线的 fake Qwen 客户端：目标规格与切片点。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake", generation={"temperature": 0.0}, client_version="1"
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        name = response_model.__name__
        self.calls.append(name)
        if name == "CountTargetSpec":
            return response_model.model_validate(
                {
                    "canonical_label": "vehicles",
                    "aliases": ["car"],
                    "inclusion_rule": "count all vehicles",
                    "exclusion_rule": "exclude none",
                }
            )
        if name == "TileCountResponse":
            return response_model.model_validate(
                {
                    "target": "vehicles",
                    "tile_id": request_meta.tile_id,
                    "points": [
                        {
                            "local_id": "p1",
                            "x": 100,
                            "y": 100,
                            "confidence": 0.9,
                            "radius": 5,
                            "short_evidence": "e",
                        }
                    ],
                    "reported_count": 1,
                    "needs_split": False,
                }
            )
        raise AssertionError(f"unexpected model {name}")


def _count_image_runtime(tmp_path: Path, client: _FakeCountClient) -> Runtime:
    """A real Runtime whose Qwen is the fake counting client.
    以 fake 计数客户端为 Qwen 的真实 Runtime。"""
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    return Runtime(settings=settings, components=components)


def _count_image_args(
    tmp_path: Path,
    *,
    image: Path,
    question: str = "how many vehicles?",
    run_id: str = "count-run",
    render: bool = False,
    resume: bool = False,
    force: bool = False,
    no_seam_verify: bool = False,
    max_qwen_calls: int | None = None,
    max_deepseek_calls: int | None = None,
    target_spec: Path | None = None,
    evaluate: bool = False,
):
    import argparse

    return argparse.Namespace(
        config=None,
        image=str(image),
        question=question,
        target_spec=str(target_spec) if target_spec else None,
        run_id=run_id,
        evaluate=evaluate,
        render=render,
        resume=resume,
        force=force,
        no_seam_verify=no_seam_verify,
        max_qwen_calls=max_qwen_calls,
        max_deepseek_calls=max_deepseek_calls,
    )


def _count_image_sample_dir(tmp_path: Path) -> Path:
    """The single current sample directory of the count-run counting task.
    count-run 的 counting 任务下唯一的当前样本目录。"""
    samples_root = tmp_path / "runs" / "count-run" / "tasks" / "counting" / "samples"
    return next(samples_root.iterdir())


def test_count_image_produces_current_artifacts_and_overlay(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_count_image(
        _count_image_args(tmp_path, image=tmp_path / "imgs" / "img.png", render=True)
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "completed"
    assert out["final_count"] == 1
    assert out["run_id"] == "count-run"
    assert out["result_path"].startswith("tasks/counting/samples/")
    assert not Path(out["result_path"]).is_absolute()
    sample_dir = _count_image_sample_dir(tmp_path)
    names = {entry.name for entry in sample_dir.iterdir()}
    assert {"sample.json", "status.json", "routing_decision.json",
            "counting_result.json", "agent_trace.json", "overlay.png"} <= names
    status = json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "succeeded"
    assert status["result_path"] == "counting_result.json"  # plain basename
    # no host absolute path in any artifact / 产物无主机绝对路径
    assert (tmp_path / "imgs").as_posix() not in (
        sample_dir / "status.json"
    ).read_text(encoding="utf-8")
    assert client.calls == [
        "CountTargetSpec",
        "_CountProposalResult",
        "TileCountResponse",
    ]


def test_count_image_resume_succeeded_zero_qwen_calls(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    args = _count_image_args(tmp_path, image=tmp_path / "imgs" / "img.png")
    assert run_count_image(args) == 0
    capsys.readouterr()  # drop the first summary
    calls_after_first = list(client.calls)
    assert calls_after_first
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="count-run",
            resume=True,
        )
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "resumed"
    assert out["final_count"] == 1
    assert client.calls == calls_after_first  # zero new Qwen calls


def test_count_image_force_reexecutes(tmp_path, monkeypatch, capsys) -> None:
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    args = _count_image_args(tmp_path, image=tmp_path / "imgs" / "img.png")
    assert run_count_image(args) == 0
    calls_after_first = len(client.calls)
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="count-run",
            resume=True,
            force=True,
        )
    )
    assert code == 0
    assert len(client.calls) > calls_after_first  # re-executed


def test_count_image_legacy_absolute_status_triggers_rerun(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    args = _count_image_args(tmp_path, image=tmp_path / "imgs" / "img.png")
    assert run_count_image(args) == 0
    # overwrite the status with a legacy absolute result_path / 用旧版绝对路径覆盖
    sample_dir = _count_image_sample_dir(tmp_path)
    status_path = sample_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["result_path"] = r"C:\legacy\absolute\counting_result.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    calls_before = len(client.calls)
    capsys.readouterr()  # drop earlier summaries
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="count-run",
            resume=True,
        )
    )
    assert code == 0
    assert len(client.calls) > calls_before  # invalid status re-runs the sample
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_count_image_budget_validation(tmp_path, monkeypatch, capsys) -> None:
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", max_qwen_calls=0
        )
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "ValueError"
    code = run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", max_deepseek_calls=-1
        )
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


def test_count_image_seam_override_is_request_local(tmp_path, monkeypatch) -> None:
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    captured = {}

    def capturing_create(cls, **kwargs):
        captured["settings"] = kwargs["settings"]
        return _count_image_runtime(tmp_path, _FakeCountClient())

    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(capturing_create)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", no_seam_verify=True
        )
    )
    assert code == 0
    assert captured["settings"].counting.seam_verify is False
    captured.clear()
    # a second fresh invocation with the same explicit run id must fail under
    # the frozen identity contract. 相同显式 run id 的第二次 fresh 调用在
    # 冻结身份契约下必须失败。
    code = run_count_image(
        _count_image_args(tmp_path, image=tmp_path / "imgs" / "img.png")
    )
    assert code == 1
    captured.clear()
    code = run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", run_id="count-run2"
        )
    )
    assert code == 0
    assert captured["settings"].counting.seam_verify is True  # default untouched


def test_count_image_target_spec_skips_target_model_call(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "canonical_label": "vehicles",
                "inclusion_rule": "count all vehicles",
                "exclusion_rule": "exclude none",
            }
        ),
        encoding="utf-8",
    )
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", target_spec=spec
        )
    )
    assert code == 0
    assert client.calls == [
        "_CountProposalResult",
        "TileCountResponse",
    ]  # No target parser call; vehicle specialist chain remains active.
    # 不调用 target parser；vehicle 专家链仍然生效。


# ── render-count / 渲染计数 ─────────────────────────────────────────────────


def _counting_result_payload(tmp_path: Path, *, width: int = 10, height: int = 10) -> Path:
    from agents.counting.schema import CountingResult, GlobalPointObservation

    payload = CountingResult(
        sample_id="s",
        target="vehicles",
        question="q",
        source_width=width,
        source_height=height,
        tile_count=1,
        global_points=[
            GlobalPointObservation(
                global_id="p1",
                target="vehicles",
                source_tile_id="whole",
                local_id="l0",
                local_x_norm=100,
                local_y_norm=100,
                local_radius_norm=5,
                global_x_px=5,
                global_y_px=5,
                global_x_norm=500,
                global_y_norm=500,
                radius_px=5.0,
                confidence=0.9,
                ownership_valid=True,
                near_core_boundary=False,
                accepted=True,
                short_evidence="e",
            )
        ],
        warnings=[],
        final_count=1,
        status="completed_with_warnings",
    )
    path = tmp_path / "counting_result.json"
    path.write_text(json.dumps(payload.model_dump(mode="json")), encoding="utf-8")
    return path


def test_render_count_success_tile_overlay_not_available(tmp_path, capsys) -> None:
    from application.commands.render_count import run_render_count

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (10, 10), (1, 2, 3)).save(img_dir / "img.png", format="PNG")
    result_path = _counting_result_payload(tmp_path)
    output = tmp_path / "out" / "overlay.png"
    code = run_render_count(
        _command_namespace(
            image=str(tmp_path / "imgs" / "img.png"),
            result=str(result_path),
            output=str(output),
        )
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["tile_overlay"] == "not_available"
    assert output.is_file()


def test_render_count_size_mismatch_fails_stably(tmp_path, capsys) -> None:
    from application.commands.render_count import run_render_count

    _make_images(tmp_path / "imgs", ["img.png"])
    result_path = _counting_result_payload(tmp_path, width=20, height=20)
    code = run_render_count(
        _command_namespace(
            image=str(tmp_path / "imgs" / "img.png"),
            result=str(result_path),
            output=str(tmp_path / "overlay.png"),
        )
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "failed"
    assert error["error"] == "ValueError"


def test_render_count_malformed_result_fails(tmp_path, capsys) -> None:
    from application.commands.render_count import run_render_count

    _make_images(tmp_path / "imgs", ["img.png"])
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code = run_render_count(
        _command_namespace(
            image=str(tmp_path / "imgs" / "img.png"),
            result=str(bad),
            output=str(tmp_path / "overlay.png"),
        )
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "JSONDecodeError"


# ── summarize-evaluations / 评估汇总 ────────────────────────────────────────


def _make_evaluation_run(tmp_path: Path) -> Path:
    """Create a run with two general_vqa records and one counting record.
    创建一个含两条 general_vqa 与一条 counting 记录的 run。"""
    from evaluation.records import EvaluationRecord
    from evaluation.metrics.vqa import VQADeterministicMetrics
    from evaluation.metrics.counting import CountDeterministicMetrics

    run_dir = tmp_path / "runs" / "eval-run"
    vqa_dir = run_dir / "tasks" / "general_vqa" / "samples" / "aaaa"
    vqa_dir.mkdir(parents=True)
    vqa2_dir = run_dir / "tasks" / "general_vqa" / "samples" / "cccc"
    vqa2_dir.mkdir(parents=True)
    count_dir = run_dir / "tasks" / "counting" / "samples" / "bbbb"
    count_dir.mkdir(parents=True)
    records = [
        EvaluationRecord(
            sample_id="a1",
            task="general_vqa",
            deterministic_metrics=VQADeterministicMetrics(exact_match=True),
            judge_status="not_requested",
        ),
        EvaluationRecord(
            sample_id="a2",
            task="general_vqa",
            deterministic_metrics=VQADeterministicMetrics(exact_match=False),
            judge_status="not_requested",
        ),
        EvaluationRecord(
            sample_id="c1",
            task="counting",
            deterministic_metrics=CountDeterministicMetrics(
                predicted_count=3,
                gold_count=3,
                exact_match=1,
                absolute_error=0,
                relative_error=0.0,
                smooth_error_score=1.0,
            ),
            judge_status="not_requested",
        ),
    ]
    (vqa_dir / "vqa_evaluation.json").write_text(
        json.dumps(records[0].model_dump(mode="json")), encoding="utf-8"
    )
    (vqa2_dir / "vqa_evaluation.json").write_text(
        json.dumps(records[1].model_dump(mode="json")), encoding="utf-8"
    )
    (count_dir / "counting_evaluation.json").write_text(
        json.dumps(records[2].model_dump(mode="json")), encoding="utf-8"
    )
    return run_dir


def test_summarize_evaluations_aggregates_deterministically(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands.summarize_evaluations import run_summarize_evaluations

    _make_evaluation_run(tmp_path)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_summarize_evaluations(_command_namespace(run_id="eval-run", input=None, output=None))
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["record_count"] == 3
    assert set(out["aggregates"]) == {"counting", "general_vqa"}
    assert out["aggregates"]["general_vqa"]["score"] == 0.5
    assert out["aggregates"]["counting"]["exact_match_accuracy"] == 1.0


def test_summarize_evaluations_malformed_fails_stably(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands.summarize_evaluations import run_summarize_evaluations

    run_dir = _make_evaluation_run(tmp_path)
    bad = (
        run_dir / "tasks" / "general_vqa" / "samples" / "aaaa" / "vqa_evaluation.json"
    )
    bad.write_text("{broken", encoding="utf-8")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_summarize_evaluations(_command_namespace(run_id="eval-run", input=None, output=None))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


def test_summarize_evaluations_empty_or_missing_fails(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands.summarize_evaluations import run_summarize_evaluations

    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_summarize_evaluations(_command_namespace(run_id="missing-run", input=None, output=None))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
    (tmp_path / "runs" / "empty-run" / "tasks").mkdir(parents=True)
    code = run_summarize_evaluations(_command_namespace(run_id="empty-run", input=None, output=None))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"

# ── offline evaluation operations (Task 11E) / 离线评估运维 ─────────────────


class _BoomCreateModel:
    """Spy: any Qwen construction through models.entry fails the test.
    spy：任何经 models.entry 的 Qwen 构造都使测试失败。"""

    def __call__(self, *args, **kwargs):
        raise AssertionError("Qwen must never be constructed in offline tools")

    @staticmethod
    def arm(monkeypatch) -> None:
        import models.entry as entry_module

        monkeypatch.setattr(entry_module, "create_model", _BoomCreateModel())


class _OfflineFakeJudgeClient:
    """Duck-typed DeepSeek judge client for offline judge tests.
    离线 judge 测试的鸭子类型 DeepSeek judge 客户端。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def judge_json(self, payload, *, response_model, request_meta, system_prompt=None):
        self.calls.append(request_meta.request_id)
        if self.fail:
            raise RuntimeError("judge transport boom")
        if response_model.__name__ == "DeepSeekJudgeResult":
            return response_model.model_validate(
                {
                    "judge_scope": "text_and_structured_evidence_only",
                    "can_verify_visual_truth": False,
                    "semantic_correctness": 1.0,
                    "answer_evidence_consistency": 1.0,
                    "constraint_following": 1.0,
                    "clarity": 1.0,
                    "verdict": "correct",
                    "concise_rationale": "ok",
                }
            )
        return response_model.model_validate(
            {"score": 1, "concise_rationale": "ok"}
        )

    def judge(self, payload, *, request_meta):
        # JudgeService.judge_counting uses client.judge (counting result
        # schema). JudgeService.judge_counting 使用 client.judge（计数结果
        # schema）。
        from evaluation.judges.base import DeepSeekJudgeResult

        return self.judge_json(
            payload,
            response_model=DeepSeekJudgeResult,
            request_meta=request_meta,
        )


def _make_offline_run(
    tmp_path: Path,
    samples: list[dict],
    *,
    run_id: str = "offline-run",
) -> Path:
    """Create a run with a valid manifest and one sample per entry; each
    entry carries sample.json/status.json (execution task) and an optional
    payload file plus optional pre-existing evaluation.
    创建带合法 manifest 的 run，每个条目一个样本；条目携带 sample.json/
    status.json（执行任务）与可选载荷文件及可选既有评估。"""
    run_dir = tmp_path / "runs" / run_id
    (run_dir / "tasks").mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "created_at": "2026-08-09T00:00:00Z",
        "git_commit": None,
        "git_dirty": None,
        "config_hash": "hash",
        "prompt_hashes": {},
        "model_ids": {"qwen": "fake"},
        "dataset": "d",
        "split": "test",
        "sample_filter": None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "config.snapshot.json").write_text(
        json.dumps({"paths": {"dataset_root": "data"}}), encoding="utf-8"
    )
    from workflows.dataset_runner import storage_key as storage_key_fn

    for entry in samples:
        task = entry.get("task") or entry["sample"]["task"]
        sample_dir = (
            run_dir
            / "tasks"
            / task
            / "samples"
            / storage_key_fn(entry["sample_id"])
        )
        sample_dir.mkdir(parents=True)
        (sample_dir / "sample.json").write_text(
            json.dumps(entry["sample"]), encoding="utf-8"
        )
        status = {
            "sample_id": entry["sample_id"],
            "task": entry.get("execution_task", task),
            "state": "succeeded",
            "error_code": None,
            "error_message": None,
            "result_path": entry.get("result_path", "agent_result.json"),
            "updated_at": "2026-08-09T00:00:00Z",
        }
        (sample_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        if entry.get("payload_file"):
            (sample_dir / entry["payload_file"]).write_text(
                json.dumps(entry["payload"]), encoding="utf-8"
            )
        if entry.get("evaluation"):
            (sample_dir / entry["evaluation_file"]).write_text(
                json.dumps(entry["evaluation"]), encoding="utf-8"
            )
        # execution-index row so build_report can read the sample
        # 执行索引行，使 build_report 可读取该样本
        (run_dir / "predictions.jsonl").open("a", encoding="utf-8").write(
            json.dumps(
                {
                    "sample_id": entry["sample_id"],
                    "run_task": task,
                    "task": entry.get("execution_task", task),
                    "status": "succeeded",
                    "result_path": entry.get("result_path", "agent_result.json"),
                    "updated_at": "2026-08-09T00:00:00Z",
                }
            )
            + chr(10)
        )
    return run_dir


def _offline_args(**overrides):
    """Complete argparse Namespace for the offline evaluation commands.
    evaluate-run/judge-vqa-run 命令的完整 argparse Namespace。"""
    import argparse

    values = dict(
        config=None,
        run_id="offline-run",
        deepseek=False,
        only_missing=False,
        force_judge=False,
        force=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _offline_vqa_sample(sample_id: str, *, answer: str = "yes") -> dict:
    return {
        "sample_id": sample_id,
        "sample": {
            "sample_id": sample_id,
            "dataset": "d",
            "split": "test",
            "task": "general_vqa",
            "images": [{"image_id": "i0", "path": "img.png", "role": "image"}],
            "question": "Is there a road?",
            "ground_truth": {"answers": ["yes"]},
        },
    }


def _offline_counting_sample(sample_id: str, *, final_count: int = 0) -> dict:
    from agents.counting.schema import CountingResult

    payload = CountingResult(
        sample_id=sample_id,
        target="vehicles",
        question="how many?",
        source_width=10,
        source_height=10,
        tile_count=1,
        global_points=[],
        warnings=[],
        final_count=final_count,
        status="completed",
    )
    return {
        "sample_id": sample_id,
        "sample": {
            "sample_id": sample_id,
            "dataset": "d",
            "split": "test",
            "task": "counting",
            "images": [{"image_id": "i0", "path": "img.png", "role": "image"}],
            "question": "how many?",
            "ground_truth": {"count": final_count},
        },
        "payload_file": "counting_result.json",
        "payload": payload.model_dump(mode="json"),
        "result_path": "counting_result.json",
    }


def _offline_caption_sample(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "sample": {
            "sample_id": sample_id,
            "dataset": "d",
            "split": "test",
            "task": "caption",
            "images": [{"image_id": "i0", "path": "img.png", "role": "image"}],
            "question": "",
            "ground_truth": None,
        },
        "payload_file": "agent_result.json",
        "payload": {
            "agent_name": "caption_agent",
            "answer": "a street",
            "status": "completed",
        },
    }


def _offline_grounding_sample(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "sample": {
            "sample_id": sample_id,
            "dataset": "d",
            "split": "test",
            "task": "grounding",
            "images": [{"image_id": "i0", "path": "img.png", "role": "image"}],
            "question": "Where is the road?",
            "ground_truth": None,
        },
        "payload_file": "agent_result.json",
        "payload": {
            "agent_name": "grounding_agent",
            "answer": "road",
            "status": "completed",
        },
    }


def test_evaluate_run_fills_missing_deterministic_zero_qwen(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    _make_offline_run(
        tmp_path,
        [
            _offline_counting_sample("c1"),
            {**_offline_vqa_sample("v1"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"}}},
            _offline_caption_sample("cap1"),
            _offline_grounding_sample("g1"),
        ],
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_evaluate_run(_offline_args())
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    by_sample = {item["sample_id"]: item for item in out["evaluated"]}
    assert set(by_sample) == {"c1", "v1"}
    assert by_sample["c1"]["filename"] == "counting_evaluation.json"
    assert by_sample["v1"]["filename"] == "vqa_evaluation.json"
    # incompatible grounding and reference-less caption: not_applicable,
    # never a fake metric file. / 不兼容 grounding 与无参考 caption：
    # not_applicable，绝不伪造指标文件。
    na = {item["sample_id"]: item for item in out["not_applicable"]}
    assert set(na) == {"cap1", "g1"}
    assert na["g1"]["reason"] == "incompatible_geometry_or_no_reference"
    run_dir = tmp_path / "runs" / "offline-run"
    counting_dir = next(
        (run_dir / "tasks" / "counting" / "samples").iterdir()
    )
    vqa_dir = next(
        (run_dir / "tasks" / "general_vqa" / "samples").iterdir()
    )
    assert (counting_dir / "counting_evaluation.json").is_file()
    assert (vqa_dir / "vqa_evaluation.json").is_file()
    grounding_dir = next(
        (run_dir / "tasks" / "grounding" / "samples").iterdir()
    )
    assert not (grounding_dir / "grounding_evaluation.json").exists()
    # the refreshed unified report is built / 刷新的统一报告已构建
    assert "report" in out
    assert set(out["report"]) >= {"total", "succeeded", "failed"}


def test_evaluate_run_e2_families_zero_qwen(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    samples = [
        {
            "sample_id": "change-qa",
            "sample": {
                "sample_id": "change-qa",
                "dataset": "d",
                "split": "test",
                "task": "change_qa",
                "images": [
                    {"image_id": "i0", "path": "t1.png", "role": "t1"},
                    {"image_id": "i1", "path": "t2.png", "role": "t2"},
                ],
                "question": "What changed?",
                "ground_truth": {"answers": ["road added"]},
            },
            "payload_file": "agent_result.json",
            "payload": {
                "agent_name": "change_agent",
                "answer": "road added",
                "status": "completed",
            },
        },
        {
            "sample_id": "spatial",
            "sample": {
                "sample_id": "spatial",
                "dataset": "d",
                "split": "test",
                "task": "spatial_relation",
                "images": [
                    {"image_id": "i0", "path": "img.png", "role": "image"}
                ],
                "question": "Where is A relative to B?",
                "ground_truth": {"answers": ["north"]},
            },
            "payload_file": "agent_result.json",
            "payload": {
                "agent_name": "general_vqa_agent",
                "answer": "north",
                "status": "completed",
            },
        },
        {
            "sample_id": "change-caption",
            "sample": {
                "sample_id": "change-caption",
                "dataset": "d",
                "split": "test",
                "task": "change_caption",
                "images": [
                    {"image_id": "i0", "path": "t1.png", "role": "t1"},
                    {"image_id": "i1", "path": "t2.png", "role": "t2"},
                ],
                "question": "",
                "ground_truth": {"answers": ["road added"]},
            },
            "payload_file": "agent_result.json",
            "payload": {
                "agent_name": "change_agent",
                "answer": "road added",
                "status": "completed",
            },
        },
    ]
    _make_offline_run(tmp_path, samples)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    assert run_evaluate_run(_offline_args()) == 0
    output = json.loads(capsys.readouterr().out)
    evaluated = {item["sample_id"]: item for item in output["evaluated"]}
    assert evaluated["change-qa"]["filename"] == "vqa_evaluation.json"
    assert evaluated["spatial"]["filename"] == "vqa_evaluation.json"
    assert evaluated["change-caption"]["filename"] == "caption_evaluation.json"


def test_evaluate_run_only_missing_skips_existing(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    from evaluation.records import EvaluationRecord
    from evaluation.metrics.vqa import VQADeterministicMetrics

    existing = EvaluationRecord(
        sample_id="v1",
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=True),
        judge_status="not_requested",
    )
    _make_offline_run(
        tmp_path,
        [
            {**_offline_vqa_sample("v1"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"},
                                             "evaluation": existing.model_dump(mode="json"),
                                             "evaluation_file": "vqa_evaluation.json"}},
            _offline_counting_sample("c1"),
        ],
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_evaluate_run(
        _offline_args(only_missing=True)
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    evaluated = {item["sample_id"] for item in out["evaluated"]}
    assert evaluated == {"c1"}  # v1 already has its evaluation / v1 已有评估
    # full mode re-evaluates everything / 全量模式重评估全部
    code = run_evaluate_run(_offline_args())
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert {item["sample_id"] for item in out["evaluated"]} == {"c1", "v1"}


def test_evaluate_run_fallback_execution_task_respected(
    tmp_path, monkeypatch, capsys
) -> None:
    """After a candidate fallback the status.task (execution task) decides the
    metric family, never the canonical resolved sample.task.
    候选兜底后由 status.task（执行任务）决定指标族，绝不按 canonical
    resolved sample.task。"""
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    entry = _offline_vqa_sample("fb1")
    entry["sample"]["task"] = "caption"  # canonical resolved task / 解析任务
    entry["execution_task"] = "general_vqa"  # executed task / 执行任务
    entry["payload_file"] = "agent_result.json"
    entry["payload"] = {
        "agent_name": "general_vqa_agent",
        "answer": "yes",
        "status": "completed",
    }
    _make_offline_run(tmp_path, [entry])
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_evaluate_run(_offline_args())
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    evaluated = {item["sample_id"]: item for item in out["evaluated"]}
    assert evaluated["fb1"]["filename"] == "vqa_evaluation.json"
    # the sample lives under its storage task namespace (caption) while the
    # evaluation family follows the execution task (general_vqa).
    # 样本位于存储任务命名空间（caption），指标族跟随执行任务（general_vqa）。
    caption_dir = next(
        (tmp_path / "runs" / "offline-run" / "tasks" / "caption" / "samples").iterdir()
    )
    assert (caption_dir / "vqa_evaluation.json").is_file()


def test_evaluate_run_deepseek_skip_and_force_judge(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import evaluate_run as evaluate_run_module
    from application.commands.evaluate_run import run_evaluate_run
    from evaluation.records import EvaluationRecord
    from evaluation.metrics.vqa import VQADeterministicMetrics

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        evaluate_run_module,
        "DeepSeekJudgeClient",
        lambda *a, **k: judge,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    existing = EvaluationRecord(
        sample_id="v1",
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=False),
        judge_status="succeeded",
    )
    _make_offline_run(
        tmp_path,
        [
            {**_offline_vqa_sample("v1"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "no", "status": "completed"},
                                             "evaluation": existing.model_dump(mode="json"),
                                             "evaluation_file": "vqa_evaluation.json"}},
            _offline_counting_sample("c1"),
        ],
    )
    code = run_evaluate_run(
        _offline_args(deepseek=True)
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    judge_results = {item["sample_id"]: item for item in out["judge"]}
    assert judge_results["v1"]["status"] == "skipped_succeeded"
    # the counting sample is judged (Fix H); the succeeded VQA judge is not.
    # counting 样本被 judge（Fix H）；succeeded 的 VQA judge 不被重判。
    assert judge.calls == ["c1:deepseek"]
    # force judge re-judges and writes the record / --force-judge 强制重判并写记录
    code = run_evaluate_run(
        _offline_args(deepseek=True, force_judge=True)
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    judge_results = {item["sample_id"]: item for item in out["judge"]}
    assert judge_results["v1"]["judge_status"] == "succeeded"
    assert judge.calls == [
        "c1:deepseek",
        "c1:deepseek",
        "v1:deepseek-vqa",
    ]  # counting re-judge + forced VQA re-judge
    vqa_dir = next(
        (tmp_path / "runs" / "offline-run" / "tasks" / "general_vqa" / "samples").iterdir()
    )
    evaluation = json.loads(
        (vqa_dir / "vqa_evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["judge_status"] == "succeeded"
    assert evaluation["deterministic_metrics"]["exact_match"] is False


def test_evaluate_run_judge_failure_preserves_deterministic(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import evaluate_run as evaluate_run_module
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    failing = _OfflineFakeJudgeClient(fail=True)
    monkeypatch.setattr(
        evaluate_run_module,
        "DeepSeekJudgeClient",
        lambda *a, **k: failing,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    _make_offline_run(
        tmp_path,
        [
            {**_offline_vqa_sample("v1"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "no", "status": "completed"}}},
        ],
    )
    code = run_evaluate_run(
        _offline_args(deepseek=True)
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    judge_results = {item["sample_id"]: item for item in out["judge"]}
    assert judge_results["v1"]["judge_status"] == "failed"
    assert judge_results["v1"]["judge_error"] == "RuntimeError"
    vqa_dir = next(
        (tmp_path / "runs" / "offline-run" / "tasks" / "general_vqa" / "samples").iterdir()
    )
    evaluation = json.loads(
        (vqa_dir / "vqa_evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["judge_status"] == "failed"
    assert evaluation["deterministic_metrics"]["exact_match"] is False  # preserved


def test_evaluate_run_exact_vqa_never_calls_deepseek_even_when_forced(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import evaluate_run as evaluate_run_module
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        evaluate_run_module, "DeepSeekJudgeClient", lambda *a, **k: judge
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    _make_offline_run(
        tmp_path,
        [
            {**_offline_vqa_sample("v1"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"}}},
        ],
    )
    code = run_evaluate_run(_offline_args(deepseek=True, force_judge=True))
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["judge"] == [{"sample_id": "v1", "status": "skipped_exact"}]
    assert judge.calls == []


def test_evaluate_run_deepseek_covers_every_runtime_vqa_family(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import evaluate_run as evaluate_run_module
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        evaluate_run_module, "DeepSeekJudgeClient", lambda *a, **k: judge
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    entries = []
    for task, agent_name in (
        ("general_vqa", "general_vqa_agent"),
        ("multiple_choice_vqa", "general_vqa_agent"),
        ("scene_classification", "general_vqa_agent"),
        ("spatial_relation", "general_vqa_agent"),
        ("change_qa", "change_agent"),
    ):
        sample_id = f"evaluate-{task}"
        entry = _offline_vqa_sample(sample_id)
        entry["sample"]["task"] = task
        entry["execution_task"] = task
        entry["payload_file"] = "agent_result.json"
        entry["payload"] = {
            "agent_name": agent_name,
            "answer": "no",
            "status": "completed",
        }
        if task == "change_qa":
            entry["sample"]["images"] = [
                {"image_id": "i0", "path": "t1.png", "role": "t1"},
                {"image_id": "i1", "path": "t2.png", "role": "t2"},
            ]
        entries.append(entry)
    _make_offline_run(tmp_path, entries)
    code = run_evaluate_run(_offline_args(deepseek=True))
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert {item["sample_id"] for item in out["judge"]} == {
        entry["sample_id"] for entry in entries
    }
    assert len(judge.calls) == 5


def test_evaluate_run_missing_deepseek_key_fails(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    _make_offline_run(tmp_path, [_offline_counting_sample("c1")])
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    code = run_evaluate_run(_offline_args(deepseek=True))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


def test_judge_vqa_run_skip_and_force(tmp_path, monkeypatch, capsys) -> None:
    from application.commands import judge_vqa_run as judge_run_module
    from application.commands.judge_vqa_run import run_judge_vqa_run
    from evaluation.records import EvaluationRecord
    from evaluation.metrics.vqa import VQADeterministicMetrics

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        judge_run_module, "DeepSeekJudgeClient", lambda *a, **k: judge
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    existing = EvaluationRecord(
        sample_id="v1",
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=False),
        judge_status="succeeded",
    )
    exact = EvaluationRecord(
        sample_id="v2",
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=True),
        judge_status="not_requested",
    )
    _make_offline_run(
        tmp_path,
        [
            {**_offline_vqa_sample("v1"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "no", "status": "completed"},
                                             "evaluation": existing.model_dump(mode="json"),
                                             "evaluation_file": "vqa_evaluation.json"}},
            {**_offline_vqa_sample("v2"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"},
                                             "evaluation": exact.model_dump(mode="json"),
                                             "evaluation_file": "vqa_evaluation.json"}},
            _offline_counting_sample("c1"),
        ],
    )
    code = run_judge_vqa_run(_offline_args())
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    judged = {item["sample_id"]: item for item in out["judged"]}
    assert judged["v1"]["status"] == "skipped_succeeded"
    assert judged["v2"]["status"] == "skipped_exact"
    assert "c1" not in judged  # only execution task general_vqa
    assert not judge.calls
    code = run_judge_vqa_run(
        _offline_args(force=True)
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    judged = {item["sample_id"]: item for item in out["judged"]}
    assert judged["v1"]["judge_status"] == "succeeded"
    assert judged["v2"]["status"] == "skipped_exact"
    assert len(judge.calls) == 1


def test_judge_vqa_run_covers_every_runtime_vqa_family(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import judge_vqa_run as judge_run_module
    from application.commands.judge_vqa_run import run_judge_vqa_run

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        judge_run_module, "DeepSeekJudgeClient", lambda *a, **k: judge
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    entries = []
    for task, agent_name in (
        ("general_vqa", "general_vqa_agent"),
        ("multiple_choice_vqa", "general_vqa_agent"),
        ("scene_classification", "general_vqa_agent"),
        ("spatial_relation", "general_vqa_agent"),
        ("change_qa", "change_agent"),
    ):
        sample_id = f"family-{task}"
        entry = _offline_vqa_sample(sample_id)
        entry["sample"]["task"] = task
        entry["execution_task"] = task
        entry["payload_file"] = "agent_result.json"
        entry["payload"] = {
            "agent_name": agent_name,
            "answer": "no",
            "status": "completed",
        }
        if task == "change_qa":
            entry["sample"]["images"] = [
                {"image_id": "i0", "path": "t1.png", "role": "t1"},
                {"image_id": "i1", "path": "t2.png", "role": "t2"},
            ]
        entries.append(entry)
    _make_offline_run(tmp_path, entries)
    code = run_judge_vqa_run(_offline_args())
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert {item["sample_id"] for item in out["judged"]} == {
        entry["sample_id"] for entry in entries
    }
    assert len(judge.calls) == 5


def test_judge_vqa_run_failure_records_stable_error(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands import judge_vqa_run as judge_run_module
    from application.commands.judge_vqa_run import run_judge_vqa_run

    _BoomCreateModel.arm(monkeypatch)
    failing = _OfflineFakeJudgeClient(fail=True)
    monkeypatch.setattr(
        judge_run_module, "DeepSeekJudgeClient", lambda *a, **k: failing
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    _make_offline_run(
        tmp_path,
        [
            {**_offline_vqa_sample("v1"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "no", "status": "completed"}}},
        ],
    )
    code = run_judge_vqa_run(_offline_args())
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    judged = {item["sample_id"]: item for item in out["judged"]}
    assert judged["v1"]["judge_status"] == "failed"
    assert judged["v1"]["judge_error"] == "RuntimeError"
    vqa_dir = next(
        (tmp_path / "runs" / "offline-run" / "tasks" / "general_vqa" / "samples").iterdir()
    )
    evaluation = json.loads(
        (vqa_dir / "vqa_evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["judge_status"] == "failed"
    assert evaluation["deterministic_metrics"]["exact_match"] is False


def test_offline_commands_missing_run_fails(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.evaluate_run import run_evaluate_run
    from application.commands.judge_vqa_run import run_judge_vqa_run

    _BoomCreateModel.arm(monkeypatch)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_evaluate_run(_offline_args(run_id="nope"))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
    code = run_judge_vqa_run(_offline_args(run_id="nope"))
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"

import sys

# ── standard & dataset evaluation seams (Task 11F) / 标准与数据集评估 seam ──


def _fake_standard_tool(
    tmp_path: Path,
    *,
    body: str,
    exit_code: int = 0,
    name: str = "eval_standard",
) -> Path:
    """Create a fake standard evaluator tool directory.
    创建 fake 标准评估器工具目录。"""
    tool_dir = tmp_path / name
    tool_dir.mkdir(parents=True)
    (tool_dir / "evaluate.py").write_text(body, encoding="utf-8")
    return tool_dir


_FAKE_EVALUATOR_OK = """import argparse, json
p = argparse.ArgumentParser()
p.add_argument('input')
p.add_argument('--output', required=True)
a = p.parse_args()
json.dump({'primary_metric': 'open_vqa_accuracy', 'primary_value': 0.75, 'score': 75.0}, open(a.output, 'w'))
"""


def test_standard_adapter_success_and_default_report_path(tmp_path) -> None:
    from evaluation.standard.adapter import (
        default_standard_report_path,
        run_standard_evaluation,
    )

    result = tmp_path / "predictions.jsonl"
    result.write_text('{"sample": {}, "prediction": {}}\n', encoding="utf-8")
    tool_dir = _fake_standard_tool(tmp_path, body=_FAKE_EVALUATOR_OK)
    report = run_standard_evaluation(
        result, tool_dir=tool_dir, python_executable=sys.executable
    )
    assert report["score"] == 75.0
    assert report["primary_metric"] == "open_vqa_accuracy"
    assert default_standard_report_path(result) == result.with_suffix(".standard.json")
    assert default_standard_report_path(result).is_file()


def test_standard_adapter_shell_false_source_read_only_and_offline(
    tmp_path,
    monkeypatch,
) -> None:
    import subprocess

    import evaluation.standard.adapter as adapter

    result = tmp_path / "predictions.jsonl"
    result.write_text('{"sample": {}, "prediction": {}}\n', encoding="utf-8")
    source_before = result.read_bytes()
    tool_dir = _fake_standard_tool(tmp_path, body=_FAKE_EVALUATOR_OK)
    real_run = subprocess.run
    observed: dict[str, object] = {}

    def _spy_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return real_run(command, **kwargs)

    monkeypatch.setattr(adapter.subprocess, "run", _spy_run)
    report = adapter.run_standard_evaluation(
        result,
        tool_dir=tool_dir,
        python_executable=sys.executable,
    )
    assert report["score"] == 75.0
    assert observed["shell"] is False
    assert isinstance(observed["command"], list)
    assert result.read_bytes() == source_before

    source = (REPO_ROOT / "evaluation/standard/adapter.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "urlopen",
        "requests",
        "httpx",
        "huggingface_hub",
        "http://",
        "https://",
    ):
        assert token not in source, token


def test_standard_adapter_nonzero_exit_fails(tmp_path) -> None:
    from evaluation.standard.adapter import run_standard_evaluation

    result = tmp_path / "predictions.jsonl"
    result.write_text("{}\n", encoding="utf-8")
    tool_dir = _fake_standard_tool(
        tmp_path, body="import sys; sys.exit(3)"
    )
    with pytest.raises(RuntimeError, match="nonzero exit"):
        run_standard_evaluation(result, tool_dir=tool_dir, python_executable=sys.executable)


def test_standard_adapter_missing_entry_or_result_fails(tmp_path) -> None:
    from evaluation.standard.adapter import run_standard_evaluation

    result = tmp_path / "predictions.jsonl"
    with pytest.raises(FileNotFoundError):
        run_standard_evaluation(
            result, tool_dir=tmp_path / "missing-tool", python_executable=sys.executable
        )
    result.write_text("{}\n", encoding="utf-8")
    tool_dir = _fake_standard_tool(
        tmp_path, body="import json; json.dump({'ok': True}, open('nowhere.json', 'w'))"
    )
    with pytest.raises(FileNotFoundError, match="entry point"):
        run_standard_evaluation(result, tool_dir=tmp_path / "no-tool", python_executable=sys.executable)
    with pytest.raises(RuntimeError, match="did not create"):
        run_standard_evaluation(result, tool_dir=tool_dir, python_executable=sys.executable)


def test_standard_adapter_no_output_and_invalid_reports_fail(tmp_path) -> None:
    from evaluation.standard.adapter import run_standard_evaluation

    result = tmp_path / "predictions.jsonl"
    result.write_text("{}\n", encoding="utf-8")
    tool_dir = _fake_standard_tool(
        tmp_path, body="import argparse, json\np = argparse.ArgumentParser()\np.add_argument('input')\np.add_argument('--output', required=True)\na = p.parse_args()\nopen(a.output, 'w').write('{not json')\n"
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        run_standard_evaluation(result, tool_dir=tool_dir, python_executable=sys.executable)
    tool_dir2 = _fake_standard_tool(
        tmp_path,
        body="import argparse, json\np = argparse.ArgumentParser()\np.add_argument('input')\np.add_argument('--output', required=True)\na = p.parse_args()\njson.dump([1, 2], open(a.output, 'w'))\n",
        name="eval_list",
    )
    with pytest.raises(ValueError, match="JSON object"):
        run_standard_evaluation(result, tool_dir=tool_dir2, python_executable=sys.executable)


def test_standard_adapter_spaces_and_cjk_paths(tmp_path) -> None:
    from evaluation.standard.adapter import run_standard_evaluation

    spaced = tmp_path / "带 空格 dir"
    result = spaced / "预测 results.jsonl"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n", encoding="utf-8")
    tool_dir = _fake_standard_tool(spaced, body=_FAKE_EVALUATOR_OK, name="标准 工具")
    report = run_standard_evaluation(
        result, tool_dir=tool_dir, python_executable=sys.executable
    )
    assert report["score"] == 75.0
    assert result.with_suffix(".standard.json").is_file()


def test_standard_evaluate_cli_prints_report_and_run_id(
    tmp_path, monkeypatch, capsys
) -> None:
    from application.commands.standard_evaluate import run_standard_evaluate

    # result inside the runs root -> run id association / 结果在 runs root 内 → 关联 run
    run_dir = tmp_path / "runs" / "std-run"
    run_dir.mkdir(parents=True)
    result = run_dir / "predictions.jsonl"
    result.write_text("{}\n", encoding="utf-8")
    tool_dir = _fake_standard_tool(tmp_path, body=_FAKE_EVALUATOR_OK)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_standard_evaluate(
        _command_namespace(
            result=str(result),
            tool_dir=str(tool_dir),
            output=None,
            python=sys.executable,
        )
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["report"]["score"] == 75.0
    assert out["run_id"] == "std-run"
    # result outside the runs root -> no run association / 结果在 runs root 外 → 无关联
    outside = tmp_path / "elsewhere.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    code = run_standard_evaluate(
        _command_namespace(
            result=str(outside),
            tool_dir=str(tool_dir),
            output=None,
            python=sys.executable,
        )
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"] is None


def test_standard_evaluate_cli_failure_stable(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.standard_evaluate import run_standard_evaluate

    result = tmp_path / "missing.jsonl"
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_standard_evaluate(
        _command_namespace(
            result=str(result), tool_dir=str(tmp_path), output=None, python=sys.executable
        )
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "FileNotFoundError"


# ── VRSBench evaluation seam / VRSBench 评估 seam ───────────────────────────


def test_vrsbench_normalize_answer_closed_vocabulary() -> None:
    from evaluation.datasets.vrsbench import VRSBENCH_CLOSED_VOCABULARY, normalize_answer

    vocab = VRSBENCH_CLOSED_VOCABULARY["existence"]
    assert normalize_answer("Yes", vocab) == "yes"
    assert normalize_answer("  NO ", vocab) == "no"
    assert normalize_answer("maybe", vocab) == "maybe"  # verbatim, never guessed
    grid = VRSBENCH_CLOSED_VOCABULARY["grid_position"]
    assert normalize_answer("Top-Left", grid) == "top-left"
    assert normalize_answer("center", grid) == "center"


def test_vrsbench_official_input_deterministic_export() -> None:
    from evaluation.datasets.vrsbench import (
        export_official_input,
        to_official_evaluator_input,
    )

    row = to_official_evaluator_input(
        question="Is there a car?",
        references=["yes", "Yes"],
        candidate_answer="yes",
        question_id="q1",
    )
    assert row["version"] == "vrsbench-official-eval-v1"
    assert row["question_id"] == "q1"
    assert row["references"] == ["yes", "Yes"]
    assert row["candidate_answer"] == "yes"
    second = to_official_evaluator_input(
        question="Is there a car?",
        references=["yes", "Yes"],
        candidate_answer="yes",
        question_id="q1",
    )
    assert second == row  # deterministic
    exported = export_official_input([row, second])
    assert exported == [row, second]
    assert exported[0] is not row  # copied, never aliased


def test_vrsbench_seam_no_model_or_routing_imports() -> None:
    """The evaluation seam must not import models, agents, or routing, and
    must not duplicate task classification. 评估 seam 绝不 import models/
    agents/routing，绝不重复任务分类。"""
    import ast

    for path in (
        "evaluation/datasets/vrsbench.py",
        "evaluation/standard/adapter.py",
    ):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in {"models", "agents", "routing"}, (
                        f"{path} must not import {top}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert top not in {"models", "agents", "routing"}, (
                    f"{path} must not import {top}"
                )
    # no task classification logic lives here / 这里没有任务分类逻辑
    text = Path("evaluation/datasets/vrsbench.py").read_text(encoding="utf-8")
    assert "normalize_task" not in text
    assert "classify_question_subtype" not in text

# ── 11G.5 functional restoration hardening / 功能恢复硬化 ───────────────────


def test_run_request_persists_actual_root_and_resume_reconstructs(
    tmp_path, monkeypatch
) -> None:
    """Fix A: the actual options root (B) survives in run_request.json even
    when the config snapshot root (A) differs; resume reconstructs B.
    Fix A：即使配置快照根（A）不同，实际 options root（B）也保存在
    run_request.json；resume 重建 B。"""
    from application.commands import resume_run as resume_run_module
    from application.commands.resume_run import run_resume_run

    data_root = tmp_path / "data"
    _make_dataset(data_root)
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = DatasetRegistry()
    registry.register(
        "auto-demo", lambda: ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    )
    runtime = Runtime(settings=settings, components=components, registry=registry)
    results = asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo",
                root=data_root,
                split="test",
                tasks=(),
                auto_task=True,
                run_id="root-run",
                evaluate=True,
                judge_policy="all",
                judge_sample_rate=0.5,
            )
        )
    )
    assert results["auto"].succeeded == 1
    request = json.loads(
        (tmp_path / "runs" / "root-run" / "run_request.json").read_text(encoding="utf-8")
    )
    # B = actual options root; A (config default) differs from B.
    # B = 实际 options root；A（配置默认）与 B 不同。
    assert request["dataset_root"] == data_root.as_posix().replace("\\", "/")
    assert request["dataset_root"] != settings.paths.dataset_root.as_posix()
    assert request["task_mode"] == "auto"
    assert request["tasks"] == []
    assert request["auto_task"] is True
    assert request["evaluate"] is True
    assert request["judge_policy"] == "all"
    assert request["judge_sample_rate"] == 0.5
    # resume reconstructs the actual invocation / resume 重建实际调用
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    captured = {}

    class _FakeRuntime:
        async def run_dataset(self, options):
            captured["options"] = options
            return {}

    monkeypatch.setattr(
        resume_run_module, "Runtime", type(
            "_FakeRuntimeClass", (), {"create": classmethod(lambda cls, **kw: _FakeRuntime())}
        )
    )
    code = run_resume_run(_command_namespace(run_id="root-run", input=None, output=None))
    assert code == 0
    options = captured["options"]
    assert options.root == data_root.resolve()
    assert options.evaluate is True
    assert options.judge_policy == "all"
    assert options.judge_sample_rate == 0.5
    assert options.auto_task is True
    assert options.tasks == ()
    assert options.resume is True


def test_run_request_judge_errors_only_survives_resume(tmp_path, monkeypatch) -> None:
    """Fix B: judge_policy=errors-only survives resume and yields the same
    deterministic participation subset. Fix B：judge_policy=errors-only 在
    resume 后存活，并产生相同确定性参与子集。"""
    from application.commands import resume_run as resume_run_module
    from application.commands.resume_run import run_resume_run
    from workflows.dataset_runner import DatasetRunner

    data_root = tmp_path / "data"
    _make_dataset(data_root)
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = DatasetRegistry()
    registry.register(
        "auto-demo", lambda: ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    )
    runtime = Runtime(settings=settings, components=components, registry=registry)
    asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo",
                root=data_root,
                split="test",
                tasks=(),
                auto_task=True,
                run_id="errors-run",
                evaluate=True,
                judge_policy="errors-only",
            )
        )
    )
    request = json.loads(
        (tmp_path / "runs" / "errors-run" / "run_request.json").read_text(encoding="utf-8")
    )
    assert request["judge_policy"] == "errors-only"
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    captured = {}

    class _FakeRuntime:
        async def run_dataset(self, options):
            captured["options"] = options
            return {}

    monkeypatch.setattr(
        resume_run_module, "Runtime", type(
            "_FakeRuntimeClass", (), {"create": classmethod(lambda cls, **kw: _FakeRuntime())}
        )
    )
    code = run_resume_run(_command_namespace(run_id="errors-run", input=None, output=None))
    assert code == 0
    assert captured["options"].judge_policy == "errors-only"
    assert captured["options"].judge_sample_rate is None


def test_run_request_adapter_default_mode_survives_resume(
    tmp_path, monkeypatch
) -> None:
    """Fix A: adapter_default task mode is reconstructed from run_request,
    never from directory names. Fix A：adapter_default 任务模式从
    run_request 重建，绝不来自目录名。"""
    from application.commands import resume_run as resume_run_module
    from application.commands.resume_run import run_resume_run

    data_root = tmp_path / "data"
    _make_dataset(data_root)
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = DatasetRegistry()
    registry.register("task-demo", lambda: _TaskAwareAdapter())
    runtime = Runtime(settings=settings, components=components, registry=registry)
    asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="task-demo", root=data_root, split="test", tasks=None
            )
        )
    )
    run_dir = tmp_path / "runs"
    request_path = next(
        path for path in run_dir.iterdir() if (path / "run_request.json").is_file()
    )
    request = json.loads((request_path / "run_request.json").read_text(encoding="utf-8"))
    assert request["task_mode"] == "adapter_default"
    monkeypatch.setenv("OUTPUT_ROOT", str(run_dir))
    captured = {}

    class _FakeRuntime:
        async def run_dataset(self, options):
            captured["options"] = options
            return {}

    monkeypatch.setattr(
        resume_run_module, "Runtime", type(
            "_FakeRuntimeClass", (), {"create": classmethod(lambda cls, **kw: _FakeRuntime())}
        )
    )
    code = run_resume_run(
        _command_namespace(run_id=request_path.name, input=None, output=None)
    )
    assert code == 0
    assert captured["options"].tasks is None
    assert captured["options"].auto_task is False


def test_render_errors_uses_execution_task_in_auto_namespace(tmp_path) -> None:
    """Fix C: an auto-task sample whose execution task is counting renders an
    overlay; run_task="auto" never decides the execution semantics.
    Fix C：执行任务为 counting 的 auto-task 样本渲染 overlay；run_task="auto"
    绝不决定执行语义。"""
    from agents.counting.schema import CountingResult
    from workflows.dataset_runner import storage_key

    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    runtime = Runtime(settings=settings, components=components)
    run_dir = tmp_path / "runs" / "auto-render"
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(data_root / "img.png", format="PNG")
    sample_dir = run_dir / "tasks" / "auto" / "samples" / storage_key("c1")
    sample_dir.mkdir(parents=True)
    (sample_dir / "sample.json").write_text(
        json.dumps(
            {
                "sample_id": "c1",
                "dataset": "d",
                "split": "t",
                "task": "counting",
                "images": [{"image_id": "i0", "path": "img.png", "role": "image"}],
                "question": "q",
                "ground_truth": None,
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / "status.json").write_text(
        json.dumps(
            {
                "sample_id": "c1",
                "task": "counting",
                "state": "failed",
                "error_code": "X",
                "error_message": "X",
                "result_path": None,
                "updated_at": "2026-08-09T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    result = CountingResult(
        sample_id="c1",
        target="vehicles",
        question="q",
        source_width=10,
        source_height=10,
        tile_count=1,
        global_points=[
            {
                **{
                    key: value
                    for key, value in _render_failed_point("p1").items()
                    if key != "accepted"
                },
                "accepted": True,
            }
        ],
        warnings=[],
        final_count=1,
        status="completed_with_warnings",
    )
    (sample_dir / "counting_result.json").write_text(
        json.dumps(result.model_dump(mode="json")), encoding="utf-8"
    )
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "c1",
                "run_task": "auto",
                "task": "counting",
                "status": "failed",
                "result_path": "counting_result.json",
                "updated_at": "now",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    notes = runtime.render_error_overlays(run_dir, data_root=data_root)
    assert (sample_dir / "error_overlay.png").is_file()
    assert not any("unsupported_task" in note["note"] for note in notes)
    # an auto-task sample whose real task is not counting is skipped
    # 真实任务非 counting 的 auto-task 样本被跳过
    cap_dir = run_dir / "tasks" / "auto" / "samples" / storage_key("x1")
    cap_dir.mkdir(parents=True)
    (cap_dir / "status.json").write_text(
        json.dumps(
            {
                "sample_id": "x1",
                "task": "caption",
                "state": "failed",
                "error_code": "X",
                "error_message": "X",
                "result_path": None,
                "updated_at": "now",
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "sample_id": "x1",
                    "run_task": "auto",
                    "task": "caption",
                    "status": "failed",
                    "result_path": None,
                    "updated_at": "now",
                }
            )
            + "\n"
        )
    notes = runtime.render_error_overlays(run_dir, data_root=data_root)
    by_sample = {note["sample_id"]: note for note in notes}
    assert by_sample["x1"]["note"] == "render_errors_skipped:unsupported_task"


def test_render_errors_path_containment_rejects_escapes(tmp_path) -> None:
    """Fix D: persisted samples with escaping or absolute image paths yield
    stable skip notes; the outside sentinel file is never opened.
    Fix D：带逃逸或绝对图片路径的持久化样本产生稳定 skip note；外部哨兵
    文件绝不被打开。"""
    from workflows.dataset_runner import storage_key

    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    runtime = Runtime(settings=settings, components=components)
    run_dir = tmp_path / "runs" / "contain-run"
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(data_root / "img.png", format="PNG")
    # outside sentinel that must never be opened / 绝不能被打开的哨兵
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"sentinel")
    cases = {
        "up1": "../outside.png",
        "up2": "../../outside.png",
        "abs": "C:/outside.png",
        "unc": r"\\server\share\outside.png",
    }
    for sample_id, path_value in cases.items():
        sample_dir = run_dir / "tasks" / "counting" / "samples" / storage_key(sample_id)
        sample_dir.mkdir(parents=True)
        (sample_dir / "sample.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "dataset": "d",
                    "split": "t",
                    "task": "counting",
                    "images": [
                        {"image_id": "i0", "path": path_value, "role": "image"}
                    ],
                    "question": "q",
                    "ground_truth": None,
                }
            ),
            encoding="utf-8",
        )
        (sample_dir / "status.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "task": "counting",
                    "state": "failed",
                    "error_code": "X",
                    "error_message": "X",
                    "result_path": None,
                    "updated_at": "now",
                }
            ),
            encoding="utf-8",
        )
        (sample_dir / "counting_result.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "target": "vehicles",
                    "question": "q",
                    "source_width": 10,
                    "source_height": 10,
                    "tile_count": 1,
                    "global_points": [],
                    "warnings": [],
                    "final_count": 0,
                    "status": "completed",
                }
            ),
            encoding="utf-8",
        )
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id in cases:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "run_task": "counting",
                        "task": "counting",
                        "status": "failed",
                        "result_path": None,
                        "updated_at": "now",
                    }
                )
                + "\n"
            )
    notes = runtime.render_error_overlays(run_dir, data_root=data_root)
    by_sample = {note["sample_id"]: note for note in notes}
    for sample_id in cases:
        assert by_sample[sample_id]["note"] == "render_errors_skipped:no_source_image"
        sample_dir = run_dir / "tasks" / "counting" / "samples" / storage_key(sample_id)
        assert not (sample_dir / "error_overlay.png").exists()
    # the sentinel was never opened: its bytes stay untouched and no overlay
    # references it. 哨兵从未被打开：字节未被触碰，也没有 overlay 引用它。
    assert outside.read_bytes() == b"sentinel"
    assert "sentinel" not in (run_dir / "render_errors_notes.json").read_text(encoding="utf-8")


def test_count_image_fresh_unique_runs_and_duplicate_rejected(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix E: fresh without --run-id creates unique RunStore ids; fresh with a
    duplicate explicit --run-id fails and old artifacts stay unchanged.
    Fix E：fresh 无 --run-id 创建唯一 RunStore id；fresh 重复显式 --run-id
    失败且旧产物保持不变。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    first = run_count_image(
        _count_image_args(tmp_path, image=tmp_path / "imgs" / "img.png", run_id=None)
    )
    assert first == 0
    out1 = json.loads(capsys.readouterr().out)
    second = run_count_image(
        _count_image_args(tmp_path, image=tmp_path / "imgs" / "img.png", run_id=None)
    )
    assert second == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out1["run_id"] != out2["run_id"]  # unique fresh runs
    assert out1["run_id"] != out1["sample_id"]  # never sample_id default
    # duplicate explicit run id fails; old artifacts unchanged
    # 重复显式 run id 失败；旧产物保持不变
    fixed_dir = tmp_path / "runs" / "fixed-run"
    code = run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", run_id="fixed-run"
        )
    )
    assert code == 0
    before = sorted(
        (path.relative_to(fixed_dir).as_posix(), path.read_bytes())
        for path in fixed_dir.rglob("*")
        if path.is_file()
    )
    code = run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", run_id="fixed-run"
        )
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "FileExistsError"
    after = sorted(
        (path.relative_to(fixed_dir).as_posix(), path.read_bytes())
        for path in fixed_dir.rglob("*")
        if path.is_file()
    )
    assert after == before


def test_count_image_resume_requires_explicit_run_id(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix E: --resume without --run-id is a contract failure before any model
    initialization. Fix E：--resume 无 --run-id 是契约失败，先于任何模型初始化。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))

    def boom(cls, **kw):
        raise AssertionError("runtime must not be created")

    monkeypatch.setattr(count_image_module.Runtime, "create", classmethod(boom))
    code = run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", resume=True
        )
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


def test_count_image_resume_and_force_reexecutes(tmp_path, monkeypatch, capsys) -> None:
    """Fix E: resume of a succeeded sample is Qwen-free; resume+force
    re-executes inside the same run without creating a new one.
    Fix E：succeeded 样本的 resume 零 Qwen；resume+force 在同一 run 内重跑，
    不创建新 run。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    args = _count_image_args(
        tmp_path, image=tmp_path / "imgs" / "img.png", run_id="force-run"
    )
    assert run_count_image(args) == 0
    capsys.readouterr()
    calls_after_first = list(client.calls)
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="force-run",
            resume=True,
        )
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "resumed"
    assert client.calls == calls_after_first  # zero Qwen
    # force re-executes inside the same run / force 在同一 run 内重跑
    run_dirs_before = {p.name for p in (tmp_path / "runs").iterdir()}
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="force-run",
            resume=True,
            force=True,
        )
    )
    assert code == 0
    assert len(client.calls) > len(calls_after_first)
    run_dirs_after = {p.name for p in (tmp_path / "runs").iterdir()}
    assert run_dirs_after == run_dirs_before  # no new run created


def test_count_image_resume_invocation_mismatch_fails(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix E: resuming a run with a different image/question identity fails
    stably. Fix E：用不同图像/问题身份 resume 一个 run 稳定失败。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png", "other.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    assert run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", run_id="match-run"
        )
    ) == 0
    capsys.readouterr()
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "other.png",
            run_id="match-run",
            resume=True,
        )
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


def test_count_image_evaluate_flag_effective(tmp_path, monkeypatch, capsys) -> None:
    """Fix F: without --evaluate no counting_evaluation.json is written; with
    --evaluate it is, with identical Qwen execution counts.
    Fix F：无 --evaluate 不写 counting_evaluation.json；有 --evaluate 写入，
    且 Qwen 执行次数相同。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    # without --evaluate / 无 --evaluate
    assert run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="eval-flag-run",
        )
    ) == 0
    calls_no_flag = list(client.calls)
    sample_dir = _count_image_sample_dir_for(tmp_path, "eval-flag-run")
    names = {entry.name for entry in sample_dir.iterdir()}
    assert "counting_evaluation.json" not in names
    # with --evaluate / 有 --evaluate
    client2 = _FakeCountClient()
    runtime2 = _count_image_runtime(tmp_path, client2)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime2)
    )
    assert run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="eval-flag-run2",
            evaluate=True,
        )
    ) == 0
    sample_dir2 = _count_image_sample_dir_for(tmp_path, "eval-flag-run2")
    assert (sample_dir2 / "counting_evaluation.json").is_file()
    assert client2.calls == calls_no_flag  # identical Qwen execution counts


def _count_image_sample_dir_for(tmp_path: Path, run_id: str) -> Path:
    samples_root = tmp_path / "runs" / run_id / "tasks" / "counting" / "samples"
    return next(samples_root.iterdir())


def test_summarize_evaluations_file_mode(tmp_path, monkeypatch, capsys) -> None:
    """Fix G: --input JSONL file mode aggregates and writes the exact output;
    --run-id and --input together, or neither, fail as argument errors.
    Fix G：--input JSONL 文件模式聚合并写入精确输出；--run-id 与 --input
    同时给出或都不给出时参数失败。"""
    from application.commands.summarize_evaluations import run_summarize_evaluations
    from evaluation.metrics.vqa import VQADeterministicMetrics
    from evaluation.records import EvaluationRecord

    records_file = tmp_path / "records.jsonl"
    records = [
        EvaluationRecord(
            sample_id="a1",
            task="general_vqa",
            deterministic_metrics=VQADeterministicMetrics(exact_match=True),
            judge_status="not_requested",
        ),
        EvaluationRecord(
            sample_id="a2",
            task="general_vqa",
            deterministic_metrics=VQADeterministicMetrics(exact_match=False),
            judge_status="not_requested",
        ),
    ]
    records_file.write_text(
        "\n".join(
            json.dumps(record.model_dump(mode="json")) for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_summarize_evaluations(
        _command_namespace(run_id=None, input=str(records_file), output=str(output))
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["record_count"] == 2
    assert out["aggregates"]["general_vqa"]["score"] == 0.5
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == out  # exact requested output file
    # mutual exclusion and requirement / 互斥与必填
    code = run_summarize_evaluations(
        _command_namespace(run_id="x", input=str(records_file), output=None)
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
    code = run_summarize_evaluations(
        _command_namespace(run_id=None, input=None, output=None)
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
    # malformed JSONL fails stably / 损坏 JSONL 稳定失败
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"sample_id": "x"}\n{broken\n', encoding="utf-8")
    code = run_summarize_evaluations(
        _command_namespace(run_id=None, input=str(bad), output=None)
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


def test_evaluate_run_counting_deepseek_zero_qwen(tmp_path, monkeypatch, capsys) -> None:
    """Fix H: counting samples with --deepseek call JudgeService.judge_counting
    exactly once with zero Qwen; deterministic metrics survive judge failure.
    Fix H：--deepseek 时计数样本恰好调用一次 JudgeService.judge_counting、
    零 Qwen；judge 失败时确定性指标存活。"""
    from application.commands import evaluate_run as evaluate_run_module
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        evaluate_run_module, "DeepSeekJudgeClient", lambda *a, **k: judge
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    counting = _offline_counting_sample("c1", final_count=0)
    counting["sample"]["metadata"] = {
        "count_target_hint": {
            "canonical_label": "vehicles",
            "inclusion_rule": "count all vehicles",
            "exclusion_rule": "exclude none",
        }
    }
    _make_offline_run(
        tmp_path,
        [counting],
    )
    code = run_evaluate_run(_offline_args(deepseek=True))
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    judge_results = {item["sample_id"]: item for item in out["judge"]}
    assert judge_results["c1"]["judge_status"] == "succeeded"
    assert len(judge.calls) == 1  # judge_counting called exactly once
    count_dir = next(
        (tmp_path / "runs" / "offline-run" / "tasks" / "counting" / "samples").iterdir()
    )
    evaluation = json.loads(
        (count_dir / "counting_evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["judge_status"] == "succeeded"
    assert evaluation["deterministic_metrics"]["exact_match"] == 1
    # judge failure keeps the deterministic record / judge 失败保留确定性记录
    failing = _OfflineFakeJudgeClient(fail=True)
    monkeypatch.setattr(
        evaluate_run_module, "DeepSeekJudgeClient", lambda *a, **k: failing
    )
    (count_dir / "counting_evaluation.json").unlink()
    code = run_evaluate_run(_offline_args(deepseek=True, force_judge=True))
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    judge_results = {item["sample_id"]: item for item in out["judge"]}
    assert judge_results["c1"]["judge_status"] == "failed"
    assert judge_results["c1"]["judge_error"] == "RuntimeError"
    evaluation = json.loads(
        (count_dir / "counting_evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["judge_status"] == "failed"
    assert evaluation["deterministic_metrics"]["exact_match"] == 1


def test_evaluate_run_counting_deepseek_reconstructed_target(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix H: without a persisted CountTargetSpec a stable neutral spec is
    reconstructed from the canonical label. Fix H：无持久化 CountTargetSpec
    时从 canonical 标签重建稳定中性 spec。"""
    from application.commands import evaluate_run as evaluate_run_module
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        evaluate_run_module, "DeepSeekJudgeClient", lambda *a, **k: judge
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    _make_offline_run(
        tmp_path,
        [_offline_counting_sample("c1", final_count=0)],
    )
    code = run_evaluate_run(_offline_args(deepseek=True))
    assert code == 0
    assert len(judge.calls) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_run_dataset_persists_report_bundle(tmp_path) -> None:
    """Fix J: a real dataset run through the application runtime persists the
    unified report bundle with zero additional Qwen calls.
    Fix J：经应用 runtime 的真实数据集运行持久化统一报告 bundle，且零额外
    Qwen 调用。"""
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = DatasetRegistry()
    registry.register(
        "auto-demo", lambda: ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    )
    runtime = Runtime(settings=settings, components=components, registry=registry)
    results = asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo",
                root=data_root,
                split="test",
                tasks=(),
                auto_task=True,
                run_id="bundle-run",
            )
        )
    )
    assert results["auto"].succeeded == 1
    report_dir = tmp_path / "runs" / "bundle-run" / "report"
    for name in ("report.html", "report.json", "samples.csv", "samples.jsonl", "metadata.json"):
        assert (report_dir / name).is_file(), name
    assert (tmp_path / "runs" / "bundle-run" / "run_request.json").is_file()
    # a resumed run adds no Qwen calls while the report bundle is refreshed
    # resume 的 run 不增加 Qwen 调用，同时报告 bundle 被刷新
    calls_after_fresh = client.calls
    resumed = asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo",
                root=data_root,
                split="test",
                tasks=(),
                auto_task=True,
                run_id="bundle-run",
                resume=True,
            )
        )
    )
    assert resumed["auto"].succeeded == 1
    assert client.calls == calls_after_fresh  # reporting added zero Qwen calls
    assert (report_dir / "report.json").is_file()


def test_standard_evaluate_persists_report_bundle(tmp_path, monkeypatch, capsys) -> None:
    """Fix I: standard-evaluate for an associated run produces real report
    bundle files including the external standard namespace.
    Fix I：standard-evaluate 对关联 run 产生真实报告 bundle 文件（含外部
    标准命名空间）。"""
    from application.commands.standard_evaluate import run_standard_evaluate

    data_root = tmp_path / "data"
    _make_dataset(data_root)
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = DatasetRegistry()
    registry.register(
        "auto-demo", lambda: ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    )
    runtime = Runtime(settings=settings, components=components, registry=registry)
    asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo",
                root=data_root,
                split="test",
                tasks=(),
                auto_task=True,
                run_id="std-bundle-run",
            )
        )
    )
    result = tmp_path / "runs" / "std-bundle-run" / "predictions.jsonl"
    tool_dir = _fake_standard_tool(tmp_path, body=_FAKE_EVALUATOR_OK)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_standard_evaluate(
        _command_namespace(
            result=str(result),
            tool_dir=str(tool_dir),
            output=None,
            python=sys.executable,
        )
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"] == "std-bundle-run"
    report_dir = tmp_path / "runs" / "std-bundle-run" / "report"
    assert (report_dir / "report.html").is_file()
    assert (report_dir / "report.json").is_file()
    assert (report_dir / "samples.csv").is_file()
    assert (report_dir / "external_standard.json").is_file()
    standard = json.loads(
        (report_dir / "external_standard.json").read_text(encoding="utf-8")
    )
    assert "external_standard" in standard


def test_deepseek_audit_real_request_metadata(tmp_path) -> None:
    """Fix K: audit rows use the real persisted RequestMeta values; missing
    metadata yields null identity fields, never a synthesized hash; counting
    uses its own judge directory. Fix K：审计行使用真实持久化 RequestMeta
    值；缺失元数据时身份字段为 null 而非合成哈希；counting 使用自己的
    judge 目录。"""
    from evaluation.judges.base import VQAAnswerJudgeResult
    from evaluation.metrics.counting import CountDeterministicMetrics
    from evaluation.metrics.vqa import VQADeterministicMetrics
    from evaluation.records import EvaluationRecord
    from reporting.exporters import write_deepseek_audit
    from reporting.schema import Report, ReportSample

    from workflows.dataset_runner import storage_key as audit_key

    run_dir = tmp_path / "runs" / "audit-run"
    vqa_dir = (
        run_dir / "tasks" / "general_vqa" / "samples" / audit_key("a1")
    )
    (vqa_dir / "deepseek_vqa_judge").mkdir(parents=True)
    (vqa_dir / "deepseek_vqa_judge" / "request_meta.json").write_text(
        json.dumps(
            {
                "request_id": "a1:deepseek-vqa",
                "request_hash": "real-hash-vqa",
                "prompt_version": "deepseek-vqa-judge-v1",
            }
        ),
        encoding="utf-8",
    )
    count_dir = (
        run_dir / "tasks" / "counting" / "samples" / audit_key("c1")
    )
    (count_dir / "deepseek").mkdir(parents=True)
    (count_dir / "deepseek" / "request_meta.json").write_text(
        json.dumps(
            {
                "request_id": "c1:deepseek",
                "request_hash": "real-hash-count",
                "prompt_version": "deepseek-judge-v1",
            }
        ),
        encoding="utf-8",
    )
    judged_vqa = EvaluationRecord(
        sample_id="a1",
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=True),
        judge_status="succeeded",
        judge_parsed=VQAAnswerJudgeResult(score=1),
    )
    judged_count = EvaluationRecord(
        sample_id="c1",
        task="counting",
        deterministic_metrics=CountDeterministicMetrics(
            predicted_count=1,
            gold_count=1,
            exact_match=1,
            absolute_error=0,
            relative_error=0.0,
            smooth_error_score=1.0,
        ),
        judge_status="succeeded",
        judge_parsed={"score": 1},
    )
    judged_no_meta = EvaluationRecord(
        sample_id="x1",
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=True),
        judge_status="succeeded",
        judge_parsed=VQAAnswerJudgeResult(score=1),
    )
    report = Report(
        run_id="audit-run",
        dataset="d",
        total=3,
        succeeded=3,
        partial=0,
        failed=0,
        skipped=0,
        samples=[
            ReportSample(
                sample_id="a1",
                run_task="general_vqa",
                task="general_vqa",
                state="succeeded",
                evaluation=judged_vqa,
            ),
            ReportSample(
                sample_id="c1",
                run_task="counting",
                task="counting",
                state="succeeded",
                evaluation=judged_count,
            ),
            ReportSample(
                sample_id="x1",
                run_task="general_vqa",
                task="general_vqa",
                state="succeeded",
                evaluation=judged_no_meta,
            ),
        ],
    )
    path = tmp_path / "deepseek_audit.jsonl"
    write_deepseek_audit(report, path, run_dir=run_dir)
    rows = {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    }
    assert rows["a1"]["request_id"] == "a1:deepseek-vqa"
    assert rows["a1"]["request_hash"] == "real-hash-vqa"
    assert rows["a1"]["prompt_version"] == "deepseek-vqa-judge-v1"
    assert rows["c1"]["request_id"] == "c1:deepseek"  # counting-shaped identity
    assert rows["c1"]["request_hash"] == "real-hash-count"
    assert rows["x1"]["request_id"] is None  # no fabricated hash
    assert rows["x1"]["request_hash"] is None
    text = path.read_text(encoding="utf-8")
    assert "api_key" not in text.lower()
    assert "authorization" not in text.lower()
    assert "sk-" not in text

# ── 11G.5.1 resume/report consistency finalization / 一致性收口 ─────────────


def test_dataset_root_canonicalized_before_persistence(tmp_path, monkeypatch) -> None:
    """Fix B: a relative --root is canonicalized once, before identity and
    persistence, so execution and run_request.dataset_root share the same
    host-resolved path. Fix B：相对 --root 在身份确立与持久化前一次性
    canonicalize，使执行与 run_request.dataset_root 共享同一主机解析路径。"""
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = DatasetRegistry()
    registry.register(
        "auto-demo", lambda: ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    )
    runtime = Runtime(settings=settings, components=components, registry=registry)
    monkeypatch.chdir(tmp_path)
    results = asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo",
                root=Path("data"),  # relative form / 相对形式
                split="test",
                tasks=(),
                auto_task=True,
            )
        )
    )
    run_id = results["auto"].run_id
    request = json.loads(
        (tmp_path / "runs" / run_id / "run_request.json").read_text(encoding="utf-8")
    )
    canonical = (tmp_path / "data").resolve().as_posix().replace("\\", "/")
    assert request["dataset_root"] == canonical


def test_run_dataset_resume_uses_run_request_and_rejects_drift(
    tmp_path, monkeypatch
) -> None:
    """Fix A: run-dataset --resume is authoritative from run_request.json;
    root/task/judge drift is rejected before model execution with zero Qwen.
    Fix A：run-dataset --resume 以 run_request.json 为权威；root/task/judge
    偏离在模型执行前被拒绝且零 Qwen。"""
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = DatasetRegistry()
    registry.register(
        "auto-demo", lambda: ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    )
    runtime = Runtime(settings=settings, components=components, registry=registry)
    fresh = DatasetRunOptions(
        dataset="auto-demo",
        root=data_root,
        split="test",
        tasks=(),
        auto_task=True,
        run_id="strict-run",
        evaluate=True,
        judge_policy="all",
        judge_sample_rate=0.5,
    )
    asyncio.run(runtime.run_dataset(fresh))
    calls_after_fresh = client.calls

    def resume_with(**overrides):
        values = dict(
            dataset="auto-demo",
            root=data_root,
            split="test",
            tasks=(),
            auto_task=True,
            run_id="strict-run",
            resume=True,
            evaluate=True,
            judge_policy="all",
            judge_sample_rate=0.5,
        )
        values.update(overrides)
        return DatasetRunOptions(**values)

    # root drift rejected / root 偏离被拒绝
    with pytest.raises(ValueError, match="dataset root mismatch"):
        asyncio.run(
            runtime.run_dataset(
                resume_with(root=tmp_path / "other-root")
            )
        )
    # task drift rejected (auto -> explicit caption) / task 偏离被拒绝
    with pytest.raises(ValueError, match="task mode mismatch"):
        asyncio.run(
            runtime.run_dataset(
                resume_with(tasks=("caption",), auto_task=False)
            )
        )
    # judge drift rejected / judge 偏离被拒绝
    with pytest.raises(ValueError, match="judge policy mismatch"):
        asyncio.run(
            runtime.run_dataset(
                resume_with(judge_policy="none", judge_sample_rate=None)
            )
        )
    assert client.calls == calls_after_fresh  # zero Qwen before failure
    # exact matching resume succeeds / 精确匹配的 resume 成功
    resumed = asyncio.run(runtime.run_dataset(resume_with()))
    assert resumed["auto"].succeeded == 1
    assert client.calls == calls_after_fresh  # succeeded samples not re-run


def test_count_image_missing_or_corrupt_result_reexecutes(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix C: a succeeded status without a valid matching CountingResult is
    incomplete — resume re-executes instead of emitting resumed/null.
    Fix C：succeeded 状态但没有合法匹配 CountingResult 视为不完整——resume
    重跑而非输出 resumed/null。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    args = _count_image_args(
        tmp_path, image=tmp_path / "imgs" / "img.png", run_id="validity-run"
    )
    assert run_count_image(args) == 0
    capsys.readouterr()
    sample_dir = _count_image_sample_dir_for(tmp_path, "validity-run")
    # missing result / 缺失结果
    (sample_dir / "counting_result.json").unlink()
    calls_before = len(client.calls)
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="validity-run",
            resume=True,
        )
    )
    assert code == 0
    assert len(client.calls) > calls_before  # re-executed
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    # corrupt result / 损坏结果
    capsys.readouterr()
    (sample_dir / "counting_result.json").write_text("{broken", encoding="utf-8")
    calls_before = len(client.calls)
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="validity-run",
            resume=True,
        )
    )
    assert code == 0
    assert len(client.calls) > calls_before
    # sample-id mismatch / sample_id 不匹配
    capsys.readouterr()
    result = json.loads((sample_dir / "counting_result.json").read_text(encoding="utf-8"))
    result["sample_id"] = "someone-else"
    (sample_dir / "counting_result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    calls_before = len(client.calls)
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="validity-run",
            resume=True,
        )
    )
    assert code == 0
    assert len(client.calls) > calls_before
    # valid result: zero-Qwen resumed / 合法结果：零 Qwen resumed
    capsys.readouterr()
    calls_before = len(client.calls)
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="validity-run",
            resume=True,
        )
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "resumed"
    assert out["final_count"] == 1
    assert len(client.calls) == calls_before  # zero new Qwen calls


def test_count_image_resume_evaluate_intent_authoritative(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix D: resume uses the persisted evaluate intent; force with evaluate
    intent false removes stale evaluation artifacts.
    Fix D：resume 使用持久化 evaluate 意图；force 且 evaluate 意图为 false
    时移除过期评估产物。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    # fresh evaluate=true, then force resume without the flag
    # fresh evaluate=true，然后无标志 force resume
    assert run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="intent-run",
            evaluate=True,
        )
    ) == 0
    capsys.readouterr()
    sample_dir = _count_image_sample_dir_for(tmp_path, "intent-run")
    assert (sample_dir / "counting_evaluation.json").is_file()
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="intent-run",
            resume=True,
            force=True,
        )
    )
    assert code == 0
    assert (sample_dir / "counting_evaluation.json").is_file()  # refreshed intent
    # fresh evaluate=false, inject a stale evaluation, force resume
    # fresh evaluate=false，注入过期评估，force resume
    assert run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="intent-run2",
        )
    ) == 0
    capsys.readouterr()
    sample_dir2 = _count_image_sample_dir_for(tmp_path, "intent-run2")
    stale = {"sample_id": "stale", "task": "counting", "judge_status": "not_requested"}
    (sample_dir2 / "counting_evaluation.json").write_text(
        json.dumps(stale), encoding="utf-8"
    )
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="intent-run2",
            resume=True,
            force=True,
        )
    )
    assert code == 0
    capsys.readouterr()
    assert not (sample_dir2 / "counting_evaluation.json").exists()  # stale removed
    # valid zero-Qwen resume must not change evaluation artifacts
    # 合法零 Qwen resume 不改变评估产物
    assert run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="intent-run",
            resume=True,
        )
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "resumed"
    assert (sample_dir / "counting_evaluation.json").is_file()


def test_evaluate_run_persists_refreshed_report_bundle(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix E: evaluate-run persists the refreshed report bundle and the
    deepseek audit reflects newly judged counting samples; zero Qwen.
    Fix E：evaluate-run 持久化刷新的报告 bundle，deepseek audit 反映新 judge
    的 counting 样本；零 Qwen。"""
    from application.commands import evaluate_run as evaluate_run_module
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        evaluate_run_module, "DeepSeekJudgeClient", lambda *a, **k: judge
    )
    secret_marker = "SUPER_SECRET_TEST_VALUE_123"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret_marker)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    _make_offline_run(
        tmp_path,
        [
            _offline_counting_sample("c1", final_count=0),
            {**_offline_vqa_sample("v1"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "no", "status": "completed"}}},
        ],
    )
    run_dir = tmp_path / "runs" / "offline-run"
    # stale pre-existing bundle / 预置过期 bundle
    stale_dir = run_dir / "report"
    stale_dir.mkdir(parents=True)
    (stale_dir / "report.json").write_text('{"stale": true}', encoding="utf-8")
    (stale_dir / "report.html").write_text("<html>stale</html>", encoding="utf-8")
    (stale_dir / "samples.csv").write_text("stale", encoding="utf-8")
    (stale_dir / "samples.jsonl").write_text("stale\n", encoding="utf-8")
    (stale_dir / "metadata.json").write_text('{"stale": true}', encoding="utf-8")
    code = run_evaluate_run(_offline_args(deepseek=True))
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    report_json = json.loads((stale_dir / "report.json").read_text(encoding="utf-8"))
    assert report_json.get("stale") is None  # refreshed, not the stale file
    assert report_json["run_id"] == "offline-run"
    assert (stale_dir / "report.html").read_text(encoding="utf-8").startswith("<!DOCTYPE")
    audit = (stale_dir / "deepseek_audit.jsonl").read_text(encoding="utf-8")
    assert '"sample_id": "c1"' in audit  # counting judge audited
    assert '"sample_id": "v1"' in audit
    assert judge.calls  # judge ran
    assert secret_marker not in json.dumps(out)
    for artifact in run_dir.rglob("*"):
        if artifact.is_file():
            assert (
                secret_marker.encode("utf-8") not in artifact.read_bytes()
            ), artifact


def test_judge_vqa_run_persists_refreshed_report_bundle(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix F: judge-vqa-run persists the refreshed report bundle reflecting
    the new judge status; zero Qwen. Fix F：judge-vqa-run 持久化刷新报告
    bundle，反映新 judge 状态；零 Qwen。"""
    from application.commands import judge_vqa_run as judge_run_module
    from application.commands.judge_vqa_run import run_judge_vqa_run

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        judge_run_module, "DeepSeekJudgeClient", lambda *a, **k: judge
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    _make_offline_run(
        tmp_path,
        [
            {**_offline_vqa_sample("v1"), **{"payload_file": "agent_result.json",
                                             "payload": {"agent_name": "general_vqa_agent", "answer": "no", "status": "completed"}}},
        ],
    )
    run_dir = tmp_path / "runs" / "offline-run"
    # pre-create a report where the sample judge status is not_requested
    # 预建 judge 状态为 not_requested 的报告
    from reporting.builder import build_report
    from reporting.exporters import persist_report_bundle

    persist_report_bundle(run_dir, build_report(run_dir))
    stale_dir = run_dir / "report"
    before = json.loads((stale_dir / "report.json").read_text(encoding="utf-8"))
    assert before["samples"][0]["judge_status"] == "not_requested"
    code = run_judge_vqa_run(_offline_args())
    assert code == 0
    after = json.loads((stale_dir / "report.json").read_text(encoding="utf-8"))
    assert after["samples"][0]["judge_status"] == "succeeded"
    csv_text = (stale_dir / "samples.csv").read_text(encoding="utf-8")
    assert "succeeded" in csv_text
    audit = (stale_dir / "deepseek_audit.jsonl").read_text(encoding="utf-8")
    assert '"sample_id": "v1"' in audit
    assert judge.calls  # judge ran


def test_counting_target_reconstruction_neutral_and_exact(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix G: an exact persisted CountTargetSpec wins; the fallback states
    only the known label and never invents rules like "exclude none".
    Fix G：精确持久化 CountTargetSpec 优先；回退只陈述已知标签，绝不虚构
    "exclude none" 等规则。"""
    from agents.counting.schema import CountingResult
    from application.commands.evaluate_run import _count_target_for
    from data.schema import GroundTruth, ImageRef, UnifiedSample

    payload = CountingResult(
        sample_id="s",
        target="vehicles",
        question="q",
        source_width=10,
        source_height=10,
        tile_count=1,
        global_points=[],
        warnings=[],
        final_count=0,
        status="completed",
    )
    exact = {
        "canonical_label": "vehicles",
        "inclusion_rule": "count every visible vehicle",
        "exclusion_rule": "exclude occluded ones",
        "aliases": ["car"],
    }
    sample_exact = UnifiedSample(
        sample_id="s",
        dataset="d",
        split="t",
        task="counting",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="q",
        ground_truth=GroundTruth(),
        metadata={"count_target_hint": exact},
    )
    spec = _count_target_for(sample_exact, payload)
    assert spec.inclusion_rule == "count every visible vehicle"
    assert spec.exclusion_rule == "exclude occluded ones"
    assert spec.aliases == ["car"]
    # neutral fallback / 中性回退
    sample_neutral = UnifiedSample(
        sample_id="s",
        dataset="d",
        split="t",
        task="counting",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="q",
        ground_truth=GroundTruth(),
    )
    spec = _count_target_for(sample_neutral, payload)
    assert spec.canonical_label == "vehicles"
    assert spec.inclusion_rule == "Persisted inclusion rule unavailable."
    assert spec.exclusion_rule == "Persisted exclusion rule unavailable."
    assert "exclude none" not in spec.exclusion_rule
    assert "count all" not in spec.inclusion_rule

# ── 11G.5.2 final invocation / judge fidelity / 调用与 Judge 保真收口 ───────


def test_count_image_fresh_persists_full_invocation_fidelity(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix A: fresh count-image persists the structured target spec snapshot
    (not the host path) with a stable hash, plus seam/budget/render/evaluate
    intent. Fix A：fresh count-image 持久化结构化 target spec 快照（而非
    主机路径）与稳定哈希，以及 seam/预算/render/evaluate 意图。"""
    from application.commands import count_image as count_image_module
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    spec = tmp_path / "cars.json"
    spec.write_text(
        json.dumps(
            {
                "canonical_label": "vehicles",
                "aliases": ["sedan"],
                "inclusion_rule": "count visible vehicles",
                "exclusion_rule": "exclude occluded vehicles",
            }
        ),
        encoding="utf-8",
    )
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="fidelity-run",
            target_spec=spec,
            evaluate=True,
            render=True,
            max_qwen_calls=7,
            max_deepseek_calls=3,
            no_seam_verify=True,
        )
    )
    assert code == 0
    request = json.loads(
        (tmp_path / "runs" / "fidelity-run" / "run_request.json").read_text(encoding="utf-8")
    )
    assert request["count_target_spec"]["canonical_label"] == "vehicles"
    assert request["count_target_spec"]["inclusion_rule"] == "count visible vehicles"
    # host path is never the authority / 主机路径绝非权威
    assert str(spec) not in json.dumps(request)
    assert len(request["count_target_spec_hash"]) == 64
    assert request["count_seam_verify"] is False
    assert request["count_max_qwen_calls"] == 7
    assert request["count_max_deepseek_calls"] == 3
    assert request["count_render"] is True
    assert request["evaluate"] is True


def test_count_image_force_uses_persisted_target_and_budgets(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix A/C: force resume uses the persisted target spec/budgets/render
    intent even when the source file changed or vanished; CLI values ignored.
    Fix A/C：即使源文件变化或消失，force resume 仍用持久化 target
    spec/预算/render 意图；CLI 值被忽略。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    spec = tmp_path / "cars.json"
    spec.write_text(
        json.dumps(
            {
                "canonical_label": "vehicles",
                "inclusion_rule": "count visible vehicles",
                "exclusion_rule": "exclude occluded vehicles",
            }
        ),
        encoding="utf-8",
    )
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    assert run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="target-run",
            target_spec=spec,
            render=True,
        )
    ) == 0
    capsys.readouterr()
    # change and delete the source file / 修改并删除源文件
    spec.write_text(
        json.dumps(
            {
                "canonical_label": "buses",
                "inclusion_rule": "count buses",
                "exclusion_rule": "exclude cars",
            }
        ),
        encoding="utf-8",
    )
    spec.unlink()
    sample_dir = _count_image_sample_dir_for(tmp_path, "target-run")
    sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
    assert sample["metadata"]["count_target_hint"]["canonical_label"] == "vehicles"
    # force resume with a different CLI target spec path is ignored
    # 带不同 CLI target spec 路径的 force resume 被忽略
    other_spec = tmp_path / "other.json"
    other_spec.write_text(
        json.dumps(
            {
                "canonical_label": "buses",
                "inclusion_rule": "count buses",
                "exclusion_rule": "exclude all",
            }
        ),
        encoding="utf-8",
    )
    calls_before = len(client.calls)
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="target-run",
            resume=True,
            force=True,
            target_spec=other_spec,
            no_seam_verify=True,
            max_qwen_calls=2,
            render=False,
        )
    )
    assert code == 0
    assert len(client.calls) > calls_before
    sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
    assert sample["metadata"]["count_target_hint"]["canonical_label"] == "vehicles"
    assert (sample_dir / "overlay.png").is_file()  # persisted render intent kept


def test_count_image_force_render_intent_consistency(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix B: forced rerun refreshes overlay when render intent is true and
    removes a stale overlay when it is false; zero-Qwen resume never mutates
    render artifacts. Fix B：render 意图为真时 force 重跑刷新 overlay，
    为假时移除 stale overlay；零 Qwen resume 绝不改动渲染产物。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    # render=true: stale overlay is rewritten / render=true：stale overlay 被重写
    assert run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="render-run",
            render=True,
        )
    ) == 0
    capsys.readouterr()
    sample_dir = _count_image_sample_dir_for(tmp_path, "render-run")
    (sample_dir / "overlay.png").write_bytes(b"stale-overlay")
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="render-run",
            resume=True,
            force=True,
        )
    )
    assert code == 0
    assert (sample_dir / "overlay.png").read_bytes() != b"stale-overlay"  # rewritten
    # render=false: stale overlay is removed / render=false：stale overlay 被移除
    assert run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="no-render-run",
        )
    ) == 0
    capsys.readouterr()
    sample_dir2 = _count_image_sample_dir_for(tmp_path, "no-render-run")
    (sample_dir2 / "overlay.png").write_bytes(b"stale-overlay")
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="no-render-run",
            resume=True,
            force=True,
        )
    )
    assert code == 0
    capsys.readouterr()
    assert not (sample_dir2 / "overlay.png").exists()
    # zero-Qwen resume never mutates render artifacts / 零 Qwen resume 不改渲染产物
    assert (sample_dir / "overlay.png").is_file()
    overlay_before = (sample_dir / "overlay.png").read_bytes()
    calls_before = len(client.calls)
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="render-run",
            resume=True,
        )
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "resumed"
    assert len(client.calls) == calls_before
    assert (sample_dir / "overlay.png").read_bytes() == overlay_before


def test_count_image_legacy_run_zero_qwen_resume_and_force_rejected(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix A (backward compat): an old current-generation count-image
    run_request without the fidelity snapshot still allows zero-Qwen reuse
    with a valid result, but force resume fails stably before model calls.
    Fix A（向后兼容）：缺少保真快照的旧当前代 count-image run_request 仍
    允许合法结果零 Qwen 复用，但 force resume 在模型调用前稳定失败。"""
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image
    from workflows.run_store import RunStore
    from workflows.schema import RunRequest

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    # create a run, then strip the fidelity fields to emulate an old request
    # 创建 run，然后移除保真字段以模拟旧请求
    assert run_count_image(
        _count_image_args(
            tmp_path, image=tmp_path / "imgs" / "img.png", run_id="legacy-run"
        )
    ) == 0
    capsys.readouterr()
    request_path = tmp_path / "runs" / "legacy-run" / "run_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    for key in (
        "count_target_spec",
        "count_target_spec_hash",
        "count_seam_verify",
        "count_max_qwen_calls",
        "count_max_deepseek_calls",
        "count_render",
    ):
        request.pop(key, None)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    # zero-Qwen reuse still allowed with a valid result / 合法结果零 Qwen 复用仍允许
    calls_before = len(client.calls)
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="legacy-run",
            resume=True,
        )
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "resumed"
    assert len(client.calls) == calls_before
    # force resume fails stably before any model call / force resume 在模型调用前稳定失败
    code = run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="legacy-run",
            resume=True,
            force=True,
        )
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
    assert len(client.calls) == calls_before  # no model call


def test_counting_judge_hint_priority_normalization_first(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix D/E/F: normalization hint wins over metadata hint; metadata hint
    second; neutral fallback third. VRSBench-style normalization rules are
    passed to judge_counting exactly. Fix D/E/F：normalization hint 优先于
    metadata hint；metadata hint 其次；中性回退第三。VRSBench 风格
    normalization 规则原样传给 judge_counting。"""
    from application.commands import evaluate_run as evaluate_run_module
    from application.commands.evaluate_run import run_evaluate_run
    from data.schema import GroundTruth, ImageRef, TaskNormalization, UnifiedSample

    # build a VRSBench-style counting sample with an audited normalization hint
    # 构建带审计 normalization hint 的 VRSBench 风格计数样本
    entry = _offline_counting_sample("vrs1", final_count=0)
    entry["sample"]["normalization"] = TaskNormalization(
        source_task="counting",
        normalized_task="counting",
        normalizer="vrsbench",
        version="1",
        reason_codes=["ontology_hint"],
        count_target_hint={
            "canonical_label": "small-vehicle",
            "aliases": ["cars", "passenger cars", "motorcycles"],
            "inclusion_rule": "cars / passenger cars / motorcycles / small vehicles",
            "exclusion_rule": "trucks / buses / trailers / large vehicles / non-vehicles",
        },
    ).model_dump(mode="json")
    entry["sample"]["metadata"] = {
        "count_target_hint": {
            "canonical_label": "metadata-wrong",
            "inclusion_rule": "metadata rule",
            "exclusion_rule": "metadata exclusion",
        }
    }
    captured = {}

    class _SpyJudge:
        def __init__(self) -> None:
            self.calls = 0

        def judge_counting(self, *, sample_id, question, target, display_answer, counting, ground_truth, artifact_dir):
            self.calls += 1
            captured["target"] = target
            from evaluation.records import EvaluationRecord
            from evaluation.metrics.counting import merge_count_evaluation

            return merge_count_evaluation(
                sample_id=sample_id,
                counting=counting,
                ground_truth=ground_truth,
            )

    spy = _SpyJudge()

    class _SpyJudgeService:
        judge_client = object()

        def judge_counting(self, **kwargs):
            return spy.judge_counting(**kwargs)

        def judge_vqa_resume(self, **kwargs):
            raise AssertionError("vqa path must not run for counting sample")

    monkeypatch.setattr(
        evaluate_run_module,
        "DeepSeekJudgeClient",
        lambda *a, **k: None,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    _make_offline_run(tmp_path, [entry])
    monkeypatch.setattr(evaluate_run_module, "JudgeService", lambda **kw: _SpyJudgeService())
    code = run_evaluate_run(_offline_args(deepseek=True))
    assert code == 0
    assert spy.calls == 1
    target = captured["target"]
    assert target.canonical_label == "small-vehicle"
    assert target.aliases == ["cars", "passenger cars", "motorcycles"]
    assert target.inclusion_rule == "cars / passenger cars / motorcycles / small vehicles"
    assert target.exclusion_rule == "trucks / buses / trailers / large vehicles / non-vehicles"
    # metadata hint second / metadata hint 其次
    entry2 = _offline_counting_sample("meta1", final_count=0)
    entry2["sample"]["metadata"] = {
        "count_target_hint": {
            "canonical_label": "metadata-cars",
            "inclusion_rule": "metadata rule",
            "exclusion_rule": "metadata exclusion",
        }
    }
    captured.clear()
    spy.calls = 0
    _make_offline_run(tmp_path, [entry2], run_id="meta-run")
    code = run_evaluate_run(_offline_args(run_id="meta-run", deepseek=True))
    assert code == 0
    assert captured["target"].canonical_label == "metadata-cars"
    # neutral fallback third / 中性回退第三
    entry3 = _offline_counting_sample("neutral1", final_count=0)
    captured.clear()
    spy.calls = 0
    _make_offline_run(tmp_path, [entry3], run_id="neutral-run")
    code = run_evaluate_run(_offline_args(run_id="neutral-run", deepseek=True))
    assert code == 0
    assert captured["target"].canonical_label == "vehicles"
    assert captured["target"].inclusion_rule == "Persisted inclusion rule unavailable."
    assert "exclude none" not in captured["target"].exclusion_rule


def test_run_dataset_resume_preflight_before_runtime_create(
    tmp_path, monkeypatch, capsys
) -> None:
    """Fix G: an invalid run-dataset --resume fails before Runtime.create is
    called (zero model construction); a matching invocation proceeds with
    exactly one create. Fix G：非法 run-dataset --resume 在 Runtime.create
    被调用前失败（零模型构造）；匹配调用恰好一次 create 并继续。"""
    from application.commands import run_dataset as run_dataset_module

    data_root = tmp_path / "data"
    _make_dataset(data_root)
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = DatasetRegistry()
    registry.register(
        "auto-demo", lambda: ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    )
    runtime = Runtime(settings=settings, components=components, registry=registry)
    asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo",
                root=data_root,
                split="test",
                tasks=(),
                auto_task=True,
                run_id="preflight-run",
                evaluate=True,
                judge_policy="all",
                judge_sample_rate=0.5,
            )
        )
    )
    creates = []

    def boom_create(cls, **kwargs):
        creates.append(kwargs)
        raise AssertionError("must not be reached for invalid resume")

    import argparse

    monkeypatch.setattr(run_dataset_module.Runtime, "create", classmethod(boom_create))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    # wrong root / root 错误
    code = run_dataset_module.run_run_dataset(
        argparse.Namespace(
            config=None,
            dataset="auto-demo",
            root=str(tmp_path / "wrong-root"),
            split="test",
            task=None,
            auto_task=True,
            sample_ids=None,
            run_id="preflight-run",
            resume=True,
            evaluate=True,
            judge_policy="all",
            judge_sample_rate=0.5,
            render_errors=False,
            fail_fast=False,
            limit=None,
            start_index=0,
            shard_index=0,
            shard_count=1,
            sample_concurrency=1,
        )
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
    assert not creates  # Runtime.create never called / create 从未被调用
    # wrong judge policy / judge 策略错误
    code = run_dataset_module.run_run_dataset(
        argparse.Namespace(
            config=None,
            dataset="auto-demo",
            root=str(data_root),
            split="test",
            task=None,
            auto_task=True,
            sample_ids=None,
            run_id="preflight-run",
            resume=True,
            evaluate=True,
            judge_policy="none",
            judge_sample_rate=None,
            render_errors=False,
            fail_fast=False,
            limit=None,
            start_index=0,
            shard_index=0,
            shard_count=1,
            sample_concurrency=1,
        )
    )
    assert code == 1
    assert not creates
    # matching invocation proceeds with exactly one create / 匹配调用恰好一次 create
    def fake_create(cls, **kwargs):
        creates.append(kwargs)
        return runtime

    monkeypatch.setattr(run_dataset_module.Runtime, "create", classmethod(fake_create))
    code = run_dataset_module.run_run_dataset(
        argparse.Namespace(
            config=None,
            dataset="auto-demo",
            root=str(data_root),
            split="test",
            task=None,
            auto_task=True,
            sample_ids=None,
            run_id="preflight-run",
            resume=True,
            evaluate=True,
            judge_policy="all",
            judge_sample_rate=0.5,
            render_errors=False,
            fail_fast=False,
            limit=None,
            start_index=0,
            shard_index=0,
            shard_count=1,
            sample_concurrency=1,
        )
    )
    assert code == 0
    assert len(creates) == 1  # exactly one create / 恰好一次 create

# ── 11G.5.2.1 seam verify effective-value fidelity / seam 有效值保真 ────────


def _count_image_with_seam_config(
    tmp_path: Path,
    monkeypatch,
    capsys,
    *,
    seam_verify: bool,
    no_seam_verify: bool,
    run_id: str,
    resume: bool = False,
    force: bool = False,
    config_seam_verify: bool | None = None,
) -> tuple[int, bool, bool]:
    """Run count-image under a YAML config and capture both the persisted
    seam value and the settings passed to Runtime.create.
    在 YAML 配置下运行 count-image，捕获持久化 seam 值与传入
    Runtime.create 的 settings。"""
    import argparse

    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    effective_config = (
        config_seam_verify if config_seam_verify is not None else seam_verify
    )
    config = tmp_path / "seam-config.yaml"
    config.write_text(
        f"counting:\n  seam_verify: {str(effective_config).lower()}\n",
        encoding="utf-8",
    )
    _make_images(tmp_path / "imgs", ["img.png"])
    captured = {}
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)

    def capturing_create(cls, **kwargs):
        captured["seam_verify"] = kwargs["settings"].counting.seam_verify
        return runtime

    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(capturing_create)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    args = argparse.Namespace(
        config=str(config),
        image=str(tmp_path / "imgs" / "img.png"),
        question="how many vehicles?",
        target_spec=None,
        run_id=run_id,
        evaluate=False,
        render=False,
        resume=resume,
        force=force,
        no_seam_verify=no_seam_verify,
        max_qwen_calls=None,
        max_deepseek_calls=None,
    )
    code = run_count_image(args)
    request = json.loads(
        (tmp_path / "runs" / run_id / "run_request.json").read_text(encoding="utf-8")
    )
    if resume:
        capsys.readouterr()
    return code, captured["seam_verify"], request["count_seam_verify"]


def test_count_image_seam_truth_table_config_false_cli_default(
    tmp_path, monkeypatch, capsys
) -> None:
    """11G.5.2.1: config=false + CLI default → persisted false, executed false.
    config=false + CLI 默认 → 持久化 false、执行 false。"""
    code, executed, persisted = _count_image_with_seam_config(
        tmp_path, monkeypatch, capsys,
        seam_verify=False, no_seam_verify=False, run_id="t-false-default",
    )
    assert code == 0
    assert persisted is False
    assert executed is False


def test_count_image_seam_truth_table_config_true_cli_default(
    tmp_path, monkeypatch, capsys
) -> None:
    """config=true + CLI default → persisted true, executed true.
    config=true + CLI 默认 → 持久化 true、执行 true。"""
    code, executed, persisted = _count_image_with_seam_config(
        tmp_path, monkeypatch, capsys,
        seam_verify=True, no_seam_verify=False, run_id="t-true-default",
    )
    assert code == 0
    assert persisted is True
    assert executed is True


def test_count_image_seam_truth_table_config_true_no_seam(
    tmp_path, monkeypatch, capsys
) -> None:
    """config=true + --no-seam-verify → persisted false, executed false.
    config=true + --no-seam-verify → 持久化 false、执行 false。"""
    code, executed, persisted = _count_image_with_seam_config(
        tmp_path, monkeypatch, capsys,
        seam_verify=True, no_seam_verify=True, run_id="t-true-no-seam",
    )
    assert code == 0
    assert persisted is False
    assert executed is False


def test_count_image_seam_truth_table_config_false_no_seam(
    tmp_path, monkeypatch, capsys
) -> None:
    """config=false + --no-seam-verify → persisted false, executed false.
    config=false + --no-seam-verify → 持久化 false、执行 false。"""
    code, executed, persisted = _count_image_with_seam_config(
        tmp_path, monkeypatch, capsys,
        seam_verify=False, no_seam_verify=True, run_id="t-false-no-seam",
    )
    assert code == 0
    assert persisted is False
    assert executed is False


def test_count_image_seam_config_drift_persisted_false_survives_true_config(
    tmp_path, monkeypatch, capsys
) -> None:
    """11G.5.2.1: persisted false must survive a later config=true on force
    resume. 持久化 false 必须在 config 变为 true 后的 force resume 中存活。"""
    code, executed, persisted = _count_image_with_seam_config(
        tmp_path, monkeypatch, capsys,
        seam_verify=False, no_seam_verify=False, run_id="drift-false",
    )
    assert code == 0
    assert persisted is False and executed is False
    capsys.readouterr()
    # config now says true; force resume must still execute with false
    # 当前 config 变为 true；force resume 仍必须以 false 执行
    code, executed, _ = _count_image_with_seam_config(
        tmp_path, monkeypatch, capsys,
        seam_verify=True, no_seam_verify=False, run_id="drift-false",
        resume=True, force=True, config_seam_verify=True,
    )
    assert code == 0
    assert executed is False


def test_count_image_seam_config_drift_persisted_true_survives_false_config(
    tmp_path, monkeypatch, capsys
) -> None:
    """Persisted true must be actively restored when the current config says
    false on force resume. 持久化 true 必须在当前 config 为 false 的 force
    resume 中被主动恢复。"""
    code, executed, persisted = _count_image_with_seam_config(
        tmp_path, monkeypatch, capsys,
        seam_verify=True, no_seam_verify=False, run_id="drift-true",
    )
    assert code == 0
    assert persisted is True and executed is True
    capsys.readouterr()
    # config now says false; force resume must still execute with true
    # 当前 config 变为 false；force resume 仍必须以 true 执行
    code, executed, _ = _count_image_with_seam_config(
        tmp_path, monkeypatch, capsys,
        seam_verify=False, no_seam_verify=False, run_id="drift-true",
        resume=True, force=True, config_seam_verify=False,
    )
    assert code == 0
    assert executed is True

# ── dataset download/loader utilities (Task 11H2) / 数据集下载与加载工具 ────


class _FakeHub:
    """Fake huggingface_hub: records snapshot_download calls and materializes
    the local snapshot. fake huggingface_hub：记录 snapshot_download 调用并
    物化本地快照。"""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.calls: list[tuple] = []
        self.files = files if files is not None else {"data.txt": b"content"}

    def snapshot_download(self, *, repo_id, local_dir, token=None):
        self.calls.append((repo_id, local_dir, token))
        local_dir.mkdir(parents=True, exist_ok=True)
        for name, content in self.files.items():
            (local_dir / name).write_bytes(content)
        return str(local_dir)


def test_downloader_official_targets_mapping() -> None:
    from data.downloader import OFFICIAL_DOWNLOAD_TARGETS, dataset_download_target

    assert dataset_download_target("vrsbench") == "xiang709/VRSBench"
    assert dataset_download_target("mme_realworld") == "yifanzhang114/MME-RealWorld"
    assert dataset_download_target("xlrs_caption") == "initiacms/XLRS-Bench_caption_en"
    assert dataset_download_target("xlrs_grounding") == "initiacms/XLRS-Bench_visual_grounding_en"
    assert dataset_download_target("xlrs_lite") == "initiacms/XLRS-Bench-lite"
    assert dataset_download_target("levir_cc") == "lcybuaa/LEVIR-CC"
    assert set(OFFICIAL_DOWNLOAD_TARGETS) == {
        "vrsbench", "mme_realworld", "xlrs_caption", "xlrs_grounding",
        "xlrs_lite", "levir_cc",
    }
    with pytest.raises(ValueError, match="unknown download dataset"):
        dataset_download_target("nope")


def test_downloader_lazy_hub_import_and_stable_missing_dependency(
    tmp_path, monkeypatch
) -> None:
    """huggingface_hub is never imported at module import time; a missing
    dependency fails stably on explicit download. huggingface_hub 绝不在
    模块导入时被 import；缺失依赖在显式下载时稳定失败。"""
    import ast

    tree = ast.parse(
        Path("data/downloader.py").read_text(encoding="utf-8")
    )
    for node in tree.body:  # module-level imports only / 仅模块级 import
        if isinstance(node, ast.ImportFrom) and node.module == "huggingface_hub":
            raise AssertionError("downloader must import huggingface_hub lazily")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "huggingface_hub"
    import builtins

    import data.downloader as downloader_module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ValueError, match="huggingface_hub is required"):
        downloader_module._import_hub()
    with pytest.raises(ValueError, match="huggingface_hub is required"):
        downloader_module.download_dataset("vrsbench", root=tmp_path)


def test_downloader_mocked_download_and_zip_extraction(tmp_path, monkeypatch) -> None:
    """download_dataset calls snapshot_download exactly once with the official
    repo id and extracts archives into the snapshot.
    download_dataset 以官方 repo id 恰好调用一次 snapshot_download 并提取
    快照内的归档。"""
    import zipfile

    import data.downloader as downloader_module

    archive_bytes = None
    with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as zf:
        zf.writestr("annotations/rows.json", '{"a": 1}')
        zf.writestr("images/img.png", b"png")
    archive_bytes = (tmp_path / "bundle.zip").read_bytes()
    hub = _FakeHub(files={"bundle.zip": archive_bytes})
    monkeypatch.setattr(downloader_module, "_import_hub", lambda: hub)
    destination = downloader_module.download_dataset("vrsbench", root=tmp_path / "root")
    assert destination == (tmp_path / "root" / "vrsbench").resolve()
    assert hub.calls == [("xiang709/VRSBench", destination, None)]
    assert (destination / "annotations" / "rows.json").is_file()
    assert (destination / "images" / "img.png").is_file()


def test_downloader_safe_extraction_rejects_zip_slip(tmp_path) -> None:
    """Unsafe archive members (.., absolute, drive, UNC, reserved names) fail
    the whole archive and are never written outside. 不安全归档成员（..、
    绝对、drive、UNC、保留名）使整个归档失败，绝不写出到外部。"""
    import zipfile

    from data.downloader import extract_archives

    cases = {
        "dotdot": "../evil.txt",
        "dotdot2": "a/../../evil.txt",
        "absolute": "/tmp/evil.txt",
        "drive": "C:/evil.txt",
        "unc": r"\\server\share\evil.txt",
        "reserved": "CON.txt",
    }
    for label, member in cases.items():
        directory = tmp_path / label
        directory.mkdir()
        archive = directory / "bad.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(member, b"evil")
        with pytest.raises(ValueError, match="unsafe archive member"):
            extract_archives(directory)
        # nothing was written outside the archive / 归档外没有任何写入
        assert not (tmp_path / "evil.txt").exists()
        assert not (directory / "evil.txt").exists()
    # a safe archive extracts fully / 安全归档完整提取
    safe = tmp_path / "safe"
    safe.mkdir()
    with zipfile.ZipFile(safe / "ok.zip", "w") as zf:
        zf.writestr("good.txt", b"ok")
    extracted = extract_archives(safe)
    assert extracted == [safe / "ok.zip"]
    assert (safe / "good.txt").is_file()


def test_download_data_cli(tmp_path, monkeypatch, capsys) -> None:
    from application.commands import download_data as download_command_module
    from application.commands.download_data import run_download_data

    calls = []

    def fake_download(dataset, *, root):
        calls.append((dataset, root))
        destination = root / dataset
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    monkeypatch.setattr(download_command_module, "download_dataset", fake_download)
    code = run_download_data(
        _command_namespace(root=str(tmp_path / "root"), datasets=["vrsbench", "levir_cc"])
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert set(out["datasets"]) == {"vrsbench", "levir_cc"}
    assert calls == [("vrsbench", tmp_path / "root"), ("levir_cc", tmp_path / "root")]
    # unknown dataset fails stably / 未知数据集稳定失败
    monkeypatch.setattr(
        download_command_module,
        "download_dataset",
        lambda dataset, *, root: (_ for _ in ()).throw(ValueError("unknown")),
    )
    code = run_download_data(
        _command_namespace(root=str(tmp_path / "root"), datasets=["nope"])
    )
    assert code == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


class _LoaderFakeAdapter:
    """Recording adapter for loader tests. loader 测试的记录型适配器。"""

    name = "loader-demo"
    supported_tasks = frozenset({"general_vqa", "caption"})

    def __init__(self) -> None:
        self.probe_calls: list[tuple] = []
        self.iter_tasks: list[tuple] = []

    def probe(self, root, task=None):
        from data.adapters.base import AdapterProbe

        self.probe_calls.append((root, task))
        return AdapterProbe(
            dataset="loader-demo",
            version="1",
            sample_file=root / "samples.jsonl",
            observed_fields=("id",),
            sample_count=2,
            task=task,
            available_tasks=("general_vqa", "caption"),
        )

    def iter_samples(self, root, split, task):
        from data.schema import ImageRef, UnifiedSample

        self.iter_tasks.append((root, split, task))
        for index in range(2):
            yield UnifiedSample(
                sample_id=f"{task}-{index}",
                dataset="loader-demo",
                split=split,
                task=task,
                images=[ImageRef(image_id="i0", path="img.png", role="image")],
                question="q",
            )


class _LoaderDraftAdapter(_LoaderFakeAdapter):
    """Adapter with iter_drafts support. 支持 iter_drafts 的适配器。"""

    supported_tasks = frozenset()

    def iter_drafts(self, root, split):
        from data.schema import ImageRef, SampleDraft

        for index in range(2):
            yield SampleDraft(
                sample_id=f"draft-{index}",
                dataset="loader-demo",
                split=split,
                images=[ImageRef(image_id="i0", path="img.png", role="image")],
                question="q",
            )


def _loader_registry(tmp_path, monkeypatch, adapter) -> None:
    from data.registry import DatasetRegistry

    import data.loader as loader_module

    registry = DatasetRegistry()
    registry.register("loader-demo", lambda: adapter)
    monkeypatch.setattr(loader_module, "build_default_registry", lambda: registry)


def test_loader_samples_task_filter_limit_and_source_read_only(
    tmp_path, monkeypatch
) -> None:
    from data.loader import load_dataset_samples

    adapter = _LoaderFakeAdapter()
    _loader_registry(tmp_path, monkeypatch, adapter)
    samples = list(
        load_dataset_samples("loader-demo", root=tmp_path / "root", split="test")
    )
    assert [s.task for s in samples] == ["caption", "caption", "general_vqa", "general_vqa"]
    assert adapter.probe_calls == [
        (tmp_path / "root", "caption"),
        (tmp_path / "root", "general_vqa"),
    ]
    # task filter / task 过滤
    filtered = list(
        load_dataset_samples(
            "loader-demo", root=tmp_path / "root", split="test", task="caption"
        )
    )
    assert [s.task for s in filtered] == ["caption", "caption"]
    # limit / 限制
    limited = list(
        load_dataset_samples(
            "loader-demo", root=tmp_path / "root", split="test", limit=1
        )
    )
    assert len(limited) == 1
    # source read-only: the loader never mutates adapter or source files
    # 源只读：loader 绝不修改适配器或源文件
    assert not hasattr(adapter, "write")
    from data.schema import UnifiedSample

    assert isinstance(samples[0], UnifiedSample)


def test_loader_unknown_dataset_and_draft_iterator(tmp_path, monkeypatch) -> None:
    import data.loader as loader_module
    from data.loader import load_dataset_drafts, load_dataset_samples

    adapter = _LoaderFakeAdapter()
    _loader_registry(tmp_path, monkeypatch, adapter)
    with pytest.raises(Exception):
        list(load_dataset_samples("nope", root=tmp_path, split="test"))
    # non-draft adapter fails stably for drafts / 非 draft 适配器稳定失败
    with pytest.raises(TypeError, match="does not yield drafts"):
        list(load_dataset_drafts("loader-demo", root=tmp_path, split="test"))
    # draft adapter yields SampleDraft, never UnifiedSample
    # draft 适配器产出 SampleDraft，绝非 UnifiedSample
    from data.schema import SampleDraft, UnifiedSample

    draft_adapter = _LoaderDraftAdapter()
    _loader_registry(tmp_path, monkeypatch, draft_adapter)
    drafts = list(load_dataset_drafts("loader-demo", root=tmp_path, split="test"))
    assert len(drafts) == 2
    assert all(isinstance(draft, SampleDraft) for draft in drafts)
    assert not any(isinstance(draft, UnifiedSample) for draft in drafts)


def test_loader_and_downloader_never_import_each_other() -> None:
    """The loader never downloads and the downloader is not imported by the
    loader; neither module imports huggingface_hub at module level.
    loader 绝不下载，也不 import downloader；两模块都不在模块级 import
    huggingface_hub。"""
    import ast

    for path in ("data/loader.py", "data/downloader.py"):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in {
                    "huggingface_hub", "application", "models", "agents",
                }, f"{path} must not import {node.module}"
    loader_text = Path("data/loader.py").read_text(encoding="utf-8")
    assert "downloader" not in loader_text
    assert "snapshot_download" not in loader_text
    downloader_text = Path("data/downloader.py").read_text(encoding="utf-8")
    assert "load_dataset_samples" not in downloader_text

import csv

# ── LEVIR harmonization evaluator (Task 11I2) / LEVIR 协调评估器 ────────────


def _make_levir_pair(
    root: Path,
    split: str,
    name: str,
    *,
    t1_value: int = 100,
    t2_value: int = 150,
    with_label: bool = True,
    label_value: int = 255,
    corrupt: bool = False,
) -> None:
    """Create one synthetic LEVIR-style A/B pair (and optional label).
    创建一对合成 LEVIR 风格 A/B 图（及可选标签）。"""
    first_dir = root / "images" / split / "A"
    second_dir = root / "images" / split / "B"
    first_dir.mkdir(parents=True, exist_ok=True)
    second_dir.mkdir(parents=True, exist_ok=True)
    if corrupt:
        (first_dir / name).write_bytes(b"not an image")
        Image.new("RGB", (64, 64), (t2_value, t2_value, t2_value)).save(
            second_dir / name, format="PNG"
        )
        return
    Image.new("RGB", (64, 64), (t1_value, t1_value, t1_value)).save(
        first_dir / name, format="PNG"
    )
    Image.new("RGB", (64, 64), (t2_value, t2_value, t2_value)).save(
        second_dir / name, format="PNG"
    )
    if with_label:
        label_dir = root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        Image.new("L", (64, 64), label_value).save(label_dir / name, format="PNG")


def _run_levir_eval(monkeypatch, capsys, args: list[str]) -> int:
    import scripts.evaluate_levir_harmonization as levir_module

    monkeypatch.setattr("sys.argv", ["evaluate_levir_harmonization.py", *args])
    return levir_module.main()


def test_levir_evaluator_no_pairs_fails(monkeypatch, capsys, tmp_path) -> None:
    with pytest.raises(SystemExit, match="No paired PNG"):
        _run_levir_eval(
            monkeypatch, capsys,
            ["--root", str(tmp_path / "root"), "--split", "test",
             "--output-dir", str(tmp_path / "out")],
        )


def test_levir_evaluator_success_and_metrics(tmp_path, monkeypatch, capsys) -> None:
    root = tmp_path / "root"
    _make_levir_pair(root, "test", "pair_001.png")
    _make_levir_pair(root, "test", "pair_002.png", t1_value=90, t2_value=160)
    code = _run_levir_eval(
        monkeypatch, capsys,
        ["--root", str(root), "--split", "test", "--max-pairs", "10",
         "--output-dir", str(tmp_path / "out")],
    )
    assert code == 0
    out_dir = tmp_path / "out"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["processed"] == 2
    assert summary["failed"] == 0
    assert summary["split"] == "test"
    assert "pif_ratio" in summary["metrics"]
    assert summary["metrics"]["pif_ratio"]["n"] == 2
    assert "bootstrap_95_ci" in summary["metrics"]["pif_ratio"]
    assert len(summary["metrics"]["pif_ratio"]["bootstrap_95_ci"]) == 2
    grouped = json.loads((out_dir / "grouped_summary.json").read_text(encoding="utf-8"))
    assert grouped["labels_available"] is True
    assert grouped["has_change"]["pif_ratio"]["n"] == 2
    # CSV rows / CSV 行
    rows = list(csv.DictReader((out_dir / "metrics.csv").open(encoding="utf-8")))
    assert {row["pair"] for row in rows} == {"pair_001", "pair_002"}
    assert all(row["has_change"] == "True" for row in rows)
    assert all(row["changed_mad_before"] != "" for row in rows)
    failed = json.loads((out_dir / "failed_pairs.json").read_text(encoding="utf-8"))
    assert failed == []
    # stdout carries the summary / stdout 携带摘要
    assert json.loads(capsys.readouterr().out)["processed"] == 2


def test_levir_evaluator_isolated_failure(tmp_path, monkeypatch, capsys) -> None:
    root = tmp_path / "root"
    _make_levir_pair(root, "val", "good_001.png")
    _make_levir_pair(root, "val", "broken_002.png", corrupt=True)
    _make_levir_pair(root, "val", "good_003.png", t1_value=80, t2_value=170)
    code = _run_levir_eval(
        monkeypatch, capsys,
        ["--root", str(root), "--split", "val", "--max-pairs", "10",
         "--output-dir", str(tmp_path / "out")],
    )
    assert code == 2  # failures visible in the exit code
    out_dir = tmp_path / "out"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["processed"] == 2  # successful rows preserved
    assert summary["failed"] == 1
    failed = json.loads((out_dir / "failed_pairs.json").read_text(encoding="utf-8"))
    assert failed[0]["pair"] == "broken_002.png"
    assert failed[0]["error_type"] != ""
    rows = list(csv.DictReader((out_dir / "metrics.csv").open(encoding="utf-8")))
    assert {row["pair"] for row in rows} == {"good_001", "good_003"}


def test_levir_evaluator_labels_absent(tmp_path, monkeypatch, capsys) -> None:
    root = tmp_path / "root"
    _make_levir_pair(root, "test", "no_label.png", with_label=False)
    code = _run_levir_eval(
        monkeypatch, capsys,
        ["--root", str(root), "--split", "test", "--output-dir", str(tmp_path / "out")],
    )
    assert code == 0
    out_dir = tmp_path / "out"
    rows = list(csv.DictReader((out_dir / "metrics.csv").open(encoding="utf-8")))
    assert rows[0]["has_change"] == ""
    assert "changed_mad_before" not in rows[0]  # no masked metrics without labels
    grouped = json.loads((out_dir / "grouped_summary.json").read_text(encoding="utf-8"))
    assert grouped["labels_available"] is False
    assert grouped["has_change"] == {}


def test_levir_bootstrap_deterministic_and_calibration(
    tmp_path, monkeypatch, capsys
) -> None:
    import scripts.evaluate_levir_harmonization as levir_module

    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    first = levir_module._bootstrap_ci(values)
    second = levir_module._bootstrap_ci(values)
    assert first == second  # fixed seed, deterministic
    root = tmp_path / "root"
    _make_levir_pair(root, "test", "c_001.png")
    _make_levir_pair(root, "test", "c_002.png", t1_value=95, t2_value=155)
    code = _run_levir_eval(
        monkeypatch, capsys,
        ["--root", str(root), "--split", "test", "--output-dir", str(tmp_path / "out"),
         "--write-calibration"],
    )
    assert code == 0
    calibration = json.loads(
        (tmp_path / "out" / "calibration.json").read_text(encoding="utf-8")
    )
    assert calibration["sample_count"] == 2
    assert set(calibration["pif_ratio"]) == {"p01", "p05", "p50", "p95", "p99"}
    assert "pif_mad_improvement" in calibration
    assert "transform_gain_abs_max" in calibration
    assert calibration["algorithm_version"] != ""


def test_levir_evaluator_source_read_only_and_no_model_imports(
    tmp_path, monkeypatch, capsys
) -> None:
    """Source images stay byte-for-byte unchanged; the script imports no
    models/application/Judge/legacy packages. 源图像逐字节不变；脚本不 import
    models/application/Judge/legacy 包。"""
    import ast

    root = tmp_path / "root"
    _make_levir_pair(root, "test", "ro_001.png")
    first_file = root / "images" / "test" / "A" / "ro_001.png"
    second_file = root / "images" / "test" / "B" / "ro_001.png"
    first_before = first_file.read_bytes()
    second_before = second_file.read_bytes()
    label_before = (root / "labels" / "test" / "ro_001.png").read_bytes()
    code = _run_levir_eval(
        monkeypatch, capsys,
        ["--root", str(root), "--split", "test", "--output-dir", str(tmp_path / "out")],
    )
    assert code == 0
    assert first_file.read_bytes() == first_before
    assert second_file.read_bytes() == second_before
    assert (root / "labels" / "test" / "ro_001.png").read_bytes() == label_before
    # no model/application/judge/legacy imports / 无模型/应用/Judge/旧包导入
    tree = ast.parse(
        Path("scripts/evaluate_levir_harmonization.py").read_text(encoding="utf-8")
    )
    forbidden = {"models", "application", "evaluation", "workflows", "spacers_agent", "eval"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden, node.module

# ── 11J full functional parity fixtures / 全功能对等 fixtures ──────────────


_PARITY_FIXTURES = Path("tests/fixtures/parity")

_PARITY_DROP_KEYS = {
    "request_id", "sample_id", "run_id", "elapsed_seconds", "artifact_dir",
    "updated_at", "created_at", "result_path", "run_dir", "root",
    "code_version", "algorithm_version", "id", "config_hash",
    "inference_seconds", "git_commit", "prompt_hashes", "input",
    # Report V2 latency summaries intentionally reflect measured wall-clock
    # inference and therefore cannot be frozen into a functional parity file.
    "latency",
    # git_dirty records the local workspace state (dirty/clean checkout),
    # which is execution/environment provenance, not stable functional
    # parity behavior; it must never be frozen into a golden contract.
    # git_dirty 记录本地工作区状态（dirty/clean checkout），属于执行/环境
    # provenance，而非稳定功能 parity 行为；绝不能被冻结进 golden 契约。
    "git_dirty",
}


def _parity_normalize(value):
    """Strip timestamps, absolute paths, and unstable identities exactly like
    the fixture generator; absolute-path string values collapse to their
    basename so host/temp roots never leak. 与 fixture 生成器相同的去不稳定
    字段逻辑；绝对路径字符串值折叠为 basename，主机/临时根绝不泄漏。"""
    if isinstance(value, dict):
        return {
            key: _parity_normalize(item)
            for key, item in value.items()
            if key not in _PARITY_DROP_KEYS
            and not (isinstance(key, str) and "path" in key.lower())
        }
    if isinstance(value, list):
        return [_parity_normalize(item) for item in value]
    if isinstance(value, str) and (
        value.startswith("/") or len(value) >= 3 and value[1] == ":"
    ):
        return Path(value).name
    return value


def _parity_fixture(name: str) -> Any:
    return json.loads((_PARITY_FIXTURES / name).read_text(encoding="utf-8"))


def _parity_runtime(tmp_path, client, *, name="auto-demo") -> Runtime:
    """Runtime wired to a fake client and the manifest draft adapter.
    以 fake client 与 manifest draft 适配器接线的 Runtime。"""
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    components = assemble_runtime(
        settings,
        project_root=tmp_path,
        prompts_root=REPO_ROOT / "prompts",
        qwen_client=client,
        api_key=None,
    )
    registry = DatasetRegistry()
    registry.register(
        name, lambda: ManifestDraftAdapter(name, {"general_vqa", "caption"})
    )
    return Runtime(settings=settings, components=components, registry=registry)


def test_parity_ask_single_and_http(tmp_path) -> None:
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(imgs / "img.png")
    client = _FakeQwenClient()
    runtime = _parity_runtime(tmp_path, client)
    answer = asyncio.run(
        runtime.ask(image_dir=imgs, question="Is there a road?")
    )
    assert _parity_normalize(answer.model_dump(mode="json")) == _parity_fixture("ask_single.json")
    health = _parity_normalize(runtime.health_payload())
    expected_health = _parity_fixture("http_health.json")
    assert health["agents"] == expected_health["agents"]
    assert health["status"] == "ready"
    http_answer = asyncio.run(
        runtime.ask(image_dir=imgs, question="q", source="http_service")
    )
    assert _parity_normalize(http_answer.model_dump(mode="json")) == _parity_fixture("http_ask.json")


def test_parity_dataset_fresh_and_resume(tmp_path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    client = _FakeQwenClient()
    runtime = _parity_runtime(tmp_path, client)
    fresh = asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo", root=data_root, split="test", tasks=(),
                auto_task=True, run_id="parity-run",
            )
        )
    )
    fresh_norm = _parity_normalize(
        {key: value.model_dump(mode="json") for key, value in fresh.items()}
    )
    assert fresh_norm == _parity_fixture("dataset_fresh.json")
    resumed = asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo", root=data_root, split="test", tasks=(),
                auto_task=True, run_id="parity-run", resume=True,
            )
        )
    )
    resumed_norm = _parity_normalize(
        {key: value.model_dump(mode="json") for key, value in resumed.items()}
    )
    assert resumed_norm == _parity_fixture("dataset_resume.json")


def test_parity_count_image_summary(tmp_path, monkeypatch, capsys) -> None:
    from application.commands import count_image as count_image_module
    from application.commands.count_image import run_count_image

    _make_images(tmp_path / "imgs", ["img.png"])
    client = _FakeCountClient()
    runtime = _count_image_runtime(tmp_path, client)
    monkeypatch.setattr(
        count_image_module.Runtime, "create", classmethod(lambda cls, **kw: runtime)
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    assert run_count_image(
        _count_image_args(
            tmp_path,
            image=tmp_path / "imgs" / "img.png",
            run_id="parity-count",
        )
    ) == 0
    out = json.loads(capsys.readouterr().out)
    assert _parity_normalize(out) == _parity_fixture("count_image_summary.json")


def test_parity_run_init_manifest(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.run_init import run_run_init

    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    assert run_run_init(
        _command_namespace(run_id="parity-init", dataset="d", split="s", sample_filter=None)
    ) == 0
    capsys.readouterr()
    manifest = json.loads(
        (tmp_path / "runs" / "parity-init" / "manifest.json").read_text(encoding="utf-8")
    )
    # field contract: git_dirty must exist and be a bool (real Git state) —
    # its concrete value is environment provenance, not a parity key.
    # 字段契约：git_dirty 必须存在且为 bool（真实 Git 状态）——其具体值是
    # 环境 provenance，而非 parity 键。
    assert "git_dirty" in manifest
    assert isinstance(manifest["git_dirty"], bool)
    # stable functional parity / 稳定功能 parity
    assert _parity_normalize(manifest) == _parity_fixture("run_init_manifest.json")


def test_parity_normalization_ignores_git_dirty_environment_state() -> None:
    """Manifests differing only in git_dirty normalize identically, and the
    original objects are never mutated in place.
    仅 git_dirty 不同的 manifest 规范化后相等，且原对象绝不被原地修改。"""
    clean = {
        "dataset": "d",
        "split": "s",
        "git_dirty": False,
        "sample_filter": None,
    }
    dirty = {
        "dataset": "d",
        "split": "s",
        "git_dirty": True,
        "sample_filter": None,
    }
    assert _parity_normalize(clean) == _parity_normalize(dirty)
    # pure function: inputs untouched / 纯函数：输入未被修改
    assert clean["git_dirty"] is False
    assert dirty["git_dirty"] is True


def test_parity_evaluate_run_report(tmp_path, monkeypatch, capsys) -> None:
    """evaluate-run --deepseek over the parity dataset run refreshes the
    report bundle. The locked fixture continues to cover legacy deterministic
    fields while E4 Judge metrics are asserted as a separate new contract.
    evaluate-run --deepseek 对 parity 数据集运行刷新报告 bundle；规范化后
    旧确定性字段仍等于锁定 fixture，E4 Judge 指标作为独立新契约断言。"""
    from application.commands import evaluate_run as evaluate_run_module
    from application.commands.evaluate_run import run_evaluate_run

    _BoomCreateModel.arm(monkeypatch)
    judge = _OfflineFakeJudgeClient()
    monkeypatch.setattr(
        evaluate_run_module, "DeepSeekJudgeClient", lambda *a, **k: judge
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    client = _FakeQwenClient()
    runtime = _parity_runtime(tmp_path, client)
    asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo", root=data_root, split="test", tasks=(),
                auto_task=True, run_id="parity-run",
            )
        )
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    assert run_evaluate_run(_offline_args(run_id="parity-run", deepseek=True)) == 0
    capsys.readouterr()
    report = json.loads(
        (tmp_path / "runs" / "parity-run" / "report" / "report.json").read_text(encoding="utf-8")
    )
    semantic = report["tasks"][0]["judge_metrics"]["vqa_semantic_equivalence"]
    assert semantic == {
        "total": 1,
        "deterministic_exact_correct": 0,
        "eligible_mismatches": 1,
        "judged_mismatches": 1,
        "semantic_equivalent_mismatches": 1,
        "semantic_non_equivalent_mismatches": 0,
        "judge_failures": 0,
        "unresolved_mismatches": 0,
        "coverage": 1.0,
        "corrected_correct": 1,
        "lower_bound_score": 1.0,
        "complete": True,
        "score": 1.0,
    }
    legacy_parity = _parity_normalize(report)
    for task in legacy_parity["tasks"]:
        task.pop("judge_metrics", None)
    assert legacy_parity == _parity_fixture("evaluate_run.json")


def test_parity_summarize_file_mode(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.summarize_evaluations import run_summarize_evaluations
    from evaluation.metrics.vqa import VQADeterministicMetrics
    from evaluation.records import EvaluationRecord

    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(
            json.dumps(
                EvaluationRecord(
                    sample_id=f"s{i}",
                    task="general_vqa",
                    deterministic_metrics=VQADeterministicMetrics(exact_match=bool(i)),
                    judge_status="not_requested",
                ).model_dump(mode="json")
            )
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    assert run_summarize_evaluations(
        _command_namespace(run_id=None, input=str(records), output=None)
    ) == 0
    out = json.loads(capsys.readouterr().out)
    assert _parity_normalize(out) == _parity_fixture("summarize_file.json")


def test_parity_standard_evaluate_mocked(tmp_path, monkeypatch, capsys) -> None:
    from application.commands.standard_evaluate import run_standard_evaluate

    data_root = tmp_path / "data"
    _make_dataset(data_root)
    client = _FakeQwenClient()
    runtime = _parity_runtime(tmp_path, client)
    asyncio.run(
        runtime.run_dataset(
            DatasetRunOptions(
                dataset="auto-demo", root=data_root, split="test", tasks=(),
                auto_task=True, run_id="parity-run",
            )
        )
    )
    tool_dir = _fake_standard_tool(tmp_path, body=_FAKE_EVALUATOR_OK)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    assert run_standard_evaluate(
        _command_namespace(
            result=str(tmp_path / "runs" / "parity-run" / "predictions.jsonl"),
            tool_dir=str(tool_dir),
            output=None,
            python=sys.executable,
        )
    ) == 0
    out = json.loads(capsys.readouterr().out)
    assert _parity_normalize(out) == _parity_fixture("standard_evaluate.json")


def test_parity_download_data_mocked(tmp_path, monkeypatch, capsys) -> None:
    from application.commands import download_data as download_command_module
    from application.commands.download_data import run_download_data

    monkeypatch.setattr(
        download_command_module,
        "download_dataset",
        lambda dataset, *, root: root / dataset,
    )
    assert run_download_data(
        _command_namespace(root=str(tmp_path / "dl"), datasets=["vrsbench", "levir_cc"])
    ) == 0
    out = json.loads(capsys.readouterr().out)
    assert _parity_normalize(out) == _parity_fixture("download_data.json")


def test_parity_levir_evaluator_synthetic(tmp_path, monkeypatch, capsys) -> None:
    import scripts.evaluate_levir_harmonization as levir_module

    root = tmp_path / "root"
    _make_levir_pair(root, "test", "p1.png")
    _make_levir_pair(root, "test", "p2.png")
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_levir_harmonization.py", "--root", str(root), "--split", "test",
         "--output-dir", str(tmp_path / "out"), "--write-calibration"],
    )
    code = levir_module.main()
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert _parity_normalize(summary) == _parity_fixture("levir_summary.json")
