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

from application.bootstrap import assemble_runtime
from application.runtime import Runtime
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
