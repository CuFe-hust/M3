"""Contract tests for the minimal public entry point: parser defaults,
mutual exclusion, options mapping, single runtime creation, stable public
errors, and summary output.

最小公开入口契约测试：parser 默认值、互斥、选项映射、单次运行时创建、稳定
公共错误与汇总输出。
"""

from __future__ import annotations

import json
from pathlib import Path

import main as main_module


def test_parser_run_dataset_defaults() -> None:
    args = main_module.build_parser().parse_args(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "data",
            "--split",
            "test",
            "--task",
            "general_vqa,caption",
        ]
    )
    assert args.command == "run-dataset"
    assert args.evaluate is True  # deterministic evaluation on by default
    assert args.judge_policy == "none"  # external DeepSeek off by default
    assert args.auto_task is False
    assert args.resume is False
    assert args.limit is None
    assert args.sample_concurrency == 1
    assert args.fail_fast is False


def test_parser_rejects_unknown_judge_policy() -> None:
    import pytest

    with pytest.raises(SystemExit):
        main_module.build_parser().parse_args(
            [
                "run-dataset",
                "--dataset",
                "d",
                "--root",
                "r",
                "--split",
                "test",
                "--task",
                "caption",
                "--judge-policy",
                "sometimes",
            ]
        )


def test_task_and_auto_task_mutually_exclusive(capsys) -> None:
    code = main_module.main(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "r",
            "--split",
            "test",
            "--task",
            "caption",
            "--auto-task",
        ]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert "mutually exclusive" in error["error"]


def test_missing_task_and_auto_task_fails(capsys) -> None:
    code = main_module.main(
        ["run-dataset", "--dataset", "d", "--root", "r", "--split", "test"]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert "--task or --auto-task" in error["error"]


def _fake_runtime_factory(runtime):
    """Patch Runtime.create so main wires a prepared runtime.
    修补 Runtime.create 使 main 使用准备好的运行时。"""

    import application.runtime as runtime_module

    def fake_create(cls, **kwargs):
        return runtime

    runtime_module.Runtime.create = classmethod(fake_create)  # type: ignore[assignment]


def test_main_maps_args_to_options_and_prints_summary(capsys, monkeypatch) -> None:
    from workflows.schema import DatasetRunOptions

    captured = {}

    class _FakeRuntime:
        async def run_dataset(self, options: DatasetRunOptions):
            captured["options"] = options
            from workflows.schema import DatasetRunSummary

            return {
                "general_vqa": DatasetRunSummary(
                    run_id="cli-run",
                    dataset="d",
                    split="test",
                    task="general_vqa",
                    total=1,
                    succeeded=1,
                    partial=0,
                    failed=0,
                    skipped=0,
                )
            }

    monkeypatch.setattr(main_module.Runtime, "create", classmethod(lambda cls, **kw: _FakeRuntime()))
    code = main_module.main(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "data",
            "--split",
            "test",
            "--task",
            "general_vqa, caption",
            "--run-id",
            "cli-run",
            "--limit",
            "5",
            "--start-index",
            "1",
            "--shard-index",
            "1",
            "--shard-count",
            "2",
            "--sample-concurrency",
            "3",
            "--fail-fast",
        ]
    )
    assert code == 0
    options = captured["options"]
    assert options.dataset == "d"
    assert options.root == Path("data")
    assert options.tasks == ("general_vqa", "caption")
    assert options.auto_task is False
    assert options.run_id == "cli-run"
    assert options.limit == 5
    assert options.start_index == 1
    assert options.shard_index == 1
    assert options.shard_count == 2
    assert options.sample_concurrency == 3
    assert options.evaluate is True
    assert options.judge_policy == "none"
    assert options.fail_fast is True
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["summaries"]["general_vqa"]["succeeded"] == 1


def test_main_disables_judge_when_not_evaluating(capsys, monkeypatch) -> None:
    captured = {}

    class _FakeRuntime:
        async def run_dataset(self, options):
            captured["options"] = options
            return {}

    monkeypatch.setattr(main_module.Runtime, "create", classmethod(lambda cls, **kw: _FakeRuntime()))
    code = main_module.main(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "r",
            "--split",
            "test",
            "--task",
            "caption",
            "--no-evaluate",
            "--judge-policy",
            "all",
        ]
    )
    assert code == 0
    assert captured["options"].evaluate is False
    assert captured["options"].judge_policy == "none"  # judge never applies


def test_no_raw_exception_public(capsys, monkeypatch) -> None:
    class _BoomRuntime:
        async def run_dataset(self, options):
            raise RuntimeError("secret-raw-detail C:\\path sk-key")

    monkeypatch.setattr(main_module.Runtime, "create", classmethod(lambda cls, **kw: _BoomRuntime()))
    code = main_module.main(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "r",
            "--split",
            "test",
            "--task",
            "caption",
        ]
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "failed"
    assert error["error"] == "RuntimeError"
    assert "secret-raw-detail" not in json.dumps(error)
    assert "sk-key" not in json.dumps(error)


def test_runtime_created_exactly_once_per_invocation(capsys, monkeypatch) -> None:
    """One run-dataset invocation creates the runtime (and therefore the Qwen
    client) exactly once — reused across tasks and samples.
    一次 run-dataset 调用恰好创建一次运行时（即 Qwen 客户端）——跨 task 与
    样本复用。"""
    calls = []

    class _FakeRuntime:
        async def run_dataset(self, options):
            return {}

    def fake_create(cls, **kwargs):
        calls.append(kwargs)
        return _FakeRuntime()

    monkeypatch.setattr(main_module.Runtime, "create", classmethod(fake_create))
    code = main_module.main(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "r",
            "--split",
            "test",
            "--task",
            "general_vqa,caption",
        ]
    )
    assert code == 0
    assert len(calls) == 1


def test_main_imports_only_application_and_stdlib() -> None:
    """The architecture rule: main.py must import only application plus the
    standard library. 架构规则：main.py 只能 import application 与标准库。"""
    import ast

    tree = ast.parse(Path(main_module.__file__).read_text(encoding="utf-8"))
    internal = {
        "data",
        "models",
        "agents",
        "routing",
        "workflows",
        "evaluation",
        "reporting",
        "application",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in internal - {"application"}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            assert node.module.split(".")[0] not in internal - {"application"}
