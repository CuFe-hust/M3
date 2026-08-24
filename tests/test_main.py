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
    assert args.judge_policy == "none"  # DeepSeek verification is opt-in
    assert args.auto_task is False
    assert args.resume is False
    assert args.limit is None
    assert args.sample_concurrency == 1
    assert args.fail_fast is False


def test_main_loads_project_dotenv_before_dispatch(monkeypatch) -> None:
    loaded = []
    monkeypatch.setattr(main_module, "load_dotenv", lambda: loaded.append(True))
    monkeypatch.setattr(main_module, "run_list_datasets", lambda args: 0)

    assert main_module.main(["list-datasets"]) == 0
    assert loaded == [True]


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


def test_parser_accepts_explicit_judge_policies() -> None:
    for policy in ("all", "errors-only"):
        args = main_module.build_parser().parse_args(
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
                policy,
            ]
        )
        assert args.judge_policy == policy


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


def test_missing_task_and_auto_task_defaults_to_adapter_tasks(
    capsys, monkeypatch
) -> None:
    """Neither --task nor --auto-task selects adapter.supported_tasks and
    never invokes visual planning. 两者都不给时选择 adapter.supported_tasks，
    绝不调用视觉规划器。"""
    import application.commands.run_dataset as run_dataset_module

    captured = {}

    class _FakeRuntime:
        async def run_dataset(self, options):
            captured["options"] = options
            return {}

    monkeypatch.setattr(
        run_dataset_module.Runtime,
        "create",
        classmethod(lambda cls, **kw: _FakeRuntime()),
    )
    code = main_module.main(
        ["run-dataset", "--dataset", "d", "--root", "r", "--split", "test"]
    )
    assert code == 0
    assert captured["options"].tasks is None  # adapter-default mode
    assert captured["options"].auto_task is False


def _fake_runtime_factory(runtime):
    """Patch Runtime.create so main wires a prepared runtime.
    修补 Runtime.create 使 main 使用准备好的运行时。"""

    import application.runtime as runtime_module

    def fake_create(cls, **kwargs):
        return runtime

    runtime_module.Runtime.create = classmethod(fake_create)  # type: ignore[assignment]


def test_main_maps_args_to_options_and_prints_summary(capsys, monkeypatch) -> None:
    import application.commands.run_dataset as run_dataset_module
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

    monkeypatch.setattr(run_dataset_module.Runtime, "create", classmethod(lambda cls, **kw: _FakeRuntime()))
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
    import application.commands.run_dataset as run_dataset_module
    captured = {}

    class _FakeRuntime:
        async def run_dataset(self, options):
            captured["options"] = options
            return {}

    monkeypatch.setattr(run_dataset_module.Runtime, "create", classmethod(lambda cls, **kw: _FakeRuntime()))
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
    import application.commands.run_dataset as run_dataset_module
    class _BoomRuntime:
        async def run_dataset(self, options):
            raise RuntimeError("secret-raw-detail C:\\path sk-key")

    monkeypatch.setattr(run_dataset_module.Runtime, "create", classmethod(lambda cls, **kw: _BoomRuntime()))
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
    import application.commands.run_dataset as run_dataset_module
    calls = []

    class _FakeRuntime:
        async def run_dataset(self, options):
            return {}

    def fake_create(cls, **kwargs):
        calls.append(kwargs)
        return _FakeRuntime()

    monkeypatch.setattr(run_dataset_module.Runtime, "create", classmethod(fake_create))
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


def test_resume_without_run_id_fails_before_runtime_init(capsys, monkeypatch) -> None:
    """--resume without --run-id is a contract failure detected before any
    runtime/model initialization. --resume 无 --run-id 是契约失败，先于任何
    运行时/模型初始化。"""
    import application.commands.run_dataset as run_dataset_module

    def boom_create(cls, **kwargs):
        raise AssertionError("runtime must not be created")

    monkeypatch.setattr(run_dataset_module.Runtime, "create", classmethod(boom_create))
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
            "--resume",
        ]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "--resume requires --run-id"


def test_cli_reports_actual_run_dir_from_summary(capsys, monkeypatch) -> None:
    """The CLI must output the actual generated run directory, never a
    recomputed dataset-split default. CLI 必须输出实际生成的 run 目录，绝不
    输出重新计算的 dataset-split 默认值。"""
    import application.commands.run_dataset as run_dataset_module
    from workflows.schema import DatasetRunSummary

    class _FakeRuntime:
        async def run_dataset(self, options):
            return {
                "auto": DatasetRunSummary(
                    run_id="20260808T120000Z-a1b2c3d4",
                    dataset="d",
                    split="test",
                    task="auto",
                    total=1,
                    succeeded=1,
                    partial=0,
                    failed=0,
                    skipped=0,
                )
            }

    monkeypatch.setattr(run_dataset_module.Runtime, "create", classmethod(lambda cls, **kw: _FakeRuntime()))
    code = main_module.main(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "r",
            "--split",
            "test",
            "--auto-task",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_dir"].endswith("20260808T120000Z-a1b2c3d4")
    assert "d-test" not in out["run_dir"]


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


# ── manual ask CLI (Task 11A) / 手动 ask CLI ───────────────────────────────


def test_parser_ask_defaults() -> None:
    args = main_module.build_parser().parse_args(
        ["ask", "--images-dir", "imgs", "--question", "q?", "--task", "caption"]
    )
    assert args.command == "ask"
    assert args.images_dir == "imgs"
    assert args.question == "q?"
    assert args.task == "caption"
    assert args.output is None


def test_parser_ask_auto_defaults_and_choices() -> None:
    import pytest

    args = main_module.build_parser().parse_args(["ask", "--images-dir", "imgs"])
    assert args.task == "auto"
    assert args.question == ""
    with pytest.raises(SystemExit):
        main_module.build_parser().parse_args(
            ["ask", "--images-dir", "imgs", "--task", "bogus"]
        )


def test_ask_help_exits_zero() -> None:
    import pytest

    with pytest.raises(SystemExit) as exc:
        main_module.build_parser().parse_args(["ask", "--help"])
    assert exc.value.code == 0


def _ask_result(request_id: str = "manual-20260808-000000-aabbcc"):
    from application.runtime import PublicAnswer

    return PublicAnswer(
        request_id=request_id,
        task="general_vqa",
        agent="general_vqa_agent",
        status="completed",
        answer="yes",
        elapsed_seconds=0.5,
        artifact_dir="service/requests/" + request_id,
    )


def test_ask_cli_maps_args_prints_and_writes_output(
    capsys, tmp_path, monkeypatch
) -> None:
    import application.commands.ask as ask_module

    captured = {}

    class _FakeRuntime:
        async def ask(self, **kwargs):
            captured["kwargs"] = kwargs
            return _ask_result()

    monkeypatch.setattr(
        ask_module.Runtime, "create", classmethod(lambda cls, **kw: _FakeRuntime())
    )
    output = tmp_path / "out" / "answer.json"
    code = main_module.main(
        [
            "ask",
            "--images-dir",
            "imgs",
            "--question",
            "q",
            "--task",
            "general_vqa",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert captured["kwargs"]["question"] == "q"
    assert captured["kwargs"]["task"] == "general_vqa"
    assert captured["kwargs"]["image_dir"] == Path("imgs")
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "completed"
    assert out["task"] == "general_vqa"
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["request_id"] == out["request_id"]


def test_ask_cli_error_prints_stable_type(capsys, monkeypatch) -> None:
    import application.commands.ask as ask_module
    class _BoomRuntime:
        async def ask(self, **kwargs):
            raise ValueError("no supported images found in: C:/secret")

    monkeypatch.setattr(
        ask_module.Runtime, "create", classmethod(lambda cls, **kw: _BoomRuntime())
    )
    code = main_module.main(["ask", "--images-dir", "imgs"])
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "failed"
    assert error["error"] == "ValueError"
    assert "secret" not in json.dumps(error)



# ── serve CLI (Task 11B) / serve CLI ───────────────────────────────────────


def test_parser_implicit_serve_defaults() -> None:
    args = main_module.build_parser().parse_args([])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_parser_serve_explicit() -> None:
    args = main_module.build_parser().parse_args(
        ["serve", "--host", "0.0.0.0", "--port", "9000"]
    )
    assert args.command == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_serve_help_exits_zero() -> None:
    import pytest

    with pytest.raises(SystemExit) as exc:
        main_module.build_parser().parse_args(["serve", "--help"])
    assert exc.value.code == 0


def test_main_no_args_dispatches_implicit_serve(monkeypatch) -> None:
    captured = {}

    def fake_run_serve(args):
        captured["command"] = args.command
        captured["host"] = args.host
        captured["port"] = args.port
        return 0

    monkeypatch.setattr(main_module, "run_serve", fake_run_serve)
    code = main_module.main([])
    assert code == 0
    assert captured == {"command": "serve", "host": "127.0.0.1", "port": 8000}


def test_main_serve_dispatches_run_serve(monkeypatch) -> None:
    captured = {}

    def fake_run_serve(args):
        captured["command"] = args.command
        return 0

    monkeypatch.setattr(main_module, "run_serve", fake_run_serve)
    code = main_module.main(["serve", "--port", "9001"])
    assert code == 0
    assert captured == {"command": "serve"}

def test_each_command_help_exits_zero() -> None:
    import pytest

    for command in (
        "serve",
        "ask",
        "run-init",
        "health",
        "list-datasets",
        "smoke-qwen",
        "resume-run",
        "inspect-data",
        "run-dataset",
    ):
        with pytest.raises(SystemExit) as exc:
            main_module.build_parser().parse_args([command, "--help"])
        assert exc.value.code == 0


def test_main_dispatches_operational_commands(monkeypatch) -> None:
    mapping = {
        "run-init": "run_run_init",
        "health": "run_health",
        "list-datasets": "run_list_datasets",
        "smoke-qwen": "run_smoke_qwen",
        "resume-run": "run_resume_run",
        "inspect-data": "run_inspect_data",
    }
    argv = {
        "run-init": ["run-init"],
        "health": ["health", "qwen"],
        "list-datasets": ["list-datasets"],
        "smoke-qwen": ["smoke-qwen", "--image", "i", "--question", "q"],
        "resume-run": ["resume-run", "--run-id", "x"],
        "inspect-data": ["inspect-data", "--root", "r"],
    }
    dispatched = []

    for command, attribute in mapping.items():

        def fake_run(args, **kwargs):
            dispatched.append(command)
            return 0

        monkeypatch.setattr(main_module, attribute, fake_run)
        code = main_module.main(argv[command])
        assert code == 0

    assert dispatched == list(mapping)


def test_parser_operational_command_defaults() -> None:
    args = main_module.build_parser().parse_args(["inspect-data", "--root", "r"])
    assert args.command == "inspect-data"
    assert args.scan_mode == "quick"
    assert args.output is None
    health = main_module.build_parser().parse_args(["health", "deepseek"])
    assert health.component == "deepseek"
    assert health.live is False
    run_init = main_module.build_parser().parse_args(["run-init", "--run-id", "x"])
    assert run_init.dataset is None
    assert run_init.split is None
    assert run_init.sample_filter is None


# ── run-dataset operational surface (Task 11C2) / run-dataset 运维面 ────────


def test_parser_shard_count_num_shards_alias() -> None:
    via_alias = main_module.build_parser().parse_args(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "r",
            "--split",
            "s",
            "--num-shards",
            "4",
        ]
    )
    assert via_alias.shard_count == 4
    via_flag = main_module.build_parser().parse_args(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "r",
            "--split",
            "s",
            "--shard-count",
            "4",
        ]
    )
    assert via_flag.shard_count == 4


def test_run_dataset_sample_ids_file(tmp_path, monkeypatch, capsys) -> None:
    import application.commands.run_dataset as run_dataset_module

    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("a1 a2" + chr(10) + "a3", encoding="utf-8")
    captured = {}

    class _FakeRuntime:
        async def run_dataset(self, options):
            captured["options"] = options
            return {}

    monkeypatch.setattr(
        run_dataset_module.Runtime,
        "create",
        classmethod(lambda cls, **kw: _FakeRuntime()),
    )
    code = main_module.main(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "r",
            "--split",
            "s",
            "--sample-ids",
            str(ids_file),
        ]
    )
    assert code == 0
    assert captured["options"].sample_ids == {"a1", "a2", "a3"}
    assert captured["options"].tasks is None  # adapter-default mode preserved


def test_run_dataset_sample_ids_missing_file_fails(tmp_path, monkeypatch, capsys) -> None:
    import application.commands.run_dataset as run_dataset_module

    def boom(cls, **kw):
        raise AssertionError("runtime must not be created")

    monkeypatch.setattr(run_dataset_module.Runtime, "create", classmethod(boom))
    code = main_module.main(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            "r",
            "--split",
            "s",
            "--sample-ids",
            str(tmp_path / "missing.txt"),
        ]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert "cannot read sample ids file" in error["error"]


# ── counting maintenance commands (Task 11D) / 计数维护命令 ─────────────────


def test_counting_commands_help_exits_zero() -> None:
    import pytest

    for command in ("count-image", "render-count", "summarize-evaluations"):
        with pytest.raises(SystemExit) as exc:
            main_module.build_parser().parse_args([command, "--help"])
        assert exc.value.code == 0


def test_main_dispatches_counting_commands(monkeypatch) -> None:
    mapping = {
        "count-image": "run_count_image",
        "render-count": "run_render_count",
        "summarize-evaluations": "run_summarize_evaluations",
    }
    argv = {
        "count-image": ["count-image", "--image", "i.png", "--question", "q"],
        "render-count": ["render-count", "--image", "i.png", "--result", "r.json", "--output", "o.png"],
        "summarize-evaluations": ["summarize-evaluations", "--run-id", "x"],
    }
    dispatched = []

    for command, attribute in mapping.items():

        def fake_run(args, **kwargs):
            dispatched.append(command)
            return 0

        monkeypatch.setattr(main_module, attribute, fake_run)
        assert main_module.main(argv[command]) == 0

    assert dispatched == list(mapping)


# ── offline evaluation commands (Task 11E) / 离线评估命令 ───────────────────


def test_offline_commands_help_exits_zero() -> None:
    import pytest

    for command in ("evaluate-run", "judge-vqa-run"):
        with pytest.raises(SystemExit) as exc:
            main_module.build_parser().parse_args([command, "--help"])
        assert exc.value.code == 0


def test_main_dispatches_offline_commands(monkeypatch) -> None:
    mapping = {
        "evaluate-run": "run_evaluate_run",
        "judge-vqa-run": "run_judge_vqa_run",
    }
    argv = {
        "evaluate-run": ["evaluate-run", "--run-id", "x", "--deepseek", "--only-missing", "--force-judge"],
        "judge-vqa-run": ["judge-vqa-run", "--run-id", "x", "--force"],
    }
    dispatched = []

    for command, attribute in mapping.items():

        def fake_run(args, **kwargs):
            dispatched.append(command)
            return 0

        monkeypatch.setattr(main_module, attribute, fake_run)
        assert main_module.main(argv[command]) == 0

    assert dispatched == list(mapping)


def test_parser_offline_command_defaults() -> None:
    args = main_module.build_parser().parse_args(["evaluate-run", "--run-id", "x"])
    assert args.deepseek is False
    assert args.only_missing is False
    assert args.force_judge is False
    judge = main_module.build_parser().parse_args(["judge-vqa-run", "--run-id", "x"])
    assert judge.force is False


# ── standard evaluate CLI (Task 11F) / 标准评估 CLI ────────────────────────


def test_standard_evaluate_help_exits_zero() -> None:
    import pytest

    with pytest.raises(SystemExit) as exc:
        main_module.build_parser().parse_args(["standard-evaluate", "--help"])
    assert exc.value.code == 0


def test_main_dispatches_standard_evaluate(monkeypatch) -> None:
    captured = []

    def fake_run(args, **kwargs):
        captured.append((args.command, args.result, args.tool_dir, args.python))
        return 0

    monkeypatch.setattr(main_module, "run_standard_evaluate", fake_run)
    code = main_module.main(
        ["standard-evaluate", "--result", "r.jsonl", "--tool-dir", "tools", "--python", "py"]
    )
    assert code == 0
    assert captured == [("standard-evaluate", "r.jsonl", "tools", "py")]


def test_parser_standard_evaluate_defaults() -> None:
    args = main_module.build_parser().parse_args(
        ["standard-evaluate", "--result", "r.jsonl"]
    )
    assert args.tool_dir is None
    assert args.output is None
    assert args.python is None


# ── download-data CLI (Task 11H2) / 数据集下载 CLI ─────────────────────────


def test_download_data_help_exits_zero() -> None:
    import pytest

    with pytest.raises(SystemExit) as exc:
        main_module.build_parser().parse_args(["download-data", "--help"])
    assert exc.value.code == 0


def test_main_dispatches_download_data(monkeypatch) -> None:
    captured = []

    def fake_run(args, **kwargs):
        captured.append((args.command, args.root, args.datasets))
        return 0

    monkeypatch.setattr(main_module, "run_download_data", fake_run)
    code = main_module.main(
        ["download-data", "--root", "r", "--datasets", "vrsbench", "levir_cc"]
    )
    assert code == 0
    assert captured == [("download-data", "r", ["vrsbench", "levir_cc"])]
