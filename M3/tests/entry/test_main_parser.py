"""Parser contract of the root main.py entry. / 根入口 main.py 的解析器契约。"""

from __future__ import annotations

import pytest

from main import build_parser


def test_no_subcommand_defaults_to_serve():
    """No subcommand defaults to serve with default host/port.
    无子命令时默认 serve 并使用默认 host/port。
    """

    args = build_parser().parse_args([])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_config_default_is_repo_default_yaml():
    args = build_parser().parse_args([])
    assert args.config.name == "default.yaml"


def test_serve_defaults():
    args = build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_serve_overrides():
    args = build_parser().parse_args(["serve", "--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_ask_defaults():
    args = build_parser().parse_args(["ask", "--images-dir", "img"])
    assert args.command == "ask"
    assert args.images_dir.name == "img"
    assert args.question == ""
    assert args.task == "auto"
    assert args.output is None


def test_ask_missing_images_dir_fails():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask", "--question", "hi"])


def test_ask_invalid_task_fails():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask", "--images-dir", "img", "--task", "bogus"])


def test_ask_all_valid_tasks_parse():
    for task in ("auto", "counting", "fine_grained_counting", "change_caption", "change_qa",
                 "grounding", "spatial_relation", "scene_classification", "general_vqa",
                 "caption", "multiple_choice_vqa"):
        args = build_parser().parse_args(["ask", "--images-dir", "img", "--task", task])
        assert args.task == task


def test_run_dataset_parses_with_defaults():
    args = build_parser().parse_args(
        ["run-dataset", "--dataset", "VRSBench", "--root", "data", "--split", "validation"]
    )
    assert args.dataset == "VRSBench"
    assert args.split == "validation"
    assert args.task is None
    assert args.run_id is None
    assert args.max_samples == 0
    assert args.start_index == 0
    assert args.sample_concurrency == 1
    assert args.resume is False
    assert args.fail_fast is False
    assert args.evaluate is True
    assert args.judge_policy == "all"


def test_run_dataset_overrides():
    args = build_parser().parse_args(
        [
            "run-dataset",
            "--dataset", "LEVIR-CC",
            "--root", "data",
            "--split", "test",
            "--task", "change_caption,change_qa",
            "--run-id", "run-1",
            "--max-samples", "5",
            "--start-index", "2",
            "--sample-concurrency", "2",
            "--resume",
            "--fail-fast",
            "--no-evaluate",
            "--judge-policy", "errors-only",
        ]
    )
    assert args.task == "change_caption,change_qa"
    assert args.run_id == "run-1"
    assert args.max_samples == 5
    assert args.start_index == 2
    assert args.sample_concurrency == 2
    assert args.resume is True
    assert args.fail_fast is True
    assert args.evaluate is False
    assert args.judge_policy == "errors-only"


def test_run_dataset_requires_root_and_split():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-dataset", "--dataset", "VRSBench"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-dataset", "--dataset", "VRSBench", "--root", "data"])
