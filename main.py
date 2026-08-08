"""Minimal public entry point: `python main.py run-dataset ...`.

最小公开入口：`python main.py run-dataset ...`。本模块保持极薄：解析参数 →
加载配置 → 组装运行时 → 构造 DatasetRunOptions → 委托
Runtime.run_dataset → 输出汇总与运行目录 → 退出码。绝不包含 Agent fallback、
数据集循环、TaskResolver 逻辑、模型业务逻辑或报告聚合。架构规则要求
main.py 只能 import application（含 stdlib）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from application.runtime import Runtime, build_dataset_run_options
from application.settings import load_settings

EXIT_OK = 0
EXIT_ARGUMENT = 2
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """The only supported public surface. 唯一受支持的公开面。"""

    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("--config", default=None, help="settings YAML path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_dataset = subparsers.add_parser("run-dataset", help="run one dataset")
    run_dataset.add_argument("--dataset", required=True)
    run_dataset.add_argument("--root", required=True)
    run_dataset.add_argument("--split", required=True)
    run_dataset.add_argument("--task", default=None, help="comma-separated tasks")
    run_dataset.add_argument("--auto-task", action="store_true", help="resolve tasks per sample")
    run_dataset.add_argument("--run-id", default=None)
    run_dataset.add_argument("--resume", action="store_true")
    run_dataset.add_argument(
        "--evaluate",
        dest="evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="deterministic evaluation on by default",
    )
    run_dataset.add_argument(
        "--judge-policy",
        choices=("none", "errors-only", "all"),
        default="none",
        help="offline by default: no external DeepSeek unless requested",
    )
    run_dataset.add_argument("--max-samples", "--limit", dest="limit", type=int, default=None)
    run_dataset.add_argument("--start-index", type=int, default=0)
    run_dataset.add_argument("--shard-index", type=int, default=0)
    run_dataset.add_argument("--shard-count", type=int, default=1)
    run_dataset.add_argument("--sample-concurrency", type=int, default=1)
    run_dataset.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested command and return the process exit code.
    运行请求的命令并返回进程退出码。"""

    args = build_parser().parse_args(argv)
    if args.command != "run-dataset":
        _print_error("unsupported command")
        return EXIT_ARGUMENT
    return _run_dataset(args)


def _run_dataset(args: argparse.Namespace) -> int:
    """run-dataset: parse → settings → runtime → options → run → summary.
    run-dataset：解析 → 配置 → 运行时 → 选项 → 运行 → 汇总。"""

    if args.resume and not args.run_id:
        # Contract failure before any runtime/model initialization.
        # 契约失败，先于任何运行时/模型初始化。
        _print_error("--resume requires --run-id")
        return EXIT_ARGUMENT
    tasks = (
        tuple(item.strip() for item in args.task.split(",") if item.strip())
        if args.task
        else ()
    )
    if tasks and args.auto_task:
        _print_error("--task and --auto-task are mutually exclusive")
        return EXIT_ARGUMENT
    if not tasks and not args.auto_task:
        _print_error("run-dataset requires --task or --auto-task")
        return EXIT_ARGUMENT
    try:
        settings = load_settings(
            Path(args.config) if args.config else None,
            environ=os.environ,
        )
        api_key = os.environ.get(settings.models.deepseek.api_key_env) or None
        runtime = Runtime.create(
            settings=settings,
            project_root=Path(__file__).resolve().parent,
            api_key=api_key,
        )
        options = build_dataset_run_options(
            dataset=args.dataset,
            root=Path(args.root),
            split=args.split,
            tasks=tasks,
            auto_task=args.auto_task,
            run_id=args.run_id,
            resume=args.resume,
            limit=args.limit,
            start_index=args.start_index,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            sample_concurrency=args.sample_concurrency,
            evaluate=args.evaluate,
            judge_policy=args.judge_policy,
            fail_fast=args.fail_fast,
        )
        summaries = asyncio.run(runtime.run_dataset(options))
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as error:
        # Public output never carries raw exception text or secrets.
        # 公共输出绝不携带原始异常文本或密钥。
        _print_error(f"{type(error).__name__}")
        return EXIT_RUNTIME
    run_id = (
        next(iter(summaries.values())).run_id
        if summaries
        else options.run_id or f"{options.dataset}-{options.split}"
    )
    payload = {
        "status": "ok",
        "run_dir": str(settings.runs.root / run_id),
        "summaries": {
            task: summary.model_dump(mode="json")
            for task, summary in summaries.items()
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


def _print_error(message: str) -> None:
    print(json.dumps({"status": "failed", "error": message}), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
