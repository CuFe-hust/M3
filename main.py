"""Single public entry point for the local multi-Agent runtime.
本地多 Agent 运行时的唯一公开入口。

Only argparse parsing, configuration loading, RuntimeApplication creation,
command dispatch, and uniform JSON output live here. No model business logic.
此处仅包含 argparse 解析、配置加载、RuntimeApplication 创建、命令分发与
统一 JSON 输出；不包含任何模型业务逻辑。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

from spacers_agent.application import RuntimeApplication, run_dataset_command, run_http_server
from spacers_agent.settings import load_settings

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# Public task names accepted by the ask command. / ask 命令接受的公开任务名。
_TASK_CHOICES = (
    "auto",
    "counting",
    "fine_grained_counting",
    "change_caption",
    "change_qa",
    "grounding",
    "spatial_relation",
    "scene_classification",
    "general_vqa",
    "caption",
    "multiple_choice_vqa",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the root parser; no subcommand defaults to ``serve``.
    构建根解析器；无子命令时默认 ``serve``。
    """

    parser = argparse.ArgumentParser(
        description="Remote-sensing multi-Agent local runtime"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to YAML configuration.",
    )
    # Defaults for the implicit serve path: without a subcommand the root
    # parser still exposes host/port so run_http_server never sees missing
    # attributes. ``set_defaults`` covers host/port; the subparsers action's
    # own ``default`` is required for command because argparse overwrites a
    # parser default with None when no subcommand is given.
    # 无子命令默认 serve 时，根解析器仍提供 host/port，使 run_http_server
    # 不会遇到缺失属性。host/port 用 set_defaults；command 必须设置 subparsers
    # action 自身的 default，因为未给子命令时 argparse 会把 parser 默认值
    # 覆盖为 None。
    parser.set_defaults(host="127.0.0.1", port=8000)
    commands = parser.add_subparsers(dest="command")
    commands.default = "serve"

    serve = commands.add_parser(
        "serve",
        help="Load the local model once and start the HTTP service.",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    ask = commands.add_parser(
        "ask",
        help="Run one question against images from a local directory.",
    )
    ask.add_argument("--images-dir", type=Path, required=True)
    ask.add_argument("--question", default="")
    ask.add_argument(
        "--task",
        choices=_TASK_CHOICES,
        default="auto",
    )
    ask.add_argument("--output", type=Path)

    dataset = commands.add_parser(
        "run-dataset",
        help="Run an existing dataset through the DatasetRunner workflow.",
    )
    dataset.add_argument(
        "--dataset",
        required=True,
        choices=("LEVIR-CC", "VRSBench", "MME-RealWorld", "XLRS-Bench-lite"),
    )
    dataset.add_argument("--root", type=Path, required=True)
    dataset.add_argument("--split", required=True)
    dataset.add_argument(
        "--task",
        help="One or more comma-separated tasks; omitted means adapter-supported tasks.",
    )
    dataset.add_argument("--run-id")
    dataset.add_argument("--max-samples", type=int, default=0)
    dataset.add_argument("--start-index", type=int, default=0)
    dataset.add_argument("--sample-concurrency", type=int, default=1)
    dataset.add_argument("--resume", action="store_true")
    dataset.add_argument("--fail-fast", action="store_true")
    dataset.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run evaluation and DeepSeek VQA judging by default; use --no-evaluate to opt out.",
    )
    dataset.add_argument(
        "--judge-policy",
        choices=("none", "errors-only", "all"),
        default="all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, load, dispatch, and return the process exit code.
    解析、加载、分发并返回进程退出码。
    """

    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)

        if args.command == "run-dataset":
            return asyncio.run(run_dataset_command(settings, args))

        app = RuntimeApplication.create(
            settings=settings,
            project_root=PROJECT_ROOT,
        )

        if args.command == "ask":
            result = asyncio.run(
                app.ask(
                    image_dir=args.images_dir,
                    question=args.question,
                    task=args.task,
                )
            )
            rendered = result.model_dump_json(indent=2)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
            return 0

        return run_http_server(
            app,
            host=args.host,
            port=args.port,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
