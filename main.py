"""Minimal public entry point: `python main.py run-dataset ...`, the manual
`ask` command, and the local HTTP `serve` service (implicit default command).

最小公开入口：`python main.py run-dataset ...`、手动 `ask` 命令与本地 HTTP
`serve` 服务（无子命令时的隐式默认）。本模块保持极薄：解析参数 → 加载配置 →
组装运行时 → 构造选项 → 委托 application 用例（serve / ask / run_dataset）→
输出与退出码。绝不包含 Agent fallback、数据集循环、TaskResolver 逻辑、模型
业务逻辑或报告聚合。架构规则要求 main.py 只能 import application（含
stdlib）。
"""

from __future__ import annotations

import argparse
import json
import sys

from application.commands.ask import run_ask
from application.commands.count_image import run_count_image
from application.commands.download_data import run_download_data
from application.commands.evaluate_run import run_evaluate_run
from application.commands.health import run_health
from application.commands.inspect_data import run_inspect_data
from application.commands.judge_vqa_run import run_judge_vqa_run
from application.commands.list_datasets import run_list_datasets
from application.commands.render_count import run_render_count
from application.commands.resume_run import run_resume_run
from application.commands.run_dataset import run_run_dataset
from application.commands.run_init import run_run_init
from application.commands.serve import run_serve
from application.commands.smoke_qwen import run_smoke_qwen
from application.commands.standard_evaluate import run_standard_evaluate
from application.commands.summarize_evaluations import run_summarize_evaluations

EXIT_ARGUMENT = 2

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
    """The only supported public surface. 唯一受支持的公开面。"""

    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("--config", default=None, help="settings YAML path")
    # Defaults for the implicit serve path: without a subcommand the root
    # parser still exposes host/port so run_serve never sees missing
    # attributes. ``set_defaults`` covers host/port; the subparsers action's
    # own ``default`` is required for command because argparse overwrites a
    # parser default with None when no subcommand is given.
    # 无子命令默认 serve 时，根解析器仍提供 host/port，使 run_serve 不会遇到
    # 缺失属性。host/port 用 set_defaults；command 必须设置 subparsers action
    # 自身的 default，因为未给子命令时 argparse 会把 parser 默认值覆盖为 None。
    parser.set_defaults(host="127.0.0.1", port=8000)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.default = "serve"
    serve = subparsers.add_parser("serve", help="start the local HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    ask = subparsers.add_parser("ask", help="run one request against local images")
    ask.add_argument("--images-dir", required=True, help="manual image directory")
    ask.add_argument("--question", default="", help="question text (may be empty)")
    ask.add_argument("--task", choices=_TASK_CHOICES, default="auto")
    ask.add_argument("--output", default=None, help="write PublicAnswer JSON here")
    run_init = subparsers.add_parser("run-init", help="create one run directory")
    run_init.add_argument("--run-id", default=None)
    run_init.add_argument("--dataset", default=None)
    run_init.add_argument("--split", default=None)
    run_init.add_argument("--sample-filter", default=None)
    health = subparsers.add_parser("health", help="show model/service readiness")
    health.add_argument("component", choices=("qwen", "deepseek"))
    health.add_argument("--live", action="store_true", help="probe once")
    subparsers.add_parser("list-datasets", help="list built-in datasets")
    smoke_qwen = subparsers.add_parser("smoke-qwen", help="one direct Qwen request")
    smoke_qwen.add_argument("--image", required=True)
    smoke_qwen.add_argument("--question", required=True)
    resume_run = subparsers.add_parser("resume-run", help="resume one run")
    resume_run.add_argument("--run-id", required=True)
    inspect_data = subparsers.add_parser("inspect-data", help="audit a dataset root")
    inspect_data.add_argument("--root", required=True)
    inspect_data.add_argument("--output", default=None)
    inspect_data.add_argument("--scan-mode", choices=("quick", "full"), default="quick")
    count_image = subparsers.add_parser(
        "count-image", help="run one-image point-derived counting"
    )
    count_image.add_argument("--image", required=True)
    count_image.add_argument("--question", required=True)
    count_image.add_argument("--target-spec", default=None)
    count_image.add_argument("--run-id", default=None)
    count_image.add_argument("--evaluate", action="store_true")
    count_image.add_argument("--render", action="store_true")
    count_image.add_argument("--resume", action="store_true")
    count_image.add_argument("--force", action="store_true")
    count_image.add_argument("--no-seam-verify", action="store_true")
    count_image.add_argument("--max-qwen-calls", type=int, default=None)
    count_image.add_argument("--max-deepseek-calls", type=int, default=None)
    download_data = subparsers.add_parser(
        "download-data", help="download official datasets explicitly"
    )
    download_data.add_argument("--root", required=True)
    download_data.add_argument(
        "--datasets", nargs="+", required=True, help="dataset keys to download"
    )
    evaluate_run = subparsers.add_parser(
        "evaluate-run", help="offline deterministic evaluation of one run"
    )
    evaluate_run.add_argument("--run-id", required=True)
    evaluate_run.add_argument("--deepseek", action="store_true", help="enable the DeepSeek judge pass")
    evaluate_run.add_argument("--only-missing", action="store_true", help="fill only missing evaluations")
    evaluate_run.add_argument("--force-judge", action="store_true", help="re-judge even succeeded judges")
    judge_vqa_run = subparsers.add_parser(
        "judge-vqa-run", help="DeepSeek judge pass for one run"
    )
    judge_vqa_run.add_argument("--run-id", required=True)
    judge_vqa_run.add_argument("--force", action="store_true", help="re-judge succeeded judges")
    render_count = subparsers.add_parser(
        "render-count", help="render a counting overlay for a persisted result"
    )
    render_count.add_argument("--image", required=True)
    render_count.add_argument("--result", required=True)
    render_count.add_argument("--output", required=True)
    summarize_evaluations = subparsers.add_parser(
        "summarize-evaluations", help="aggregate evaluation records of one run or file"
    )
    summarize_evaluations.add_argument("--run-id", default=None, help="scan one run")
    summarize_evaluations.add_argument("--input", default=None, help="EvaluationRecord JSONL file")
    summarize_evaluations.add_argument("--output", default=None, help="write the summary here")
    standard_evaluate = subparsers.add_parser(
        "standard-evaluate", help="run the external team standard evaluator"
    )
    standard_evaluate.add_argument("--result", required=True, help="canonical result JSONL")
    standard_evaluate.add_argument("--tool-dir", default=None, help="directory containing evaluate.py")
    standard_evaluate.add_argument("--output", default=None, help="report path (default beside the result)")
    standard_evaluate.add_argument("--python", default=None, help="python executable for the tool")
    run_dataset = subparsers.add_parser("run-dataset", help="run one dataset")
    run_dataset.add_argument("--dataset", required=True)
    run_dataset.add_argument("--root", required=True)
    run_dataset.add_argument("--split", required=True)
    run_dataset.add_argument(
        "--task", default=None, help="comma-separated tasks (default: adapter-supported)"
    )
    run_dataset.add_argument("--auto-task", action="store_true", help="resolve tasks per sample")
    run_dataset.add_argument("--sample-ids", default=None, help="file of whitespace sample ids")
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
    run_dataset.add_argument(
        "--judge-sample-rate",
        type=float,
        default=None,
        help="deterministic judge sampling rate within [0.0, 1.0]",
    )
    run_dataset.add_argument(
        "--render-errors",
        action="store_true",
        help="render counting overlays for failed samples after execution",
    )
    run_dataset.add_argument("--max-samples", "--limit", dest="limit", type=int, default=None)
    run_dataset.add_argument("--start-index", type=int, default=0)
    run_dataset.add_argument("--shard-index", type=int, default=0)
    run_dataset.add_argument(
        "--shard-count",
        "--num-shards",
        dest="shard_count",
        type=int,
        default=1,
        help="shard count (--num-shards is an alias)",
    )
    run_dataset.add_argument("--sample-concurrency", type=int, default=1)
    run_dataset.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested command and return the process exit code.
    运行请求的命令并返回进程退出码。"""

    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return run_serve(args)
    if args.command == "ask":
        return run_ask(args)
    if args.command == "run-init":
        return run_run_init(args)
    if args.command == "health":
        return run_health(args)
    if args.command == "list-datasets":
        return run_list_datasets(args)
    if args.command == "smoke-qwen":
        return run_smoke_qwen(args)
    if args.command == "resume-run":
        return run_resume_run(args)
    if args.command == "inspect-data":
        return run_inspect_data(args)
    if args.command == "count-image":
        return run_count_image(args)
    if args.command == "download-data":
        return run_download_data(args)
    if args.command == "evaluate-run":
        return run_evaluate_run(args)
    if args.command == "judge-vqa-run":
        return run_judge_vqa_run(args)
    if args.command == "standard-evaluate":
        return run_standard_evaluate(args)
    if args.command == "render-count":
        return run_render_count(args)
    if args.command == "summarize-evaluations":
        return run_summarize_evaluations(args)
    if args.command == "run-dataset":
        return run_run_dataset(args)
    _print_error("unsupported command")
    return EXIT_ARGUMENT


def _print_error(message: str) -> None:
    print(json.dumps({"status": "failed", "error": message}), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
