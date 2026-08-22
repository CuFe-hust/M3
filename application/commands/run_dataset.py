"""Public `run-dataset` CLI command: run one dataset through the DatasetRunner.

公开 `run-dataset` CLI 命令：经 DatasetRunner 运行一个数据集。本模块是薄
接线：参数校验（先于任何运行时/模型初始化）→ 配置 → Runtime.create（Qwen
一次）→ build_dataset_run_options → Runtime.run_dataset → 汇总 JSON 与
run_dir → 退出码（0/1/2/130）。任务选择模式：--task 显式；--auto-task
逐样本进入唯一 VisualTaskPlanner；两者都不给 → adapter.supported_tasks；
两者都给 → 参数错误。公共错误只输出稳定类型名。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from application.runtime import (
    Runtime,
    build_dataset_run_options,
    evidence_preprocessing_identity,
    preflight_dataset_resume,
)
from application.settings import load_settings

EXIT_OK = 0
EXIT_ARGUMENT = 2
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


def run_run_dataset(args: argparse.Namespace) -> int:
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
        else () if args.auto_task else None
    )
    if tasks and args.auto_task:
        _print_error("--task and --auto-task are mutually exclusive")
        return EXIT_ARGUMENT
    try:
        sample_ids = _read_sample_ids(args.sample_ids)
    except ValueError as error:
        _print_error(str(error))
        return EXIT_ARGUMENT
    try:
        settings = load_settings(
            Path(args.config) if args.config else None,
            environ=os.environ,
        )
        project_root = Path(__file__).resolve().parents[2]
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
            sample_ids=sample_ids,
            evaluate=args.evaluate,
            judge_policy=args.judge_policy,
            judge_sample_rate=args.judge_sample_rate,
            render_errors=args.render_errors,
            fail_fast=args.fail_fast,
            # Fresh runs freeze the current evidence preprocessing identity
            # explicitly; resume defers to the persisted run request.
            # 新鲜运行显式冻结当前 evidence 预处理身份；resume 服从持久化
            # run request。
            evidence_preprocessing=evidence_preprocessing_identity(settings),
        )
        if options.resume:
            # Validate the resume invocation against the persisted run
            # before any model construction: an invalid resume must fail
            # without loading Qwen weights. 在任何模型构造前按持久化 run
            # 校验 resume 调用：非法 resume 必须在加载 Qwen 权重前失败。
            options = preflight_dataset_resume(
                options=options,
                runs_root=settings.runs.root,
                project_root=project_root,
            )
        api_key = os.environ.get(settings.models.deepseek.api_key_env) or None
        runtime = Runtime.create(
            settings=settings,
            project_root=project_root,
            api_key=api_key,
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


def _read_sample_ids(path_value: str | None) -> set[str] | None:
    """Read whitespace-separated sample ids from one file; the ids feed the
    existing selection pipeline before execution and limit.
    从一个文件读取空白分隔的 sample id；id 喂给既有选择管线，先于执行与
    limit。"""

    if not path_value:
        return None
    try:
        text = Path(path_value).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("cannot read sample ids file") from error
    ids = {part for part in text.split() if part}
    if not ids:
        raise ValueError("sample ids file is empty")
    return ids


def _print_error(message: str) -> None:
    print(json.dumps({"status": "failed", "error": message}), file=sys.stderr)
