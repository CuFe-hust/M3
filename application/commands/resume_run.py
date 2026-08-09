"""Public `resume-run` CLI command: resume one run from its persisted
invocation.

公开 `resume-run` CLI 命令：从持久化调用恢复一个 run。读取当前
RunManifest 与 run_request.json（具体运行调用），重建 DatasetRunOptions
并委托 Runtime（resume=True）；绝不复制 DatasetRunner 循环，绝不由默认
配置/目录名/汇总猜测调用。manifest 缺失/损坏/身份不匹配或 run_request
缺失/损坏时稳定失败。
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
from workflows.run_store import RunManifest, RunStore

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


def run_resume_run(args: argparse.Namespace) -> int:
    """Reconstruct run options from the persisted invocation and delegate to
    Runtime. 从持久化调用重建运行选项并委托给 Runtime。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        project_root = Path(__file__).resolve().parents[2]
        run_dir = settings.runs.root / args.run_id
        manifest = _read_manifest(run_dir, args.run_id)
        store = RunStore(settings.runs.root, project_root)
        request = store.read_run_request(run_dir)
        root = Path(request.dataset_root)
        if request.task_mode == "auto":
            tasks: tuple[str, ...] | None = ()
            auto_task = True
        elif request.task_mode == "adapter_default":
            tasks = None
            auto_task = False
        else:
            tasks = tuple(request.tasks)
            auto_task = False
        options = build_dataset_run_options(
            dataset=request.dataset,
            root=root,
            split=request.split,
            tasks=tasks,
            auto_task=auto_task,
            run_id=args.run_id,
            resume=True,
            limit=request.limit,
            start_index=request.start_index,
            shard_index=request.shard_index,
            shard_count=request.shard_count,
            sample_concurrency=request.sample_concurrency,
            sample_ids=(
                set(request.sample_ids)
                if request.sample_ids is not None
                else None
            ),
            evaluate=request.evaluate,
            judge_policy=request.judge_policy,
            judge_sample_rate=request.judge_sample_rate,
            render_errors=request.render_errors,
            fail_fast=request.fail_fast,
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
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}"}),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": args.run_id,
                "summaries": {
                    task: summary.model_dump(mode="json")
                    for task, summary in summaries.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK


def _read_manifest(run_dir: Path, run_id: str) -> RunManifest:
    """Read and validate the run manifest; identity mismatches fail stably.
    读取并校验 run manifest；身份不匹配稳定失败。"""

    manifest_path = run_dir / "manifest.json"
    if not run_dir.is_dir() or not manifest_path.is_file():
        raise ValueError("resume run does not exist")
    try:
        manifest = RunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("resume run manifest is invalid") from exc
    if manifest.run_id != run_id:
        raise ValueError("resume run id mismatch")
    return manifest
