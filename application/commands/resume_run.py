"""Public `resume-run` CLI command: resume one run from its manifest.

公开 `resume-run` CLI 命令：从 manifest 恢复一个 run。读取当前
RunManifest 与 config 快照，重建 DatasetRunOptions 并委托 Runtime
（resume=True）；绝不复制 DatasetRunner 循环。manifest 缺失/损坏/
信息不足/身份不匹配时稳定失败。
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
from workflows.run_store import RunManifest

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


def run_resume_run(args: argparse.Namespace) -> int:
    """Reconstruct run options from the manifest and delegate to Runtime.
    从 manifest 重建运行选项并委托给 Runtime。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        project_root = Path(__file__).resolve().parents[2]
        run_dir = settings.runs.root / args.run_id
        manifest = _read_manifest(run_dir, args.run_id)
        if manifest.dataset is None or manifest.split is None:
            raise ValueError("resume manifest lacks dataset or split")
        root = _read_dataset_root(run_dir)
        tasks = _run_tasks(run_dir)
        auto_task = tasks == ["auto"]
        if not tasks:
            raise ValueError("resume run has no task directories")
        options = build_dataset_run_options(
            dataset=manifest.dataset,
            root=root,
            split=manifest.split,
            tasks=() if auto_task else tuple(sorted(tasks)),
            auto_task=auto_task,
            run_id=args.run_id,
            resume=True,
            sample_ids=_parse_sample_filter(manifest.sample_filter),
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


def _read_dataset_root(run_dir: Path) -> Path:
    """Recover the dataset root from the config snapshot; missing or invalid
    snapshots fail stably. 从配置快照恢复数据集根；缺失或无效稳定失败。"""

    snapshot_path = run_dir / "config.snapshot.json"
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("resume config snapshot is invalid") from exc
    root_value = (snapshot.get("paths") or {}).get("dataset_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("resume config snapshot lacks dataset_root")
    return Path(root_value)


def _run_tasks(run_dir: Path) -> list[str]:
    """Task namespaces present in the run directory; 'auto' means the
    auto-task namespace. 运行目录中存在的 task 命名空间；'auto' 表示
    auto-task 命名空间。"""

    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        return []
    return sorted(
        entry.name for entry in tasks_dir.iterdir() if entry.is_dir()
    )


def _parse_sample_filter(sample_filter: str | None) -> set[str] | None:
    """Reconstruct sample_ids from the persisted comma-separated filter.
    从持久化的逗号分隔过滤器重建 sample_ids。"""

    if not sample_filter:
        return None
    return set(part for part in sample_filter.split(",") if part)
