"""Public `run-init` CLI command: create one run directory without running.

公开 `run-init` CLI 命令：只创建 run 目录，不运行任何模型。使用当前
RunStore.create_run，快照配方与 fresh Runtime 完全一致（config payload /
model ids / Prompt 副本）；重复显式 run id 稳定失败。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from application.prompts import PromptCatalog
from application.settings import load_settings
from workflows.run_store import RunStore

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


def run_run_init(args: argparse.Namespace) -> int:
    """Create a run directory and print its identity as JSON.
    创建 run 目录并以 JSON 输出其身份。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        project_root = Path(__file__).resolve().parents[2]
        catalog = PromptCatalog(project_root / "prompts")
        manifest = RunStore(settings.runs.root, project_root).create_run(
            config_payload=settings.to_config_payload(),
            model_ids={
                "qwen": settings.models.qwen.effective_cache_model_id,
                "deepseek": settings.models.deepseek.model,
            },
            prompt_paths=catalog.snapshot_paths(),
            run_id=args.run_id,
            dataset=args.dataset,
            split=args.split,
            sample_filter=args.sample_filter,
        )
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as error:
        # Public output never carries raw exception text or secrets.
        # 公共输出绝不携带原始异常文本或密钥。
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}"}),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": manifest.run_id,
                "run_dir": (settings.runs.root / manifest.run_id).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK
