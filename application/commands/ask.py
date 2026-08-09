"""Public `ask` CLI command: one manual local-image request.

公开 `ask` CLI 命令：一次手动本地图片请求。本模块是薄接线：加载配置 →
Runtime.create（Qwen 一次）→ Runtime.ask → 输出 PublicAnswer JSON 与可选
--output 文件。不包含任何 Agent 业务逻辑。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from application.runtime import Runtime
from application.settings import load_settings

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


def run_ask(args: argparse.Namespace) -> int:
    """Run one manual ask request and print the PublicAnswer JSON.
    运行一次手动 ask 请求并输出 PublicAnswer JSON。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
            environ=os.environ,
        )
        runtime = Runtime.create(
            settings=settings,
            project_root=Path(__file__).resolve().parents[2],
            api_key=None,
        )
        result = asyncio.run(
            runtime.ask(
                image_dir=Path(args.images_dir),
                question=args.question,
                task=args.task,
            )
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
    rendered = result.model_dump_json(indent=2)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return EXIT_OK
