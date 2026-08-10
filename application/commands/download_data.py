"""Public `download-data` CLI command: explicit dataset download.

公开 `download-data` CLI 命令：显式数据集下载。这是唯一自动下载路径；
run-dataset/ask/serve/adapters/loader 绝不隐式调用 downloader。命令是
data.downloader 之上的薄适配；公共错误只输出稳定类型名。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from application.settings import load_settings
from data.downloader import download_dataset

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


def run_download_data(args: argparse.Namespace) -> int:
    """Download the requested datasets into the root and print the result.
    将请求的数据集下载到 root 并输出结果。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        root = Path(args.root)
        results: dict[str, str] = {}
        for dataset in args.datasets:
            destination = download_dataset(dataset, root=root)
            results[dataset] = destination.as_posix()
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
                "root": root.as_posix(),
                "datasets": results,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK
