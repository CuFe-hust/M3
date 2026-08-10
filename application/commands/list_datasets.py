"""Public `list-datasets` CLI command: list built-in dataset adapters.

公开 `list-datasets` CLI 命令：列出内建数据集适配器。使用当前
DatasetRegistry.names()；绝不使用旧 ADAPTERS 表。
"""

from __future__ import annotations

import argparse
import json
import sys

from data.registry import build_default_registry

EXIT_OK = 0
EXIT_RUNTIME = 1


def run_list_datasets(args: argparse.Namespace) -> int:
    """Print the sorted built-in dataset names as JSON.
    以 JSON 输出排序后的内建数据集名。"""

    try:
        datasets = list(build_default_registry().names())
    except Exception as error:
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}"}),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    print(
        json.dumps(
            {"status": "ok", "datasets": datasets},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK
