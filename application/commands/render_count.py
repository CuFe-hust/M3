"""Public `render-count` CLI command: counting overlay for a persisted result.

公开 `render-count` CLI 命令：为持久化计数结果渲染标注图。使用当前
Reporting overlay（纯本地，无模型调用）。tile 级调试只在持久化
CountingResult 携带足够几何时渲染——绝不猜测；几何不足时点 overlay 仍
成功并输出 `tile_overlay=not_available` 摘要。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents.counting.schema import CountingResult
from models.images import read_normalized_image
from reporting.visualization import render_counting_overlay

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


def run_render_count(args: argparse.Namespace) -> int:
    """Render one persisted CountingResult overlay and print a summary.
    渲染一份持久化 CountingResult 标注图并输出摘要。"""

    try:
        result = CountingResult.model_validate(
            json.loads(Path(args.result).read_text(encoding="utf-8"))
        )
        image = read_normalized_image(Path(args.image))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        render_counting_overlay(image, result=result, output_path=output)
        tile_overlay = "available" if _tile_geometry_available(result) else "not_available"
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
                "tile_overlay": tile_overlay,
                "output": output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK


def _tile_geometry_available(result: CountingResult) -> bool:
    """Tile-aware debugging requires persisted tile boundary geometry. The
    current CountingResult schema carries tile ids and per-tile point
    provenance but no tile rectangles, so this stays False unless a future
    schema adds explicit tile geometry — never guessed.
    tile 级调试需要持久化的 tile 边界几何。当前 CountingResult schema 携带
    tile id 与逐 tile 点来源，但没有 tile 矩形，因此除非未来 schema 显式
    提供 tile 几何，恒为 False——绝不猜测。"""

    return bool(getattr(result, "tile_geometry", None))
