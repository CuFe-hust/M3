"""Offline CLI skeleton for the multi-Agent runtime.
多 Agent 运行时的离线 CLI 骨架。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from spacers_agent.imaging import build_core_halo_tiles, read_normalized_image
from spacers_agent.reporting import summarize_evaluations
from spacers_agent.run_store import RunStore
from spacers_agent.schemas import CountingResult
from spacers_agent.settings import load_settings
from spacers_agent.data_audit import inspect_dataset_root, write_dataset_audit
from spacers_agent.evaluation import EvaluationRecord
from spacers_agent.visualization import render_counting_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_COUNT_PROMPT = PROJECT_ROOT / "prompts" / "count_tile_v2.md"
DEFAULT_JSON_REPAIR_PROMPT = PROJECT_ROOT / "prompts" / "json_repair_v1.md"
DEFAULT_PROMPT_PATHS = [
    DEFAULT_COUNT_PROMPT,
    DEFAULT_JSON_REPAIR_PROMPT,
    PROJECT_ROOT / "prompts" / "router_v1.md",
    PROJECT_ROOT / "prompts" / "target_parse_v1.md",
    PROJECT_ROOT / "prompts" / "count_repair_v1.md",
    PROJECT_ROOT / "prompts" / "seam_verify_v1.md",
    PROJECT_ROOT / "prompts" / "missing_point_review_v1.md",
    PROJECT_ROOT / "prompts" / "change_v1.md",
    PROJECT_ROOT / "prompts" / "spatial_v1.md",
    PROJECT_ROOT / "prompts" / "general_vqa_v1.md",
    PROJECT_ROOT / "prompts" / "deepseek_judge_v1.md",
    PROJECT_ROOT / "prompts" / "deepseek_judge_repair_v1.md",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the offline-safe Phase 1 command-line interface.
    构建离线安全的 Phase 1 命令行接口。
    """

    parser = argparse.ArgumentParser(description="Remote-sensing multi-Agent local runtime")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to a YAML configuration file.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_init = commands.add_parser("run-init", help="Create a local reproducible run directory.")
    run_init.add_argument("--run-id", help="Optional stable local run identifier.")
    run_init.add_argument("--dataset", help="Dataset name recorded in the manifest.")
    run_init.add_argument("--split", help="Dataset split recorded in the manifest.")
    run_init.add_argument("--sample-filter", help="Sample filter recorded in the manifest.")
    health = commands.add_parser("health", help="Show configured endpoint metadata without making a request.")
    health.add_argument("service", choices=("qwen", "deepseek"))
    inspect_data = commands.add_parser("inspect-data", help="Read a local dataset layout without modifying source files.")
    inspect_data.add_argument("--root", type=Path, required=True, help="Local dataset root to inspect read-only.")
    inspect_data.add_argument("--output", type=Path, required=True, help="Separate JSON report output path.")
    render_count = commands.add_parser("render-count", help="Render a local counting result without calling a model.")
    render_count.add_argument("--image", type=Path, required=True, help="Source image used for the local counting result.")
    render_count.add_argument("--result", type=Path, required=True, help="CountingResult JSON path.")
    render_count.add_argument("--output", type=Path, required=True, help="Output PNG or JPEG path.")
    summarize = commands.add_parser("summarize-evaluations", help="Summarize local EvaluationRecord JSONL without a model call.")
    summarize.add_argument("--input", type=Path, required=True, help="EvaluationRecord JSONL path.")
    summarize.add_argument("--output", type=Path, required=True, help="Summary JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an offline CLI command and return its process code.
    运行离线 CLI 命令并返回进程状态码。
    """

    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    if args.command == "health":
        return _show_health_metadata(settings, args.service)
    if args.command == "inspect-data":
        report = inspect_dataset_root(args.root)
        write_dataset_audit(report, args.output)
        print(report.model_dump_json(indent=2))
        return 0
    if args.command == "render-count":
        return _render_count(settings, args.image, args.result, args.output)
    if args.command == "summarize-evaluations":
        return _summarize_evaluations(args.input, args.output)
    store = RunStore(settings.runs.root, PROJECT_ROOT)
    manifest = store.create_run(
        settings,
        prompt_paths=DEFAULT_PROMPT_PATHS,
        run_id=args.run_id,
        dataset=args.dataset,
        split=args.split,
        sample_filter=args.sample_filter,
    )
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _render_count(settings: object, image_path: Path, result_path: Path, output_path: Path) -> int:
    """Render persisted point evidence locally without any model invocation.
    在不调用任何模型的情况下本地渲染持久化点证据。
    """

    image = read_normalized_image(image_path)
    result = CountingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    if (result.source_width, result.source_height) != image.size:
        raise ValueError("result dimensions do not match normalized source image")
    tiles = build_core_halo_tiles(
        *image.size,
        core_size=settings.counting.tile_core_size,
        halo_size=settings.counting.halo_size,
        model_max_side=settings.counting.model_max_side,
    )
    render_counting_overlay(image, result=result, tiles=tiles, output_path=output_path)
    print(output_path)
    return 0


def _summarize_evaluations(input_path: Path, output_path: Path) -> int:
    """Read local JSONL evaluation records and emit one deterministic summary.
    读取本地 JSONL 评估记录并输出一个确定性汇总。
    """

    records = [
        EvaluationRecord.model_validate_json(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_evaluations(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(summary.model_dump_json(indent=2))
    return 0


def _show_health_metadata(settings: object, service: str) -> int:
    """Expose endpoint metadata while Phase 1 intentionally avoids networking.
    在 Phase 1 有意不联网时展示端点元数据。
    """

    model_settings = settings.models.qwen if service == "qwen" else settings.models.deepseek
    print(
        json.dumps(
            {
                "service": service,
                "base_url": model_settings.base_url,
                "model": model_settings.model,
                "network_check": "deferred_until_phase_2_and_explicit_authorization",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
