"""Offline CLI skeleton for the multi-Agent runtime.
多 Agent 运行时的离线 CLI 骨架。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Sequence

import httpx

from spacers_agent.imaging import build_core_halo_tiles, read_normalized_image
from spacers_agent.reporting import summarize_evaluations
from spacers_agent.run_store import RunStore
from spacers_agent.schemas import CountingResult
from spacers_agent.settings import load_settings
from spacers_agent.data_audit import inspect_dataset_root, write_dataset_audit
from spacers_agent.evaluation import EvaluationRecord
from spacers_agent.visualization import render_counting_overlay
from spacers_agent.clients.base import JsonResponseCache, RequestMeta, VisionLanguageClient
from spacers_agent.clients.deepseek import DeepSeekJudgeClient
from spacers_agent.clients.qwen_vllm import QwenVLLMClient
from spacers_agent.clients.qwen_transformers import QwenTransformersClient
from spacers_agent.counting import PointCountingOrchestrator
from spacers_agent.dataset_adapters import get_adapter
from spacers_agent.evaluation import build_count_judge_payload, build_judge_request_hash, merge_count_evaluation
from spacers_agent.workflow import DatasetRunner, TargetParser, atomic_write_json
from spacers_agent.vqa_report import build_multiagent_vqa_report
from spacers_agent.commands import count_image as count_image_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_COUNT_PROMPT = PROJECT_ROOT / "prompts" / "count_tile_v4.md"
DEFAULT_JSON_REPAIR_PROMPT = PROJECT_ROOT / "prompts" / "json_repair_v1.md"
DEFAULT_PROMPT_PATHS = [
    DEFAULT_COUNT_PROMPT,
    DEFAULT_JSON_REPAIR_PROMPT,
    PROJECT_ROOT / "prompts" / "router_v1.md",
    PROJECT_ROOT / "prompts" / "target_parse_v1.md",
    PROJECT_ROOT / "prompts" / "count_repair_v1.md",
    PROJECT_ROOT / "prompts" / "seam_verify_v1.md",
    PROJECT_ROOT / "prompts" / "missing_point_review_v1.md",
    PROJECT_ROOT / "prompts" / "missing_point_review_v2.md",
    PROJECT_ROOT / "prompts" / "missing_point_review_v3.md",
    PROJECT_ROOT / "prompts" / "change_v1.md",
    PROJECT_ROOT / "prompts" / "spatial_v2.md",
    PROJECT_ROOT / "prompts" / "spatial_v3.md",
    PROJECT_ROOT / "prompts" / "spatial_v4.md",
    PROJECT_ROOT / "prompts" / "spatial_candidate_review_v1.md",
    PROJECT_ROOT / "prompts" / "spatial_candidate_review_v2.md",
    PROJECT_ROOT / "prompts" / "general_vqa_v2.md",
    PROJECT_ROOT / "prompts" / "deepseek_judge_v1.md",
    PROJECT_ROOT / "prompts" / "deepseek_judge_repair_v1.md",
    PROJECT_ROOT / "prompts" / "deepseek_vqa_judge_v1.md",
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
    health.add_argument("--live", action="store_true", help="Perform an authenticated endpoint health request.")
    commands.add_parser("list-datasets", help="List registered strict read-only dataset adapters.")
    smoke = commands.add_parser("smoke-qwen", help="Send one image and question to the configured local Qwen endpoint.")
    smoke.add_argument("--image", type=Path, required=True)
    smoke.add_argument("--question", required=True)
    count_image_command.add_parser(commands)
    dataset = commands.add_parser("run-dataset", help="Probe then run an explicit read-only dataset adapter.")
    dataset.add_argument("--dataset", required=True, choices=("LEVIR-CC", "VRSBench", "MME-RealWorld", "XLRS-Bench-lite"))
    dataset.add_argument("--root", type=Path, required=True)
    dataset.add_argument("--split", required=True)
    dataset.add_argument("--task", help="One or more comma-separated tasks; omitted means adapter-supported tasks.")
    dataset.add_argument("--run-id")
    dataset.add_argument("--resume", action="store_true")
    dataset.add_argument("--evaluate", action="store_true")
    dataset.add_argument("--judge-policy", choices=("none", "errors-only", "all"), default="errors-only")
    dataset.add_argument("--judge-sample-rate", type=float, default=0.1)
    dataset.add_argument("--max-samples", "--limit", dest="limit", type=int, default=0)
    dataset.add_argument("--sample-ids", type=Path)
    dataset.add_argument("--start-index", type=int, default=0)
    dataset.add_argument("--shard-index", type=int, default=0)
    dataset.add_argument("--num-shards", "--shard-count", dest="shard_count", type=int, default=1)
    dataset.add_argument("--sample-concurrency", type=int, default=1)
    dataset.add_argument("--render-errors", action="store_true")
    dataset.add_argument("--fail-fast", action="store_true")
    resume = commands.add_parser("resume-run", help="Resume an existing dataset run from its saved manifest.")
    resume.add_argument("--run-id", required=True)
    evaluate = commands.add_parser("evaluate-run", help="Evaluate persisted counting results without Qwen inference.")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--deepseek", action="store_true")
    evaluate.add_argument("--only-missing", action="store_true")
    evaluate.add_argument("--force-judge", action="store_true")
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
        return _show_health_metadata(settings, args.service, live=args.live)
    if args.command == "list-datasets":
        return _list_datasets()
    if args.command == "inspect-data":
        report = inspect_dataset_root(args.root)
        write_dataset_audit(report, args.output)
        print(report.model_dump_json(indent=2))
        return 0
    if args.command == "render-count":
        return _render_count(settings, args.image, args.result, args.output)
    if args.command == "summarize-evaluations":
        return _summarize_evaluations(args.input, args.output)
    if args.command == "smoke-qwen":
        return asyncio.run(_smoke_qwen(settings, args.image, args.question))
    if args.command == "count-image":
        return count_image_command.run(settings, args)
    if args.command == "run-dataset":
        return asyncio.run(_run_dataset(settings, args))
    if args.command == "resume-run":
        return asyncio.run(_resume_run(settings, args.run_id))
    if args.command == "evaluate-run":
        return asyncio.run(_evaluate_run(settings, args.run_id, args.deepseek))
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


def _show_health_metadata(settings: object, service: str, *, live: bool = False) -> int:
    """Expose endpoint metadata while Phase 1 intentionally avoids networking.
    在 Phase 1 有意不联网时展示端点元数据。
    """

    model_settings = settings.models.qwen if service == "qwen" else settings.models.deepseek
    payload = {
        "service": service,
        "base_url": model_settings.base_url,
        "model": model_settings.model,
        "network_check": "deferred_until_phase_2_and_explicit_authorization",
    }
    if live:
        api_key = os.environ.get(model_settings.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable: {model_settings.api_key_env}")
        endpoint = model_settings.base_url.rstrip("/") + "/models"
        response = httpx.get(endpoint, headers={"Authorization": f"Bearer {api_key}"}, timeout=model_settings.timeout_seconds)
        response.raise_for_status()
        payload["network_check"] = "ok"
        payload["status_code"] = response.status_code
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _client(settings: object, run_dir: Path) -> VisionLanguageClient:
    """Create a live Qwen client with run-scoped safe cache. / 创建带运行范围安全缓存的在线 Qwen 客户端。"""

    cache = JsonResponseCache(run_dir / "cache")
    if settings.models.qwen.backend == "transformers":
        return QwenTransformersClient(
            settings.models.qwen,
            repair_prompt=DEFAULT_JSON_REPAIR_PROMPT.read_text(encoding="utf-8"),
            cache=cache,
        )
    return QwenVLLMClient(
        settings.models.qwen,
        repair_prompt=DEFAULT_JSON_REPAIR_PROMPT.read_text(encoding="utf-8"),
        cache=cache,
    )


def _prompts() -> dict[str, str]:
    """Load the versioned prompt assets used by runnable workflows. / 加载可运行工作流使用的版本化 Prompt 资源。"""

    return {
        "count": DEFAULT_COUNT_PROMPT.read_text(encoding="utf-8"),
        "count_zero_review": (PROJECT_ROOT / "prompts" / "missing_point_review_v3.md").read_text(encoding="utf-8"),
        "target": (PROJECT_ROOT / "prompts" / "target_parse_v1.md").read_text(encoding="utf-8"),
        "change": (PROJECT_ROOT / "prompts" / "change_v1.md").read_text(encoding="utf-8"),
        "spatial": (PROJECT_ROOT / "prompts" / "spatial_v4.md").read_text(encoding="utf-8"),
        "spatial_review": (PROJECT_ROOT / "prompts" / "spatial_candidate_review_v2.md").read_text(encoding="utf-8"),
        "general": (PROJECT_ROOT / "prompts" / "general_vqa_v2.md").read_text(encoding="utf-8"),
        "seam": (PROJECT_ROOT / "prompts" / "seam_verify_v1.md").read_text(encoding="utf-8"),
    }


async def _smoke_qwen(settings: object, image_path: Path, question: str) -> int:
    """Perform one explicit local Qwen visual smoke request. / 执行一次明确的本地 Qwen 视觉 smoke 请求。"""

    from spacers_agent.schemas import ExpertResult, UnifiedSample, ImageRef
    client = _client(settings, settings.runs.root / "smoke")
    data = image_path.read_bytes()
    messages = [{"role": "system", "content": _prompts()["general"]}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + __import__("base64").b64encode(data).decode("ascii")}}, {"type": "text", "text": question}]}]
    result = await client.complete_json(messages=messages, response_model=ExpertResult, request_meta=RequestMeta(request_id="smoke-qwen", request_hash=__import__("hashlib").sha256((question + str(image_path.stat().st_size)).encode()).hexdigest(), prompt_version="general-vqa-v2", artifact_dir=settings.runs.root / "smoke" / "artifacts"))
    print(result.model_dump_json(indent=2))
    return 0


async def _count_image(settings: object, image_path: Path, question: str, run_id: str, evaluate: bool, render: bool) -> int:
    """Run target parsing and point-derived single-image counting. / 运行目标解析和点导出的单图计数。"""

    store = RunStore(settings.runs.root, PROJECT_ROOT)
    manifest = store.create_run(settings, prompt_paths=DEFAULT_PROMPT_PATHS, run_id=run_id)
    run_dir = settings.runs.root / manifest.run_id
    client = _client(settings, run_dir)
    target = await TargetParser(client, _prompts()["target"], settings.models.qwen.model).parse(question, sample_id=run_id, artifact_dir=run_dir)
    image = read_normalized_image(image_path)
    result = await PointCountingOrchestrator(client, counting=settings.counting, qwen=settings.models.qwen, system_prompt=_prompts()["count"], run_dir=run_dir, seam_prompt=_prompts()["seam"]).count_image(image, sample_id=run_id, question=question, target=target)
    result_path = run_dir / "counting_result.json"
    atomic_write_json(result_path, result.model_dump(mode="json"))
    if evaluate:
        record = merge_count_evaluation(sample_id=run_id, counting=result, ground_truth=None)
        atomic_write_json(run_dir / "evaluation_records.json", [record.model_dump(mode="json")])
    if render:
        return _render_count(settings, image_path, result_path, run_dir / "counting_overlay.png")
    print(result.model_dump_json(indent=2))
    return 0


async def _run_dataset(settings: object, args: object) -> int:
    """Create or resume a dataset run after adapter probing. / 在适配器探测后创建或恢复数据集运行。"""

    settings.paths.dataset_root = args.root
    if args.limit < 0 or args.start_index < 0 or args.sample_concurrency < 1:
        raise ValueError("max-samples, start-index, and sample-concurrency must be valid")
    run_id = args.run_id or f"{args.dataset}-{args.split}"
    run_dir = settings.runs.root / run_id
    if not run_dir.exists():
        RunStore(settings.runs.root, PROJECT_ROOT).create_run(settings, prompt_paths=DEFAULT_PROMPT_PATHS, run_id=run_id, dataset=args.dataset, split=args.split, sample_filter=args.task)
    adapter = get_adapter(args.dataset)
    requested = args.task.split(",") if args.task else sorted(getattr(adapter, "supported_tasks", ()))
    selected_ids = set(args.sample_ids.read_text(encoding="utf-8").split()) if args.sample_ids else None
    qwen_client = _client(settings, run_dir)
    judge_client = None
    if args.evaluate and args.judge_policy != "none" and os.environ.get(settings.models.deepseek.api_key_env):
        judge_client = DeepSeekJudgeClient(
            settings.models.deepseek,
            judge_prompt=(PROJECT_ROOT / "prompts" / "deepseek_vqa_judge_v1.md").read_text(encoding="utf-8"),
            repair_prompt=(PROJECT_ROOT / "prompts" / "deepseek_judge_repair_v1.md").read_text(encoding="utf-8"),
            cache=JsonResponseCache(run_dir / "deepseek_vqa_cache"),
        )
    summaries = []
    for task in requested:
        summaries.append(await DatasetRunner(settings, adapter, run_dir=run_dir, client=qwen_client, prompts=_prompts(), judge_client=judge_client, judge_policy=args.judge_policy if args.evaluate else "none").run(split=args.split, task=task, resume=args.resume, limit=None if args.limit == 0 else args.limit, shard_index=args.shard_index, shard_count=args.shard_count, start_index=args.start_index, sample_ids=selected_ids, fail_fast=args.fail_fast, sample_concurrency=args.sample_concurrency))
    if args.evaluate and any(task in {"counting", "fine_grained_counting"} for task in requested):
        # A missing key still produces deterministic records; it never silently skips evaluation.
        # 缺少密钥时仍生成确定性记录，绝不静默跳过评估。
        await _evaluate_run(settings, run_id, bool(os.environ.get(settings.models.deepseek.api_key_env)))
    if "general_vqa" in requested:
        report_path = build_multiagent_vqa_report(
            run_dir,
            qwen=settings.models.qwen,
            model_load_seconds=float(getattr(qwen_client, "load_seconds", 0.0)),
        )
        if report_path is not None:
            print(json.dumps({"html_report": str(report_path)}, ensure_ascii=False))
    print(json.dumps([summary.model_dump(mode="json") for summary in summaries], ensure_ascii=False, indent=2))
    return 0


async def _resume_run(settings: object, run_id: str) -> int:
    """Resume a saved dataset run using its manifest fields. / 使用其清单字段恢复已保存的数据集运行。"""

    manifest_path = settings.runs.root / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("dataset") or not manifest.get("split"):
        raise ValueError("run manifest is not a dataset run")
    class Args: pass
    args = Args(); args.root = settings.paths.dataset_root; args.run_id = run_id; args.dataset = manifest["dataset"]; args.split = manifest["split"]; args.task = manifest.get("sample_filter") or "counting"; args.resume = True; args.limit = None; args.shard_index = 0; args.shard_count = 1; args.evaluate = False
    return await _run_dataset(settings, args)


async def _evaluate_run(settings: object, run_id: str, deepseek: bool) -> int:
    """Evaluate stored counting results without reissuing Qwen requests. / 不重新发起 Qwen 请求地评估已存计数结果。"""

    run_dir = settings.runs.root / run_id
    records = []
    judge_client = None
    if deepseek:
        judge_client = DeepSeekJudgeClient(
            settings.models.deepseek,
            judge_prompt=(PROJECT_ROOT / "prompts" / "deepseek_judge_v1.md").read_text(encoding="utf-8"),
            repair_prompt=(PROJECT_ROOT / "prompts" / "deepseek_judge_repair_v1.md").read_text(encoding="utf-8"),
            cache=JsonResponseCache(run_dir / "deepseek_cache"),
        )
    for result_path in run_dir.rglob("counting_result.json"):
        from spacers_agent.schemas import CountTargetSpec, CountingResult, UnifiedSample
        result = CountingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        sample_path = result_path.parent / "sample.json"
        sample = UnifiedSample.model_validate_json(sample_path.read_text(encoding="utf-8")) if sample_path.is_file() else None
        ground_truth = sample.ground_truth if sample is not None else None
        target = CountTargetSpec(canonical_label=result.target, inclusion_rule="Persisted target specification unavailable.", exclusion_rule="Persisted target specification unavailable.")
        if judge_client is None:
            records.append(merge_count_evaluation(sample_id=result.sample_id, counting=result, ground_truth=ground_truth))
            continue
        payload = build_count_judge_payload(question=result.question, target=target, display_answer=f"{result.final_count} accepted global points", counting=result, ground_truth=ground_truth, min_confidence=settings.counting.min_confidence)
        try:
            verdict = await judge_client.judge(payload, request_meta=RequestMeta(request_id=f"{result.sample_id}:deepseek", request_hash=build_judge_request_hash(model=settings.models.deepseek.model, prompt_text=judge_client.judge_prompt, sample_id=result.sample_id, payload=payload), prompt_version="deepseek-judge-v1", sample_id=result.sample_id, artifact_dir=result_path.parent / "deepseek"))
            records.append(merge_count_evaluation(sample_id=result.sample_id, counting=result, ground_truth=ground_truth, judge_parsed=verdict))
        except Exception as error:
            records.append(merge_count_evaluation(sample_id=result.sample_id, counting=result, ground_truth=ground_truth, judge_error=f"{type(error).__name__}: {error}"))
    payload = [record.model_dump(mode="json") for record in records]
    atomic_write_json(run_dir / "evaluation.json", payload)
    atomic_write_json(run_dir / "evaluation_records.json", payload)
    jsonl_path = run_dir / "evaluations.jsonl"
    temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    temporary.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")
    temporary.replace(jsonl_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _list_datasets() -> int:
    """Print registered datasets and their strict mapping requirement. / 输出注册数据集及其严格映射要求。"""

    from spacers_agent.dataset_adapters import ADAPTERS
    print(json.dumps({"datasets": sorted(ADAPTERS), "adapter_manifest": "spacers_adapter.json", "version": "1"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
