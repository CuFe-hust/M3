"""Public dataset run command shared by main.py and the internal CLI.
main.py 与内部 CLI 共用的公开数据集运行命令。

One implementation of the dataset loop is used by both the single-entry
``main.py run-dataset`` and the internal maintenance CLI; no caller re-implements
the loop. The loop reuses the existing composition points only: ``get_adapter``,
``RunStore``, ``create_model``, ``assemble_runtime``, and ``build_dataset_runner``.
main.py run-dataset 与内部维护 CLI 共用同一份数据集循环实现，调用方不重复实现；
循环只复用现有组合点：``get_adapter``、``RunStore``、``create_model``、
``assemble_runtime`` 与 ``build_dataset_runner``。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.base import JsonResponseCache, RequestMeta
from models.entry import create_model
from spacers_agent.bootstrap import assemble_runtime, build_dataset_runner
from spacers_agent.clients.deepseek import DeepSeekJudgeClient
from spacers_agent.dataset_adapters import get_adapter
from spacers_agent.evaluation import (
    build_count_judge_payload,
    build_judge_request_hash,
    merge_count_evaluation,
)
from spacers_agent.run_store import RunStore
from spacers_agent.schemas import CountTargetSpec, CountingResult, UnifiedSample
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.artifact_writer import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Versioned prompt assets snapshotted into every dataset run manifest.
# 每次数据集运行 manifest 快照的版本化 Prompt 资源。
PROMPT_PATHS = [
    PROJECT_ROOT / "prompts" / "count_tile_v4.md",
    PROJECT_ROOT / "prompts" / "json_repair_v1.md",
    PROJECT_ROOT / "prompts" / "router_v1.md",
    PROJECT_ROOT / "prompts" / "target_parse_v1.md",
    PROJECT_ROOT / "prompts" / "count_repair_v1.md",
    PROJECT_ROOT / "prompts" / "seam_verify_v1.md",
    PROJECT_ROOT / "prompts" / "missing_point_review_v1.md",
    PROJECT_ROOT / "prompts" / "missing_point_review_v2.md",
    PROJECT_ROOT / "prompts" / "missing_point_review_v3.md",
    PROJECT_ROOT / "prompts" / "change_dual_path_v1.md",
    PROJECT_ROOT / "prompts" / "spatial_v2.md",
    PROJECT_ROOT / "prompts" / "spatial_v3.md",
    PROJECT_ROOT / "prompts" / "spatial_v4.md",
    PROJECT_ROOT / "prompts" / "spatial_v5.md",
    PROJECT_ROOT / "prompts" / "spatial_candidate_review_v1.md",
    PROJECT_ROOT / "prompts" / "spatial_candidate_review_v2.md",
    PROJECT_ROOT / "prompts" / "spatial_candidate_review_v3.md",
    PROJECT_ROOT / "prompts" / "spatial_candidate_review_v4.md",
    PROJECT_ROOT / "prompts" / "spatial_candidate_review_v5.md",
    PROJECT_ROOT / "prompts" / "general_vqa_v2.md",
    PROJECT_ROOT / "prompts" / "deepseek_judge_v1.md",
    PROJECT_ROOT / "prompts" / "deepseek_judge_repair_v1.md",
    PROJECT_ROOT / "prompts" / "deepseek_vqa_judge_v1.md",
]


@dataclass(frozen=True)
class RunDatasetOptions:
    """Typed options for one dataset run. / 一次数据集运行的定型选项。"""

    dataset: str
    root: Path
    split: str
    task: str | None
    run_id: str | None
    max_samples: int
    start_index: int
    sample_concurrency: int
    resume: bool
    fail_fast: bool
    evaluate: bool
    judge_policy: str
    sample_ids: set[str] | None = None
    shard_index: int = 0
    shard_count: int = 1


async def run_dataset(settings: AppSettings, options: RunDatasetOptions) -> int:
    """Create or resume a dataset run after adapter probing.
    在适配器探测后创建或恢复数据集运行。

    Keeps SampleRunner fallback, Judge, Resume, Artifact, and Report behavior.
    保留 SampleRunner fallback、Judge、Resume、Artifact 与 Report 行为。
    """

    if (
        options.max_samples < 0
        or options.start_index < 0
        or options.sample_concurrency < 1
        or options.shard_count < 1
        or not 0 <= options.shard_index < options.shard_count
    ):
        raise ValueError(
            "max-samples, start-index, sample-concurrency, and shard selection must be valid"
        )
    settings.paths.dataset_root = options.root
    run_id = options.run_id or f"{options.dataset}-{options.split}"
    run_dir = settings.runs.root / run_id
    if not run_dir.exists():
        RunStore(settings.runs.root, PROJECT_ROOT).create_run(
            settings,
            prompt_paths=PROMPT_PATHS,
            run_id=run_id,
            dataset=options.dataset,
            split=options.split,
            sample_filter=options.task,
        )
    adapter = get_adapter(options.dataset)
    requested = (
        options.task.split(",")
        if options.task
        else sorted(getattr(adapter, "supported_tasks", ()))
    )
    qwen_client = _client(settings, run_dir)
    judge_client = None
    if options.evaluate and options.judge_policy != "none" and "general_vqa" in requested:
        judge_client = _vqa_judge_client(settings, run_dir)
    runtime = assemble_runtime(
        settings,
        qwen_client=qwen_client,
        judge_client=judge_client,
        prompt_root=PROJECT_ROOT / "prompts",
    )
    summaries: list[Any] = []
    for task in requested:
        runner = build_dataset_runner(
            runtime,
            adapter=adapter,
            run_dir=run_dir,
            settings=settings,
            judge_policy=options.judge_policy if options.evaluate else "none",
        )
        summaries.append(
            await runner.run(
                split=options.split,
                task=task,
                resume=options.resume,
                limit=None if options.max_samples == 0 else options.max_samples,
                shard_index=options.shard_index,
                shard_count=options.shard_count,
                start_index=options.start_index,
                sample_ids=options.sample_ids,
                fail_fast=options.fail_fast,
                sample_concurrency=options.sample_concurrency,
            )
        )
    if options.evaluate and any(
        task in {"counting", "fine_grained_counting"} for task in requested
    ):
        # A missing key still produces deterministic records; it never silently
        # skips evaluation. 缺少密钥时仍生成确定性记录，绝不静默跳过评估。
        deepseek = bool(os.environ.get(settings.models.deepseek.api_key_env))
        await _evaluate_counting_results(settings, run_dir, deepseek=deepseek)
    if any(task in {"counting", "fine_grained_counting"} for task in requested):
        from spacers_agent.counting_report import build_multiagent_counting_report

        report_path = build_multiagent_counting_report(
            run_dir,
            qwen=settings.models.qwen,
            model_load_seconds=float(getattr(qwen_client, "load_seconds", 0.0)),
        )
        if report_path is not None:
            print(json.dumps({"counting_html_report": str(report_path)}, ensure_ascii=False))
    if "general_vqa" in requested:
        from spacers_agent.vqa_report import build_multiagent_vqa_report

        report_path = build_multiagent_vqa_report(
            run_dir,
            qwen=settings.models.qwen,
            model_load_seconds=float(getattr(qwen_client, "load_seconds", 0.0)),
        )
        if report_path is not None:
            print(json.dumps({"html_report": str(report_path)}, ensure_ascii=False))
    print(
        json.dumps(
            [summary.model_dump(mode="json") for summary in summaries],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _client(settings: AppSettings, run_dir: Path) -> Any:
    """Create the local Transformers Qwen client with run-scoped safe cache.
    创建带运行范围安全缓存的本地 Transformers Qwen 客户端。
    """

    cache = JsonResponseCache(run_dir / "cache")
    return create_model(
        "qwen_transformers",
        settings=settings.models.qwen,
        repair_prompt=(PROJECT_ROOT / "prompts" / "json_repair_v1.md").read_text(
            encoding="utf-8"
        ),
        cache=cache,
    )


def _vqa_judge_client(settings: AppSettings, run_dir: Path) -> DeepSeekJudgeClient:
    """Create the default text-only VQA Judge or fail visibly when its key is absent.
    创建默认的纯文本 VQA Judge；密钥缺失时明确失败。
    """

    return DeepSeekJudgeClient(
        settings.models.deepseek,
        judge_prompt=(PROJECT_ROOT / "prompts" / "deepseek_vqa_judge_v1.md").read_text(
            encoding="utf-8"
        ),
        repair_prompt=(
            PROJECT_ROOT / "prompts" / "deepseek_judge_repair_v1.md"
        ).read_text(encoding="utf-8"),
        cache=JsonResponseCache(run_dir / "deepseek_vqa_cache"),
    )


async def _evaluate_counting_results(
    settings: AppSettings, run_dir: Path, *, deepseek: bool
) -> None:
    """Evaluate persisted counting results without reissuing Qwen requests.
    不重新发起 Qwen 请求地评估已存计数结果。
    """

    records: list[Any] = []
    judge_client = None
    if deepseek:
        judge_client = DeepSeekJudgeClient(
            settings.models.deepseek,
            judge_prompt=(PROJECT_ROOT / "prompts" / "deepseek_judge_v1.md").read_text(
                encoding="utf-8"
            ),
            repair_prompt=(
                PROJECT_ROOT / "prompts" / "deepseek_judge_repair_v1.md"
            ).read_text(encoding="utf-8"),
            cache=JsonResponseCache(run_dir / "deepseek_cache"),
        )
    for result_path in run_dir.rglob("counting_result.json"):
        result = CountingResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        sample_path = result_path.parent / "sample.json"
        sample = (
            UnifiedSample.model_validate_json(sample_path.read_text(encoding="utf-8"))
            if sample_path.is_file()
            else None
        )
        ground_truth = sample.ground_truth if sample is not None else None
        target = CountTargetSpec(
            canonical_label=result.target,
            inclusion_rule="Persisted target specification unavailable.",
            exclusion_rule="Persisted target specification unavailable.",
        )
        if judge_client is None:
            records.append(
                merge_count_evaluation(
                    sample_id=result.sample_id,
                    counting=result,
                    ground_truth=ground_truth,
                )
            )
            continue
        payload = build_count_judge_payload(
            question=result.question,
            target=target,
            display_answer=f"{result.final_count} accepted global points",
            counting=result,
            ground_truth=ground_truth,
            min_confidence=settings.counting.min_confidence,
        )
        try:
            verdict = await judge_client.judge(
                payload,
                request_meta=RequestMeta(
                    request_id=f"{result.sample_id}:deepseek",
                    request_hash=build_judge_request_hash(
                        model=settings.models.deepseek.model,
                        prompt_text=judge_client.judge_prompt,
                        sample_id=result.sample_id,
                        payload=payload,
                    ),
                    prompt_version="deepseek-judge-v1",
                    sample_id=result.sample_id,
                    artifact_dir=result_path.parent / "deepseek",
                ),
            )
            records.append(
                merge_count_evaluation(
                    sample_id=result.sample_id,
                    counting=result,
                    ground_truth=ground_truth,
                    judge_parsed=verdict,
                )
            )
        except Exception as error:
            records.append(
                merge_count_evaluation(
                    sample_id=result.sample_id,
                    counting=result,
                    ground_truth=ground_truth,
                    judge_error=f"{type(error).__name__}: {error}",
                )
            )
    payload = [record.model_dump(mode="json") for record in records]
    atomic_write_json(run_dir / "evaluation.json", payload)
    atomic_write_json(run_dir / "evaluation_records.json", payload)
    jsonl_path = run_dir / "evaluations.jsonl"
    temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    temporary.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(jsonl_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
