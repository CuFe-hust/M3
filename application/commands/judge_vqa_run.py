"""Public `judge-vqa-run` CLI command: DeepSeek judge pass for one run.

All runtime tasks in the canonical VQA evaluation family are eligible. By
default only deterministic mismatches are judged; exact matches never call
DeepSeek, and --force only re-judges an otherwise eligible mismatch.

公开 `judge-vqa-run` CLI 命令：单个 run 的 DeepSeek judge 补判。处理 canonical
VQA evaluation family 中的全部 succeeded mismatch 样本；exact 样本始终跳过，
已 succeeded 的 eligible judge 默认跳过，--force 仅强制重判 eligible mismatch。
绝不构造/调用 Qwen。judge 失败保留确定性记录并记录稳定 judge_error。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from application.prompts import PromptCatalog
from application.settings import load_settings
from evaluation.judges.deepseek import DeepSeekJudgeClient
from evaluation.records import (
    EVALUATION_FILENAME_BY_TASK,
    evaluation_task_for_runtime_task,
)
from models.cache import JsonResponseCache
from reporting.adapters import load_evaluation, load_sample, load_status
from reporting.builder import build_report
from workflows.artifact_writer import ArtifactWriter
from workflows.judge_service import JudgeService
from workflows.run_store import RunManifest

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130

_AGENT_RESULT_FILENAME = "agent_result.json"


def run_judge_vqa_run(args: argparse.Namespace) -> int:
    """Judge every eligible VQA sample of one run and print the summary.
    审核一个 run 的每个合格 VQA 样本并输出摘要。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        project_root = Path(__file__).resolve().parents[2]
        run_dir = settings.runs.root / args.run_id
        if not run_dir.is_dir():
            raise ValueError("run does not exist")
        _validate_manifest(run_dir, args.run_id)
        catalog = PromptCatalog(project_root / "prompts")
        judge_service = _build_judge_service(settings, catalog)
        artifact_writer = ArtifactWriter()
        judged = _judge_run(
            run_dir,
            judge_service,
            artifact_writer,
            force=args.force,
        )
        # Persist the refreshed unified report bundle so the disk artifacts
        # reflect the new judge state; failure fails the command stably.
        # 持久化刷新的统一报告 bundle，使磁盘产物反映新 judge 状态；失败使
        # 命令稳定失败。
        from reporting.exporters import persist_report_bundle

        report = build_report(run_dir)
        persist_report_bundle(run_dir, report)
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
                "judged": judged,
                "report": {
                    "total": report.total,
                    "succeeded": report.succeeded,
                    "partial": report.partial,
                    "failed": report.failed,
                    "skipped": report.skipped,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK


def _validate_manifest(run_dir: Path, run_id: str) -> None:
    """The run must carry a parseable manifest whose identity matches.
    run 必须携带可解析且身份匹配的 manifest。"""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("run manifest is missing")
    try:
        manifest = RunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("run manifest is invalid") from exc
    if manifest.run_id != run_id:
        raise ValueError("run id mismatch")


def _build_judge_service(
    settings: object, catalog: PromptCatalog
) -> JudgeService:
    """Build the judge service with the DeepSeek client; the api key comes
    from the declared environment variable only (never echoed).
    用 DeepSeek 客户端构建 judge 服务；api key 只来自声明的环境变量
    （绝不回显）。"""

    api_key_env = settings.models.deepseek.api_key_env
    api_key = os.environ.get(api_key_env) or None
    if not api_key:
        raise ValueError("deepseek judge requires the api key environment")
    client = DeepSeekJudgeClient(
        settings.models.deepseek,
        api_key=api_key,
        judge_prompt=catalog["count_judge"],
        repair_prompt=catalog["json_repair"],
        cache=JsonResponseCache(settings.runs.root / "service" / "deepseek_cache"),
    )
    return JudgeService(
        judge_prompt=catalog["count_judge"],
        judge_prompt_version=catalog.version("count_judge"),
        vqa_judge_prompt=catalog["vqa_judge"],
        vqa_judge_prompt_version=catalog.version("vqa_judge"),
        judge_client=client,
        model_id=settings.models.deepseek.model,
        counting_min_confidence=settings.counting.min_confidence,
    )


def _judge_run(
    run_dir: Path,
    judge_service: JudgeService,
    artifact_writer: ArtifactWriter,
    *,
    force: bool,
) -> list[dict]:
    """Judge every succeeded VQA-family mismatch; skip exact matches and a
    persisted succeeded judge unless force.
    审核 VQA family 的全部 succeeded mismatch；exact 始终跳过，持久化
    succeeded judge 除非 force 否则跳过。"""

    judged: list[dict] = []
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        return judged
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        samples_root = task_dir / "samples"
        if not samples_root.is_dir():
            continue
        for sample_dir in sorted(samples_root.iterdir()):
            if not sample_dir.is_dir():
                continue
            status = load_status(sample_dir)
            if status is None or status.state != "succeeded":
                continue
            execution_task = status.task  # executed task, authoritative / 执行任务
            if evaluation_task_for_runtime_task(execution_task) != "general_vqa":
                continue
            sample = load_sample(sample_dir)
            if sample is None:
                judged.append(
                    {"sample_id": status.sample_id, "status": "skipped_missing_sample"}
                )
                continue
            existing = load_evaluation(sample_dir, execution_task)
            if existing is not None and _is_exact_vqa(existing):
                judged.append(
                    {"sample_id": status.sample_id, "status": "skipped_exact"}
                )
                continue
            if not force:
                if existing is not None and existing.judge_status == "succeeded":
                    judged.append(
                        {"sample_id": status.sample_id, "status": "skipped_succeeded"}
                    )
                    continue
            try:
                if force:
                    record = _force_judge(judge_service, sample, sample_dir)
                else:
                    record = judge_service.judge_vqa_resume(
                        sample=sample,
                        candidate_answer="",
                        sample_dir=sample_dir,
                        judge_policy="errors-only",
                        call_budget=None,
                    )
                artifact_writer.write_evaluation(
                    sample_dir,
                    record,
                    filename=EVALUATION_FILENAME_BY_TASK["general_vqa"],
                )
                judged.append(
                    (
                        {"sample_id": status.sample_id, "status": "skipped_exact"}
                        if _is_exact_vqa(record)
                        else {
                            "sample_id": status.sample_id,
                            "judge_status": record.judge_status,
                            "judge_error": record.judge_error,
                        }
                    )
                )
            except Exception as error:
                # The deterministic record stays untouched; only the stable
                # error type is recorded. 确定性记录保持不变；只记录稳定错误类型。
                judged.append(
                    {
                        "sample_id": status.sample_id,
                        "judge_status": "failed",
                        "judge_error": type(error).__name__,
                    }
                )
    return judged


def _is_exact_vqa(record: object) -> bool:
    """Return the deterministic VQA exact result without consulting judge data."""

    metrics = getattr(record, "deterministic_metrics", None)
    return bool(getattr(metrics, "exact_match", False))


def _force_judge(
    judge_service: JudgeService,
    sample: object,
    sample_dir: Path,
):
    """Re-judge a VQA sample regardless of the persisted judge status, using
    the persisted agent answer. 无视持久化 judge 状态，用持久化 agent 答案
    重新审核一个 VQA 样本。"""

    result_path = sample_dir / _AGENT_RESULT_FILENAME
    if not result_path.is_file():
        raise ValueError("agent_result.json is missing for force judge")
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    answer = str(raw.get("answer", "")) if isinstance(raw, dict) else ""
    return judge_service.judge_vqa(
        sample=sample,
        candidate_answer=answer,
        sample_dir=sample_dir,
        judge_policy="errors-only",
        call_budget=None,
    )
