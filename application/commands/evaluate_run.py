"""Public `evaluate-run` CLI command: offline deterministic evaluation for one
run, with an optional DeepSeek judge.

公开 `evaluate-run` CLI 命令：单个 run 的离线确定性评估与可选 DeepSeek
judge。与 fresh/resume 共用同一确定性分派（build_deterministic_evaluation），
按执行任务（status.task）定键——候选兜底后绝不按 canonical resolved
sample.task 生成错误指标族。覆盖 VQA 族/counting 族/兼容 grounding/caption；
不支持的确定性任务与不兼容几何记录 not_applicable（绝不伪造指标）。
DeepSeek 仅用于 Judge（失败保留确定性记录）。本命令绝不构造/调用 Qwen。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from application.prompts import PromptCatalog
from application.settings import load_settings
from agents.counting.schema import CountingResult, CountTargetSpec
from data.schema import UnifiedSample
from evaluation.judges.deepseek import DeepSeekJudgeClient
from evaluation.records import EvaluationRecord
from models.cache import JsonResponseCache
from reporting.adapters import (
    evaluation_filename_for_task,
    load_evaluation,
    load_payload,
    load_sample,
    load_status,
)
from reporting.builder import build_report
from workflows.artifact_writer import ArtifactWriter
from workflows.judge_service import JudgeService
from workflows.run_store import RunManifest
from workflows.sample_runner import _COUNTING_TASKS, build_deterministic_evaluation

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130

_VQA_EVALUATION_FILENAME = "vqa_evaluation.json"
_COUNTING_EVALUATION_FILENAME = "counting_evaluation.json"
_AGENT_RESULT_FILENAME = "agent_result.json"


def run_evaluate_run(args: argparse.Namespace) -> int:
    """Evaluate one run offline and print the summary as JSON.
    离线评估一个 run 并以 JSON 输出摘要。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        project_root = Path(__file__).resolve().parents[2]
        run_dir = settings.runs.root / args.run_id
        if not run_dir.is_dir():
            raise ValueError("run does not exist")
        _validate_manifest(run_dir, args.run_id)
        artifact_writer = ArtifactWriter()
        judge_service = None
        if args.deepseek:
            catalog = PromptCatalog(project_root / "prompts")
            judge_service = _build_judge_service(settings, catalog)
        evaluated, not_applicable, judge_results = _evaluate_run(
            run_dir,
            artifact_writer,
            judge_service,
            only_missing=args.only_missing,
            force_judge=args.force_judge,
        )
        report = build_report(run_dir)  # refreshed unified report / 刷新的统一报告
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
                "evaluated": evaluated,
                "not_applicable": not_applicable,
                "judge": judge_results,
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
        vqa_judge_prompt=catalog["vqa_judge"],
        judge_client=client,
        model_id=settings.models.deepseek.model,
        counting_min_confidence=settings.counting.min_confidence,
    )


def _evaluate_run(
    run_dir: Path,
    artifact_writer: ArtifactWriter,
    judge_service: JudgeService | None,
    *,
    only_missing: bool,
    force_judge: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Evaluate every succeeded sample by its execution task; missing sample
    artifacts or incompatible geometry become stable not_applicable entries,
    never fabricated metrics. 按执行任务评估全部 succeeded 样本；样本产物
    缺失或不兼容几何转为稳定 not_applicable 条目，绝不伪造指标。"""

    evaluated: list[dict] = []
    not_applicable: list[dict] = []
    judge_results: list[dict] = []
    for task_dir, sample_dir in _iter_sample_dirs(run_dir):
        status = load_status(sample_dir)
        if status is None or status.state != "succeeded":
            continue
        execution_task = status.task  # executed task, authoritative / 执行任务，权威
        filename = evaluation_filename_for_task(execution_task)
        if filename is None:
            not_applicable.append(
                {
                    "sample_id": status.sample_id,
                    "task": execution_task,
                    "reason": "unsupported_task",
                }
            )
            continue
        evaluation_path = sample_dir / filename
        if only_missing and evaluation_path.is_file():
            continue
        sample = load_sample(sample_dir)
        payload = load_payload(sample_dir, execution_task)
        if sample is None or payload is None:
            not_applicable.append(
                {
                    "sample_id": status.sample_id,
                    "task": execution_task,
                    "reason": "missing_sample_or_payload",
                }
            )
            continue
        # The judge skip decision must use the record as it was before the
        # deterministic rewrite below. 判定 skip 必须使用下方确定性重写前的
        # 记录。
        evaluation_before = load_evaluation(sample_dir, execution_task)
        evaluation, name = build_deterministic_evaluation(
            sample=sample,
            execution_payload=payload,
            execution_task=execution_task,
        )
        if evaluation is None or name is None:
            # Fail closed: incompatible coordinate frames or missing
            # references never produce a fake metric file.
            # fail-closed：不兼容坐标系或缺失参考绝不产生伪造指标文件。
            not_applicable.append(
                {
                    "sample_id": status.sample_id,
                    "task": execution_task,
                    "reason": "incompatible_geometry_or_no_reference",
                }
            )
            continue
        artifact_writer.write_evaluation(sample_dir, evaluation, filename=name)
        evaluated.append(
            {
                "sample_id": status.sample_id,
                "task": execution_task,
                "filename": name,
            }
        )
        if judge_service is not None and execution_task == "general_vqa":
            judge_results.append(
                _judge_one(
                    judge_service,
                    sample,
                    sample_dir,
                    # The skip decision uses the pre-recompute record so a
                    # succeeded judge survives the deterministic rewrite.
                    # skip 判断使用重算前的记录，使 succeeded judge 在确定性
                    # 重写后仍被跳过。
                    existing=evaluation_before,
                    force=force_judge,
                )
            )
        elif judge_service is not None and execution_task in _COUNTING_TASKS:
            judge_results.append(
                _judge_counting_one(
                    judge_service,
                    sample,
                    payload,
                    sample_dir,
                    existing=evaluation_before,
                    force=force_judge,
                )
            )
    return evaluated, not_applicable, judge_results


def _judge_counting_one(
    judge_service: JudgeService,
    sample: UnifiedSample,
    payload: object,
    sample_dir: Path,
    *,
    existing: EvaluationRecord | None,
    force: bool,
) -> dict:
    """Judge one counting sample through JudgeService.judge_counting; a
    persisted succeeded judge is skipped unless force. The ground truth stays
    authoritative, the persisted CountingResult.target provides the label, and
    an exact persisted CountTargetSpec is preferred over a stable neutral
    reconstruction. Judge failures keep the deterministic record and are
    recorded as a stable judge_error; zero Qwen calls.
    经 JudgeService.judge_counting 审核一个计数样本；持久化 succeeded judge
    除非 force 否则跳过。ground truth 保持权威，持久化 CountingResult.target
    提供标签，精确持久化 CountTargetSpec 优先于稳定中性重建。judge 失败
    保留确定性记录并记录稳定 judge_error；零 Qwen 调用。"""

    if not force and existing is not None and existing.judge_status == "succeeded":
        return {"sample_id": sample.sample_id, "status": "skipped_succeeded"}
    if not isinstance(payload, CountingResult):
        return {
            "sample_id": sample.sample_id,
            "judge_status": "failed",
            "judge_error": "TypeError",
        }
    try:
        target = _count_target_for(sample, payload)
        record = judge_service.judge_counting(
            sample_id=sample.sample_id,
            question=sample.question,
            target=target,
            display_answer=str(payload.final_count),
            counting=payload,
            ground_truth=sample.ground_truth,
            artifact_dir=sample_dir / "deepseek",
        )
        artifact_writer = ArtifactWriter()
        artifact_writer.write_evaluation(
            sample_dir, record, filename=_COUNTING_EVALUATION_FILENAME
        )
        return {
            "sample_id": sample.sample_id,
            "judge_status": record.judge_status,
            "judge_error": record.judge_error,
        }
    except Exception as error:
        return {
            "sample_id": sample.sample_id,
            "judge_status": "failed",
            "judge_error": type(error).__name__,
        }


def _count_target_for(
    sample: UnifiedSample,
    payload: CountingResult,
) -> CountTargetSpec:
    """Prefer the exact persisted CountTargetSpec hint; otherwise reconstruct
    a stable neutral spec from the persisted canonical label — never invent
    visual facts. 优先使用精确持久化 CountTargetSpec hint；否则从持久化
    canonical 标签重建稳定中性 spec——绝不虚构视觉事实。"""

    hint = (sample.metadata or {}).get("count_target_hint")
    if isinstance(hint, dict):
        try:
            return CountTargetSpec.model_validate(hint)
        except ValueError:
            pass
    return CountTargetSpec(
        canonical_label=payload.target,
        inclusion_rule=f"count all {payload.target}",
        exclusion_rule="exclude none",
    )


def _judge_one(
    judge_service: JudgeService,
    sample: UnifiedSample,
    sample_dir: Path,
    *,
    existing: EvaluationRecord | None,
    force: bool,
) -> dict:
    """Judge one VQA sample; a persisted succeeded judge is skipped unless
    force. Judge failures keep the deterministic record and are recorded as a
    stable judge_error. 审核一个 VQA 样本；持久化 succeeded judge 除非 force
    否则跳过。judge 失败保留确定性记录并记录稳定 judge_error。"""

    if not force and existing is not None and existing.judge_status == "succeeded":
        return {"sample_id": sample.sample_id, "status": "skipped_succeeded"}
    try:
        if force:
            record = _force_judge(judge_service, sample, sample_dir)
        else:
            record = judge_service.judge_vqa_resume(
                sample=sample,
                candidate_answer="",
                sample_dir=sample_dir,
                judge_policy="all",
                call_budget=None,
            )
        artifact_writer = ArtifactWriter()
        artifact_writer.write_evaluation(
            sample_dir, record, filename=_VQA_EVALUATION_FILENAME
        )
        return {
            "sample_id": sample.sample_id,
            "judge_status": record.judge_status,
            "judge_error": record.judge_error,
        }
    except Exception as error:
        return {
            "sample_id": sample.sample_id,
            "judge_status": "failed",
            "judge_error": type(error).__name__,
        }


def _force_judge(
    judge_service: JudgeService,
    sample: UnifiedSample,
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
        judge_policy="all",
        call_budget=None,
    )


def _iter_sample_dirs(run_dir: Path):
    """Yield (task_dir, sample_dir) pairs in deterministic order.
    按确定性顺序产出 (task_dir, sample_dir) 对。"""

    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        return
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        samples_root = task_dir / "samples"
        if not samples_root.is_dir():
            continue
        for sample_dir in sorted(samples_root.iterdir()):
            if sample_dir.is_dir():
                yield task_dir, sample_dir
