#!/usr/bin/env python
"""evaluate_caption.py — dataset -> visual planner -> caption agent -> metrics.

单脚本 caption 评测入口，全部复用既有公开接口，不复制指标/模型构造/路由
逻辑：

1. 数据集输入：data 层既有 adapter（VRSBench val caption、XLRS-Bench
   caption en 等）的 UnifiedSample；脚本把每条样本的 question 统一注入为
   "Describe the image in detail."（不改 adapter、不影响 sample_id）。
2. 执行：复用 Runtime 组装好的组件——每条 fresh 样本经过唯一
   VisualTaskPlanner（visual-task-plan-v5）规划（输入只含归一化图像预览 +
   question），物化后由 caption agent（复用 AgentRegistry 的 caption_agent
   与绑定 Qwen 客户端）执行；plan.task 非 caption 的样本如实计数跳过。
3. 产物：写入与系统 run 相同布局的 runs/<run_id>/tasks/caption/samples/
   <key>/（agent_result.json、caption_evaluation.json、status.json、
   predictions.jsonl），可直接被 --run-id 复用与系统 reporting 读取。
4. 指标：evaluation.metrics.caption.evaluate_caption —— 语料级 BLEU_1..4 /
   METEOR / ROUGE_L / CIDEr（pycocoevalcap，METEOR 需 Java）。CIDEr
   （pycocoevalcap Cider）自带 min 计数裁剪与高斯长度惩罚，即 CIDEr-D
   行为。脚本补充 Avg_L（候选平均词数，仅诊断）。
5. CLAIR 无封闭公式，需固定裁判模型/Prompt 打分，只能作为辅助指标，
   本脚本不内置（如需可复用 evaluation.judges.deepseek.DeepSeekJudgeClient）。

--run-id 给定时跳过推理，直接对已有 run 产物计算指标。

用法：
  python scripts/evaluate_caption.py --dataset VRSBench --root <root> --split val
  python scripts/evaluate_caption.py --dataset XLRS-Bench --root <root> --split <split>
  python scripts/evaluate_caption.py --run-id <run_id> [--output report.json]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the repository root importable when the script is run directly
# (same pattern as the other scripts under scripts/).
# 使仓库根可导入，支持直接运行本脚本（与 scripts/ 下其他脚本一致）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.runtime import Runtime  # noqa: E402
from application.settings import load_settings  # noqa: E402
from agents.base import AgentContext  # noqa: E402
from data.schema import materialize_sample  # noqa: E402
from evaluation.metrics.caption import evaluate_caption  # noqa: E402
from evaluation.records import (  # noqa: E402
    CaptionDeterministicMetrics,
    EvaluationRecord,
    evaluation_task_for_runtime_task,
)
from reporting.adapters import (  # noqa: E402
    iter_current_predictions,
    load_evaluation,
    sample_dir_for_row,
)
from workflows.dataset_runner import storage_key  # noqa: E402

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130

# Unified question injected into every caption sample before planning.
# 规划前注入到每条 caption 样本的统一 question。
CAPTION_QUESTION = "Describe the image in detail."
_AGENT_NAME = "caption_agent"
_RUN_TASK = "caption"


def _collect_caption_records(
    run_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, int]]:
    """Collect candidate/references of every current caption-family record
    from the run's execution index; missing records are counted, never
    fabricated. 从 run 执行索引收集全部当前 caption 族候选/参考答案；缺失
    记录只计数，绝不伪造。"""

    references: dict[str, list[str]] = {}
    candidates: dict[str, list[str]] = {}
    skipped: dict[str, int] = defaultdict(int)
    for row in iter_current_predictions(run_dir):
        run_task = row.get("run_task")
        sample_id = row.get("sample_id")
        if not isinstance(run_task, str) or not isinstance(sample_id, str):
            skipped["malformed_row"] += 1
            continue
        if evaluation_task_for_runtime_task(run_task) != "caption":
            skipped["non_caption_task"] += 1
            continue
        sample_dir = sample_dir_for_row(run_dir, row)
        if sample_dir is None:
            skipped["missing_sample_dir"] += 1
            continue
        evaluation = load_evaluation(sample_dir, run_task)
        metrics = (
            evaluation.deterministic_metrics if evaluation is not None else None
        )
        if not isinstance(metrics, CaptionDeterministicMetrics):
            skipped["missing_caption_metrics"] += 1
            continue
        references[sample_id] = list(metrics.references)
        candidates[sample_id] = [metrics.candidate]
    return references, candidates, dict(skipped)


def _average_length(candidates: dict[str, list[str]]) -> float:
    """Avg_L: mean candidate length in whitespace tokens; diagnostic only.
    Avg_L：候选平均词数（空格分词）；仅诊断。"""

    lengths = [candidate_list[0].split() for candidate_list in candidates.values()]
    return sum(len(tokens) for tokens in lengths) / len(lengths) if lengths else 0.0


async def _run_fresh(
    args: argparse.Namespace,
    settings: Any,
    project_root: Path,
) -> tuple[str, dict[str, int]]:
    """Run one caption pass in-process: adapter samples -> question injection
    -> visual planner -> caption agent, persisting system-layout artifacts.
    进程内执行一次 caption 评测：adapter 样本 → question 注入 → visual
    planner → caption agent，持久化系统同布局产物。"""

    api_key = os.environ.get(settings.models.deepseek.api_key_env) or None
    runtime = Runtime.create(
        settings=settings, project_root=project_root, api_key=api_key
    )
    root = Path(args.root).expanduser().resolve()
    adapter = runtime.registry.get(args.dataset)
    adapter.probe(root, _RUN_TASK)

    def sample_image_root(sample: Any) -> Path:
        """Resolve the image root one sample's ImageRef paths are relative
        to. XLRS materializes bytes/PIL images into an external cache and
        records image_root_kind in metadata, so planner previews and agents
        must resolve against that root, never the dataset root. Other
        adapters (VRSBench) keep paths relative to the dataset root.
        解析单样本 ImageRef 路径所相对的图片根：XLRS 把 bytes/PIL 图片物化
        到外部 cache 并在 metadata 记录 image_root_kind，planner 预览与
        agent 必须按该根解析，而不是数据集根；其他 adapter（VRSBench）的
        路径相对数据集根。"""
        if "image_root_kind" not in sample.metadata:
            return root
        return adapter.image_root_for_sample(sample, root)

    planner = runtime.components.visual_task_planner
    budget_factory = runtime.components.call_budget_factory
    agent = runtime.components.agent_registry.get(_AGENT_NAME)
    qwen_client = runtime.components.qwen_clients[_AGENT_NAME]

    run_id = args.run_id or (
        "caption-eval-"
        + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    )
    run_dir = settings.runs.root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_root = run_dir / "tasks" / _RUN_TASK / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)

    iterator = adapter.iter_samples(root, args.split, _RUN_TASK)
    if args.start_index:
        iterator = itertools.islice(iterator, args.start_index, None)
    if args.limit:
        iterator = itertools.islice(iterator, args.limit)

    for sample in iterator:
        sample.question = CAPTION_QUESTION  # 注入统一 question
        sample_dir = samples_root / storage_key(sample.sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        budget = budget_factory.create_for_sample(sample.sample_id)
        try:
            plan, views = await planner.plan_with_views(
                sample,
                data_root=root,
                artifact_dir=sample_dir,
                budget=budget,
            )
        except Exception as error:
            skipped[f"plan_failed:{type(error).__name__}"] += 1
            continue
        if plan.task != _RUN_TASK:
            # The model-selected task is authoritative; only caption samples
            # can run through the caption agent. 模型选定 task 权威；只有
            # caption 样本才走 caption agent。
            skipped[f"plan_task:{plan.task}"] += 1
            continue
        materialized = materialize_sample(sample, plan.task)
        context = AgentContext(
            artifact_dir=sample_dir,
            qwen_client=qwen_client,
            call_budget=budget,
            data_root=root,
            judge_client=None,
            visual_bindings=runtime.components.visual_bindings,
            visual_task_plan=plan,
            visual_views=views,
        )
        try:
            execution = await agent.run(materialized, context)
        except Exception as error:
            skipped[f"agent_failed:{type(error).__name__}"] += 1
            continue
        _write_sample_artifacts(
            sample_dir, materialized, execution, run_dir, predictions
        )
    _write_predictions(run_dir, predictions)
    return run_id, dict(skipped)


def _write_sample_artifacts(
    sample_dir: Path,
    sample: Any,
    execution: Any,
    run_dir: Path,
    predictions: list[dict[str, Any]],
) -> None:
    """Persist the same sample-level artifacts the system pipeline writes:
    agent_result.json / caption_evaluation.json / status.json, and append the
    execution-index row. 持久化与系统管线相同的样本级产物：
    agent_result.json / caption_evaluation.json / status.json，并追加执行
    索引行。"""

    payload = execution.payload
    (sample_dir / "agent_result.json").write_text(
        json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    references = (
        list(sample.ground_truth.answers)
        if sample.ground_truth is not None
        else []
    )
    record = EvaluationRecord(
        sample_id=sample.sample_id,
        task="caption",
        deterministic_metrics=CaptionDeterministicMetrics(
            candidate=str(getattr(payload, "answer", "")),
            references=references,
        ),
        judge_status="not_requested",
    )
    (sample_dir / "caption_evaluation.json").write_text(
        record.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    updated_at = datetime.now(timezone.utc).isoformat()
    (sample_dir / "status.json").write_text(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "state": "succeeded",
                "task": _RUN_TASK,
                "result_path": "agent_result.json",
                "error_code": None,
                "error_message": None,
                "updated_at": updated_at,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.append(
        {
            "run_task": _RUN_TASK,
            "sample_id": sample.sample_id,
            "task": _RUN_TASK,
            "status": "succeeded",
            "result_path": (
                f"tasks/{_RUN_TASK}/samples/{storage_key(sample.sample_id)}/"
                "agent_result.json"
            ),
            "updated_at": updated_at,
        }
    )


def _write_predictions(run_dir: Path, predictions: list[dict[str, Any]]) -> None:
    """Append the execution-index rows (append-only, one line per sample).
    追加执行索引行（append-only，每样本一行）。"""

    if not predictions:
        return
    path = run_dir / "predictions.jsonl"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    with path.open("a", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not existing and path.is_file():
        pass  # freshly created; nothing else to do / 新建文件，无需其他处理


def run_evaluate_caption(args: argparse.Namespace) -> int:
    """Run (or reuse) a caption evaluation and print corpus metrics.
    运行（或复用）一次 caption 评测并输出语料级指标。"""

    started_at = time.perf_counter()
    try:
        settings = load_settings(
            Path(args.config) if args.config else None,
            environ=os.environ,
        )
        project_root = Path(__file__).resolve().parents[1]
        if args.run_id:
            skipped: dict[str, int] = {}
        else:
            args.run_id, skipped = asyncio.run(
                _run_fresh(args, settings, project_root)
            )
        run_dir = settings.runs.root / args.run_id
        if not run_dir.is_dir():
            raise ValueError("run does not exist")
        references, candidates, collect_skipped = _collect_caption_records(run_dir)
        for key, count in collect_skipped.items():
            skipped[key] = skipped.get(key, 0) + count
        if not references:
            raise ValueError("no caption evaluation records found in run")
        # pycocoevalcap scorers print progress to stdout; suppress so the
        # JSON report stays clean. 抑制 pycocoevalcap 的 stdout 打印，保持
        # JSON 报告干净。
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            metrics = evaluate_caption(references, candidates)
        metrics["Avg_L"] = _average_length(candidates)
        payload: dict[str, Any] = {
            "status": "ok",
            "run_id": args.run_id,
            "run_dir": str(run_dir),
            "dataset": args.dataset,
            "split": args.split,
            "question": CAPTION_QUESTION,
            "record_count": len(references),
            "skipped": skipped,
            "metrics": metrics,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return EXIT_OK
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


def build_parser() -> argparse.ArgumentParser:
    """Thin CLI surface of this script only. 仅本脚本的薄命令行面。"""

    parser = argparse.ArgumentParser(
        prog="evaluate_caption.py",
        description="dataset -> visual planner -> caption agent -> caption metrics",
    )
    parser.add_argument("--config", default=None, help="settings YAML path")
    parser.add_argument("--dataset", default=None, help="dataset key, e.g. VRSBench / XLRS-Bench")
    parser.add_argument("--root", default=None, help="dataset root directory")
    parser.add_argument("--split", default=None, help="dataset split, e.g. val")
    parser.add_argument(
        "--run-id", default=None, help="reuse an existing run (skips inference)"
    )
    parser.add_argument("--limit", type=int, default=None, help="sample limit")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", default=None, help="write the JSON report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_id and not (args.dataset and args.root and args.split):
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "require --dataset/--root/--split, or --run-id",
                }
            ),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    return run_evaluate_caption(args)


if __name__ == "__main__":
    sys.exit(main())
