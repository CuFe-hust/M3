"""One-command point-derived counting workflow. / 单命令点导出计数工作流。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from spacers_agent.commands.common import EXIT_DATA, EXIT_INVARIANT, EXIT_OK, EXIT_PARTIAL, EXIT_QWEN_FAILED, emit_summary, prompts, qwen_client
from spacers_agent.counting import PointCountingOrchestrator
from spacers_agent.imaging import read_normalized_image
from spacers_agent.run_store import RunStore
from spacers_agent.schemas import CountTargetSpec, ImageRef, UnifiedSample, stable_sample_id
from spacers_agent.settings import AppSettings
from spacers_agent.targeting import CountTargetParser
from spacers_agent.workflows.counting_workflow import CountingWorkflow, EvidenceReviewer, EvaluationService, SeamVerifier


def add_parser(commands: Any) -> None:
    """Register the complete count-image argument contract. / 注册完整的 count-image 参数契约。"""

    parser = commands.add_parser("count-image", help="Run the complete point-derived count workflow.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--target-spec", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-seam-verify", action="store_true")
    parser.add_argument("--max-qwen-calls", type=int)
    parser.add_argument("--max-deepseek-calls", type=int)


def run(settings: AppSettings, args: Any) -> int:
    """Run one image and always emit a final JSON summary. / 运行单张图并始终输出最终 JSON 摘要。"""

    try:
        return asyncio.run(_run(settings, args))
    except FileNotFoundError as error:
        emit_summary({"status": "data_error", "error": str(error)})
        return EXIT_DATA
    except ValueError as error:
        emit_summary({"status": "argument_error", "error": str(error)})
        return EXIT_INVARIANT if "final_count" in str(error) else EXIT_DATA
    except Exception as error:
        emit_summary({"status": "qwen_failed", "error_code": type(error).__name__})
        return EXIT_QWEN_FAILED


async def _run(settings: AppSettings, args: Any) -> int:
    if args.max_qwen_calls is not None and args.max_qwen_calls < 1:
        raise ValueError("--max-qwen-calls must be positive")
    if args.max_deepseek_calls is not None and args.max_deepseek_calls < 0:
        raise ValueError("--max-deepseek-calls must not be negative")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    sample_id = stable_sample_id(None, args.image, args.question, 0)
    run_id = args.run_id or sample_id
    run_dir = settings.runs.root / run_id
    sample_dir = run_dir / "samples" / sample_id
    result_path = sample_dir / "counting_result.json"
    if args.resume and result_path.is_file() and not args.force:
        result = result_path.read_text(encoding="utf-8")
        emit_summary({"run_id": run_id, "sample_id": sample_id, "status": "resumed", "result_path": str(result_path)})
        return EXIT_OK
    if not run_dir.exists():
        RunStore(settings.runs.root, Path(__file__).resolve().parents[2]).create_run(
            settings,
            prompt_paths=[path for path in (Path(__file__).resolve().parents[2] / "prompts").glob("*_v1.md")]
            + [Path(__file__).resolve().parents[2] / "prompts" / "count_tile_v3.md"],
            run_id=run_id,
        )
    image = read_normalized_image(args.image)
    counting = settings.counting.model_copy(update={"seam_verify": not args.no_seam_verify})
    client = qwen_client(settings, run_dir)
    store = RunStore(settings.runs.root, Path(__file__).resolve().parents[2])
    sample = UnifiedSample(sample_id=sample_id, dataset="single-image", split="adhoc", task="counting", images=[ImageRef(image_id=sample_id, path=args.image, role="image", width=image.width, height=image.height)], question=args.question)
    workflow = CountingWorkflow(target_parser=CountTargetParser(client, prompts()["target"], settings.models.qwen.model), orchestrator_factory=lambda directory: PointCountingOrchestrator(client, counting=counting, qwen=settings.models.qwen, system_prompt=prompts()["count"], seam_prompt=prompts()["seam"], run_dir=directory), seam_verifier=SeamVerifier(), evidence_reviewer=EvidenceReviewer(), evaluation_service=EvaluationService(), run_store=store)
    target = CountTargetSpec.model_validate_json(args.target_spec.read_text(encoding="utf-8")) if args.target_spec else None
    execution = await workflow.run(sample, run_dir=run_dir, target_override=target, evaluate=args.evaluate, render=args.render, resume=args.resume and not args.force)
    result = execution.result
    status = "completed" if result.status in {"completed", "completed_with_warnings"} else "partial"
    emit_summary({"run_id": run_id, "sample_id": sample_id, "status": status, "final_count": result.final_count, "result_path": str(result_path)})
    return EXIT_OK if status == "completed" else EXIT_PARTIAL
