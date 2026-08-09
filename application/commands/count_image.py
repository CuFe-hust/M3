"""Public `count-image` CLI command: one-image point-derived counting.

公开 `count-image` CLI 命令：单图点导出计数。使用当前 CountingAgent 与当前
run/sample 存储布局（`runs/<run_id>/tasks/counting/samples/<sha256[:24]>`），
绝不使用旧布局。SampleRunner 自动产出 sample/status/routing/counting_result/
agent_trace 与（有 GT 时）counting_evaluation；--render 附加 overlay。
resume 遇 succeeded 当前结果零 Qwen 调用；--force 重跑；旧版绝对路径
status 校验失败视为无效并重跑；budget 覆盖仅请求局部。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from agents.counting.schema import CountingResult, CountTargetSpec
from application.prompts import PromptCatalog
from application.runtime import Runtime
from application.settings import load_settings
from data.schema import GroundTruth, ImageRef, UnifiedSample, stable_sample_id
from reporting.visualization import render_counting_overlay
from workflows.call_budget import CallBudget
from workflows.dataset_runner import storage_key
from workflows.run_store import RunStore
from workflows.schema import SampleRunStatus

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130

_STATUS_FILENAME = "status.json"
_COUNTING_RESULT_FILENAME = "counting_result.json"


def run_count_image(args: argparse.Namespace) -> int:
    """Run one counting request and always emit a final JSON summary.
    运行一次计数请求并始终输出最终 JSON 摘要。"""

    try:
        return asyncio.run(_run(args))
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


async def _run(args: argparse.Namespace) -> int:
    if args.max_qwen_calls is not None and args.max_qwen_calls < 1:
        raise ValueError("--max-qwen-calls must be positive")
    if args.max_deepseek_calls is not None and args.max_deepseek_calls < 0:
        raise ValueError("--max-deepseek-calls must not be negative")
    settings = load_settings(
        Path(args.config) if getattr(args, "config", None) else None,
    )
    project_root = Path(__file__).resolve().parents[2]
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError("image file does not exist")
    with Image.open(image_path) as source:
        width, height = source.size

    sample_id = stable_sample_id(
        dataset="single-image",
        split="adhoc",
        source_id=None,
        relative_image_paths=[image_path.name],
        question=args.question,
        source_index=0,
    )
    run_id = args.run_id or sample_id
    run_dir = settings.runs.root / run_id
    sample_dir = run_dir / "tasks" / "counting" / "samples" / storage_key(sample_id)

    if args.resume and not args.force:
        persisted = _read_status(sample_dir)
        if persisted is not None and persisted.state == "succeeded":
            # Resume succeeded current result without any new Qwen call.
            # resume 已成功的当前结果，不发起任何新 Qwen 调用。
            final_count = _read_final_count(sample_dir)
            _emit(
                {
                    "status": "resumed",
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "final_count": final_count,
                    "run_dir": run_dir.as_posix(),
                }
            )
            return EXIT_OK
    if not run_dir.exists():
        catalog = PromptCatalog(project_root / "prompts")
        RunStore(settings.runs.root, project_root).create_run(
            config_payload=settings.to_config_payload(),
            model_ids={
                "qwen": settings.models.qwen.effective_cache_model_id,
                "deepseek": settings.models.deepseek.model,
            },
            prompt_paths=catalog.snapshot_paths(),
            run_id=run_id,
        )

    metadata: dict[str, Any] = {}
    if args.target_spec:
        spec = CountTargetSpec.model_validate(
            json.loads(Path(args.target_spec).read_text(encoding="utf-8"))
        )
        metadata["count_target_hint"] = spec.model_dump(mode="json")
    effective_settings = settings
    if args.no_seam_verify:
        # Request-local override only; global settings stay untouched.
        # 仅请求局部覆盖；全局配置保持不变。
        effective_settings = settings.model_copy(
            update={
                "counting": settings.counting.model_copy(
                    update={"seam_verify": False}
                )
            }
        )
    runtime = Runtime.create(
        settings=effective_settings,
        project_root=project_root,
        api_key=None,
    )
    budget = _build_budget(runtime, args)
    sample = UnifiedSample(
        sample_id=sample_id,
        dataset="single-image",
        split="adhoc",
        task="counting",
        images=[
            ImageRef(
                image_id="image-0",
                path=Path(image_path.name),
                role="image",
                width=width,
                height=height,
            )
        ],
        question=args.question,
        ground_truth=GroundTruth(),
        metadata=metadata,
    )
    runner = runtime.components.sample_runner_factory(data_root=image_path.parent)
    outcome = await runner.run_one(
        sample, sample_dir, judge_policy="none", budget=budget
    )
    execution = outcome.execution
    if execution is None or not isinstance(execution.payload, CountingResult):
        raise TypeError("count-image requires the native CountingAgent result")
    result: CountingResult = execution.payload
    if args.render:
        with Image.open(image_path) as source:
            render_counting_overlay(
                source,
                result=result,
                output_path=sample_dir / "overlay.png",
            )
    status = (
        "completed"
        if result.status in {"completed", "completed_with_warnings"}
        else "partial"
    )
    _emit(
        {
            "status": status,
            "run_id": run_id,
            "sample_id": sample_id,
            "final_count": result.final_count,
            "result_path": (
                sample_dir.relative_to(run_dir) / _COUNTING_RESULT_FILENAME
            ).as_posix(),
            "run_dir": run_dir.as_posix(),
        }
    )
    return EXIT_OK


def _build_budget(runtime: Runtime, args: argparse.Namespace) -> CallBudget:
    """Request-local budget override; default limits apply when unset.
    请求局部预算覆盖；未设置时使用默认限制。"""

    default = runtime.components.call_budget_factory.create_for_sample("counting")
    return CallBudget(
        max_qwen_calls=args.max_qwen_calls or default.max_qwen_calls,
        max_deepseek_calls=(
            args.max_deepseek_calls
            if args.max_deepseek_calls is not None
            else default.max_deepseek_calls
        ),
    )


def _read_status(sample_dir: Path) -> SampleRunStatus | None:
    """Read the persisted sample status; corrupt or schema-invalid files
    (including legacy absolute result_path values) count as absent and
    trigger a re-run. 读取持久化样本状态；损坏或 schema 非法文件（含旧版
    绝对 result_path）视为不存在并触发重跑。"""

    path = sample_dir / _STATUS_FILENAME
    if not path.is_file():
        return None
    try:
        return SampleRunStatus.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError):
        return None


def _read_final_count(sample_dir: Path) -> int | None:
    """Read the persisted final count for a resumed summary; corrupt or
    missing results yield None. 为 resumed 摘要读取持久化最终数量；损坏或
    缺失返回 None。"""

    path = sample_dir / _COUNTING_RESULT_FILENAME
    if not path.is_file():
        return None
    try:
        result = CountingResult.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError):
        return None
    return result.final_count


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
