"""Public `count-image` CLI command: one-image point-derived counting.

公开 `count-image` CLI 命令：单图点导出计数。使用当前 CountingAgent 与当前
run/sample 存储布局（`runs/<run_id>/tasks/counting/samples/<sha256[:24]>`），
绝不使用旧布局。SampleRunner 产出 sample/status/routing/counting_result/
agent_trace；--evaluate 时才产出 counting_evaluation；--render 附加 overlay。

运行身份遵循冻结契约：fresh 无 --run-id 恒创建唯一 RunStore run id（绝不
用 sample_id 默认）；fresh 显式 --run-id 已存在稳定失败；--resume 必须显式
--run-id 且校验既有 run 的 manifest 与 count-image 调用身份（run_request）；
--force 只在 resume 的既有 run 内重跑样本，绝不弱化 fresh 身份规则。budget
与 seam_verify 覆盖仅请求局部。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from workflows.run_store import RunManifest, RunStore
from workflows.schema import RunRequest, SampleRunStatus

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
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    sample_id = stable_sample_id(
        dataset="single-image",
        split="adhoc",
        source_id=None,
        relative_image_paths=[image_path.name],
        question=args.question,
        source_index=0,
    )
    store = RunStore(settings.runs.root, project_root)
    request: RunRequest | None = None
    if args.resume:
        # Resume requires an explicit run id and a matching count-image
        # invocation; never guess a run by sample id.
        # resume 要求显式 run id 与匹配的 count-image 调用；绝不按 sample id
        # 猜测 run。
        if not args.run_id:
            raise ValueError("--resume requires --run-id")
        run_id = args.run_id
        run_dir = settings.runs.root / run_id
        request = _validate_resume_run(store, run_dir, run_id, sample_id, image_sha256)
    else:
        catalog = PromptCatalog(project_root / "prompts")
        manifest = store.create_run(
            config_payload=settings.to_config_payload(),
            model_ids={
                "qwen": settings.models.qwen.effective_cache_model_id,
                "deepseek": settings.models.deepseek.model,
            },
            prompt_paths=catalog.snapshot_paths(),
            run_id=args.run_id,
            dataset="single-image",
            split="adhoc",
        )
        run_id = manifest.run_id
        run_dir = settings.runs.root / run_id
        store.write_run_request(
            run_dir,
            _count_image_request(
                sample_id=sample_id,
                image_sha256=image_sha256,
                question=args.question,
                image_dir=image_path.parent,
                evaluate=args.evaluate,
            ),
        )
    sample_dir = run_dir / "tasks" / "counting" / "samples" / storage_key(sample_id)

    if args.resume and not args.force:
        persisted = _read_status(sample_dir)
        if persisted is not None and persisted.state == "succeeded":
            # Zero-Qwen resume requires a valid persisted CountingResult that
            # matches this sample; anything else counts as incomplete and
            # re-executes. 零 Qwen resume 需要与样本匹配的合法持久化
            # CountingResult；否则视为不完整并重跑。
            result = _read_valid_counting_result(sample_dir, sample_id)
            if result is not None:
                _emit(
                    {
                        "status": "resumed",
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "final_count": result.final_count,
                        "run_dir": run_dir.as_posix(),
                    }
                )
                return EXIT_OK

    # On resume the persisted invocation's evaluate intent is authoritative;
    # CLI --evaluate is ignored (post-hoc evaluation changes belong to
    # evaluate-run). 在 resume 上持久化调用的 evaluate 意图权威；CLI
    # --evaluate 被忽略（事后评估变更属于 evaluate-run）。
    effective_evaluate = request.evaluate if request is not None else args.evaluate
    if args.resume and not effective_evaluate:
        # Remove any stale evaluation artifact from a previous inconsistent
        # state before re-execution — narrowly scoped to the validated sample
        # directory. 在重跑前移除先前不一致状态遗留的评估产物——严格限定在
        # 已校验的样本目录内。
        evaluation_path = sample_dir / "counting_evaluation.json"
        if evaluation_path.is_file():
            evaluation_path.unlink()

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
        sample,
        sample_dir,
        judge_policy="none",
        budget=budget,
        evaluate=effective_evaluate,
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


def _count_image_request(
    *,
    sample_id: str,
    image_sha256: str,
    question: str,
    image_dir: Path,
    evaluate: bool,
) -> RunRequest:
    """The persisted single-image invocation identity. The image itself is
    referenced by its stable content hash, never by a host absolute path.
    持久化单图调用身份。图像以稳定内容哈希引用，绝不使用主机绝对路径。"""

    return RunRequest(
        dataset="single-image",
        dataset_root=image_dir.as_posix().replace("\\", "/"),
        split="adhoc",
        task_mode="explicit",
        tasks=["counting"],
        auto_task=False,
        evaluate=evaluate,
        judge_policy="none",
        command="count-image",
        image_identity=image_sha256,
        question=question,
        sample_id=sample_id,
    )


def _validate_resume_run(
    store: RunStore,
    run_dir: Path,
    run_id: str,
    sample_id: str,
    image_sha256: str,
) -> RunRequest:
    """The resumed run must exist, carry a valid matching manifest, and match
    the expected single-image counting invocation identity; the persisted
    invocation is returned so resume uses its authoritative intent.
    resume 的 run 必须存在、携带合法匹配的 manifest，并匹配预期的单图计数
    调用身份；返回持久化调用，使 resume 使用其权威意图。"""

    manifest_path = run_dir / "manifest.json"
    if not run_dir.is_dir() or not manifest_path.is_file():
        raise ValueError("resume run does not exist")
    try:
        manifest = RunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("resume run manifest is invalid") from exc
    if manifest.run_id != run_id:
        raise ValueError("resume run id mismatch")
    request = store.read_run_request(run_dir)
    if request.command != "count-image":
        raise ValueError("resume run is not a count-image invocation")
    if request.sample_id != sample_id or request.image_identity != image_sha256:
        raise ValueError("resume run invocation mismatch")
    return request


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


def _read_valid_counting_result(
    sample_dir: Path,
    sample_id: str,
) -> CountingResult | None:
    """Read a valid persisted CountingResult that matches this sample;
    missing, corrupt, or mismatched results yield None so resume re-executes
    instead of emitting resumed/null. 读取与样本匹配的合法持久化
    CountingResult；缺失、损坏或不匹配返回 None，使 resume 重跑而非输出
    resumed/null。"""

    path = sample_dir / _COUNTING_RESULT_FILENAME
    if not path.is_file():
        return None
    try:
        result = CountingResult.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError):
        return None
    if result.sample_id != sample_id:
        return None
    return result


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
