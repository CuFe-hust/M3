"""Public `count-image` CLI command: one-image point-derived counting.

公开 `count-image` CLI 命令：单图点导出计数。使用当前 CountingAgent 与当前
run/sample 存储布局（`runs/<run_id>/tasks/counting/samples/<sha256[:24]>`），
绝不使用旧布局。SampleRunner 产出 sample/status/routing/counting_result/
agent_trace；--evaluate 时才产出 counting_evaluation；--render 附加 overlay。

运行身份遵循冻结契约：fresh 无 --run-id 恒创建唯一 RunStore run id（绝不
用 sample_id 默认）；fresh 显式 --run-id 已存在稳定失败；--resume 必须显式
--run-id 且校验既有 run 的 manifest 与 count-image 调用身份（run_request）。

调用保真（11G.5.2）：fresh 持久化全部行为影响参数（结构化 target spec
快照 + 稳定哈希——绝非主机路径、seam 校验模式、Qwen/DeepSeek 预算、render
意图、evaluate 意图）；resume/force 一律以持久化调用为权威，绝不从当前
CLI 默认值重算，绝不重读外部 target-spec 文件。以下为 fresh 专用调用选项：
--target-spec / --no-seam-verify / --max-qwen-calls / --max-deepseek-calls /
--evaluate / --render（事后评估归 evaluate-run，事后渲染可用 render-count）。
旧当前代 run（缺保真快照）：非 force 零 Qwen 复用仍允许（只需合法匹配
结果）；force 重跑以稳定
COUNT_IMAGE_INVOCATION_METADATA_INCOMPLETE 失败，绝不猜测缺失设置。
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
from application.runtime import Runtime, _validate_qwen_resume_identity
from application.settings import load_settings
from data.schema import GroundTruth, ImageRef, UnifiedSample, stable_sample_id
from evaluation.records import EVALUATION_FILENAME_BY_TASK
from reporting.visualization import render_counting_overlay
from workflows.call_budget import CallBudget
from workflows.dataset_runner import storage_key
from workflows.run_store import RunManifest, RunStore
from workflows.schema import QwenRuntimeAuditIdentity, RunRequest, SampleRunStatus

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130

_STATUS_FILENAME = "status.json"
_COUNTING_RESULT_FILENAME = "counting_result.json"
_OVERLAY_FILENAME = "overlay.png"


class CountImageCommandError(ValueError):
    """Stable public count-image failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


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
            json.dumps({
                "status": "failed",
                "error": getattr(error, "code", type(error).__name__),
            }),
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
    request: RunRequest
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
        target_spec, target_hash = _load_target_spec(args.target_spec)
        manifest = store.create_run(
            config_payload=settings.to_config_payload(),
            model_ids={
                "qwen": settings.models.qwen.effective_cache_model_id,
                "deepseek": settings.models.deepseek.model,
                **settings.models.qwen_manifest_model_ids(),
            },
            prompt_paths=catalog.snapshot_paths(),
            run_id=args.run_id,
            dataset="single-image",
            split="adhoc",
        )
        run_id = manifest.run_id
        run_dir = settings.runs.root / run_id
        # The actual fresh effective seam value is computed exactly once:
        # config may disable seam verification, and the CLI may only disable
        # it further — never enable it against config. The same value is
        # persisted and used for execution. 实际 fresh 生效 seam 值只计算
        # 一次：config 可禁用 seam 校验，CLI 只能进一步禁用——绝不能在
        # config 禁用时启用它。同一值用于持久化与执行。
        effective_seam_verify = (
            settings.counting.seam_verify and not args.no_seam_verify
        )
        # Unsupplied budgets are normalized to the explicit configured
        # defaults so the persisted snapshot is always a complete
        # re-executable invocation (distinguishable from legacy runs that
        # predate the fidelity fields). 未提供的预算归一化为显式配置默认值，
        # 使持久化快照始终是完整可重跑调用（可与早于保真字段的旧运行区分）。
        max_qwen = (
            args.max_qwen_calls
            if args.max_qwen_calls is not None
            else settings.router.default_qwen_calls
        )
        max_deepseek = (
            args.max_deepseek_calls
            if args.max_deepseek_calls is not None
            else settings.router.default_deepseek_calls
        )
        request = _count_image_request(
            sample_id=sample_id,
            image_sha256=image_sha256,
            question=args.question,
            image_dir=image_path.parent,
            evaluate=args.evaluate,
            target_spec=target_spec,
            target_hash=target_hash,
            seam_verify=effective_seam_verify,
            max_qwen_calls=max_qwen,
            max_deepseek_calls=max_deepseek,
            render=args.render,
        )
        store.write_run_request(run_dir, request)
    sample_dir = run_dir / "tasks" / "counting" / "samples" / storage_key(sample_id)

    if args.resume and not args.force:
        # Zero-Qwen reuse requires both a valid succeeded status AND a valid
        # matching CountingResult; missing/corrupt/mismatched results
        # re-execute. 零 Qwen 复用需要合法 succeeded 状态与合法匹配的
        # CountingResult 同时成立；缺失/损坏/不匹配结果重跑。
        persisted_status = _read_status(sample_dir)
        if persisted_status is not None and persisted_status.state == "succeeded":
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

    # Any re-execution (force or invalid-result rerun) must reconstruct the
    # full behavior-affecting invocation from the persisted snapshot; old
    # current-generation runs without it fail stably instead of guessing.
    # 任何重跑（force 或结果无效重跑）都必须从持久化快照重建完整行为影响
    # 调用；缺少快照的旧当前代运行稳定失败而非猜测。
    if args.resume and not _has_full_invocation(request):
        raise ValueError("COUNT_IMAGE_INVOCATION_METADATA_INCOMPLETE")

    # The persisted invocation is authoritative for every resumed execution;
    # fresh-only CLI options are ignored on resume (documented). 持久化调用
    # 对每次 resume 执行权威；fresh 专用 CLI 选项在 resume 时被忽略（已
    # 文档化）。
    effective_evaluate = request.evaluate
    effective_render = bool(request.count_render)
    effective_max_qwen = request.count_max_qwen_calls
    effective_max_deepseek = request.count_max_deepseek_calls
    effective_target_spec = request.count_target_spec
    if args.resume:
        # Narrowly scoped cleanup inside the validated sample directory for
        # artifacts the persisted intent does not produce. 对持久化意图不
        # 产出的产物，在已校验样本目录内窄范围清理。
        if not effective_evaluate:
            evaluation_path = (
                sample_dir / EVALUATION_FILENAME_BY_TASK["counting"]
            )
            if evaluation_path.is_file():
                evaluation_path.unlink()
        if not effective_render:
            overlay_path = sample_dir / _OVERLAY_FILENAME
            if overlay_path.is_file():
                overlay_path.unlink()

    metadata: dict[str, Any] = {}
    if effective_target_spec is not None:
        metadata["count_target_hint"] = effective_target_spec
    # The persisted seam value is authoritative in BOTH directions: for a
    # fresh run it equals the single effective value computed above; for a
    # resumed execution it unconditionally overwrites the current config so
    # config drift can never change the original seam mode.
    # 持久化 seam 值在两个方向都权威：fresh 时它等于上面计算的唯一有效值；
    # resume 执行时无条件覆盖当前 config，使 config 漂移绝不可能改变原始
    # seam 模式。
    effective_settings = settings.model_copy(
        update={
            "counting": settings.counting.model_copy(
                update={"seam_verify": bool(request.count_seam_verify)}
            )
        }
    )
    runtime = Runtime.create(
        settings=effective_settings,
        project_root=project_root,
        api_key=None,
    )
    if args.resume:
        _validate_qwen_resume_identity(
            request.qwen_runtime_identity,
            runtime.components.qwen_runtime_identity,
        )
    else:
        # Replace the declaration-only fresh snapshot with the verified engine
        # identity before the first planner/Agent call. 首次 planner/Agent 调用前，
        # 用已验证 engine 身份替换仅声明的 fresh 快照。
        request = request.model_copy(
            update={
                "qwen_runtime_identity": QwenRuntimeAuditIdentity.model_validate(
                    runtime.components.qwen_runtime_identity
                )
            }
        )
        store.write_run_request(run_dir, request)
    budget = _build_budget(runtime, effective_max_qwen, effective_max_deepseek)
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
    visual_task_plan = None
    visual_views = ()
    if effective_target_spec is None:
        planner = runtime.components.visual_task_planner
        if planner is None:
            raise CountImageCommandError("COUNT_IMAGE_PLANNER_NOT_ASSEMBLED")
        visual_task_plan, visual_views = await planner.plan_with_views(
            sample,
            data_root=image_path.parent,
            artifact_dir=sample_dir,
            budget=budget,
        )
        if visual_task_plan.task != "counting":
            raise CountImageCommandError("COUNT_IMAGE_PLANNER_TASK_MISMATCH")
    outcome = await runner.run_one(
        sample,
        sample_dir,
        visual_task_plan=visual_task_plan,
        visual_views=visual_views,
        judge_policy="none",
        budget=budget,
        evaluate=effective_evaluate,
    )
    execution = outcome.execution
    if execution is None or not isinstance(execution.payload, CountingResult):
        raise TypeError("count-image requires the native CountingAgent result")
    result: CountingResult = execution.payload
    if effective_render:
        with Image.open(image_path) as source:
            render_counting_overlay(
                source,
                result=result,
                output_path=sample_dir / _OVERLAY_FILENAME,
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


def _load_target_spec(
    path_value: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the user target spec and return its canonical JSON snapshot
    plus a stable content hash; None when no spec was supplied. A host path
    is never the persisted authority — only the validated structured content.
    校验用户目标 spec，返回其 canonical JSON 快照与稳定内容哈希；未提供时
    返回 None。主机路径绝非持久化权威——只有校验后的结构化内容才是。"""

    if not path_value:
        return None, None
    spec = CountTargetSpec.model_validate(
        json.loads(Path(path_value).read_text(encoding="utf-8"))
    )
    snapshot = spec.model_dump(mode="json")
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return snapshot, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _count_image_request(
    *,
    sample_id: str,
    image_sha256: str,
    question: str,
    image_dir: Path,
    evaluate: bool,
    target_spec: dict[str, Any] | None,
    target_hash: str | None,
    seam_verify: bool,
    max_qwen_calls: int | None,
    max_deepseek_calls: int | None,
    render: bool,
) -> RunRequest:
    """The persisted single-image invocation. The image is referenced by its
    stable content hash and the target by its structured snapshot — never by
    host absolute paths. 持久化单图调用。图像以稳定内容哈希引用，目标以
    结构化快照引用——绝不使用主机绝对路径。"""

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
        count_target_spec=target_spec,
        count_target_spec_hash=target_hash,
        count_seam_verify=seam_verify,
        count_max_qwen_calls=max_qwen_calls,
        count_max_deepseek_calls=max_deepseek_calls,
        count_render=render,
        planning_mode=(
            "direct" if target_spec is not None else "visual-task-plan-v5"
        ),
    )


def _has_full_invocation(request: RunRequest) -> bool:
    """A re-executable count-image invocation needs every behavior-affecting
    snapshot field; count_target_spec may legitimately be None (no spec).
    可重跑的 count-image 调用需要全部行为影响快照字段；count_target_spec
    允许合法为 None（无 spec）。"""

    return (
        request.count_seam_verify is not None
        and request.count_render is not None
        and request.count_max_qwen_calls is not None
        and request.count_max_deepseek_calls is not None
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


def _build_budget(
    runtime: Runtime,
    max_qwen_calls: int | None,
    max_deepseek_calls: int | None,
) -> CallBudget:
    """Request-local budget override; default limits apply when unset.
    请求局部预算覆盖；未设置时使用默认限制。"""

    default = runtime.components.call_budget_factory.create_for_sample("counting")
    return CallBudget(
        max_qwen_calls=max_qwen_calls or default.max_qwen_calls,
        max_deepseek_calls=(
            max_deepseek_calls
            if max_deepseek_calls is not None
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
