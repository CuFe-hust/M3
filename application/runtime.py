"""High-level runtime use cases: dataset runs, reports, and the manual `ask`
path. The runtime never writes the dataset loop — it only delegates to
DatasetRunner per task. The manual `ask` path executes exactly one primary
Agent per request; no Judge, evaluation, fallback, report, or model reload
happens there.

高层运行时用例：数据集运行、报告与手动 ask 路径。Runtime 不写数据集循环——
只按 task 委托 DatasetRunner。手动 ask 路径每次请求执行恰好一个主 Agent；
不执行 Judge、评测、fallback、报告或模型重载。
"""

from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict, Field
from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.counting.schema import CountingResult
from agents.schema import AgentResult
from application.bootstrap import RuntimeComponents, assemble_runtime
from application.settings import AppSettings, load_settings
from data.registry import DatasetRegistry, build_default_registry
from data.schema import CHANGE_TASKS, GroundTruth, ImageRef, TaskName, UnifiedSample
from reporting.schema import Report
from routing.schema import SampleCapabilities, TaskResolutionRequest
from workflows.artifact_writer import atomic_write_json
from workflows.run_store import RunManifest
from workflows.schema import DatasetRunOptions, DatasetRunSummary, RunRequest

# Only the first level of a manual image directory is scanned. / 手动图片目录只扫描第一层。
MAX_MANUAL_IMAGES = 8
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")

# Legal task names; the manual path accepts 'auto' plus every TaskName.
# 合法任务名；手动路径接受 'auto' 与全部 TaskName。
_ALL_TASK_NAMES = frozenset(get_args(TaskName))

# Request-id prefixes by source; the HTTP service reuses the same helper.
# 按来源区分请求 ID 前缀；HTTP 服务复用同一 helper。
_REQUEST_PREFIX = {"main_cli": "manual", "http_service": "http"}


def build_dataset_run_options(
    *,
    dataset: str,
    root: Path,
    split: str,
    tasks: tuple[str, ...] | None = None,
    auto_task: bool = False,
    run_id: str | None = None,
    resume: bool = False,
    limit: int | None = None,
    start_index: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    sample_concurrency: int = 1,
    sample_ids: set[str] | None = None,
    evaluate: bool = True,
    judge_policy: str = "none",
    judge_sample_rate: float | None = None,
    render_errors: bool = False,
    fail_fast: bool = False,
) -> DatasetRunOptions:
    """Thin options construction for the public entry point: the architecture
    rule forbids main.py from importing workflows, so construction lives here.
    Validation (task/auto-task exclusivity, judge rate bounds) comes from
    DatasetRunOptions itself. tasks=None selects adapter.supported_tasks.
    公开入口的薄选项构造：架构规则禁止 main.py 导入 workflows，因此构造在此
    完成。互斥校验（task/auto-task）与 judge 率边界由 DatasetRunOptions 自身
    承担。tasks=None 选择 adapter.supported_tasks。"""

    return DatasetRunOptions(
        dataset=dataset,
        root=root,
        split=split,
        tasks=tasks,
        auto_task=auto_task,
        run_id=run_id,
        resume=resume,
        limit=limit,
        start_index=start_index,
        shard_index=shard_index,
        shard_count=shard_count,
        sample_concurrency=sample_concurrency,
        sample_ids=sample_ids,
        evaluate=evaluate,
        judge_policy=judge_policy if evaluate else "none",
        judge_sample_rate=judge_sample_rate if evaluate else None,
        render_errors=render_errors,
        fail_fast=fail_fast,
    )


class PublicAnswer(BaseModel):
    """Uniform public result returned by ask and (later) the HTTP service.
    ask 与（后续）HTTP 服务返回的统一公开结果。

    artifact_dir is run-root-relative (e.g. ``service/requests/<id>``) and
    never a host absolute path. artifact_dir 为 run-root 相对路径（如
    ``service/requests/<id>``），绝非主机绝对路径。
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    task: str
    agent: str
    status: str
    answer: str
    target: str | None = None
    count: int | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_seconds: float
    artifact_dir: str


@dataclass(frozen=True)
class CollectedImage:
    """One image discovered in a manual image directory.
    手动图片目录中发现的一张图片。"""

    path: Path
    width: int
    height: int


def natural_key(path: Path) -> list[object]:
    """Natural sort key so image2.png precedes image10.png.
    自然排序键，使 image2.png 排在 image10.png 之前。"""

    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def collect_images(image_dir: Path) -> list[CollectedImage]:
    """Collect first-level images in natural order with strict errors.
    以自然顺序收集第一层图片，出错时严格失败。

    Rules / 规则: must exist and be a directory; only the first level; hidden
    and non-image files ignored; corrupt images are never skipped silently;
    more than ``MAX_MANUAL_IMAGES`` images is an error.
    必须存在且为目录；只扫描第一层；忽略隐藏与非图片文件；损坏图片绝不静默
    跳过；超过 ``MAX_MANUAL_IMAGES`` 张图片时报错。
    """

    resolved = image_dir.expanduser().resolve()
    if not resolved.exists():
        raise ValueError("image directory does not exist")
    if not resolved.is_dir():
        raise ValueError("image path is not a directory")
    candidates = [
        entry
        for entry in resolved.iterdir()
        if entry.is_file()
        and not entry.name.startswith(".")
        and entry.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    candidates.sort(key=natural_key)
    if not candidates:
        raise ValueError("no supported images found in the image directory")
    if len(candidates) > MAX_MANUAL_IMAGES:
        raise ValueError(f"too many images (max {MAX_MANUAL_IMAGES})")
    collected: list[CollectedImage] = []
    for path in candidates:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception as error:
            raise ValueError(
                f"cannot open image {path.name}: {type(error).__name__}"
            ) from error
        collected.append(CollectedImage(path=path, width=width, height=height))
    return collected


def validate_image_count(task: str, collected: list[CollectedImage]) -> None:
    """Fail when the resolved task is incompatible with the image count.
    当解析出的任务与图片数量不兼容时直接失败。"""

    count = len(collected)
    if task in CHANGE_TASKS and count != 2:
        raise ValueError(f"{task} requires exactly two images, got {count}")
    if count < 1:
        raise ValueError("at least one image is required")


def build_image_refs(
    task: str,
    collected: list[CollectedImage],
    image_root: Path,
) -> list[ImageRef]:
    """Build canonical ImageRefs relative to the manual image directory:
    t1/t2 for change tasks, image+context otherwise.
    构建相对手动图片目录的统一 ImageRef：变化任务使用 t1/t2，其余使用
    image+context。"""

    if task in CHANGE_TASKS:
        return [
            ImageRef(
                image_id="t1",
                path=_relative_path(collected[0].path, image_root),
                role="t1",
                width=collected[0].width,
                height=collected[0].height,
            ),
            ImageRef(
                image_id="t2",
                path=_relative_path(collected[1].path, image_root),
                role="t2",
                width=collected[1].width,
                height=collected[1].height,
            ),
        ]
    images = [
        ImageRef(
            image_id="image-0",
            path=_relative_path(collected[0].path, image_root),
            role="image",
            width=collected[0].width,
            height=collected[0].height,
        )
    ]
    for index, item in enumerate(collected[1:], start=1):
        images.append(
            ImageRef(
                image_id=f"context-{index}",
                path=_relative_path(item.path, image_root),
                role="context",
                width=item.width,
                height=item.height,
            )
        )
    return images


def _relative_path(path: Path, image_root: Path) -> Path:
    """Return one collected image as a path relative to the manual image
    directory; collected paths always live under it.
    返回一张已收集图片相对手动图片目录的路径；收集的路径恒在其下。"""

    return path.relative_to(image_root)


def to_public_answer(
    *,
    request_id: str,
    resolved_task: str,
    execution: AgentExecution,
    artifact_dir: str,
    elapsed_seconds: float,
) -> PublicAnswer:
    """Map one Agent payload to the uniform PublicAnswer without leaking
    internals. 将一条 Agent 载荷映射为统一 PublicAnswer，不泄露内部载荷类型。
    """

    payload = execution.payload
    if isinstance(payload, CountingResult):
        accepted = [point for point in payload.global_points if point.accepted]
        return PublicAnswer(
            request_id=request_id,
            task=resolved_task,
            agent=execution.agent_name,
            status=payload.status,
            answer=str(payload.final_count),
            target=payload.target,
            count=payload.final_count,
            evidence=[
                {
                    "point": [point.global_x_norm, point.global_y_norm],
                    "confidence": point.confidence,
                    "image_id": "image-0",
                    "source_tile_id": point.source_tile_id,
                }
                for point in accepted
            ],
            warnings=[item.model_dump(mode="json") for item in payload.warnings],
            elapsed_seconds=elapsed_seconds,
            artifact_dir=artifact_dir,
        )
    if isinstance(payload, AgentResult):
        return PublicAnswer(
            request_id=request_id,
            task=resolved_task,
            agent=execution.agent_name,
            status=payload.status,
            answer=payload.answer,
            evidence=[item.model_dump(mode="json") for item in payload.evidence_items],
            warnings=[],
            elapsed_seconds=elapsed_seconds,
            artifact_dir=artifact_dir,
        )
    raise TypeError(f"unsupported agent payload type: {type(payload).__name__}")


def _new_request_id(source: str) -> str:
    """Build a unique request id for the current second.
    为当前秒构建唯一请求 ID。"""

    prefix = _REQUEST_PREFIX.get(source)
    if prefix is None:
        raise ValueError("unknown request source")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def _utc_now() -> str:
    """Return an RFC-3339 UTC timestamp ending with Z. / 返回以 Z 结尾的 UTC 时间戳。"""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _build_run_request(options: DatasetRunOptions) -> RunRequest:
    """Map the concrete dataset run options into the persisted invocation
    artifact. judge_policy/rate are stored as the original intent; the
    evaluate gate is re-applied on resume by the runtime itself.
    将具体数据集运行选项映射为持久化调用产物。judge_policy/rate 按原始意图
    存储；evaluate 门控在 resume 时由 runtime 重新应用。"""

    if options.auto_task:
        task_mode: str = "auto"
        tasks: list[str] = []
    elif options.tasks is None:
        task_mode = "adapter_default"
        tasks = []
    else:
        task_mode = "explicit"
        tasks = list(options.tasks)
    return RunRequest(
        dataset=options.dataset,
        dataset_root=_posix(options.root),
        split=options.split,
        task_mode=task_mode,  # type: ignore[arg-type]
        tasks=tasks,
        auto_task=options.auto_task,
        sample_ids=sorted(options.sample_ids) if options.sample_ids else None,
        limit=options.limit,
        start_index=options.start_index,
        shard_index=options.shard_index,
        shard_count=options.shard_count,
        sample_concurrency=options.sample_concurrency,
        evaluate=options.evaluate,
        judge_policy=options.judge_policy,
        judge_sample_rate=options.judge_sample_rate,
        render_errors=options.render_errors,
        fail_fast=options.fail_fast,
    )


def _posix(path: Path) -> str:
    """POSIX serialization with forward-slash separators on every platform.
    所有平台统一正斜杠的 POSIX 序列化。"""

    return path.as_posix().replace("\\", "/")


@dataclass(frozen=True)
class Runtime:
    """One composition-root runtime with high-level use cases.
    带高层用例的单一组合根运行时。"""

    settings: AppSettings
    components: RuntimeComponents
    registry: DatasetRegistry = field(default_factory=build_default_registry)

    @classmethod
    def create(
        cls,
        *,
        settings: AppSettings | None = None,
        project_root: Path | None = None,
        api_key: str | None = None,
        qwen_client=None,
        config_path: Path | None = None,
        prompts_root: Path | None = None,
    ) -> "Runtime":
        """Create the runtime from settings and an optional injected Qwen
        client (tests) and DeepSeek api_key. 从配置与可选注入的 Qwen 客户端
        （测试）与 DeepSeek api_key 创建运行时。"""

        resolved_settings = settings or load_settings(config_path)
        components = assemble_runtime(
            resolved_settings,
            project_root=project_root or Path.cwd(),
            qwen_client=qwen_client,
            api_key=api_key,
            prompts_root=prompts_root,
        )
        return cls(settings=resolved_settings, components=components)

    async def run_dataset(
        self,
        options: DatasetRunOptions,
    ) -> dict[str, DatasetRunSummary]:
        """Run one dataset under the frozen run-identity contract: a fresh
        run without an explicit run_id always creates a unique run; a fresh
        run with an explicit run_id fails stably when the run already exists;
        resume requires an explicit run_id and a valid matching manifest.
        Then each task (or the auto-task namespace) is delegated to a
        DatasetRunner. Judge policy only applies when evaluate is enabled.
        按冻结 run identity 契约运行一个数据集：fresh 无显式 run_id 恒创建
        唯一 run；fresh 带显式 run_id 且已存在时稳定失败；resume 要求显式
        run_id 与合法匹配的 manifest。随后每个 task（或 auto-task 命名空间）
        委托给 DatasetRunner。仅 evaluate 启用时应用 judge 策略。"""

        if options.resume:
            if options.run_id is None:
                raise ValueError("resume requires an explicit run_id")
            run_id = options.run_id
            run_dir = self.settings.runs.root / run_id
            self._validate_existing_run(run_dir, options, run_id)
        else:
            manifest = self.components.run_store.create_run(
                config_payload=self.settings.to_config_payload(),
                model_ids={
                    "qwen": self.settings.models.qwen.effective_cache_model_id,
                    "deepseek": self.settings.models.deepseek.model,
                },
                prompt_paths=self.components.prompt_catalog.snapshot_paths(),
                run_id=options.run_id,
                dataset=options.dataset,
                split=options.split,
                sample_filter=(
                    ",".join(sorted(options.sample_ids))
                    if options.sample_ids
                    else None
                ),
            )
            run_id = manifest.run_id
            run_dir = self.settings.runs.root / run_id
            # Persist the concrete invocation after identity is established
            # and before any sample/model execution; failure fails the fresh
            # run before inference. 在身份确立后、任何样本/模型执行前持久化
            # 具体调用；失败使 fresh run 在推理前失败。
            self.components.run_store.write_run_request(
                run_dir, _build_run_request(options)
            )
        adapter = self.registry.get(options.dataset)
        judge_policy = options.judge_policy if options.evaluate else "none"
        if options.auto_task:
            task_names: list[str | None] = [None]
        elif options.tasks is None:
            # Adapter-default mode: run every supported task with no
            # TaskResolver involvement. / adapter 默认模式：运行全部受支持
            # 任务，不涉及 TaskResolver。
            task_names = sorted(adapter.supported_tasks)
        else:
            task_names = list(options.tasks)
        results: dict[str, DatasetRunSummary] = {}
        for task in task_names:
            runner = self.components.dataset_runner_factory(
                adapter,
                run_dir,
                judge_policy=judge_policy,
                judge_sample_rate=options.judge_sample_rate,
                data_root=options.root,
            )
            results[task or "auto"] = await runner.run(
                root=options.root,
                split=options.split,
                task=task,
                resume=options.resume,
                limit=options.limit,
                shard_index=options.shard_index,
                shard_count=options.shard_count,
                start_index=options.start_index,
                sample_ids=options.sample_ids,
                fail_fast=options.fail_fast,
                sample_concurrency=options.sample_concurrency,
            )
        if options.render_errors:
            self.render_error_overlays(run_dir, data_root=options.root)
        self._persist_report(run_dir)
        return results

    def _persist_report(self, run_dir: Path) -> None:
        """Build and persist the unified current-generation report bundle
        after a terminal dataset run; the report builder stays read-only with
        respect to execution artifacts. Persistence failures propagate as
        stable command failures instead of silently claiming success.
        在数据集运行终态后构建并持久化统一当前代报告 bundle；报告构建器对
        执行产物保持只读。持久化失败作为稳定命令失败传播，绝不静默宣称
        成功。"""

        from reporting.builder import build_report
        from reporting.exporters import persist_report_bundle

        report = build_report(run_dir)
        persist_report_bundle(run_dir, report)

    def render_error_overlays(
        self,
        run_dir: Path,
        *,
        data_root: Path,
    ) -> list[dict[str, Any]]:
        """Render counting overlays for failed samples after execution; never
        calls a model. Unsupported samples or missing artifacts become stable
        notes, never a whole-run failure; notes persist to
        ``render_errors_notes.json`` in the run root.
        执行后为 failed 样本渲染计数标注图；绝不调用模型。不支持的样本或
        缺失产物转为稳定 note，绝不导致整轮失败；note 持久化到 run 根的
        ``render_errors_notes.json``。"""

        from agents.counting.schema import CountingResult
        from reporting.adapters import iter_current_predictions, load_status
        from reporting.visualization import render_counting_overlay
        from workflows.dataset_runner import storage_key
        from workflows.sample_runner import _COUNTING_TASKS

        notes: list[dict[str, Any]] = []
        for row in iter_current_predictions(run_dir):
            if row.get("status") != "failed":
                continue
            sample_id = row.get("sample_id")
            run_task = row.get("run_task")
            if not isinstance(sample_id, str) or not isinstance(run_task, str):
                continue
            sample_dir = (
                run_dir / "tasks" / run_task / "samples" / storage_key(sample_id)
            )
            # The storage namespace never decides execution semantics: in
            # auto-task mode run_task is "auto" while the executed task comes
            # from status.task / prediction.task. 存储命名空间绝不决定执行
            # 语义：auto-task 模式下 run_task 是 "auto"，执行任务来自
            # status.task / prediction.task。
            status = load_status(sample_dir)
            execution_task = status.task if status is not None else row.get("task")
            if execution_task not in _COUNTING_TASKS:
                notes.append(
                    {
                        "sample_id": sample_id,
                        "task": execution_task or run_task,
                        "note": "render_errors_skipped:unsupported_task",
                    }
                )
                continue
            result_path = sample_dir / "counting_result.json"
            if not result_path.is_file():
                notes.append(
                    {
                        "sample_id": sample_id,
                        "task": execution_task,
                        "note": "render_errors_skipped:no_counting_result",
                    }
                )
                continue
            try:
                result = CountingResult.model_validate(
                    json.loads(result_path.read_text(encoding="utf-8"))
                )
                image_path = self._sample_source_image(sample_dir, data_root)
                if image_path is None:
                    notes.append(
                        {
                            "sample_id": sample_id,
                            "task": execution_task,
                            "note": "render_errors_skipped:no_source_image",
                        }
                    )
                    continue
                with Image.open(image_path) as source:
                    render_counting_overlay(
                        source,
                        result=result,
                        output_path=sample_dir / "error_overlay.png",
                    )
            except Exception as error:
                notes.append(
                    {
                        "sample_id": sample_id,
                        "task": execution_task,
                        "note": f"render_errors_failed:{type(error).__name__}",
                    }
                )
        atomic_write_json(run_dir / "render_errors_notes.json", notes)
        return notes

    def _sample_source_image(
        self,
        sample_dir: Path,
        data_root: Path,
    ) -> Path | None:
        """Resolve the first sample image against the dataset root with a
        canonical containment check. The persisted sample is validated as a
        UnifiedSample; the candidate path is resolved and must stay inside
        the dataset root and exist. Anything else yields None (stable skip),
        and no absolute machine path is ever reported.
        以 canonical containment 校验按数据集根解析首图。持久化样本经
        UnifiedSample 校验；候选路径 resolve 后必须位于数据集根内且存在。
        否则返回 None（稳定跳过），绝不报告任何主机绝对路径。"""

        sample_path = sample_dir / "sample.json"
        try:
            raw = json.loads(sample_path.read_text(encoding="utf-8"))
            sample = UnifiedSample.model_validate(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        if not sample.images:
            return None
        root = data_root.resolve()
        candidate = (root / sample.images[0].path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
        return candidate

    async def ask(
        self,
        *,
        image_dir: Path,
        question: str,
        task: str = "auto",
        source: str = "main_cli",
    ) -> PublicAnswer:
        """Run exactly one primary Agent against one local image directory.
        The manual path has no Judge, evaluation, fallback, or reload.
        针对一个本地图片目录执行恰好一个主 Agent。手动路径无 Judge、评测、
        fallback 或重载。

        Explicit tasks skip the TaskResolver; auto resolves through the
        current TaskResolver (deterministic rules for empty questions, one
        model call otherwise). Low-confidence manual requests still execute
        only the resolved primary task — no multi-attempt fallback.
        显式任务跳过 TaskResolver；auto 经当前 TaskResolver 解析（空问题走
        确定性规则，否则一次模型调用）。低置信度手动请求仍只执行解析出的
        主任务——无多尝试兜底。"""

        if task != "auto" and task not in _ALL_TASK_NAMES:
            raise ValueError(
                "unknown task; expected 'auto' or one of the known task names"
            )
        request_id = _new_request_id(source)
        request_dir = self.settings.runs.root / "service" / "requests" / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()

        collected = collect_images(image_dir)
        image_root = image_dir.expanduser().resolve()
        high_resolution = any(
            item.width * item.height > self.settings.counting.max_pixels_without_tiling
            for item in collected
        )

        if task == "auto":
            resolution = await self.components.task_resolver.resolve(
                TaskResolutionRequest(
                    question=question,
                    image_count=len(collected),
                ),
                sample_id=request_id,
                artifact_dir=request_dir,
                budget=self.components.call_budget_factory.create_for_sample(
                    "general_vqa"
                ),
            )
            resolved_task = resolution.task
        else:
            resolved_task = task
        decision = self.components.router.route(
            resolved_task,
            capabilities=SampleCapabilities(high_resolution=high_resolution),
        )
        resolved_task = decision.task
        primary_agent = decision.primary_agent
        validate_image_count(resolved_task, collected)

        images = build_image_refs(resolved_task, collected, image_root)
        sample = UnifiedSample(
            sample_id=request_id,
            dataset="manual",
            split="user",
            task=resolved_task,  # type: ignore[arg-type]
            images=images,
            question=question,
            ground_truth=GroundTruth(),
            metadata={
                "source": source,
                "image_dir": "manual://input",
            },
        )

        request_payload = {
            "request_id": request_id,
            "source": source,
            "image_dir": "manual://input",
            "images": [
                {
                    "path": str(ref.path),
                    "role": ref.role,
                    "width": ref.width,
                    "height": ref.height,
                }
                for ref in images
            ],
            "question": question,
            "requested_task": task,
            "resolved_task": resolved_task,
            "created_at": _utc_now(),
        }
        atomic_write_json(request_dir / "request.json", request_payload)

        context = AgentContext(
            artifact_dir=request_dir / "agent",
            qwen_client=self.components.qwen_client,
            call_budget=self.components.call_budget_factory.create_for_sample(
                resolved_task
            ),
            data_root=image_root,
            judge_client=None,
        )
        agent = self.components.agent_registry.get(primary_agent)
        execution = await agent.run(sample, context)

        answer = to_public_answer(
            request_id=request_id,
            resolved_task=resolved_task,
            execution=execution,
            artifact_dir=request_dir.relative_to(self.settings.runs.root).as_posix(),
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
        )
        atomic_write_json(request_dir / "result.json", answer.model_dump(mode="json"))
        return answer

    def _validate_existing_run(
        self,
        run_dir: Path,
        options: DatasetRunOptions,
        run_id: str,
    ) -> RunManifest:
        """Resume validation: the run must exist with a parseable manifest
        whose identity matches the requested run and dataset/split; any
        mismatch is a stable failure so a run from dataset A can never be
        resumed as dataset B. resume 校验：run 必须存在且 manifest 可解析，
        其身份必须与请求的 run 及 dataset/split 一致；任何不一致都是稳定
        失败，绝不允许把 dataset A 的 run 当作 dataset B resume。"""

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
        if manifest.dataset is not None and manifest.dataset != options.dataset:
            raise ValueError("resume dataset mismatch")
        if manifest.split is not None and manifest.split != options.split:
            raise ValueError("resume split mismatch")
        return manifest

    def health_payload(self) -> dict[str, Any]:
        """Report readiness without loading the model or calling any endpoint.
        报告就绪状态，不加载模型、不调用任何端点。"""

        return {
            "status": "ready",
            "model": self.settings.models.qwen.model,
            "model_load_seconds": float(
                getattr(self.components.qwen_client, "load_seconds", 0.0) or 0.0
            ),
            "agents": list(self.components.agent_registry.names()),
        }

    def build_report(self, run_id: str) -> Report:
        """Build the read-only report for a run id. 为 run id 构建只读报告。"""

        return self.components.build_report(self.settings.runs.root / run_id)
