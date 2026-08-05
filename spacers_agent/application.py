"""Single-entry application facade for the local multi-Agent runtime.
单一入口应用门面：本地多 Agent 运行时。

``RuntimeApplication`` loads the local Qwen model exactly once, assembles the
existing Agent Runtime through ``assemble_runtime()``, and exposes three
manual-entry paths: one-shot ``ask()``, the serial HTTP service, and the
dataset command that reuses the existing ``DatasetRunner`` workflow.
``RuntimeApplication`` 只加载一次本地 Qwen 模型，通过 ``assemble_runtime()``
组装现有 Agent Runtime，并提供三条手动入口：单次 ``ask()``、串行 HTTP 服务，
以及复用现有 ``DatasetRunner`` 工作流的数据集命令。

Boundaries / 边界:
- No Judge, evaluation, fallback, report, or model reload on the manual path.
- 手动路径不执行 Judge、评测、fallback、报告或模型重载。
- The service creates no DeepSeek client, starts no vLLM, and never downloads.
- 服务不创建 DeepSeek Client、不启动 vLLM、从不下载模型。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, get_args

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from models.base import JsonResponseCache
from models.entry import create_model
from spacers_agent.agents.base import AgentContext, AgentExecution
from spacers_agent.bootstrap import RuntimeComponents, assemble_runtime
from spacers_agent.routing.schemas import RoutingDecision
from spacers_agent.schemas import (
    AgentResult,
    CountingResult,
    GroundTruth,
    ImageRef,
    TaskName,
    UnifiedSample,
)
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.artifact_writer import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Only the first level of a manual image directory is scanned. / 手动图片目录只扫描第一层。
MAX_MANUAL_IMAGES = 8
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
MAX_HTTP_BODY_BYTES = 1 << 20  # 1 MiB request body cap / 1 MiB 请求体上限

_KNOWN_TASKS: frozenset[str] = frozenset(get_args(TaskName))
_CHANGE_TASKS: frozenset[str] = frozenset({"change_caption", "change_qa"})

# Request-id prefixes by source. / 按来源区分请求 ID 前缀。
_REQUEST_PREFIX = {"main_cli": "manual", "http_service": "http"}


@dataclass(frozen=True)
class CollectedImage:
    """One image discovered in a manual image directory.
    手动图片目录中发现的一张图片。
    """

    path: Path
    width: int
    height: int


class PublicAnswer(BaseModel):
    """Uniform public result returned by ask and the HTTP service.
    ask 与 HTTP 服务返回的统一公开结果。
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


@dataclass
class RuntimeApplication:
    """One application instance sharing one model, one runtime, one registry.
    共享一个模型、一个 Runtime、一个注册表的应用实例。
    """

    settings: AppSettings
    project_root: Path
    qwen_client: Any
    runtime: RuntimeComponents
    service_root: Path

    @classmethod
    def create(
        cls,
        *,
        settings: AppSettings,
        project_root: Path,
    ) -> "RuntimeApplication":
        """Load the model once and assemble the runtime exactly once.
        只加载一次模型，且只组装一次 Runtime。
        """

        prompt_root = project_root / "prompts"
        service_root = settings.runs.root / "service"
        service_root.mkdir(parents=True, exist_ok=True)

        repair_prompt = (prompt_root / "json_repair_v1.md").read_text(encoding="utf-8")
        cache = JsonResponseCache(service_root / "cache")

        qwen_client = create_model(
            "qwen_transformers",
            settings=settings.models.qwen,
            repair_prompt=repair_prompt,
            cache=cache,
        )
        runtime = assemble_runtime(
            settings,
            qwen_client=qwen_client,
            judge_client=None,
            prompt_root=prompt_root,
        )
        return cls(
            settings=settings,
            project_root=project_root,
            qwen_client=qwen_client,
            runtime=runtime,
            service_root=service_root,
        )

    def health_payload(self) -> dict[str, Any]:
        """Report readiness without loading the model or calling any endpoint.
        报告就绪状态，不加载模型、不调用任何端点。
        """

        return {
            "status": "ready",
            "model": self.settings.models.qwen.model,
            "model_load_seconds": float(getattr(self.qwen_client, "load_seconds", 0.0) or 0.0),
            "agents": list(self.runtime.agent_registry.names()),
        }

    async def ask(
        self,
        *,
        image_dir: Path,
        question: str,
        task: str = "auto",
        source: str = "main_cli",
    ) -> PublicAnswer:
        """Run exactly one primary Agent against one local image directory.
        针对一个本地图片目录执行恰好一个主 Agent。

        Raises on every failure; no fallback Agent, Judge, or reload happens here.
        所有失败直接抛出；此处不执行 fallback Agent、Judge 或重载。
        """

        if task != "auto" and task not in _KNOWN_TASKS:
            raise ValueError(
                f"unknown task {task!r}; expected 'auto' or one of {sorted(_KNOWN_TASKS)}"
            )
        request_id = _new_request_id(source)
        request_dir = self.service_root / "requests" / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()

        collected = collect_images(image_dir)
        high_resolution = any(
            item.width * item.height > self.settings.counting.max_pixels_without_tiling
            for item in collected
        )

        if task == "auto":
            auto_task = resolve_task_rules(question, len(collected))
            if auto_task == "__router__":
                decision = await self._route_unknown(request_id, request_dir, question)
            else:
                decision = self.runtime.router.route_known(
                    auto_task, high_resolution=high_resolution
                )
        else:
            decision = self.runtime.router.route_known(task, high_resolution=high_resolution)

        resolved_task = decision.task
        primary_agent = decision.primary_agent
        validate_image_count(resolved_task, collected)

        images = build_image_refs(resolved_task, collected)
        sample = UnifiedSample(
            sample_id=request_id,
            dataset="manual",
            split="user",
            task=resolved_task,
            images=images,
            question=question,
            ground_truth=GroundTruth(),
            metadata={
                "source": source,
                "image_dir": str(image_dir.expanduser().resolve()),
            },
        )

        request_payload = {
            "request_id": request_id,
            "source": source,
            "image_dir": str(image_dir.expanduser().resolve()),
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
            settings=self.settings,
            qwen_client=self.qwen_client,
            call_budget=self.runtime.call_budget_factory.create_for_sample(resolved_task),
            prompt_catalog=self.runtime.prompt_catalog,
            judge_client=None,
        )
        agent = self.runtime.agent_registry.get(primary_agent)
        execution = await agent.run(sample, context)

        answer = to_public_answer(
            request_id=request_id,
            resolved_task=resolved_task,
            execution=execution,
            request_dir=request_dir,
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
        )
        atomic_write_json(request_dir / "result.json", answer.model_dump(mode="json"))
        return answer

    async def _route_unknown(
        self,
        request_id: str,
        request_dir: Path,
        question: str,
    ) -> RoutingDecision:
        """Call the Router Agent at most once; failures propagate unchanged.
        最多调用一次 Router Agent；失败原样传播。
        """

        budget = self.runtime.call_budget_factory.create_for_sample("general_vqa")
        return await self.runtime.router.route_unknown(
            question,
            budget=budget,
            sample_id=request_id,
            artifact_dir=request_dir / "router",
        )


# ── image collection / 图片收集 ────────────────────────────────────────────


def natural_key(path: Path) -> list[object]:
    """Natural sort key so image2.png precedes image10.png.
    自然排序键，使 image2.png 排在 image10.png 之前。
    """

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
    必须存在且为目录；只扫描第一层；忽略隐藏与非图片文件；损坏图片绝不静默跳过；
    超过 ``MAX_MANUAL_IMAGES`` 张图片时报错。
    """

    resolved = image_dir.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"image directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"image path is not a directory: {resolved}")
    candidates = [
        entry
        for entry in resolved.iterdir()
        if entry.is_file()
        and not entry.name.startswith(".")
        and entry.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    candidates.sort(key=natural_key)
    if not candidates:
        raise ValueError(f"no supported images found in: {resolved}")
    if len(candidates) > MAX_MANUAL_IMAGES:
        raise ValueError(
            f"too many images: {len(candidates)} (max {MAX_MANUAL_IMAGES}) in {resolved}"
        )
    collected: list[CollectedImage] = []
    for path in candidates:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception as error:
            raise ValueError(
                f"cannot open image {path}: {type(error).__name__}: {error}"
            ) from error
        collected.append(CollectedImage(path=path, width=width, height=height))
    return collected


# ── task resolution / 任务解析 ─────────────────────────────────────────────


def resolve_task_rules(question: str, image_count: int) -> str:
    """Apply the fixed auto-task rules; ``__router__`` means call the Router Agent.
    应用固定的自动任务规则；``__router__`` 表示调用 Router Agent。
    """

    if image_count == 2 and not question:
        return "change_caption"
    if image_count == 2 and question:
        return "change_qa"
    if image_count == 1 and not question:
        return "caption"
    if question:
        return "__router__"
    raise ValueError("empty question requires exactly one or two images")


def validate_image_count(task: str, collected: list[CollectedImage]) -> None:
    """Fail when the resolved task is incompatible with the image count.
    当解析出的任务与图片数量不兼容时直接失败。
    """

    count = len(collected)
    if task in _CHANGE_TASKS and count != 2:
        raise ValueError(f"{task} requires exactly two images, got {count}")
    if count < 1:
        raise ValueError("at least one image is required")


def build_image_refs(task: str, collected: list[CollectedImage]) -> list[ImageRef]:
    """Build canonical ImageRefs: t1/t2 for change tasks, image+context otherwise.
    构建统一 ImageRef：变化任务使用 t1/t2，其余使用 image+context。
    """

    if task in _CHANGE_TASKS:
        return [
            ImageRef(
                image_id="t1",
                path=collected[0].path,
                role="t1",
                width=collected[0].width,
                height=collected[0].height,
            ),
            ImageRef(
                image_id="t2",
                path=collected[1].path,
                role="t2",
                width=collected[1].width,
                height=collected[1].height,
            ),
        ]
    images = [
        ImageRef(
            image_id="image-0",
            path=collected[0].path,
            role="image",
            width=collected[0].width,
            height=collected[0].height,
        )
    ]
    for index, item in enumerate(collected[1:], start=1):
        images.append(
            ImageRef(
                image_id=f"context-{index}",
                path=item.path,
                role="context",
                width=item.width,
                height=item.height,
            )
        )
    return images


# ── public result mapping / 统一公开结果映射 ───────────────────────────────


def to_public_answer(
    *,
    request_id: str,
    resolved_task: str,
    execution: AgentExecution,
    request_dir: Path,
    elapsed_seconds: float,
) -> PublicAnswer:
    """Map one Agent payload to the uniform PublicAnswer without leaking internals.
    将一条 Agent 载荷映射为统一 PublicAnswer，不泄露内部载荷类型。
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
            artifact_dir=str(request_dir),
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
            artifact_dir=str(request_dir),
        )
    raise TypeError(f"unsupported agent payload type: {type(payload).__name__}")


# ── request identity / 请求标识 ────────────────────────────────────────────


def _new_request_id(source: str) -> str:
    """Build a unique request id for the current second.
    为当前秒构建唯一请求 ID。
    """

    prefix = _REQUEST_PREFIX.get(source)
    if prefix is None:
        raise ValueError(f"unknown request source: {source!r}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def _utc_now() -> str:
    """Return an RFC-3339 UTC timestamp ending with Z. / 返回以 Z 结尾的 UTC 时间戳。"""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


# ── serial HTTP service / 串行 HTTP 服务 ───────────────────────────────────


class RuntimeRequestHandler(BaseHTTPRequestHandler):
    """Serial handler exposing only ``GET /health`` and ``POST /ask``.
    仅暴露 ``GET /health`` 与 ``POST /ask`` 的串行处理器。
    """

    application: RuntimeApplication
    max_body_bytes: int = MAX_HTTP_BODY_BYTES

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._send_json(404, {"status": "failed", "error": "not found"})
            return
        self._send_json(200, self.application.health_payload())

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/ask":
            # Consume the request body so the connection can close cleanly on
            # Windows; otherwise the client may see an aborted connection.
            # 消费请求体以便连接在 Windows 上干净关闭，否则客户端可能遇到连接中止。
            self._read_body()
            self._send_json(404, {"status": "failed", "error": "not found"})
            return
        body = self._read_body()
        if body is None:
            self._send_json(413, {"status": "failed", "error": "request body too large"})
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"status": "failed", "error": "invalid JSON body"})
            return
        if not isinstance(payload, dict):
            self._send_json(
                400, {"status": "failed", "error": "request body must be a JSON object"}
            )
            return
        image_dir = payload.get("image_dir")
        if not isinstance(image_dir, str) or not image_dir:
            self._send_json(400, {"status": "failed", "error": "image_dir is required"})
            return
        try:
            result = asyncio.run(
                self.application.ask(
                    image_dir=Path(image_dir),
                    question=str(payload.get("question", "")),
                    task=str(payload.get("task", "auto")),
                    source="http_service",
                )
            )
        except ValueError as error:
            self._send_json(400, {"status": "failed", "error": str(error)})
            return
        except Exception as error:
            self._send_json(
                500,
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            )
            return
        self._send_json(200, result.model_dump(mode="json"))

    def _read_body(self) -> bytes | None:
        """Read the request body with a hard size cap; None means too large.
        以硬上限读取请求体；返回 None 表示超限。

        Oversized bodies are drained chunk-wise before the 413 response so the
        client can finish sending; this avoids aborted connections on Windows.
        超限请求体在返回 413 前被分块排空，使客户端可以完成发送；
        这避免了 Windows 上的连接中止。
        """

        length_header = self.headers.get("Content-Length")
        if length_header is not None:
            try:
                length = int(length_header)
            except ValueError:
                length = 0
            if length > self.max_body_bytes:
                _drain_body(self.rfile, length)
                return None
            return self.rfile.read(length)
        body = bytearray()
        while True:
            chunk = self.rfile.read(65536)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > self.max_body_bytes:
                _drain_body(self.rfile, None)
                return None
        return bytes(body)


def _drain_body(fileobj: Any, length: int | None) -> None:
    """Discard a request body in bounded chunks. / 以受限分块丢弃请求体。"""

    remaining = length
    while remaining is None or remaining > 0:
        size = min(65536, remaining) if remaining is not None else 65536
        chunk = fileobj.read(size)
        if not chunk:
            break
        if remaining is not None:
            remaining -= len(chunk)


def run_http_server(
    app: RuntimeApplication,
    *,
    host: str,
    port: int,
) -> int:
    """Start the serial HTTP service and block until interrupted.
    启动串行 HTTP 服务并阻塞直到被中断。
    """

    if not (1 <= port <= 65535):
        raise ValueError("port must be between 1 and 65535")
    handler = type(
        "BoundRuntimeRequestHandler",
        (RuntimeRequestHandler,),
        {"application": app},
    )
    server = HTTPServer((host, port), handler)
    payload = dict(app.health_payload())
    payload.update({"host": host, "port": port})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


# ── dataset command / 数据集命令 ───────────────────────────────────────────


async def run_dataset_command(settings: AppSettings, args: argparse.Namespace) -> int:
    """Run a dataset through the shared public DatasetRunner command.
    通过共享的公开 DatasetRunner 命令运行数据集。

    Delegates to ``spacers_agent.commands.run_dataset.run_dataset`` so the manual
    entry, the internal CLI, and resume-run share one dataset loop; SampleRunner
    fallback, Judge, Resume, Artifact, and Report behavior stay in one place.
    委托给 ``spacers_agent.commands.run_dataset.run_dataset``，使手动入口、内部
    CLI 与 resume-run 共享同一数据集循环；SampleRunner fallback、Judge、Resume、
    Artifact 与 Report 行为保持单一实现。
    """

    if args.max_samples < 0 or args.start_index < 0 or args.sample_concurrency < 1:
        raise ValueError("max-samples, start-index, and sample-concurrency must be valid")
    # Lazy import keeps the manual service/ask path free of dataset imports.
    # 惰性导入使手动服务/ask 路径不引入数据集相关依赖。
    from spacers_agent.commands.run_dataset import RunDatasetOptions, run_dataset

    options = RunDatasetOptions(
        dataset=args.dataset,
        root=args.root,
        split=args.split,
        task=args.task,
        run_id=args.run_id,
        max_samples=args.max_samples,
        start_index=args.start_index,
        sample_concurrency=args.sample_concurrency,
        resume=args.resume,
        fail_fast=args.fail_fast,
        evaluate=args.evaluate,
        judge_policy=args.judge_policy,
    )
    return await run_dataset(settings, options)
