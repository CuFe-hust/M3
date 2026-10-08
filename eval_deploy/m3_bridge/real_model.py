"""M3-backed RealModel for the official participant API shell (Protocol 2.0).

官方协议壳（participant_api_starter/server.py）的 RealModel 实现：
- 启动时恰好一次构建 M3 Runtime（模型只加载一次）；
- 每题把请求映射为 UnifiedSample，并走
  ``SampleRunner.run_one(..., visual_task_plan=None)`` 直连路径
  （源码契约："Direct tools may omit them for an explicitly selected
  sample task; such calls never plan."）；
- 输出经 protocol_map 规范化成协议答案；
- 失败返回稳定非 2xx 错误码，绝不伪造答案。

Design notes / 设计要点:
- A persistent single-thread event loop owns every M3 coroutine because M3's
  generation lock lives on the loop; ThreadingHTTPServer threads submit into it.
  常驻单线程事件循环：M3 生成锁绑定在 loop 上，HTTP 线程通过
  run_coroutine_threadsafe 提交。
- The runner is constructed once per package directory via the composition
  root's ``sample_runner_factory`` (lightweight wiring; the model itself is
  never reloaded).
- evaluate=False / judge_policy="none": we have no ground truth on this path
  and DeepSeek must never be touched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from .errors import AdapterError
from .protocol_map import (
    CHANGE_TASKS,
    choice_pairs,
    compose_question,
    m3_choices,
    normalize_answer,
    task_for,
)

log = logging.getLogger("m3_bridge")

_SAMPLE_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class M3RealModel:
    """Loads the M3 runtime once and answers Protocol 2.0 requests."""

    def __init__(self) -> None:
        self.m3_root = Path(os.environ.get("M3_ROOT", "/home/tongyi/M3")).resolve()
        self.config_path = Path(
            os.environ.get("M3_CONFIG", "/home/tongyi/eval/m3_bridge/m3_eval.yaml")
        )
        self.runs_root = Path(
            os.environ.get("OUTPUT_ROOT", "/home/tongyi/eval/runs")
        ).resolve()
        self.run_tag = os.environ.get("RUN_TAG") or time.strftime(
            "%Y%m%dT%H%M%SZ", time.gmtime()
        )
        self.hard_timeout = float(os.environ.get("BRIDGE_TIMEOUT_SECONDS", "110"))
        self.task_mode = os.environ.get("BRIDGE_TASK_MODE", "constraint")
        self.load_timeout = float(
            os.environ.get("BRIDGE_LOAD_TIMEOUT_SECONDS", "900")
        )
        self._runtime: Any = None
        self._runners: dict[str, Any] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="m3-loop", daemon=True
        )
        self._thread.start()
        setup = asyncio.run_coroutine_threadsafe(self._async_setup(), self._loop)
        try:
            setup.result(timeout=self.load_timeout)
        except Exception as exc:  # bare stable failure during startup only
            raise RuntimeError("M3 runtime assembly failed") from exc
        log.info("M3 runtime ready; run_tag=%s", self.run_tag)

    # -- setup ---------------------------------------------------------------

    async def _async_setup(self) -> None:
        # Keep M3-side run artifacts (service cache etc.) under the eval root,
        # not inside the M3 repository.
        # 让 M3 侧运行产物（服务缓存等）落在评测目录而非 M3 仓库内。
        os.environ.setdefault("OUTPUT_ROOT", str(self.runs_root))
        if str(self.m3_root) not in sys.path:
            sys.path.insert(0, str(self.m3_root))
        from application.runtime import Runtime
        from application.settings import load_settings

        settings = load_settings(self.config_path)
        self._runtime = Runtime.create(settings=settings, project_root=self.m3_root)

    def _runner_for(self, data_root: Path) -> Any:
        key = str(data_root)
        runner = self._runners.get(key)
        if runner is None:
            runner = self._runtime.components.sample_runner_factory(data_root)
            self._runners[key] = runner
        return runner

    # -- predict -------------------------------------------------------------

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> Any:
        """Protocol entry: one question in, one answer out (or AdapterError)."""
        future = asyncio.run_coroutine_threadsafe(
            self._infer(request, image_paths), self._loop
        )
        try:
            return future.result(timeout=self.hard_timeout)
        except FutureTimeoutError:
            # The underlying generation cannot be cancelled (M3 lock); the
            # request is answered with a stable 504 and the site moves on.
            # 底层生成无法取消（M3 锁）；以稳定 504 结束本请求。
            log.warning("item=%s timeout after %.0fs", request.get("item_id"), self.hard_timeout)
            raise AdapterError(504, "inference_timeout")
        except AdapterError:
            raise
        except Exception:
            raise AdapterError(500, "inference_failed")

    async def _infer(self, request: dict[str, Any], image_paths: list[Path]) -> Any:
        started = time.perf_counter()
        kind = request["response_constraint"]["type"]
        n_images = len(request.get("images", []))
        # Observed 2026-10-08: for change-region enum requests the planner can
        # pick change_caption (description) and the ChangeAgent then answers a
        # full sentence, which is not one of the nine canonical labels. The
        # response constraint fully declares the task here, so skip planning.
        # For bbox, the planner's grounding evidence chain fails schema
        # validation on every observed item (0 planner successes; each attempt
        # burns ~9s before the identical constraint fallback).
        # 实测：区域枚举题 planner 可能误判为 change_caption，导致整句被判畸形；
        # bbox 题 planner 证据链从未通过 schema（每答必回退，白耗 ~9s）。
        # 约束已完全确定任务，直接跳过 planner（类比计数策略）。
        if (kind == "enum" and n_images == 2) or kind == "bbox":
            log.info(
                "item=%s %s constrained task; planner skipped by policy",
                request.get("item_id"),
                kind,
            )
        elif getattr(self, "task_mode", "constraint") == "planner":
            planned = await self._try_planner(request, image_paths, started)
            if planned is not None:
                return planned
            log.warning(
                "item=%s planner path unavailable; falling back to constraint mode",
                request.get("item_id"),
            )
        return await self._infer_constraint(request, image_paths, started)

    async def _infer_constraint(
        self, request: dict[str, Any], image_paths: list[Path], started: float
    ) -> Any:
        kind = request["response_constraint"]["type"]
        task = task_for(request)
        variants = self._question_variants(request, task)
        last_error: AdapterError = AdapterError(502, "unanswerable")
        for index, question in enumerate(variants):
            sample, sample_dir = self._build_sample(
                request,
                image_paths,
                task,
                question=question,
                dir_suffix="" if index == 0 else f"_retry{index}",
            )
            payload, agent = await self._run_once(sample, sample_dir, kind, task)
            if payload is None:
                mined = self._mine_bbox(request, sample_dir["dir"])
                if mined is not None:
                    log.info(
                        "item=%s type=%s task=%s attempt=%d agent=mined elapsed=%.1fs answer=%s",
                        sample.sample_id, kind, task, index + 1,
                        time.perf_counter() - started, _preview(mined),
                    )
                    return mined
                continue
            try:
                answer = normalize_answer(request, payload)
            except AdapterError as exc:
                last_error = exc
                if exc.code == "no_localization":
                    # Malformed-JSON boxes can be recovered by mining digits.
                    # 破损 JSON 里的框可通过数字挖掘恢复。
                    mined = self._mine_bbox(request, sample_dir["dir"])
                    if mined is not None:
                        log.info(
                            "item=%s type=%s task=%s attempt=%d agent=mined elapsed=%.1fs answer=%s",
                            sample.sample_id, kind, task, index + 1,
                            time.perf_counter() - started, _preview(mined),
                        )
                        return mined
                    # A missing localization gets one escalated retry.
                    # 定位缺失时做一次升级重试（更强硬的提示词）。
                    if index + 1 < len(variants):
                        continue
                raise
            log.info(
                "item=%s type=%s task=%s attempt=%d agent=%s elapsed=%.1fs answer=%s",
                sample.sample_id,
                kind,
                task,
                index + 1,
                agent,
                time.perf_counter() - started,
                _preview(answer),
            )
            return answer
        raise last_error

    async def _run_once(
        self, sample: Any, sample_dir: dict[str, Any], kind: str, task: str
    ) -> tuple[Any | None, str | None]:
        """Run one SampleRunner attempt; salvage raw output when strict
        validation rejected the model's genuine answer.

        执行一次 SampleRunner；严格校验拒收模型真实输出时做原始响应救援。"""
        runner = self._runner_for(sample_dir["data_root"])
        try:
            outcome = await runner.run_one(
                sample,
                sample_dir["dir"],
                visual_task_plan=None,
                judge_policy="none",
                evaluate=False,
            )
        except Exception:
            log.warning(
                "item=%s type=%s task=%s run_one failed", sample.sample_id, kind, task
            )
            return None, None
        if outcome.execution is not None:
            return outcome.execution.payload, outcome.execution.agent_name
        salvaged = self._salvage(sample_dir["dir"])
        if salvaged is not None:
            log.warning(
                "item=%s type=%s task=%s salvage=raw_response",
                sample.sample_id,
                kind,
                task,
            )
            return salvaged, "salvage"
        log.warning("item=%s type=%s task=%s no execution", sample.sample_id, kind, task)
        return None, None

    def _question_variants(self, request: dict[str, Any], task: str) -> tuple[str, ...]:
        """Question texts to try in order; localization gets one escalated
        retry when the first reply carries no usable box.

        按序尝试的题干变体；定位题在首答无可用框时做一次升级重试。"""
        base = compose_question(request)
        if task != "grounding":
            return (base,)
        retry = base + (
            "\n\nIMPORTANT: your previous reply contained no bounding box. You MUST "
            "answer with exactly one bounding box as the first entry of boxes, using "
            "integer whole-image 0..999 xyxy coordinates. Return your best estimate "
            "even if you are not fully certain."
        )
        return (base, retry)

    # -- planner-driven mode (full M3 system) --------------------------------

    async def _try_planner(
        self, request: dict[str, Any], image_paths: list[Path], started: float
    ) -> Any | None:
        """One v5-planner-driven attempt (the full M3 system). Returns None on
        any plumbing failure so the caller can fall back to constraint mode.

        一次 v5 planner 驱动的完整系统尝试；任何链路故障都返回 None，
        由调用方退回 constraint 直连模式（双层保险）。"""
        kind = request["response_constraint"]["type"]
        task_guess = task_for(request)
        sample_dir: dict[str, Any] | None = None
        try:
            sample, sample_dir, plan, views = await self._build_planned(
                request, image_paths, task_guess
            )
            # Observed 2026-10-06: this model's wrapped tile-count JSON
            # ({"count_tile_response": {...}}) never satisfies the counting
            # pipeline's strict schemas; go straight to the constraint answer.
            # 实测：该模型的计数 JSON 习惯包一层包装键，无法通过 counting 严格
            # Schema；判定为计数题时直接走直答回退，省掉必然失败的尝试。
            if plan.task in {"counting", "fine_grained_counting"}:
                log.info(
                    "item=%s planner decided counting; using constraint fallback "
                    "by policy",
                    request.get("item_id"),
                )
                return None
            runner = self._runner_for(sample_dir["data_root"])
            outcome = await runner.run_one(
                sample,
                sample_dir["dir"],
                visual_task_plan=plan,
                visual_views=views,
                judge_policy="none",
                evaluate=False,
            )
        except Exception as exc:
            # Keep the diagnostic short and prompt-free: type + first chars.
            # 诊断信息仅保留异常类型与前若干字符，避免完整题干入日志。
            log.warning(
                "item=%s type=%s planner path failed: %s: %s",
                request.get("item_id"),
                kind,
                type(exc).__name__,
                str(exc)[:300],
            )
            return None
        payload: Any | None = None
        agent: str | None = None
        if outcome.execution is not None:
            payload, agent = outcome.execution.payload, outcome.execution.agent_name
        elif sample_dir is not None:
            salvaged = self._salvage(sample_dir["dir"])
            if salvaged is not None:
                payload, agent = salvaged, "salvage"
        if payload is None:
            return None
        try:
            answer = normalize_answer(request, payload)
        except AdapterError:
            return None
        log.info(
            "item=%s type=%s task=%s mode=planner planned=%s agent=%s elapsed=%.1fs answer=%s",
            sample.sample_id,
            kind,
            task_guess,
            getattr(plan, "task", None),
            agent,
            time.perf_counter() - started,
            _preview(answer),
        )
        return answer

    async def _build_planned(
        self, request: dict[str, Any], image_paths: list[Path], task_guess: str
    ) -> tuple[Any, dict[str, Any], Any, Any]:
        """Build a SampleDraft, run the v5 planner, and materialize the sample.
        Draft has placeholder roles; the planner's task is authoritative.

        构造 SampleDraft → v5 planner → materialize；draft 中角色为占位，
        以 planner 判定的任务为准。"""
        from data.schema import (
            GroundTruth,
            ImageRef,
            SampleDraft,
            TaskNormalization,
            materialize_sample,
        )

        resolved_paths = [Path(p).resolve() for p in image_paths]
        data_root = resolved_paths[0].parent.parent
        images = []
        for index, (spec, path) in enumerate(zip(request["images"], resolved_paths)):
            images.append(
                ImageRef(
                    image_id=f"image-{index}",
                    path=path.relative_to(data_root),
                    role="image",
                    sha256=spec.get("sha256"),
                )
            )
        normalization = None
        if task_guess == "multiple_choice_vqa":
            normalization = TaskNormalization(
                source_task="protocol_2_0",
                normalized_task="multiple_choice_vqa",
                normalizer="protocol_2_0",
                version="2.0",
                choices=m3_choices(choice_pairs(request)),
                allow_multiple=False,
            )
        sample_id = _SAMPLE_ID_SAFE.sub("_", str(request.get("item_id", ""))).strip("_")[:96]
        if not sample_id:
            sample_id = "item"
        draft = SampleDraft(
            sample_id=sample_id,
            dataset="rsu-protocol-2",
            split="eval",
            images=images,
            question=compose_question(request),
            explicit_task=None,
            ground_truth=GroundTruth(),
            metadata={"source": "protocol_2_0"},
            normalization=normalization,
        )
        sample_dir = (
            self.runs_root / "service" / "protocol" / self.run_tag / f"{sample_id}_planner"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        planner = self._runtime.components.visual_task_planner
        budget = self._runtime.components.call_budget_factory.create_for_sample("draft")
        plan, views = await planner.plan_with_views(
            draft,
            data_root=data_root,
            artifact_dir=sample_dir / "planner",
            budget=budget,
        )
        sample = materialize_sample(draft, plan.task)
        return sample, {"dir": sample_dir, "data_root": data_root}, plan, views

    # -- sample construction -------------------------------------------------

    def _build_sample(
        self,
        request: dict[str, Any],
        image_paths: list[Path],
        task: str,
        *,
        question: str,
        dir_suffix: str = "",
    ) -> tuple[Any, dict[str, Any]]:
        from data.schema import GroundTruth, ImageRef, TaskNormalization, UnifiedSample

        resolved_paths = [Path(p).resolve() for p in image_paths]
        data_root = resolved_paths[0].parent.parent
        specs = request["images"]
        count = len(specs)
        if task in CHANGE_TASKS:
            if count != 2:
                raise AdapterError(502, "unanswerable")
            image_ids = ["t1", "t2"]
            roles = ["t1", "t2"]
        else:
            image_ids = ["image-0"] + [f"context-{i}" for i in range(1, count)]
            roles = ["image"] + ["context"] * (count - 1)
        images = []
        for spec, path, image_id, role in zip(specs, resolved_paths, image_ids, roles):
            relative = path.relative_to(data_root)
            images.append(
                ImageRef(
                    image_id=image_id,
                    path=relative,
                    role=role,
                    sha256=spec.get("sha256"),
                )
            )
        normalization = None
        if task == "multiple_choice_vqa":
            normalization = TaskNormalization(
                source_task="protocol_2_0",
                normalized_task=task,
                normalizer="protocol_2_0",
                version="2.0",
                choices=m3_choices(choice_pairs(request)),
                allow_multiple=False,
            )
        sample_id = _SAMPLE_ID_SAFE.sub("_", str(request.get("item_id", ""))).strip("_")[:96]
        if not sample_id:
            sample_id = "item"
        sample = UnifiedSample(
            sample_id=sample_id,
            dataset="rsu-protocol-2",
            split="eval",
            task=task,
            images=images,
            question=question,
            ground_truth=GroundTruth(),
            metadata={"source": "protocol_2_0"},
            normalization=normalization,
        )
        dir_name = f"{sample_id}{dir_suffix}"
        sample_dir = self.runs_root / "service" / "protocol" / self.run_tag / dir_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample, {"dir": sample_dir, "data_root": data_root}

    def _mine_bbox(self, request: dict[str, Any], sample_dir: Path) -> list[int] | None:
        """Last-resort recovery for bbox items: mine four integers that follow
        a "boxes"/"box" key in ANY attempt's raw response, even when the JSON
        is malformed (observed: the model writes corner pairs as
        (x1,y1),(x2,y2) or [x1,y1],[x2,y2], which breaks json.loads).

        定位题最后兜底：即使 JSON 破损（实测模型会把角点写成 (x1,y1),(x2,y2)
        或 [x1,y1],[x2,y2]），也从任一尝试的原始响应中 "boxes"/"box" 键后
        挖出 4 个整数组成框。内容仍是模型自己的输出，只做协议格式规范化。
        """
        if request["response_constraint"]["type"] != "bbox":
            return None
        from types import SimpleNamespace

        base = sample_dir.name.split("_retry")[0]
        files: list[Path] = []
        for directory in sorted(sample_dir.parent.glob(base + "*")):
            files.extend(sorted(directory.glob("*/raw_response.txt")))
            direct = directory / "raw_response.txt"
            if direct.is_file():
                files.append(direct)
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for key in ('"boxes"', '"box"'):
                for match in re.finditer(re.escape(key), text):
                    tail = text[match.end() : match.end() + 400]
                    numbers = re.findall(r"\d+", tail)
                    if len(numbers) < 4:
                        continue
                    box = [int(value) for value in numbers[:4]]
                    payload = SimpleNamespace(answer="", boxes=[box], evidence_items=[])
                    try:
                        return normalize_answer(request, payload)
                    except AdapterError:
                        continue
        return None

    def _salvage(self, sample_dir: Path) -> Any | None:
        """Recover the model's genuine output when strict schema validation
        failed and no execution was produced.

        严格 Schema 校验失败、无执行结果时，从原始响应（raw_response.txt）
        中抢救模型的真实输出。这不是伪造答案：内容就是模型自己的输出，
        只是被更严格的内部 Schema 拒收；抢救只按协议做格式规范化。
        """
        from types import SimpleNamespace

        candidates = sorted(sample_dir.glob("*/raw_response.txt"))
        if not candidates:
            candidates = sorted(sample_dir.glob("raw_response.txt"))
        for path in candidates:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:65536]
            except OSError:
                continue
            parsed: dict[str, Any] | None = None
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    parsed = obj  # keep the last valid attempt
            if parsed is None:
                continue
            boxes: list[list[int]] = []
            for box in parsed.get("boxes") or []:
                if (
                    isinstance(box, (list, tuple))
                    and len(box) == 4
                    and all(isinstance(v, (int, float)) for v in box)
                ):
                    boxes.append([int(round(float(v))) for v in box])
            evidence: list[Any] = []
            for item in parsed.get("evidence_items") or []:
                if isinstance(item, dict):
                    box = item.get("box")
                    if (
                        isinstance(box, (list, tuple))
                        and len(box) == 4
                        and all(isinstance(v, (int, float)) for v in box)
                    ):
                        evidence.append(
                            SimpleNamespace(box=[int(round(float(v))) for v in box])
                        )
            answer = str(parsed.get("answer") or "")
            if boxes or evidence or answer:
                return SimpleNamespace(answer=answer, boxes=boxes, evidence_items=evidence)
        return None


def _preview(answer: Any) -> str:
    """Short, log-safe preview of the normalized answer (never the question).

    规范化答案的短预览（绝不记录题干）；用于本地日志排错。
    """
    text = str(answer)
    return text[:120]
