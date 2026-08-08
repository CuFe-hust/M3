"""Dataset-level orchestration: probe, selection, resume, shard, limits,
concurrency, SampleRunner invocation, and summary.

数据集级编排：probe、选择、resume、分片、限制、并发、SampleRunner 调用与
汇总。本模块不含任何 Agent 特定逻辑。

- selection 固定顺序：adapter 稳定顺序 → start_index → shard → sample_ids
  → limit；shard 用 SHA256（非 Python hash），稳定跨进程/跨平台。
- 目录布局：runs/<run_id>/tasks/<task>/samples/<sha256(sample_id)[:24]>，
  不直接使用 sample_id 作为目录（解决 Windows 危险名与多 task 同 id 冲突）；
  predictions.jsonl 位于 run 根，dataset_summary.json 位于 task 目录。
- resume：succeeded 默认不重新推理，只补缺失的确定性评估与缺失/失败的
  VQA judge（补判异常 → skipped，稳定 code）；partial/failed/running/
  pending/缺失状态一律重新执行 SampleRunner。
- 并发只承诺单进程 asyncio；fail-fast 后不再提交新任务、cancel/await 已
  启动任务、被取消样本写 skipped（FAIL_FAST_CANCELLED），绝不遗留永久
  running 状态。
- 所有失败只记录稳定 code；原始异常/路径/密钥绝不进入产物。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.counting.schema import CountingResult
from data.adapters.base import DatasetAdapter
from data.adapters.manifest import update_manifest_probe
from data.schema import SampleDraft, UnifiedSample
from evaluation.metrics.counting import merge_count_evaluation
from evaluation.metrics.vqa import merge_vqa_evaluation
from routing.schema import TaskResolutionRequest
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudgetFactory
from workflows.sample_runner import SampleRunner
from workflows.schema import DatasetRunSummary, SampleRunStatus
from workflows.task_resolver import (
    SampleMaterializationError,
    TaskResolutionError,
    TaskResolver,
    materialize_sample,
)

# Storage key length: sha256(sample_id) hex digest, truncated for directory
# names. / 存储键长度：sha256(sample_id) 十六进制摘要截断为目录名。
STORAGE_KEY_LENGTH = 24

# Artifact filenames shared with SampleRunner. / 与 SampleRunner 共享的产物文件名。
_VQA_EVALUATION_FILENAME = "vqa_evaluation.json"
_COUNTING_EVALUATION_FILENAME = "counting_evaluation.json"
_AGENT_RESULT_FILENAME = "agent_result.json"
_COUNTING_RESULT_FILENAME = "counting_result.json"
_STATUS_FILENAME = "status.json"


class ResumeSupplementError(ValueError):
    """Stable error for resume-supplement failures; the code never leaks raw
    content. resume 补判失败的稳定错误；code 绝不泄漏原始内容。"""

    def __init__(self, code: str) -> None:
        super().__init__(f"RESUME_SUPPLEMENT_FAILED:{code}")
        self.code = code


def storage_key(sample_id: str) -> str:
    """Deterministic, path-safe sample storage key derived from the sample id;
    Windows-dangerous ids (CON, separators, trailing dots, drive paths) can
    never escape the samples root. 由 sample id 派生的确定性、路径安全样本
    存储键；Windows 危险 id（CON、分隔符、尾点、盘符路径）绝不逃逸样本根。"""

    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:STORAGE_KEY_LENGTH]


def _shard_matches(sample_id: str, shard_index: int, shard_count: int) -> bool:
    """Stable SHA256-based shard assignment; never Python's hash(). 基于
    SHA256 的稳定分片分配；绝不使用 Python hash()。"""

    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % shard_count == shard_index


def select_samples(
    samples: Iterable[UnifiedSample],
    *,
    start_index: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    sample_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[UnifiedSample]:
    """Select samples in the fixed order: adapter stable order → start_index
    → shard → sample_ids → limit. limit caps the filtered selection.
    按固定顺序选择样本：adapter 稳定顺序 → start_index → shard → sample_ids
    → limit；limit 是过滤后选择数的上限。"""

    selected: list[UnifiedSample] = []
    for index, sample in enumerate(samples):
        if limit is not None and len(selected) >= limit:
            break
        if index < start_index:
            continue
        if not _shard_matches(sample.sample_id, shard_index, shard_count):
            continue
        if sample_ids is not None and sample.sample_id not in sample_ids:
            continue
        selected.append(sample)
    return selected


def _stable_error_code(error: Exception) -> str:
    """Prefer an explicit stable code on the exception, else the class name;
    never the raw message. 优先取异常上的显式稳定 code，否则用类名；绝不取
    原始消息。"""

    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    return type(error).__name__


class DatasetRunner:
    """Iterate one dataset task through the injected SampleRunner.
    通过注入的 SampleRunner 迭代一个数据集任务。"""

    def __init__(
        self,
        adapter: DatasetAdapter,
        sample_runner: SampleRunner,
        *,
        run_dir: Path,
        artifact_writer: ArtifactWriter,
        judge_policy: str = "none",
        task_resolver: TaskResolver | None = None,
        call_budget_factory: CallBudgetFactory | None = None,
    ) -> None:
        self.adapter = adapter
        self.sample_runner = sample_runner
        self.run_dir = run_dir
        self.artifact_writer = artifact_writer
        self.judge_policy = judge_policy
        self.task_resolver = task_resolver
        self.call_budget_factory = call_budget_factory

    async def run(
        self,
        *,
        root: Path,
        split: str,
        task: str | None = None,
        resume: bool = False,
        limit: int | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
        start_index: int = 0,
        sample_ids: set[str] | None = None,
        fail_fast: bool = False,
        sample_concurrency: int = 1,
    ) -> DatasetRunSummary:
        """Run one task over the selected samples and persist predictions,
        summary, and the dataset probe in the run manifest. With task=None the
        adapter is treated as a DraftDatasetAdapter: drafts are resolved,
        materialized, and executed under the 'auto' task directory.
        在选中样本上运行一个任务，并持久化 predictions、summary 与 run
        manifest 中的数据集 probe。task=None 时把适配器视为
        DraftDatasetAdapter：drafts 经解析、物化后在 'auto' 任务目录执行。"""

        if shard_count < 1:
            raise ValueError("shard_count must be >= 1")
        if not 0 <= shard_index < shard_count:
            raise ValueError("shard_index must be within [0, shard_count)")
        if sample_concurrency < 1:
            raise ValueError("sample_concurrency must be >= 1")
        if task is None:
            return await self._run_draft_task(
                root=root,
                split=split,
                resume=resume,
                limit=limit,
                shard_index=shard_index,
                shard_count=shard_count,
                start_index=start_index,
                sample_ids=sample_ids,
                fail_fast=fail_fast,
                sample_concurrency=sample_concurrency,
            )
        probe = self.adapter.probe(root, task)
        update_manifest_probe(self.run_dir, probe)
        selected = select_samples(
            self.adapter.iter_samples(root, split, task),
            start_index=start_index,
            shard_index=shard_index,
            shard_count=shard_count,
            sample_ids=sample_ids,
            limit=limit,
        )
        return await self._run_selected(
            selected,
            split=split,
            task=task,
            resume=resume,
            fail_fast=fail_fast,
            sample_concurrency=sample_concurrency,
            run_item=self._run_sample,
        )

    async def _run_draft_task(
        self,
        *,
        root: Path,
        split: str,
        resume: bool,
        limit: int | None,
        shard_index: int,
        shard_count: int,
        start_index: int,
        sample_ids: set[str] | None,
        fail_fast: bool,
        sample_concurrency: int,
    ) -> DatasetRunSummary:
        """Draft mode: resolve each draft (or use its explicit task),
        materialize, and execute; all samples live under tasks/auto/ so resume
        lookup never requires re-resolution. Draft 模式：解析每条 draft（或
        使用其显式 task）、物化并执行；所有样本位于 tasks/auto/ 下，resume
        查找无需重新解析。"""

        if self.task_resolver is None or self.call_budget_factory is None:
            raise ValueError(
                "draft task mode requires task_resolver and call_budget_factory"
            )
        if not hasattr(self.adapter, "iter_drafts"):
            raise TypeError(f"adapter {self.adapter.name!r} does not yield drafts")
        probe = self.adapter.probe(root, None)
        update_manifest_probe(self.run_dir, probe)
        drafts = select_samples(
            self.adapter.iter_drafts(root, split),
            start_index=start_index,
            shard_index=shard_index,
            shard_count=shard_count,
            sample_ids=sample_ids,
            limit=limit,
        )
        return await self._run_selected(
            drafts,
            split=split,
            task="auto",
            resume=resume,
            fail_fast=fail_fast,
            sample_concurrency=sample_concurrency,
            run_item=self._run_draft,
        )

    async def _run_selected(
        self,
        selected: list[Any],
        *,
        split: str,
        task: str,
        resume: bool,
        fail_fast: bool,
        sample_concurrency: int,
        run_item: Any,
    ) -> DatasetRunSummary:
        """Shared concurrency loop: bounded asyncio batching, fail-fast
        cancellation, per-sample predictions, and the task summary.
        共享并发循环：有界 asyncio 批次、fail-fast 取消、逐样本 predictions
        与任务汇总。"""

        task_dir = self.run_dir / "tasks" / task
        samples_root = task_dir / "samples"
        semaphore = asyncio.Semaphore(sample_concurrency)
        statuses: list[SampleRunStatus] = []
        fail_fast_triggered = False

        def record_status(status: SampleRunStatus) -> None:
            nonlocal fail_fast_triggered
            statuses.append(status)
            self.artifact_writer.append_prediction(
                self.run_dir,
                sample_id=status.sample_id,
                task=status.task,
                status=status,
            )
            if fail_fast and status.state == "failed":
                fail_fast_triggered = True

        async def run_one_item(item: Any) -> SampleRunStatus:
            async with semaphore:
                try:
                    return await run_item(item, samples_root, resume=resume)
                except asyncio.CancelledError:
                    status = _cancelled_status(item)
                    self.artifact_writer.write_final_status(
                        samples_root / storage_key(item.sample_id), status
                    )
                    record_status(status)
                    raise

        pending: set[asyncio.Task] = set()
        for item in selected:
            if fail_fast_triggered:
                break
            pending_task = asyncio.create_task(run_one_item(item))
            pending.add(pending_task)
            if len(pending) >= sample_concurrency:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for done_task in done:
                    if not done_task.cancelled():
                        record_status(done_task.result())
        if pending:
            if fail_fast_triggered:
                for pending_task in pending:
                    pending_task.cancel()
            done, _ = await asyncio.wait(pending)
            for done_task in done:
                if not done_task.cancelled():
                    record_status(done_task.result())
        summary = DatasetRunSummary(
            run_id=self.run_dir.name,
            dataset=self.adapter.name,
            split=split,
            task=task,
            total=len(selected),
            succeeded=sum(1 for status in statuses if status.state == "succeeded"),
            partial=sum(1 for status in statuses if status.state == "partial"),
            failed=sum(1 for status in statuses if status.state == "failed"),
            skipped=sum(1 for status in statuses if status.state == "skipped"),
        )
        self.artifact_writer.write_summary(task_dir, summary)
        return summary

    async def _run_sample(
        self,
        sample: UnifiedSample,
        samples_root: Path,
        *,
        resume: bool,
    ) -> SampleRunStatus:
        """Run one sample, or supplement a resumed succeeded sample. Sample
        failures never raise: they collapse into a failed status with stable
        codes. 运行一条样本，或补判 resume 的 succeeded 样本。样本失败绝不
        抛出：收敛为携带稳定 code 的 failed 状态。"""

        sample_dir = samples_root / storage_key(sample.sample_id)
        if resume:
            persisted = self._read_status(sample_dir)
            if persisted is not None and persisted.state == "succeeded":
                return await self._resume_supplement(sample, sample_dir, sample.task)
        try:
            outcome = await self.sample_runner.run_one(
                sample, sample_dir, judge_policy=self.judge_policy
            )
        except Exception as error:
            # SampleRunner is not expected to raise for sample-level failures;
            # this is a defensive net with stable codes only.
            # SampleRunner 不应因样本级失败而抛出；这里是只带稳定 code 的
            # 防御网。
            status = _defensive_failed_status(sample, sample.task, error)
            self.artifact_writer.write_final_status(sample_dir, status)
            return status
        return outcome.status

    async def _run_draft(
        self,
        draft: SampleDraft,
        samples_root: Path,
        *,
        resume: bool,
    ) -> SampleRunStatus:
        """Run one draft through resolution, materialization, and the
        SampleRunner. A shared default CallBudget spans the resolver call and
        every agent attempt. Pre-task failures collapse into a failed status
        with a stable code and an honest task label — never a guessed
        general_vqa. 将一条 draft 经解析、物化与 SampleRunner 执行。共享默认
        CallBudget 贯穿 resolver 调用与所有 agent attempt。预 task 失败收敛
        为携带稳定 code 与诚实 task 标签（绝不猜测 general_vqa）的 failed
        状态。"""

        sample_dir = samples_root / storage_key(draft.sample_id)
        if resume:
            persisted = self._read_status(sample_dir)
            if persisted is not None and persisted.state == "succeeded":
                persisted_sample = self._read_persisted_sample(sample_dir)
                if persisted_sample is not None:
                    return await self._resume_supplement(
                        persisted_sample, sample_dir, persisted_sample.task
                    )
        if draft.explicit_task is not None:
            try:
                sample = materialize_sample(draft, draft.explicit_task)
            except SampleMaterializationError as error:
                return self._write_draft_failure(
                    draft, sample_dir, task=draft.explicit_task, code=error.code
                )
            outcome = await self.sample_runner.run_one(
                sample, sample_dir, judge_policy=self.judge_policy
            )
            return outcome.status
        budget = self.call_budget_factory.create_for_sample("draft")
        try:
            resolution = await self.task_resolver.resolve(
                TaskResolutionRequest(
                    explicit_task=None,
                    question=draft.question,
                    image_count=len(draft.images),
                    metadata_hints=draft.metadata,
                ),
                sample_id=draft.sample_id,
                artifact_dir=sample_dir / "task_resolution",
                budget=budget,
            )
            sample = materialize_sample(draft, resolution.task)
        except TaskResolutionError as error:
            return self._write_draft_failure(
                draft, sample_dir, task=None, code=error.code
            )
        except SampleMaterializationError as error:
            return self._write_draft_failure(
                draft, sample_dir, task=resolution.task, code=error.code
            )
        outcome = await self.sample_runner.run_one(
            sample,
            sample_dir,
            resolution=resolution,
            budget=budget,
            judge_policy=self.judge_policy,
        )
        return outcome.status

    def _write_draft_failure(
        self,
        draft: SampleDraft,
        sample_dir: Path,
        *,
        task: str | None,
        code: str,
    ) -> SampleRunStatus:
        """Persist a failed status for a pre-task draft failure; the task
        label is the known task or the honest sentinel 'unknown' — never a
        guessed general_vqa. 持久化预 task draft 失败的 failed 状态；task
        标签为已知任务或诚实哨兵 'unknown'——绝不猜测 general_vqa。"""

        status = SampleRunStatus(
            sample_id=draft.sample_id,
            task=task or "unknown",
            state="failed",
            error_code=code,
            error_message=code,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self.artifact_writer.write_final_status(sample_dir, status)
        return status

    def _read_persisted_sample(self, sample_dir: Path) -> UnifiedSample | None:
        """Read the persisted sample for a resume supplement; corrupt or
        missing files count as absent and trigger a re-run. 读取持久化样本
        用于 resume 补判；损坏或缺失视为不存在并触发重跑。"""

        path = sample_dir / "sample.json"
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return UnifiedSample.model_validate(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError):
            return None

    def _read_status(self, sample_dir: Path) -> SampleRunStatus | None:
        """Read the persisted sample status; corrupt or missing files count as
        absent and trigger a re-run. 读取持久化样本状态；损坏或缺失一律视为
        不存在并触发重跑。"""

        status_path = sample_dir / _STATUS_FILENAME
        if not status_path.is_file():
            return None
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            return SampleRunStatus.model_validate(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError):
            return None

    async def _resume_supplement(
        self,
        sample: UnifiedSample,
        sample_dir: Path,
        task: str,
    ) -> SampleRunStatus:
        """Supplement a resumed succeeded sample: missing deterministic
        evaluation and missing/failed VQA judge only. Supplement exceptions
        degrade the sample to skipped with a stable code; a failed re-judge
        keeps the succeeded state (the failure is visible in the evaluation).
        补判 resume 的 succeeded 样本：只补缺失的确定性评估与缺失/失败的
        VQA judge。补判异常将样本降级为 skipped（稳定 code）；重判失败保留
        succeeded 状态（失败在 evaluation 中可见）。"""

        persisted = self._read_status(sample_dir)
        if persisted is None:
            raise ResumeSupplementError("STATUS_MISSING")
        try:
            await self._supplement_evaluation(sample, sample_dir, task)
        except Exception as error:
            skipped = persisted.model_copy(
                update={
                    "state": "skipped",
                    "error_code": _stable_error_code(error),
                    "error_message": _stable_error_code(error),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self.artifact_writer.write_final_status(sample_dir, skipped)
            return skipped
        return persisted

    async def _supplement_evaluation(
        self,
        sample: UnifiedSample,
        sample_dir: Path,
        task: str,
    ) -> None:
        """Write the missing deterministic evaluation and re-run a missing or
        failed VQA judge. 补写缺失的确定性评估，并重跑缺失或失败的 VQA
        judge。"""

        if task == "general_vqa":
            evaluation_path = sample_dir / _VQA_EVALUATION_FILENAME
            result_path = sample_dir / _AGENT_RESULT_FILENAME
            if not evaluation_path.is_file():
                if not result_path.is_file():
                    raise ResumeSupplementError("AGENT_RESULT_MISSING")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                answer = str(result.get("answer", ""))
                evaluation = merge_vqa_evaluation(
                    sample_id=sample.sample_id,
                    question=sample.question,
                    reference_answers=(
                        list(sample.ground_truth.answers)
                        if sample.ground_truth is not None
                        else []
                    ),
                    candidate_answer=answer,
                )
                self.artifact_writer.write_evaluation(
                    sample_dir, evaluation, filename=_VQA_EVALUATION_FILENAME
                )
            judge_service = self.sample_runner.judge_service
            if judge_service is not None and self.judge_policy != "none":
                evaluation = judge_service.judge_vqa_resume(
                    sample=sample,
                    candidate_answer="",
                    sample_dir=sample_dir,
                    judge_policy=self.judge_policy,
                    call_budget=None,
                )
                self.artifact_writer.write_evaluation(
                    sample_dir, evaluation, filename=_VQA_EVALUATION_FILENAME
                )
            return
        if task == "counting":
            evaluation_path = sample_dir / _COUNTING_EVALUATION_FILENAME
            if evaluation_path.is_file():
                return
            result_path = sample_dir / _COUNTING_RESULT_FILENAME
            if not result_path.is_file():
                raise ResumeSupplementError("COUNTING_RESULT_MISSING")
            counting = CountingResult.model_validate(
                json.loads(result_path.read_text(encoding="utf-8"))
            )
            evaluation = merge_count_evaluation(
                sample_id=sample.sample_id,
                counting=counting,
                ground_truth=sample.ground_truth,
            )
            self.artifact_writer.write_evaluation(
                sample_dir, evaluation, filename=_COUNTING_EVALUATION_FILENAME
            )


def _cancelled_status(item: Any) -> SampleRunStatus:
    """Final status for a sample cancelled by fail-fast; never leaves a
    permanent running state. The task label is the item's known task or the
    honest sentinel 'unknown' for unresolved drafts. fail-fast 取消样本的最终
    状态；绝不遗留永久 running 状态。task 标签为条目已知任务，未解析 draft
    为诚实哨兵 'unknown'。"""

    task = (
        getattr(item, "task", None)
        or getattr(item, "explicit_task", None)
        or "unknown"
    )
    return SampleRunStatus(
        sample_id=item.sample_id,
        task=task,
        state="skipped",
        error_code="FAIL_FAST_CANCELLED",
        error_message="FAIL_FAST_CANCELLED",
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _defensive_failed_status(
    sample: UnifiedSample, task: str, error: Exception
) -> SampleRunStatus:
    """Stable failed status for an unexpected sample-runner exception.
    SampleRunner 意外异常时的稳定 failed 状态。"""

    return SampleRunStatus(
        sample_id=sample.sample_id,
        task=task,
        state="failed",
        error_code=_stable_error_code(error),
        error_message=_stable_error_code(error),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
