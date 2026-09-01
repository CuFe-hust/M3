"""Dataset-level orchestration for the canonical visual-plan path: probe,
selection, resume, shard, limits, concurrency, SampleRunner invocation, and
summary.

数据集级编排：probe、选择、resume、分片、限制、并发、SampleRunner 调用与
汇总。本模块不含任何 Agent 特定逻辑；fresh execution 统一使用
``VisualTaskPlanner``。

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
from agents.schema import (
    GENERAL_VQA_AGENT_TASKS,
    AgentResult,
)
from data.adapters.base import DatasetAdapter
from data.schema import SampleDraft, SampleMaterializationError, UnifiedSample, materialize_sample
from evaluation.records import (
    EVALUATION_FILENAME_BY_TASK,
    evaluation_filename_for_runtime_task,
    evaluation_task_for_runtime_task,
)
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudgetFactory
from workflows.sample_runner import (
    SampleRunner,
    _rebuild_sample_for_task,
    build_deterministic_evaluation,
)
from workflows.schema import (
    DatasetRunSummary,
    EvidencePreprocessingIdentity,
    SampleRunStatus,
)
from workflows.visual_planner import VisualTaskPlanError, VisualTaskPlanner

# Storage key length: sha256(sample_id) hex digest, truncated for directory
# names. / 存储键长度：sha256(sample_id) 十六进制摘要截断为目录名。
STORAGE_KEY_LENGTH = 24

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
        judge_sample_rate: float | None = None,
        call_budget_factory: CallBudgetFactory | None = None,
        visual_task_planner: VisualTaskPlanner | None = None,
        planning_mode: str = "visual-task-plan-v5",
        data_root: Path | None = None,
        evidence_preprocessing: EvidencePreprocessingIdentity | None = None,
        vqa_assistance_scope: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.sample_runner = sample_runner
        self.run_dir = run_dir
        self.artifact_writer = artifact_writer
        self.judge_policy = judge_policy
        self.judge_sample_rate = judge_sample_rate
        self.call_budget_factory = call_budget_factory
        self.visual_task_planner = visual_task_planner
        self.planning_mode = planning_mode
        self.data_root = data_root
        self.evidence_preprocessing = evidence_preprocessing
        self.vqa_assistance_scope = vqa_assistance_scope

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
        if not resume and self.planning_mode != "visual-task-plan-v5":
            raise ValueError("fresh dataset runs require visual-task-plan-v5")
        if not resume and (
            self.call_budget_factory is None
            or self.visual_task_planner is None
            or self.data_root is None
        ):
            raise ValueError("fresh dataset runs require assembled visual planner")
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
        task_dir = self.run_dir / "tasks" / task
        self.artifact_writer.write_dataset_probe(task_dir, probe, dataset_root=root)
        if resume:
            self._restore_persisted_judge_rate(task_dir)
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
        """Draft mode: plan each draft, materialize, and execute; all samples
        live under tasks/auto/ so resume lookup never requires replanning.
        Draft 模式：规划、物化并执行每条 draft；所有样本位于 tasks/auto/ 下，
        resume 查找无需重新规划。"""

        if not resume and (
            self.call_budget_factory is None
            or self.visual_task_planner is None
            or self.data_root is None
        ):
            raise ValueError("draft task mode requires assembled visual planner")
        if not hasattr(self.adapter, "iter_drafts"):
            raise TypeError(f"adapter {self.adapter.name!r} does not yield drafts")
        probe = self.adapter.probe(root, None)
        task_dir = self.run_dir / "tasks" / "auto"
        self.artifact_writer.write_dataset_probe(task_dir, probe, dataset_root=root)
        if resume:
            self._restore_persisted_judge_rate(task_dir)
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

        def record_status(status: SampleRunStatus, sample_dir: Path) -> None:
            nonlocal fail_fast_triggered
            statuses.append(status)
            result_path = None
            if status.result_path is not None:
                # The row path is run-relative and always derived from the
                # actual sample directory — never from status.task, which may
                # be an execution task different from the storage namespace
                # (e.g. tasks/auto/ with task=caption).
                # 行路径为 run 相对且恒由实际样本目录推导——绝不根据
                # status.task 拼目录（如 tasks/auto/ 下 task=caption）。
                result_path = (
                    sample_dir.relative_to(self.run_dir) / status.result_path
                ).as_posix()
            self.artifact_writer.append_prediction(
                self.run_dir,
                sample_id=status.sample_id,
                run_task=task,
                task=status.task,
                status=status,
                result_path=result_path,
            )
            if fail_fast and status.state == "failed":
                fail_fast_triggered = True

        async def run_one_item(item: Any) -> SampleRunStatus:
            async with semaphore:
                sample_dir = samples_root / storage_key(item.sample_id)
                try:
                    return await run_item(item, samples_root, resume=resume)
                except asyncio.CancelledError:
                    status = _cancelled_status(item)
                    self.artifact_writer.write_final_status(sample_dir, status)
                    record_status(status, sample_dir)
                    raise

        pending: set[asyncio.Task] = set()
        not_started: list[Any] = []
        for item in selected:
            if fail_fast_triggered:
                not_started.append(item)
                continue
            pending_task = asyncio.create_task(run_one_item(item))
            pending.add(pending_task)
            if len(pending) >= sample_concurrency:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for done_task in done:
                    if not done_task.cancelled():
                        status = done_task.result()
                        sample_dir = samples_root / storage_key(status.sample_id)
                        record_status(status, sample_dir)
        if pending:
            if fail_fast_triggered:
                for pending_task in pending:
                    pending_task.cancel()
            done, _ = await asyncio.wait(pending)
            for done_task in done:
                if not done_task.cancelled():
                    status = done_task.result()
                    sample_dir = samples_root / storage_key(status.sample_id)
                    record_status(status, sample_dir)
        # fail-fast accounting: selected-but-never-started samples receive a
        # terminal skipped status so the summary counts always close.
        # fail-fast 记账：已选中但从未启动的样本写入终态 skipped，使汇总计数
        # 永远闭合。
        if not_started:
            for item in not_started:
                status = _not_started_status(item)
                sample_dir = samples_root / storage_key(item.sample_id)
                self.artifact_writer.write_final_status(sample_dir, status)
                record_status(status, sample_dir)
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
            judge_sample_rate=self.judge_sample_rate,
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
                # Supplement by persisted execution task; this path never
                # calls a model. 按持久化执行 task 补判；此路径绝不调用模型。
                return await self._resume_supplement(sample, sample_dir, persisted.task)
            if self.planning_mode != "visual-task-plan-v5":
                return self._write_planning_resume_failure(
                    sample,
                    sample_dir,
                    persisted_task=persisted.task if persisted is not None else None,
                )
            # The planner may rewrite the adapter source task, so the
            # persisted execution task is the only authoritative task for the
            # legacy gates; the source task is a fallback only when no status
            # was persisted (missing status reruns per the documented
            # contract). The honest 'unknown' sentinel (pre-task failure)
            # cannot prove the replan stays outside the VQA family and fails
            # closed. planner 可能改写 adapter 的 source task，因此 legacy
            # 门禁只认持久化 execution task；仅在无持久化状态（缺失状态按
            # 文档契约重跑）时才回退 source task。诚实的 'unknown' 哨兵（预
            # task 失败）无法证明重规划会留在 VQA 族之外，按 fail-closed
            # 处理。
            persisted_task = persisted.task if persisted is not None else None
            gate_task = persisted_task if persisted_task is not None else sample.task
            # A legacy run (no frozen evidence preprocessing identity) that
            # needs to rerun VQA evidence would silently switch to the new
            # greedy-1024-stretch-v1 semantics; fail closed instead.
            # 历史运行（无冻结 evidence 预处理身份）需要重跑 VQA evidence 时
            # 会悄悄切换成新 greedy-1024-stretch-v1 语义；因此严格失败。
            if self.evidence_preprocessing is None and gate_task == "general_vqa":
                return self._write_planning_resume_failure(
                    sample,
                    sample_dir,
                    persisted_task=persisted_task,
                    code="LEGACY_VQA_EVIDENCE_PREPROCESSING_UNSUPPORTED",
                )
            # A legacy run without the frozen VQA assistance scope could
            # silently let any GeneralVQAAgent task replan into the new
            # evidence path; fail closed instead. The scope and the tile
            # preprocessing identity are two independent frozen identities.
            # 历史运行若缺少冻结的 VQA assistance scope，重新规划时可能让
            # 任一 GeneralVQAAgent task 静默进入新证据路径；因此严格失败。
            # scope 与 tile 预处理是两个独立的冻结身份。
            if self.vqa_assistance_scope is None and (
                gate_task in GENERAL_VQA_AGENT_TASKS or gate_task == "unknown"
            ):
                return self._write_planning_resume_failure(
                    sample,
                    sample_dir,
                    persisted_task=persisted_task,
                    code="LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED",
                )
        return await self._run_sample_visual(sample, sample_dir)

    async def _run_sample_visual(
        self,
        sample: UnifiedSample,
        sample_dir: Path,
    ) -> SampleRunStatus:
        """Run one explicit sample through the canonical visual-only planner.
        将显式样本经规范纯视觉规划器执行。"""
        if (
            self.call_budget_factory is None
            or self.data_root is None
            or self.visual_task_planner is None
        ):
            return self._write_planning_failure(
                sample, sample_dir, task=sample.task, code="PLANNER_NOT_ASSEMBLED"
            )
        budget = self.call_budget_factory.create_for_sample("sample")
        try:
            planner_data_root = self.data_root
            image_root_for_sample = getattr(self.adapter, "image_root_for_sample", None)
            if callable(image_root_for_sample):
                planner_data_root = image_root_for_sample(sample, self.data_root)
            plan, views = await self.visual_task_planner.plan_with_views(
                sample,
                data_root=planner_data_root,
                artifact_dir=sample_dir,
                budget=budget,
            )
            rebuilt = (
                sample
                if plan.task == sample.task
                else _rebuild_sample_for_task(sample, plan.task)
            )
            if rebuilt is None:
                return self._write_planning_failure(
                    sample,
                    sample_dir,
                    task=plan.task,
                    code="INCOMPATIBLE_VISUAL_TASK",
                )
            self.artifact_writer.write_visual_task_plan(
                sample_dir, plan, materialized_views=views
            )
        except VisualTaskPlanError as error:
            return self._write_planning_failure(
                sample, sample_dir, task=None, code=error.code
            )
        except Exception as error:
            return self._write_planning_failure(
                sample, sample_dir, task=None, code=_stable_error_code(error)
            )
        runner_has_data_root = hasattr(self.sample_runner, "data_root")
        previous_data_root = getattr(self.sample_runner, "data_root", None)
        if runner_has_data_root:
            self.sample_runner.data_root = planner_data_root
        try:
            outcome = await self.sample_runner.run_one(
                rebuilt,
                sample_dir,
                visual_task_plan=plan,
                visual_views=views,
                budget=budget,
                judge_policy=self._judge_policy_for(sample.sample_id),
            )
        except Exception as error:
            return self._write_planning_failure(
                sample, sample_dir, task=plan.task, code=_stable_error_code(error)
            )
        finally:
            if runner_has_data_root:
                self.sample_runner.data_root = previous_data_root
        return outcome.status

    async def _run_draft(
        self,
        draft: SampleDraft,
        samples_root: Path,
        *,
        resume: bool,
    ) -> SampleRunStatus:
        """Plan, materialize, and execute one draft through the v5 seam.
        通过 v5 seam 规划、物化并执行一条 draft。
        """

        sample_dir = samples_root / storage_key(draft.sample_id)
        if resume:
            persisted = self._read_status(sample_dir)
            if persisted is not None and persisted.state == "succeeded":
                persisted_sample = self._read_persisted_sample(sample_dir)
                if persisted_sample is not None:
                    return await self._resume_supplement(
                        persisted_sample, sample_dir, persisted.task
                    )
            if self.planning_mode != "visual-task-plan-v5":
                return self._write_planning_resume_failure(
                    draft,
                    sample_dir,
                    persisted_task=persisted.task if persisted is not None else None,
                )
            # A draft has no task field to inspect on the in-memory object;
            # the persisted execution task is the only authority. The honest
            # 'unknown' sentinel (pre-task planning failure) cannot prove the
            # replan stays outside the VQA family and fails closed.
            # SampleDraft 没有可检查的内存 task；持久化 execution task 是唯一
            # 权威。诚实 'unknown' 哨兵（预 task 规划失败）无法证明重规划会
            # 留在 VQA 族之外，按 fail-closed 处理。
            persisted_task = persisted.task if persisted is not None else None
            if self.evidence_preprocessing is None and persisted_task == "general_vqa":
                return self._write_planning_resume_failure(
                    draft,
                    sample_dir,
                    persisted_task=persisted_task,
                    code="LEGACY_VQA_EVIDENCE_PREPROCESSING_UNSUPPORTED",
                )
            if self.vqa_assistance_scope is None and (
                persisted_task in GENERAL_VQA_AGENT_TASKS or persisted_task == "unknown"
            ):
                return self._write_planning_resume_failure(
                    draft,
                    sample_dir,
                    persisted_task=persisted_task,
                    code="LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED",
                )
        return await self._run_draft_visual(draft, sample_dir)


    async def _run_draft_visual(
        self,
        draft: SampleDraft,
        sample_dir: Path,
    ) -> SampleRunStatus:
        """Plan and materialize one draft with the same v5 call as all entries.
        用与所有入口相同的 v5 调用规划并物化一条 draft。"""
        if (
            self.call_budget_factory is None
            or self.data_root is None
            or self.visual_task_planner is None
        ):
            return self._write_planning_failure(
                draft, sample_dir, task=None, code="PLANNER_NOT_ASSEMBLED"
            )
        budget = self.call_budget_factory.create_for_sample("draft")
        task_known: str | None = None
        try:
            plan, views = await self.visual_task_planner.plan_with_views(
                draft,
                data_root=self.data_root,
                artifact_dir=sample_dir,
                budget=budget,
            )
            task_known = plan.task
            sample = materialize_sample(draft, plan.task)
            self.artifact_writer.write_visual_task_plan(
                sample_dir, plan, materialized_views=views
            )
        except VisualTaskPlanError as error:
            return self._write_planning_failure(
                draft, sample_dir, task=None, code=error.code
            )
        except SampleMaterializationError as error:
            return self._write_planning_failure(
                draft, sample_dir, task=task_known, code=error.code
            )
        except Exception as error:
            return self._write_planning_failure(
                draft, sample_dir, task=task_known, code=_stable_error_code(error)
            )
        try:
            outcome = await self.sample_runner.run_one(
                sample,
                sample_dir,
                visual_task_plan=plan,
                visual_views=views,
                budget=budget,
                judge_policy=self._judge_policy_for(sample.sample_id),
            )
        except Exception as error:
            return self._write_planning_failure(
                draft, sample_dir, task=task_known, code=_stable_error_code(error)
            )
        return outcome.status

    def _write_planning_failure(
        self,
        sample: SampleDraft | UnifiedSample,
        sample_dir: Path,
        *,
        task: str | None,
        code: str,
    ) -> SampleRunStatus:
        """Persist a stable planner failure without guessing a task.
        持久化稳定规划失败，绝不猜测 task。"""
        return self._write_draft_failure(
            sample,
            sample_dir,
            task=task,
            code=code,
        )

    def _write_planning_resume_failure(
        self,
        sample: SampleDraft | UnifiedSample,
        sample_dir: Path,
        *,
        persisted_task: str | None = None,
        code: str = "LEGACY_PLANNING_RESUME_UNSUPPORTED",
    ) -> SampleRunStatus:
        """Reject inference reruns for historical planner modes while keeping
        old success supplements model-free. When a persisted status exists, its
        execution task remains authoritative even for a failed resume. 历史规划模式
        只允许无模型成功补判，拒绝重新推理；如果存在持久化状态，即使 resume
        失败也继续使用其中的实际执行 task。"""
        task = (
            persisted_task
            if persisted_task is not None
            else sample.task if isinstance(sample, UnifiedSample) else None
        )
        return self._write_planning_failure(
            sample,
            sample_dir,
            task=task,
            code=code,
        )

    def _write_draft_failure(
        self,
        draft: SampleDraft | UnifiedSample,
        sample_dir: Path,
        *,
        task: str | None,
        code: str,
    ) -> SampleRunStatus:
        """Persist a failed status for a pre-task failure; the task label is
        the known task or the honest sentinel 'unknown' — never a guessed
        general_vqa. Accepts drafts and task-rebuilt samples, which share the
        sample_id. 持久化预 task 失败的 failed 状态；task 标签为已知任务或
        诚实哨兵 'unknown'——绝不猜测 general_vqa。接受 draft 与重建 task
        样本，二者共享 sample_id。"""

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

    def _judge_policy_for(self, sample_id: str) -> str:
        """Deterministic judge participation sampled from the SHA256 of the
        run/sample identity — never random. rate <= 0 disables judge, rate
        >= 1 keeps the configured policy, intermediate rates select a fixed
        subset that is identical across fresh runs and resume.
        由 run/sample 身份的 SHA256 确定性抽样的 judge 参与——绝不随机。
        rate<=0 禁用 judge，rate>=1 保留配置策略，中间值选择固定子集，
        fresh 与 resume 完全一致。"""

        rate = self.judge_sample_rate
        if rate is None or self.judge_policy == "none":
            return self.judge_policy
        if rate <= 0.0:
            return "none"
        if rate >= 1.0:
            return self.judge_policy
        digest = hashlib.sha256(
            f"{self.run_dir.name}:{sample_id}".encode("utf-8")
        ).hexdigest()
        if int(digest[:16], 16) % 10000 < rate * 10000:
            return self.judge_policy
        return "none"

    def _restore_persisted_judge_rate(self, task_dir: Path) -> None:
        """Resume with the same judge sampling policy as the original run:
        the CLI rate wins when explicitly given, otherwise the persisted
        summary rate is restored so resume is identical.
        resume 使用与原运行相同的 judge 抽样策略：显式 CLI rate 优先，
        否则恢复 summary 持久化 rate，使 resume 一致。"""

        if self.judge_sample_rate is not None:
            return
        summary_path = task_dir / "dataset_summary.json"
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return
        value = raw.get("judge_sample_rate") if isinstance(raw, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.judge_sample_rate = float(value)

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
        """Write the missing deterministic evaluation through the same shared
        dispatch a fresh run uses, then re-run a missing or failed VQA judge
        for general_vqa only. Never fabricates metrics to fill files.
        经 fresh run 使用的同一共享分派补写缺失的确定性评估，然后仅对
        general_vqa 重跑缺失或失败的 VQA judge。绝不为了补文件而伪造指标。"""

        filename = evaluation_filename_for_runtime_task(task)
        if filename is None:
            return
        evaluation_path = sample_dir / filename
        if not evaluation_path.is_file():
            payload = self._load_persisted_payload(sample_dir, task)
            # The persisted status.task is the execution task; pass it
            # explicitly so the metric family never follows the canonical
            # resolved sample.task after a candidate fallback.
            # 持久化 status.task 即执行任务；显式传入，使指标族绝不因候选
            # 兜底而跟随 canonical resolved sample.task。
            evaluation, _ = build_deterministic_evaluation(
                sample=sample,
                execution_payload=payload,
                execution_task=task,
            )
            if evaluation is None:
                return  # fail closed: coordinate/geometry/reference mismatch
            self.artifact_writer.write_evaluation(sample_dir, evaluation, filename=filename)
        evaluation_family = evaluation_task_for_runtime_task(task)
        judge_caption = task == "change_caption"
        if evaluation_family == "general_vqa" or judge_caption:
            judge_service = self.sample_runner.judge_service
            if judge_service is not None and self._judge_policy_for(sample.sample_id) != "none":
                judge_method = (
                    judge_service.judge_caption_resume
                    if judge_caption
                    else judge_service.judge_vqa_resume
                )
                evaluation = await asyncio.to_thread(
                    judge_method,
                    sample=sample,
                    candidate_answer="",
                    sample_dir=sample_dir,
                    judge_policy=self.judge_policy,
                    call_budget=None,
                )
                self.artifact_writer.write_evaluation(
                    sample_dir,
                    evaluation,
                    filename=EVALUATION_FILENAME_BY_TASK[evaluation_family],
                )

    def _load_persisted_payload(
        self,
        sample_dir: Path,
        task: str,
    ) -> object:
        """Load the persisted execution payload for a task: counting tasks
        read counting_result.json, every other evaluated task reads
        agent_result.json; a missing or corrupt artifact fails with a stable
        code. 按任务加载持久化执行载荷：计数任务读 counting_result.json，
        其余已评估任务读 agent_result.json；缺失或损坏以稳定 code 失败。"""

        family = evaluation_task_for_runtime_task(task)
        if family == "counting":
            result_path = sample_dir / _COUNTING_RESULT_FILENAME
            model = CountingResult
        elif family is not None:
            result_path = sample_dir / _AGENT_RESULT_FILENAME
            model = AgentResult
        else:
            raise ResumeSupplementError("UNSUPPORTED_EVALUATION_TASK")
        if not result_path.is_file():
            raise ResumeSupplementError("PERSISTED_RESULT_MISSING")
        try:
            return model.model_validate(
                json.loads(result_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError) as exc:
            raise ResumeSupplementError("PERSISTED_RESULT_INVALID") from exc


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


def _not_started_status(item: Any) -> SampleRunStatus:
    """Terminal status for a selected sample that never started under
    fail-fast; keeps summary accounting closed and lets resume re-run it.
    fail-fast 下已选中但从未启动样本的终态；保持汇总记账闭合并允许 resume
    重跑。"""

    task = (
        getattr(item, "task", None)
        or getattr(item, "explicit_task", None)
        or "unknown"
    )
    return SampleRunStatus(
        sample_id=item.sample_id,
        task=task,
        state="skipped",
        error_code="FAIL_FAST_NOT_STARTED",
        error_message="FAIL_FAST_NOT_STARTED",
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
