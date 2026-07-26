"""Dataset-level runner — data iteration, resume, concurrency, summary.
数据集级运行器 — 数据遍历、恢复、并发、汇总。

Delegates per-sample execution to SampleRunner. No agent-specific logic.
将单样本执行委托给 SampleRunner。无 Agent 特定逻辑。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from spacers_agent.dataset_adapters import DatasetAdapter
from spacers_agent.schemas import DatasetRunSummary, SampleRunStatus
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.sample_runner import SampleRunner, failed_sample_status
from spacers_agent.workflows.artifact_writer import atomic_write_json


class DatasetRunner:
    """Iterate samples, call SampleRunner, collect results. No agent knowledge.
    遍历样本、调用 SampleRunner、收集结果。不包含 Agent 知识。
    """

    def __init__(
        self,
        adapter: DatasetAdapter,
        sample_runner: SampleRunner,
        *,
        run_dir: Path,
        settings: AppSettings,
        judge_policy: str = "none",
    ) -> None:
        self.adapter = adapter
        self.sample_runner = sample_runner
        self.run_dir = run_dir
        self.settings = settings
        self.judge_policy = judge_policy
        self.artifact_writer = sample_runner.artifact_writer

    async def run(
        self, *, split: str, task: str,
        resume: bool = False, limit: int | None = None,
        shard_index: int = 0, shard_count: int = 1,
        start_index: int = 0, sample_ids: set[str] | None = None,
        fail_fast: bool = False, sample_concurrency: int = 1,
    ) -> DatasetRunSummary:
        """Run selected samples and keep every failure visible. / 运行选中样本并保持每个失败可见。"""

        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("invalid shard selection")
        if sample_concurrency < 1:
            raise ValueError("sample_concurrency must be positive")

        # Probe adapter and record observed layout / 探测适配器并记录观察到的布局
        probe = self.adapter.probe(self.settings.paths.dataset_root)
        _update_manifest_probe(self.run_dir, probe)

        statuses: list[SampleRunStatus] = []
        pending: dict[asyncio.Task[SampleRunStatus], Any] = {}

        async def collect_one() -> bool:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            stop = False
            for future in done:
                sample = pending.pop(future)
                status = await future
                statuses.append(status)
                self.artifact_writer.append_prediction(
                    self.run_dir,
                    sample_id=sample.sample_id,
                    task=sample.task,
                    status=status,
                )
                stop = stop or (fail_fast and status.state == "failed")
            return stop

        selected = 0
        for index, sample in enumerate(self.adapter.iter_samples(
            self.settings.paths.dataset_root, split, task,
        )):
            if index < start_index or _shard_for_sample(sample.sample_id, shard_count) != shard_index:
                continue
            if sample_ids is not None and sample.sample_id not in sample_ids:
                continue
            if limit is not None and selected >= limit:
                break
            selected += 1

            sample_dir = self.run_dir / "samples" / sample.sample_id
            status_path = sample_dir / "status.json"

            if resume and status_path.is_file():
                previous = SampleRunStatus.model_validate_json(status_path.read_text(encoding="utf-8"))
                if previous.state == "succeeded":
                    if await self._resume_judge(sample, sample_dir):
                        statuses.append(previous)
                        continue
                    statuses.append(previous.model_copy(update={"state": "skipped"}))
                    continue

            pending[asyncio.create_task(self._run_sample(sample, sample_dir))] = sample
            if len(pending) >= sample_concurrency and await collect_one():
                break

        while pending:
            if await collect_one():
                for future in pending:
                    future.cancel()
                break

        summary = DatasetRunSummary(
            run_id=self.run_dir.name, dataset=getattr(self.adapter, "name", "unknown"),
            split=split, task=task, total=len(statuses),
            succeeded=sum(s.state == "succeeded" for s in statuses),
            partial=sum(s.state == "partial" for s in statuses),
            failed=sum(s.state == "failed" for s in statuses),
            skipped=sum(s.state == "skipped" for s in statuses),
        )
        self.artifact_writer.write_summary(self.run_dir, summary)
        return summary

    async def _run_sample(self, sample: Any, sample_dir: Path) -> SampleRunStatus:
        """Execute one sample via SampleRunner; translate exceptions to status.
        通过 SampleRunner 执行一条样本；将异常转换为状态。
        """
        try:
            outcome = await self.sample_runner.run_one(
                sample,
                sample_dir,
                judge_policy=self.judge_policy,
            )
            status = outcome.status
        except Exception as error:
            status = failed_sample_status(sample, error)
            self.artifact_writer.write_final_status(sample_dir, status)
        return status

    async def _resume_judge(self, sample: Any, sample_dir: Path) -> bool:
        """Retry missing/failed VQA judge on resume via JudgeService.
        通过 JudgeService 在 resume 时重试缺失/失败的 VQA 审核。
        """

        if (
            getattr(sample, "task", "") != "general_vqa"
            or self.judge_policy == "none"
            or self.sample_runner.judge_service.judge_client is None
        ):
            return False
        try:
            evaluation = await self.sample_runner.judge_service.judge_vqa_resume(
                sample=sample,
                candidate_answer="",
                sample_dir=sample_dir,
                judge_policy=self.judge_policy,
                call_budget=self.sample_runner.call_budget_factory.create_for_sample(
                    getattr(sample, "task", "general_vqa")
                ),
            )
            evaluation_payload = (
                evaluation.model_dump(mode="json")
                if hasattr(evaluation, "model_dump")
                else evaluation
            )
            self.artifact_writer.write_evaluation(
                sample_dir,
                evaluation_payload,
                filename="vqa_evaluation.json",
            )
            trace_path = sample_dir / "agent_trace.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.is_file() else {}
            judge_status = getattr(evaluation, "judge_status", None)
            if isinstance(evaluation, dict):
                judge_status = evaluation.get("judge_status")
            trace["judge_status"] = judge_status or "failed"
            self.artifact_writer.write_trace(sample_dir, trace)
            return True
        except FileNotFoundError as error:
            self._persist_resume_judge_error(sample, sample_dir, error)
            return False
        except Exception as error:
            self._persist_resume_judge_error(sample, sample_dir, error)
            return False

    def _persist_resume_judge_error(self, sample: Any, sample_dir: Path, error: Exception) -> None:
        """Persist a failed resume Judge attempt instead of silently discarding it.
        持久化失败的 resume Judge 尝试，绝不静默丢弃。
        """

        evaluation = {
            "sample_id": sample.sample_id,
            "judge_status": "failed",
            "judge_error": f"{type(error).__name__}: {error}",
        }
        self.artifact_writer.write_evaluation(
            sample_dir,
            evaluation,
            filename="vqa_evaluation.json",
        )
        trace_path = sample_dir / "agent_trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.is_file() else {}
        trace["judge_status"] = "failed"
        trace["judge_error"] = evaluation["judge_error"]
        self.artifact_writer.write_trace(sample_dir, trace)


def _update_manifest_probe(run_dir: Path, probe: Any) -> None:
    """Record dataset probe results in the manifest. / 在 manifest 中记录数据集探测结果。"""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_probe"] = {
        "dataset": probe.dataset, "version": probe.version,
        "sample_file": str(probe.sample_file),
        "observed_fields": list(probe.observed_fields),
        "sample_count": probe.sample_count,
    }
    atomic_write_json(manifest_path, manifest)


def _shard_for_sample(sample_id: str, shard_count: int) -> int:
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % shard_count
