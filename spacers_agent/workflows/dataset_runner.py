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
from spacers_agent.workflows.sample_runner import SampleRunner
from spacers_agent.workflow import atomic_write_json


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
        settings: Any,
    ) -> None:
        self.adapter = adapter
        self.sample_runner = sample_runner
        self.run_dir = run_dir
        self.settings = settings

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
                _append_jsonl(self.run_dir / "predictions.jsonl", {
                    "sample_id": sample.sample_id, "task": sample.task,
                    "state": status.state, "result_path": str(status.result_path) if status.result_path else None,
                })
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
        atomic_write_json(self.run_dir / "dataset_summary.json", summary.model_dump(mode="json"))
        return summary

    async def _run_sample(self, sample: Any, sample_dir: Path) -> SampleRunStatus:
        """Execute one sample via SampleRunner; translate exceptions to status.
        通过 SampleRunner 执行一条样本；将异常转换为状态。
        """
        from datetime import datetime, timezone
        try:
            await self.sample_runner.run_one(sample, sample_dir)
            status = SampleRunStatus(
                sample_id=sample.sample_id, task=sample.task,
                state="succeeded", updated_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as error:
            status = SampleRunStatus(
                sample_id=sample.sample_id, task=sample.task,
                state="failed", error_code=type(error).__name__,
                error_message=str(error),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        atomic_write_json(sample_dir / "status.json", status.model_dump(mode="json"))
        return status

    async def _resume_judge(self, sample: Any, sample_dir: Path) -> bool:
        """Retry missing/failed VQA judge on resume. / resume 时重试缺失/失败的 VQA 审核。"""
        if getattr(sample, "task", "") != "general_vqa" or not self.sample_runner.judge_service:
            return False
        evaluation_path = sample_dir / "vqa_evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.is_file() else {}
        if evaluation.get("judge_status") == "succeeded":
            return False
        result_path = sample_dir / "expert_result.json"
        if not result_path.is_file():
            return False
        from spacers_agent.schemas import ExpertResult
        result = ExpertResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        try:
            await self.sample_runner._judge_vqa(sample, type("Exec", (), {"payload": result})(), sample_dir)
        except Exception:
            pass
        return True


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


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
