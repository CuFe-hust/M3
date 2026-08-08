"""High-level runtime use cases: run a dataset and build a report. No product
entry points yet (no ask / serve / CLI). The runtime never writes the dataset
loop — it only delegates to DatasetRunner per task.

高层运行时用例：运行数据集与构建报告。尚无产品入口（无 ask / serve /
CLI）。Runtime 不写数据集循环——只按 task 委托 DatasetRunner。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from application.bootstrap import RuntimeComponents, assemble_runtime
from application.settings import AppSettings, load_settings
from data.registry import DatasetRegistry, build_default_registry
from reporting.schema import Report
from workflows.schema import DatasetRunOptions, DatasetRunSummary


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
        )
        return cls(settings=resolved_settings, components=components)

    async def run_dataset(
        self,
        options: DatasetRunOptions,
    ) -> dict[str, DatasetRunSummary]:
        """Run one dataset: create the run manifest when missing, then
        delegate each task (or the auto-task namespace) to a DatasetRunner.
        Judge policy only applies when evaluate is enabled.
        运行一个数据集：缺失时创建 run manifest，然后把每个 task（或
        auto-task 命名空间）委托给 DatasetRunner。仅 evaluate 启用时应用
        judge 策略。"""

        run_id = options.run_id or f"{options.dataset}-{options.split}"
        run_dir = self.settings.runs.root / run_id
        if not run_dir.exists():
            self.components.run_store.create_run(
                config_payload=self.settings.to_config_payload(),
                model_ids={
                    "qwen": self.settings.models.qwen.effective_cache_model_id,
                    "deepseek": self.settings.models.deepseek.model,
                },
                prompt_paths=self.components.prompt_catalog.snapshot_paths(),
                run_id=run_id,
                dataset=options.dataset,
                split=options.split,
                sample_filter=(
                    ",".join(sorted(options.sample_ids))
                    if options.sample_ids
                    else None
                ),
            )
        adapter = self.registry.get(options.dataset)
        judge_policy = options.judge_policy if options.evaluate else "none"
        tasks: list[str | None] = [None] if options.auto_task else list(options.tasks)
        results: dict[str, DatasetRunSummary] = {}
        for task in tasks:
            runner = self.components.dataset_runner_factory(
                adapter,
                run_dir,
                judge_policy=judge_policy,
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
        return results

    def build_report(self, run_id: str) -> Report:
        """Build the read-only report for a run id. 为 run id 构建只读报告。"""

        return self.components.build_report(self.settings.runs.root / run_id)
