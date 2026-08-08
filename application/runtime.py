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
from workflows.run_store import RunManifest
from workflows.schema import DatasetRunOptions, DatasetRunSummary


def build_dataset_run_options(
    *,
    dataset: str,
    root: Path,
    split: str,
    tasks: tuple[str, ...] = (),
    auto_task: bool = False,
    run_id: str | None = None,
    resume: bool = False,
    limit: int | None = None,
    start_index: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    sample_concurrency: int = 1,
    evaluate: bool = True,
    judge_policy: str = "none",
    fail_fast: bool = False,
) -> DatasetRunOptions:
    """Thin options construction for the public entry point: the architecture
    rule forbids main.py from importing workflows, so construction lives here.
    Validation (task/auto-task exclusivity) comes from DatasetRunOptions
    itself. 公开入口的薄选项构造：架构规则禁止 main.py 导入 workflows，因此
    构造在此完成。互斥校验（task/auto-task）由 DatasetRunOptions 自身承担。"""

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
        evaluate=evaluate,
        judge_policy=judge_policy if evaluate else "none",
        fail_fast=fail_fast,
    )


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

    def build_report(self, run_id: str) -> Report:
        """Build the read-only report for a run id. 为 run id 构建只读报告。"""

        return self.components.build_report(self.settings.runs.root / run_id)
