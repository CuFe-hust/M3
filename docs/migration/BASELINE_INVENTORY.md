# BASELINE_INVENTORY.md — try_yolo 迁移基线清单

> Task 00 产物。记录迁移开始前 `try_yolo` 锁定提交的受跟踪文件、公开 CLI、测试基线、
> 运行产物文件名与核心 Schema 定义位置。本清单只读记录，不修改任何生产代码。

## 1. 锁定基线确认

- 仓库: `CuFe-hust/M3`
- 参考分支: `try_yolo`
- 锁定提交: `ec962eb87c3ad0b8c1502efcbd08db0daec48868`
- 本地验证: 分支 `try_yolo` HEAD 与该提交一致；远程 `refs/heads/try_yolo` 指向同一提交 ✅
- 验证环境: Windows + `C:\Users\TZDEZACR\miniconda3\envs\m3` (Python 3.11.15)
- 工作区注意: `code/.gitignore` 存在用户本地修改（非本任务改动，未触碰）

## 2. 受 Git 跟踪的文件清单

### 2.1 顶层受跟踪条目

```text
.env.example  .gitattributes  .gitignore  AGENTS.md  CLAUDE.md  DETAILS.md
README.md  main.py  pyproject.toml  requirements.txt  requirements-models.txt
.github/  configs/  data/  docs/  eval/  models/  prompts/  scripts/  spacers_agent/  tests/
```

### 2.2 受跟踪 Python 文件（非 tests，共 98 个）

```text
main.py
data/__init__.py  data/downloader.py  data/loader.py  data/loaders.py
data/schema.py  data/validator.py
eval/__init__.py  eval/audit_report.py  eval/metrics.py  eval/standard_adapter.py
models/__init__.py  models/base.py  models/entry.py  models/qwen3vl.py
models/qwen_transformers.py
models/qwen3_5/__init__.py  models/qwen3_5/model.py
models/qwen3_vl/__init__.py  models/qwen3_vl/baseline.py
scripts/evaluate_levir_harmonization.py
spacers_agent/__init__.py  spacers_agent/application.py  spacers_agent/bootstrap.py
spacers_agent/cli.py  spacers_agent/clients/__init__.py  spacers_agent/clients/deepseek.py
spacers_agent/clients/mock.py  spacers_agent/commands/__init__.py
spacers_agent/commands/common.py  spacers_agent/commands/count_image.py
spacers_agent/commands/run_dataset.py  spacers_agent/counting_report.py
spacers_agent/data_audit.py  spacers_agent/dataset_adapters.py  spacers_agent/errors.py
spacers_agent/evaluation.py  spacers_agent/events.py  spacers_agent/imaging.py
spacers_agent/prompt_catalog.py  spacers_agent/reporting.py
spacers_agent/run_store.py  spacers_agent/schemas.py  spacers_agent/seam.py
spacers_agent/settings.py  spacers_agent/targeting.py  spacers_agent/visualization.py
spacers_agent/vqa_geometry.py  spacers_agent/vqa_report.py
spacers_agent/agents/__init__.py  spacers_agent/agents/base.py
spacers_agent/agents/errors.py  spacers_agent/agents/registry.py
spacers_agent/agents/visual_base.py
spacers_agent/agents/caption/__init__.py  spacers_agent/agents/caption/agent.py
spacers_agent/agents/change/__init__.py  spacers_agent/agents/change/agent.py
spacers_agent/agents/change/difference_proposal.py
spacers_agent/agents/change/harmonizer.py  spacers_agent/agents/change/pair_validator.py
spacers_agent/agents/change/preprocess.py  spacers_agent/agents/change/reviewer.py
spacers_agent/agents/change/schemas.py
spacers_agent/agents/counting/__init__.py  spacers_agent/agents/counting/agent.py
spacers_agent/agents/counting/evidence.py  spacers_agent/agents/counting/point_pipeline.py
spacers_agent/agents/counting/target_parser.py
spacers_agent/agents/counting/backends/__init__.py
spacers_agent/agents/counting/backends/base.py
spacers_agent/agents/counting/backends/qwen_point.py
spacers_agent/agents/counting/backends/registry.py
spacers_agent/agents/counting/backends/selector.py
spacers_agent/agents/counting/backends/vrsbench_qwen_count.py
spacers_agent/agents/counting/backends/yolo_adapter.py
spacers_agent/agents/counting/backends/yolo_model_store.py
spacers_agent/agents/counting/backends/yolo_obb.py
spacers_agent/agents/counting/backends/yolov5_obb_onnx.py
spacers_agent/agents/general_vqa/__init__.py  spacers_agent/agents/general_vqa/agent.py
spacers_agent/agents/grounding/__init__.py  spacers_agent/agents/grounding/agent.py
spacers_agent/agents/spatial/__init__.py  spacers_agent/agents/spatial/agent.py
spacers_agent/agents/spatial/candidate_review.py  spacers_agent/agents/spatial/evidence_merge.py
spacers_agent/routing/__init__.py  spacers_agent/routing/budget.py
spacers_agent/routing/policies.py  spacers_agent/routing/router.py
spacers_agent/routing/routes.py  spacers_agent/routing/schemas.py
spacers_agent/workflows/__init__.py  spacers_agent/workflows/artifact_writer.py
spacers_agent/workflows/dataset_runner.py  spacers_agent/workflows/judge_service.py
spacers_agent/workflows/sample_runner.py
```

### 2.3 受跟踪测试文件（tests/，共 79 个 .py）

```text
tests/__init__.py
tests/test_agent_registry.py  tests/test_backend_registry.py  tests/test_baseline_adapters.py
tests/test_baseline_audit_report.py  tests/test_counting_report.py  tests/test_dataset_audit.py
tests/test_dataset_validator.py  tests/test_downloader.py  tests/test_loader.py
tests/test_model_entry.py  tests/test_multiagent_vqa_pipeline.py  tests/test_packaging.py
tests/test_phase1_foundation.py  tests/test_phase2_clients.py  tests/test_phase3_geometry.py
tests/test_phase4_point_counting.py  tests/test_phase5_routing.py
tests/test_phase6_deepseek_evaluation.py  tests/test_phase7_operations.py
tests/test_qwen3vl_local_loading.py  tests/test_routing_package.py  tests/test_schema.py
tests/test_stage_a_to_g_contracts.py  tests/test_standard_eval_integration.py
tests/test_targeting_and_seam.py  tests/test_vrsbench_vqa_geometry.py
tests/test_yolo_backend.py  tests/test_yolov5_obb_onnx.py
tests/agents/__init__.py  tests/agents/test_agent_contract.py  tests/agents/test_registry.py
tests/agents/caption/test_caption_agent.py
tests/agents/change/test_change_agent.py  tests/agents/change/test_difference_proposal.py
tests/agents/change/test_harmonizer.py  tests/agents/change/test_pair_validator.py
tests/agents/counting/__init__.py  tests/agents/counting/test_status_propagation.py
tests/agents/counting/test_target_selection.py  tests/agents/counting/test_vrsbench_yolo_result.py
tests/agents/counting/test_yolo_runtime.py
tests/agents/general_vqa/test_general_vqa_agent.py
tests/agents/grounding/test_grounding_agent.py
tests/agents/spatial/test_candidate_review.py  tests/agents/spatial/test_evidence_merge.py
tests/agents/spatial/test_spatial_agent.py
tests/architecture/test_cli_contract.py  tests/architecture/test_dependency_boundaries.py
tests/architecture/test_no_legacy_agent_api.py  tests/architecture/test_persisted_schema_contracts.py
tests/architecture/test_routing_coverage.py
tests/cli/test_count_image_runtime.py  tests/cli/test_resume_run_contract.py
tests/entry/__init__.py  tests/entry/test_application.py  tests/entry/test_http_service.py
tests/entry/test_main_execution.py  tests/entry/test_main_parser.py
tests/entry/test_manual_images.py  tests/entry/test_single_model_load.py
tests/parity/__init__.py  tests/parity/canonicalize.py  tests/parity/fake_clients.py
tests/parity/fixture_harness.py  tests/parity/test_native_counting_parity.py
tests/parity/test_visual_base_parity.py  tests/parity/test_vrsbench_counting_parity.py
tests/routing/test_router.py  tests/routing/test_router_routing.py
tests/runtime/__init__.py  tests/runtime/test_artifact_writer.py
tests/runtime/test_bootstrap_wiring.py  tests/runtime/test_budget_factory.py
tests/runtime/test_runtime_no_yolo.py  tests/runtime/test_status_mapping.py
tests/workflows/test_judge_service_errors.py  tests/workflows/test_sample_runner.py
tests/workflows/test_sample_runner_fallback.py  tests/workflows/test_sample_runner_status.py
```

另有测试子目录：`tests/fixtures/`（含 `tests/parity/fixtures/` 下 20 个任务 fixture 目录）。

### 2.4 其他受跟踪资产

- prompts/ 共 31 个版本化 Prompt（`caption_v1` … `spatial_v5`，含 `count_tile_v1..v4`、
  `missing_point_review_v1..v3`、`spatial_candidate_review_v1..v5`、`router_v1` 等）
- configs/: `default.yaml`、`yolo.example.yaml`
- docs/: 修改记录 `docs/changes/`、架构文档 `docs/architecture/`、runbook 等
- .github/workflows/: `offline-tests.yml`

## 3. 公开 CLI 与退出码

### 3.1 唯一公开入口 `python main.py`（详见 BASELINE_COMMANDS.txt）

| 命令 | 参数要点 | help 退出码 |
|---|---|---|
| `main.py --help` | `--config`（默认 configs/default.yaml）；子命令 `{serve,ask,run-dataset}`，默认 `serve` | 0 |
| `main.py serve --help` | `--host`（默认 127.0.0.1）、`--port`（默认 8000） | 0 |
| `main.py ask --help` | `--images-dir`（必填）、`--question`、`--task`（auto/counting/fine_grained_counting/change_caption/change_qa/grounding/spatial_relation/scene_classification/general_vqa/caption/multiple_choice_vqa）、`--output` | 0 |
| `main.py run-dataset --help` | `--dataset`（LEVIR-CC/VRSBench/MME-RealWorld/XLRS-Bench-lite）、`--root`、`--split`、`--task`、`--run-id`、`--max-samples`、`--start-index`、`--sample-concurrency`、`--resume`、`--fail-fast`、`--evaluate/--no-evaluate`（默认 evaluate）、`--judge-policy`（none/errors-only/all，默认 all） | 0 |

### 3.2 内部维护 CLI `python -m spacers_agent.cli`

命令：`run-init`、`health`（qwen/deepseek，`--live`）、`list-datasets`、`smoke-qwen`、
`count-image`、`run-dataset`、`resume-run`、`evaluate-run`（`--deepseek`）、
`judge-vqa-run`、`standard-evaluate`、`inspect-data`、`render-count`、`summarize-evaluations`。

## 4. pytest 与离线测试基线

- 收集数量: **478 tests**（`python -m pytest --collect-only -q`）
- 离线全量结果（本地 Windows + m3 环境，`--basetemp=tmp/pytest-basetemp`）:
  **477 passed, 1 failed**
- 唯一失败: `tests/test_dataset_validator.py::test_levir_cc_missing_image_side_fails`
  - 原因: 断言 `"test/B" in str(error)`，Windows 上 `Path` 渲染为 `test\B`（反斜杠），
    属于平台路径分隔符差异；CI（ubuntu）不受影响。
- 首次运行在默认临时目录时出现 `PermissionError`（Windows 对
  `AppData\Local\Temp\pytest-of-*` 的访问问题），改用工作区内 `--basetemp` 后全部正常执行；
  这是本地环境限制，与代码无关。

### 4.1 未运行的 live 验证（明确标注原因）

| 项 | 原因 |
|---|---|
| `live_qwen` / `live_deepseek` / `live_dataset` 标记测试 | 需要显式授权的模型服务、DeepSeek API key 或真实数据集，本任务不运行 |
| `python main.py serve` 实际启动 | 需加载本地 Qwen 模型与权重，未授权 |
| `python main.py ask` 实际推理 | 需模型与图片，未授权 |
| `python main.py run-dataset` 实际运行 | 需真实数据集根目录，未授权 |
| 内部 CLI live 命令（`health --live`、`smoke-qwen` 等） | 需授权端点/模型，未运行 |

## 5. 运行产物文件名清单

### 5.1 run 级（`outputs/runs/<run-id>/`）

| 文件 | 写入方 |
|---|---|
| `manifest.json` | `spacers_agent/run_store.py`（含 git commit/dirty、config hash、prompt 快照信息） |
| `config.snapshot.yaml` | `spacers_agent/run_store.py` |
| `prompts.snapshot/` | `spacers_agent/run_store.py`（Prompt 资产快照目录） |
| `events.jsonl` | `spacers_agent/events.py`（RUN_CREATED 等事件） |
| `predictions.jsonl` | `spacers_agent/workflows/artifact_writer.py` |
| `dataset_summary.json` | `spacers_agent/workflows/artifact_writer.py` |
| `evaluation.json` / `evaluation_records.json` / `evaluations.jsonl` | `spacers_agent/cli.py::_evaluate_run`（计数评估输出） |
| `deepseek_cache/` | `JsonResponseCache(run_dir / "deepseek_cache")`（`spacers_agent/cli.py`） |

### 5.2 sample 级（`outputs/runs/<run-id>/samples/<sample-id>/`）

| 文件 | 写入方 |
|---|---|
| `sample.json` | `artifact_writer.py::write_sample` |
| `status.json` | `artifact_writer.py::write_status` |
| `routing_decision.json` | `artifact_writer.py::write_routing` |
| `agent_result.json` | Agent `AgentExecution(result_filename="agent_result.json")`（caption/change/grounding/general_vqa/spatial） |
| `counting_result.json` | `CountingAgent` `AgentExecution(result_filename="counting_result.json")` |
| `agent_trace.json` | `artifact_writer.py::write_trace` |
| `vqa_evaluation.json` | `dataset_runner.py` / `sample_runner.py` / `judge_service.py`（VQA 与 resume Judge 评估） |

### 5.3 报告级

| 文件 | 写入方 |
|---|---|
| `<result-stem>.report/report.html` | `eval/audit_report.py`（baseline 审计报告）、`spacers_agent/vqa_report.py`（`vrsbench_vqa.report/report.html`）、`spacers_agent/counting_report.py`（`counting.report/report.html`） |
| `<result-stem>.metadata.json` | baseline 推理（`vqa_report.py` / `counting_report.py`） |
| `<result-stem>.report/samples.csv`、`samples.jsonl` | `eval/audit_report.py` |
| `<result-stem>.report/deepseek_audit.jsonl` | `eval/audit_report.py`（--deepseek-proxy 时） |
| `<result-stem>.standard.json` | `eval/standard_adapter.py`（external evaluator 输出） |
| `mme_real_rs.official.json` | baseline MME 推理（官方记录 Output 替换） |
| `outputs/dataset_audit.json` | `inspect-data` 命令 |

## 6. 公开导入与核心 Schema 定义位置

### 6.1 核心数据 Schema

| 符号 | 定义位置 |
|---|---|
| `CanonicalSample`、`CanonicalPrediction` | `data/schema.py`（外部兼容/报告记录） |
| `UnifiedSample`、`ImageRef`、`GroundTruth`、`VisualEvidence`、`AgentResult` | `spacers_agent/schemas.py` |
| `PixelRect`、`TileSpec`、`LocalPointObservation`、`GlobalPointObservation`、`PointProvenance` | `spacers_agent/schemas.py` |
| `CountTargetSpec`、`TileCountResponse`、`CountingDraft`、`CountingResult`、`IssueRecord` | `spacers_agent/schemas.py` |
| `SampleRunStatus`、`DatasetRunSummary`、`TargetParseResult` | `spacers_agent/schemas.py` |
| `YoloDetectorSettings`、`YoloCountingSettings`、`BackendConfig` | `spacers_agent/schemas.py` |
| `RoutingDecision` | `spacers_agent/routing/schemas.py` |
| `DatasetRunOptions`、`SampleRunOutcome`、`SampleRunner` | `spacers_agent/workflows/sample_runner.py` |
| `CountDeterministicMetrics`、`DeepSeekJudgeResult`、`VQAAnswerJudgeResult`、`VQAEvaluationRecord`、`CountJudgeResult`、`EvaluationRecord` | `spacers_agent/evaluation.py` |
| `RequestMeta`、`VisionLanguageClient`、`CacheEntry`、`JsonResponseCache` | `models/base.py` |

### 6.2 关键运行时入口

| 符号 | 定义位置 |
|---|---|
| `main` / `build_parser`（公开入口） | `main.py` |
| `build_parser` / `main`（内部 CLI） | `spacers_agent/cli.py` |
| `RuntimeApplication`（create/ask/health_payload）、`PublicAnswer`、`CollectedImage`、`run_http_server`、`run_dataset_command` | `spacers_agent/application.py` |
| `RuntimeComponents`、`assemble_runtime` | `spacers_agent/bootstrap.py` |
| `create_model` | `models/entry.py` |
| `QwenTransformersClient` | `models/qwen_transformers.py` |
| `DeepSeekJudgeClient` | `spacers_agent/clients/deepseek.py` |
| `PromptCatalog`、`PromptAsset` | `spacers_agent/prompt_catalog.py` |
| `CallBudgetFactory` | `spacers_agent/routing/budget.py` |
| `TaskRouter` | `spacers_agent/routing/router.py` |
| `AgentRegistry`、`AgentContext`、`AgentExecution` | `spacers_agent/agents/base.py`、`spacers_agent/agents/registry.py` |
| `PointCountingOrchestrator` | `spacers_agent/agents/counting/point_pipeline.py` |
| `RunStore` | `spacers_agent/run_store.py` |
| `run_dataset`（唯一数据集循环实现） | `spacers_agent/commands/run_dataset.py` |
| `load_settings`、`AppSettings` | `spacers_agent/settings.py` |

## 7. 已知限制 / 后续注意

- 本地环境存在用户未提交改动（`code/.gitignore`），清单以锁定提交 `ec962eb` 为准。
- 迁移时目标结构（顶层 `data/models/agents/routing/workflows/evaluation/reporting/application`）
  必须保留上述产物文件名与公开导入契约；`spacers_agent/` 与旧 `eval/` 最终删除。
- 477/478 的通过率以本机 Windows 环境为准；CI（ubuntu）预期 478/478（路径分隔符差异仅影响本机）。

## 8. 分支关系与锁定参考（Task 03.5 补充）

- `new_structure` 与 `try_yolo` **没有共同祖先**（本分支从空仓库重建），
  功能是否遗漏必须依赖本清单、Golden fixtures 与行为测试判断，不能依赖 Git diff。
- 参考提交：`ec962eb87c3ad0b8c1502efcbd08db0daec48868`
  （`try_yolo` 分支后续可前进；如需锁定 checkout 可用
  `git worktree add <tmp> ec962eb` 提供只读工作树）。
- 本地绝对环境路径（如 `C:\Users\TZDEZACR\miniconda3\envs\m3`、`code/` 目录）
  仅作为**非稳定生成环境备注**，不属于任何契约；可复现生成器见
  `scripts/generate_migration_fixtures.py`。

### 8.1 必须保持

- 数据集原始字段转换后的关键事实（question、ground truth、图片顺序与角色）；
- sample ID 稳定性；图片顺序；ground truth；
- 最终任务专家选择（同一问题规范化到同一标准任务）；
- 运行产物文件名与指标稳定字段（status/trace/evaluation/report record 形状）；
- 三种样本状态（succeeded / partial / failed）的持久化语义。

### 8.2 有意改变

- VRSBench 语义任务判断从运行时 Router **前移到 Adapter**；
  `router_source` 从 `vrsbench_semantic_rule` 改为 `normalized_task_policy`，
  不再作为最终稳定契约锁定；
- `CanonicalSample`/`CanonicalPrediction` 不再是内部主 Schema（仅外部兼容记录）；
- 新代码不再保留 `spacers_agent` 兼容层（本分支从零重建，旧包目录永久禁止）；
- `stable_sample_id` 升级为多图版（含 dataset/split/有序图片路径），
  不安全源 ID 改由稳定摘要替代。
