# 01 — 旧功能 / 新状态 / 缺口矩阵（只读审计）

> Task 01 产物。只读审计：确认旧实现（`try_yolo`）功能清单、新架构
> （`new_structure`）当前状态，以及两者之间的**真正缺口**，为后续任务
> 02–09 提供输入。本任务未修改任何生产代码、白名单与测试。
>
> 审计时间：2026-08-08。所有结论均基于下面的只读命令与文件读取证据。

## 1. 仓库状态快照

| 项 | 值 | 证据 |
|---|---|---|
| 旧仓库 HEAD | `b86d02bd6d868eac5e0e1d6e4cadf66a9030141e`（`try_yolo`），工作树干净 | `git status --short` 空、`git rev-parse HEAD` |
| 冻结行为基线 | `ec962eb87c3ad0b8c1502efcbd08db0daec48868`（存在于旧仓库，`cat-file -t` 返回 `commit`） | 同上 |
| 旧 HEAD 与基线差异 | **仅 2 个提交**：`9403d25`（SegFormer MiT-B2 OEM/iSAID 权重，Git LFS）+ `b86d02b`（类标签映射修正） | `git log ec962eb8..b86d02bd` |
| 新仓库 HEAD | `a1ed20b319273c15fbf84c54cd159fe9e46d5daf`（分支 `new_structure`），工作树干净 | 同上（与计划起点一致，无需适配） |
| 新仓库全量测试基线 | **1190 passed**（38.2s，2 个无关 warning） | `python -m pytest -q --basetemp=tmp/pytest-basetemp` |
| 实施状态 | `implementation_status.json`：Task 0–33 + 13 个 hardening 全部完成，`pending_files` 为空 | 直接读取 |

**结论**：旧 HEAD 相对冻结基线的增量属于分割实验（OEM/iSAID SegFormer），
归入计划任务 12 独立扩展线，不影响核心能力链（Judge→Runner→Reporting→
Application→Entry）的审计结论。本审计以冻结基线 `ec962eb8` 的行为为对照
（与 Golden fixtures 一致），以旧 HEAD 为文件级参考。

## 2. 新架构现状总览

已实现（`implementation_status.json` implemented_files，共 108 个生产 .py）：

- **data/**：`UnifiedSample`/`ImageRef`/`GroundTruth`/`TaskNormalization`、
  校验/选择/审计、4 个数据集 Adapter（LEVIR-CC/MME-RealWorld/VRSBench/XLRS）
  + `DatasetAdapter` 协议（`probe()` / `iter_samples()` / `AdapterProbe`）
- **models/**：协议/缓存/图像工具/纯声明配置、统一工厂、Qwen Transformers
  客户端、Qwen3-VL 基线（旧 `models/qwen3vl.py` 由 `models/qwen3_vl/baseline.py` 替代）
- **agents/**：契约/Registry/错误/VisualAgentBase + general_vqa/caption/
  grounding + 计数子系统（全）+ 空间子系统（全）+ 变化子系统（全）
- **routing/**：同步确定性 Thin Router（不读 question、不调用模型）
- **workflows/**：`CallBudget`（含 `reserve_deepseek`）、
  `CallBudgetFactory.create_for_sample`、`EventWriter`/`RunStore`、
  `ArtifactWriter`（全部旧产物文件名常量与方法就位）、`TaskResolver`
  （explicit→rule→model 三路径 + 低置信度结构化候选）、运行契约
  `SampleRunState`（含 `skipped`）/`SampleRunStatus`/`DatasetRunSummary`/
  `DatasetRunOptions`/`SampleRunOutcome`（与旧契约字段一致）
- **evaluation/**：统一 `EvaluationRecord`（judge 旁路字段
  `judge_status/judge_raw/judge_parsed/judge_inconsistency/judge_error` 已
  在记录层就位）、四类确定性指标 + aggregate、`merge_count_evaluation`/
  `merge_vqa_evaluation` 的 judge 合并语义已实现（judge 永不覆盖确定性指标）
- **prompts/**：16 个现行版本 Prompt 已迁移（含 3 个 deepseek judge prompt）
- **tests/**：1190 个测试全绿，含 Golden parity 与架构守卫

已批准但**尚未创建**（白名单未来路径，`implementation_status.json` 未声明）：

- `main.py`；`workflows/{judge_service,sample_runner,dataset_runner}.py`
- `evaluation/judges/{__init__,base,deepseek}.py`、
  `evaluation/standard/{__init__,adapter}.py`、
  `evaluation/datasets/{__init__,vrsbench}.py`
- `reporting/{__init__,schema,adapters,builder,html,exporters,visualization}.py`
- `application/{__init__,settings,prompts,runtime,bootstrap}.py` +
  `application/commands/` 下 13 个命令文件（含 `run_dataset.py`）

**空目录**：`reporting/`、`application/`（含 `commands/`）、
`evaluation/judges/`、`evaluation/standard/`、`evaluation/datasets/`、
`docs/architecture/`、`tests/application/`、`tests/reporting/`。
`configs/*.yaml`（default/local/spark）均为 **0 字节空文件**。

## 3. 缺口矩阵（按能力链分组）

判定列：`已覆盖` = 新实现已具备该能力；`缺口` = 需本轮恢复/新建；
`不恢复` = 明确不在本轮范围。

### 3.1 Judge / Evaluation 链 → 任务 03

| 旧功能 | 旧位置（基线） | 新状态 | 判定 |
|---|---|---|---|
| DeepSeek judge 客户端（纯文本强制、重试/修复一次/退避、缓存、产物 request_meta/raw_response/validation/parsed） | `spacers_agent/clients/deepseek.py` | `evaluation/judges/` 空；`evaluation/judges/{base,deepseek}.py` 已批未建 | **缺口**：judge 客户端 + judge 载荷类型（`DeepSeekJudgeResult`/`VQAAnswerJudgeResult`/`CountJudgeResult`） |
| JudgeService（`judge_counting`/`judge_vqa`/`judge_vqa_resume`；policy none/errors-only/all；`reserve_deepseek` 预算；judge 异常不覆盖确定性记录） | `workflows/judge_service.py` | 白名单已批未建；`CallBudget.reserve_deepseek` 与 `make_budget_guard` 已就位 | **缺口** |
| judge 旁路记录字段与合并语义 | `spacers_agent/evaluation.py` | `evaluation/records.py` 已含全部 judge 字段；`merge_count_evaluation`/`merge_vqa_evaluation` 已实现"judge 只旁路、冲突显式标记（`judge_inconsistency`）" | 已覆盖（记录层）；judge 载荷类型按设计解耦到 `evaluation/judges` |
| 确定性指标（counting/VQA/grounding/caption + aggregate） | `spacers_agent/evaluation.py`、`eval/metrics.py` | `evaluation/metrics/*` 完成并有测试 | 已覆盖 |
| DeepSeek judge prompts（`deepseek_judge_v1`/`deepseek_vqa_judge_v1`/`deepseek_judge_repair_v1`/`json_repair_v1`） | `prompts/` | 4 个文件均已迁移 | 已覆盖 |
| 外部标准评测器适配（subprocess） | `eval/standard_adapter.py` | `evaluation/standard/adapter.py` 已批未建 | **暂缓**：仅 CLI `standard-evaluate` 使用，非本轮核心链；白名单路径保留待未来 |
| VRSBench 官方评估数据解析 | （旧无直接对应） | `evaluation/datasets/vrsbench.py` 已批未建 | **暂缓**：本轮用途未定义，见开放问题 O3 |

### 3.2 SampleRunner → 任务 04

| 旧功能 | 旧位置 | 新状态 | 判定 |
|---|---|---|---|
| 单样本执行序列（写 sample → running 状态 → 每样本 CallBudget → high_resolution 判定 → 路由 → 写 routing_decision → 组装 AgentContext → 执行 → 写执行/状态/trace） | `workflows/sample_runner.py` | `workflows/sample_runner.py` 已批未建；**全部支撑已就位**：`ArtifactWriter`（含 `SAMPLE_FILENAME`…`AGENT_TRACE_FILENAME` 与方法）、`CallBudgetFactory`、`TaskResolver`、`workflows/schema.py` 契约类型 | **缺口**：仅剩 SampleRunner 实现 + 测试 |
| 三态映射（completed/completed_with_warnings→succeeded、partial→partial、failed→failed）与 `failed_sample_status` | 同上 | 无对应实现 | **缺口** |
| fallback 双触发（primary 异常 + execution_mode=fallback；`fallback_on_partial` 组装默认 False）与顺序 fallback | 同上 | 无 | **缺口**（Task 04 按旧契约实现；`fallback_on_partial` 是否保留为新配置待任务 04 定） |
| judge 旁路（仅 general_vqa；judge 异常永不使样本失败；写 `vqa_evaluation.json`；trace `judge_status`） | 同上 | `ArtifactWriter.write_evaluation(filename=...)` 已支持任意安全文件名 | **缺口**：SampleRunner 内接线 |
| `TaskResolver` 低置信度候选兜底执行 | `TaskResolver` 已有设计注释"留给未来 SampleRunner" | `workflows/task_resolver.py` 已实现 | **缺口**：SampleRunner 需消费 `needs_candidate_fallback` 与 `candidate_tasks` |

### 3.3 DatasetRunner / run-dataset 入口 → 任务 05

| 旧功能 | 旧位置 | 新状态 | 判定 |
|---|---|---|---|
| 数据集迭代（asyncio 协程并发、sha256 分片、start_index/limit/sample_ids 过滤、fail-fast 取消） | `workflows/dataset_runner.py` | 已批未建；`DatasetAdapter` 协议（probe/iter_samples）与 `AdapterProbe` 已就位 | **缺口** |
| resume（仅 `succeeded` 跳过；partial/failed/running/pending 重跑；general_vqa 时 resume judge，失败→skipped） | 同上 | `SampleRunState` 含 `skipped`；无 runner | **缺口** |
| 运行产物（`predictions.jsonl` 追加、`dataset_summary.json`、probe 写回 manifest `dataset_probe`） | `workflows/dataset_runner.py` + `run_store.py` | `ArtifactWriter.append_prediction/write_summary` 已实现；**manifest `dataset_probe` 写入无对应能力** | **缺口**（probe 写回 manifest 需 Task 02 的 manifest adapter，见开放问题 O1） |
| `run_dataset` 单一入口（参数校验、judge client 创建条件、逐 task 顺序跑、counting 后处理评估三文件、报告输出） | `spacers_agent/commands/run_dataset.py` | `application/commands/run_dataset.py` 已批未建 | **缺口**（命令本体归 05，main 接线归 09） |
| `RunStore.create_run`（run 目录不存在才创建；resume 复用不覆盖 manifest） | `run_store.py` | 新 `workflows/run_store.py` 已实现（`FileExistsError` 语义一致） | 已覆盖（API 差异见 §5） |

### 3.4 无 task 数据集 TaskResolver seam → 任务 06

| 旧功能 | 旧位置 | 新状态 | 判定 |
|---|---|---|---|
| 未知任务解析（router-agent 文本分类 + 规则兜底，位于 TaskRouter 内） | `routing/router.py` + `routing/routes.py` | `TaskResolver` 独立于 Router 已实现（explicit→rule→model；低置信度只返回结构化候选；稳定错误 `TASK_RESOLUTION_FAILED:*`） | 已覆盖（架构红线：Resolver/Router 分离） |
| `UnifiedSample.task` 缺失时的样本构造路径 | （旧：适配器直接产出 task） | **无 `SampleDraft`**；`UnifiedSample.task` 必填 | **缺口**：SampleDraft 桥接（无 task 样本 → TaskResolver → UnifiedSample），Task 06 |
| router prompt（`router_v1.md`，版本串 router-v2） | `prompts/router_v1.md` | **未迁移**；新 TaskResolver 需要 system_prompt（`prompt_version="task-resolver-v1"`，用户载荷含 allowed_tasks/结构化字段，与旧 router 载荷不同） | **缺口**：需新增 `prompts/task_resolver_v1.md`（.md 不受 Python 白名单约束；任务 06 或 08） |

### 3.5 Reporting → 任务 07

| 旧功能 | 旧位置 | 新状态 | 判定 |
|---|---|---|---|
| 汇总（`EvaluationSummary`/`summarize_evaluations`，确定性指标与 judge 分开汇总） | `spacers_agent/reporting.py` | `reporting/` 空（6 文件已批） | **缺口** |
| counting 报告（`counting.jsonl` CanonicalSample/Prediction 对、`counting.metadata.json`、`.report/report.html`，无样本返回 None） | `spacers_agent/counting_report.py` | 空 | **缺口** |
| VQA 报告（`vrsbench_vqa.jsonl`/`.metrics.json`/`.deepseek_audit.jsonl`/`.metadata.json`） | `spacers_agent/vqa_report.py` | 空 | **缺口** |
| 审计报告（`samples.jsonl`/图像哈希去重/`samples.csv`（utf-8-sig）/单页 HTML） | `eval/audit_report.py` | 空 | **缺口** |
| 可视化（`render_counting_overlay`） | `spacers_agent/visualization.py` | 空（`reporting/visualization.py` 已批） | **缺口** |

### 3.6 Application / Bootstrap → 任务 08

| 旧功能 | 旧位置 | 新状态 | 判定 |
|---|---|---|---|
| `AppSettings`（models/counting/runs/paths/router/agents/backend 节 + YAML/环境加载） | `spacers_agent/settings.py` | `models/settings.py` 只有 `QwenSettings`/`DeepSeekSettings`/`ModelSettings`；**`configs/*.yaml` 全空** | **缺口**：`application/settings.py` + configs 重建（内容可参照旧 default.yaml 结构） |
| `PromptCatalog`（绑定表/版本/快照） | `spacers_agent/prompt_catalog.py` | `application/prompts.py` 已批未建 | **缺口**（prompt 文件 16/17 已就位，缺 task-resolver 版） |
| `assemble_runtime`/`RuntimeComponents`（注册顺序、BackendRegistry 组装、ROUTES 校验、JudgeService/ArtifactWriter/CallBudgetFactory/SampleRunner 装配） | `spacers_agent/bootstrap.py` | `application/bootstrap.py` 已批未建 | **缺口** |
| 最小 runtime（模型工厂选择、health 等价物） | `spacers_agent/application.py`（RuntimeApplication 部分） | `application/runtime.py` 已批未建 | **缺口**（只建组合所需最小面；`ask`/`serve`/HTTP 明确不恢复） |
| `CallBudgetFactory` | `routing/budget.py` | `workflows/call_budget.py`（含 `create_for_sample`/`task_limits`/`make_budget_guard`） | 已覆盖 |
| `MockVisionClient` | `clients/mock.py` | 无生产对应（测试内 fake 即可） | 不恢复（测试辅助） |

### 3.7 最小 run-dataset 入口 → 任务 09

| 旧功能 | 旧位置 | 新状态 | 判定 |
|---|---|---|---|
| `main.py run-dataset`（精简参数、`--evaluate` 默认 true、`--judge-policy` 默认 all、异常 stderr JSON `{status,error_type,error}` 退出码 1） | 顶层 `main.py` | `main.py` 已批未建 | **缺口** |
| 内部 CLI 13 命令（run-init/health/list-datasets/smoke-qwen/count-image/resume-run/evaluate-run/judge-vqa-run/standard-evaluate/inspect-data/render-count/summarize-evaluations）与 `serve`/`ask` | `spacers_agent/cli.py`、`application.py` | `application/commands/` 13 个文件全部已批未建 | **不恢复**（计划明示；未来需要时加薄壳） |
| 退出码常量（EXIT_OK/ARGUMENT/DATA/…/INVARIANT） | `commands/common.py` | 无 | 随 09 按需最小化（本轮 CLI 面小，不建 `common.py` 泛化文件——白名单也禁止该名） |

### 3.8 已覆盖 / 明确不恢复（非缺口）

| 旧功能 | 旧位置 | 新归属/结论 |
|---|---|---|
| Thin Router 与 ROUTES 表 | `routing/*` | 新 `routing/` 已实现（不读 question、不调用模型——有意变化，更严格）；ROUTES 语义并入 `routing/policies.py` |
| VRSBench 语义路由与几何改写 | `vqa_geometry.py` | 语义判断**前移到 Adapter**（`data/adapters/vrsbench/`）——有意变化，已在 BASELINE_INVENTORY §8.2 记录 |
| 计数/空间/变化 Agent 全链 | `agents/*`、`seam.py`、`targeting.py`、`imaging.py` | 新 `agents/counting|spatial|change` 已完整覆盖（含 seam finalization、target parser、图像几何） |
| 旧 data 包（loader/downloader/validator） | `data/*` | 新 `data/` 已实现；**downloader 不恢复**（硬性规则 6：不得默认联网） |
| 数据集审计 | `data_audit.py` | 能力由新 `data/validation.py`+`selection.py` 承担；`inspect-data` CLI 不恢复 |
| 错误码枚举 | `errors.py` | 新 `agents/errors.py` 稳定错误体系覆盖（语义不同——有意变化，已记录于 DETAILS.md） |
| 事件/产物写入 | `events.py`、`artifact_writer.py` | 新 `workflows/` 已实现（含 JSONL 进程内并发安全——增强） |

## 4. 白名单与测试路径状态（任务 02 输入）

- **生产路径**：任务 03–09 所需生产 .py **全部已在白名单**（§2 第二组清单），
  本轮**无需生产路径白名单变更**。
- **测试路径缺口**（白名单 §5 要求测试文件显式列出，当前缺失，任务 02 需
  批准）。建议清单（按任务归属）：

```text
# 任务 03
tests/workflows/test_judge_service.py
tests/evaluation/test_judge_models.py
tests/evaluation/test_judges_deepseek.py
# 任务 04
tests/workflows/test_sample_runner.py
# 任务 05
tests/workflows/test_dataset_runner.py
# 任务 06
tests/integration/test_auto_task_vertical_slice.py
# 任务 07
tests/reporting/test_schema.py
tests/reporting/test_builder.py
tests/reporting/test_html.py
tests/reporting/test_exporters.py
tests/reporting/test_visualization.py
# 任务 08
tests/application/test_settings.py
tests/application/test_prompts.py
tests/application/test_bootstrap.py
# 任务 09
tests/application/test_main_parser.py
tests/application/test_main_execution.py
```

（`tests/integration/test_general_vqa_vertical_slice.py` 已批准；新增纵向切片
按上表命名。任务 02 应与用户确认此清单。）

## 5. 产物文件名契约对照（新 RunStore 差异）

| 旧契约（基线） | 新实现 | 差异性质 |
|---|---|---|
| `manifest.json`（run_id/created_at/git_commit/git_dirty/config_hash/prompt_hashes/model_ids/dataset/split/sample_filter） | `workflows/run_store.py` 同名同字段 | 一致 |
| `config.snapshot.yaml` | `config.snapshot.json`（sort_keys 稳定布局） | **有意变化，需按 AGENTS.md 规则 5 记录到 `docs/migration/`**（任务 02/05 时补记） |
| `prompts.snapshot/`（重名抛错） | 同 | 一致 |
| `events.jsonl`（RUN_CREATED） | 同 | 一致 |
| run 目录已存在 → `FileExistsError` | 同 | 一致 |
| resume 复用 run 目录、不覆盖 manifest | 由 runner 层保证（未实现） | 任务 05 |
| `dataset_probe` 写回 manifest | 无对应能力 | **缺口 → manifest adapter（开放问题 O1）** |

## 6. 开放问题（需任务 02/08 与用户确认）

- **O1 — "manifest adapter"的确切范围**（计划任务 02 原文"仅批准后续必要
  测试路径 + manifest adapter"）。候选解读：
  1. 让新 `DatasetRunner` 能把 `AdapterProbe` 写回已创建 run 的
     `manifest.json`（对应旧 dataset_runner 的 probe 行为）；
  2. 读取旧格式 run 产物（`manifest.json`/`config.snapshot.yaml`）的适配层
     （resume 兼容旧 run 目录）；
  3. `application/settings.py` 产出 `RunStore.create_run(config_payload…)`
     的配置序列化适配。
  建议任务 02 与用户确认后按选定的最小路径实现。
- **O2 — task-resolver prompt**：新 `TaskResolver` 需要与
  `task-resolver-v1` 载荷（allowed_tasks/结构化 user payload）配套的
  system prompt；旧 `router_v1.md` 未迁移且载荷不同。建议新建
  `prompts/task_resolver_v1.md`（.md 不在 Python 白名单范围，无需架构变更；
  归任务 06 或 08）。
- **O3 — `evaluation/standard/adapter.py` 与
  `evaluation/datasets/vrsbench.py`**：白名单已批但本轮计划未定义用途。
  建议本轮暂缓实现（CLI standard-evaluate 不在范围；VRSBench 官方评估输出
  解析待报告需求确定），保持为已批准未来路径。
- **O4 — CI 更新**：`offline-tests.yml` 的 compileall 目录列表与 wheel
  smoke import 清单需随任务 07/08/09 增加 `reporting`/`application`/`main`。
- **O5 — 旧 `count_repair_v1`/`count_tile_v1..v3` 等历史 prompt**：旧 catalog
  本就不绑定（仅历史），不迁移——与基线行为一致，无需动作。

## 7. 结论

1. **核心链路的支撑层已全部就位**：契约类型（`workflows/schema.py`）、
   产物写入（`ArtifactWriter`）、预算（`CallBudget`+Factory）、任务解析
   （`TaskResolver`）、确定性评估（`evaluation/metrics` + 记录层 judge 字段）、
   数据适配（`DatasetAdapter` 协议）、模型层（Qwen 客户端/缓存/身份）。
2. **真正的缺口是 9 个已批准文件**：`judge 客户端+载荷类型`、
   `JudgeService`、`SampleRunner`、`DatasetRunner`、`run_dataset 命令`、
   `reporting 6 文件`、`application 4 文件`、`main.py`，外加
   `SampleDraft`（任务 06）与 task-resolver prompt——全部有白名单路径。
3. **任务 02 的范围**：批准 §4 测试路径清单 + 确认 O1（manifest adapter）
   语义；无需生产路径白名单变更。
4. **后续每个实现任务**需同步：更新 `implementation_status.json`、
   README、DETAILS、CI（O4）、按 AGENTS.md 规则 5 记录有意变化（§5）。
