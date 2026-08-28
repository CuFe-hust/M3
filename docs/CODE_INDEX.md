# M3 功能代码索引

> 本索引按“要找什么功能”组织，指向稳定的职责入口，而不是枚举每个私有函数。当前接口事实以 [`DETAILS.md`](../DETAILS.md) 为准；文件发生变化时应同步维护本索引。

## 1. 快速入口

| 需求 | 首要入口 | 继续阅读 |
|---|---|---|
| 查看或新增 CLI 命令 | `main.py` | `application/commands/` |
| 组装完整运行时 | `application/bootstrap.py` | `application/runtime.py` |
| 修改应用配置 | `application/settings.py` | `configs/default.yaml`、`configs/local.yaml` |
| 修改统一样本 | `data/schema.py` | `data/validation.py` |
| 新增数据集 | `data/adapters/base.py` | `data/registry.py`、现有 adapter |
| 修改任务规划 | `workflows/visual_planner.py` | `workflows/sample_runner.py` |
| 修改 task → Agent 映射 | `routing/policies.py` | `routing/router.py`、`agents/registry.py` |
| 修改单样本执行 | `workflows/sample_runner.py` | `workflows/artifact_writer.py` |
| 修改数据集运行/resume | `workflows/dataset_runner.py` | `workflows/schema.py`、`application/runtime.py` |
| 新增主流程模型 | `models/entry.py` | `models/base.py`、`application/bootstrap.py` |
| 修改评测 | `evaluation/records.py` | `evaluation/metrics/` |
| 修改报告 | `reporting/builder.py` | `reporting/schema.py`、`reporting/exporters.py` |
| 修改离线训练 | `training/` | `scripts/`、`docs/train/`、`docs/training/` |

## 2. 公共入口与应用组装

| 功能 | 代码位置 | 说明 |
|---|---|---|
| 唯一公共 CLI | `main.py` | 定义参数并委托命令；不放模型逻辑、数据集循环或评测聚合 |
| 手动图像问答 | `application/commands/ask.py` | `ask` 命令适配器 |
| 数据集批量运行 | `application/commands/run_dataset.py` | `run-dataset` 命令适配器 |
| 恢复运行 | `application/commands/resume_run.py` | `resume-run` 命令适配器 |
| 本地 HTTP 服务 | `application/commands/serve.py` | 输入校验、Runtime 调用与稳定响应 |
| 单图计数 | `application/commands/count_image.py` | 单图计数 use case |
| 创建运行目录 | `application/commands/run_init.py` | 不加载模型的 run 初始化 |
| 组件健康检查 | `application/commands/health.py` | Qwen/DeepSeek readiness，可选 live probe |
| Qwen smoke | `application/commands/smoke_qwen.py` | 显式单次模型冒烟测试 |
| 数据集查看/下载/审计 | `application/commands/list_datasets.py`、`download_data.py`、`inspect_data.py` | 数据运维入口 |
| 运行后评测 | `application/commands/evaluate_run.py`、`judge_vqa_run.py` | 确定性补评测与可选 Judge |
| 评测汇总/标准评测 | `application/commands/summarize_evaluations.py`、`standard_evaluate.py` | 聚合内部记录或调用显式外部工具 |
| 计数结果渲染 | `application/commands/render_count.py` | 从已持久化结果生成 overlay |
| 组件组装 | `application/bootstrap.py` | 唯一 composition root，构造模型、Agent、workflow、Judge 与证据服务 |
| 应用门面 | `application/runtime.py` | 手动请求、数据集运行、resume、报告等高层用例 |
| 配置模型 | `application/settings.py` | `AppSettings`、YAML/env 加载、严格字段校验 |
| Prompt 加载 | `application/prompts.py` | Prompt catalog 与版本化内容 |

## 3. 数据与样本契约

| 功能 | 代码位置 | 关键对象/职责 |
|---|---|---|
| 统一 Schema | `data/schema.py` | `ImageRef`、`GroundTruth`、`TaskNormalization`、`UnifiedSample`、`SampleDraft` |
| Draft 物化 | `data/schema.py` | `materialize_sample(...)` |
| 稳定样本 ID | `data/schema.py` | `stable_sample_id(...)`，避免机器绝对路径进入身份 |
| Adapter 协议 | `data/adapters/base.py` | `DatasetAdapter` 及 probe/read 合约 |
| VRSBench | `data/adapters/vrsbench/adapter.py` | 样本读取与标准化；ontology/任务归一化在同目录 |
| LEVIR-CC | `data/adapters/levir_cc.py` | 双时相变化描述数据接入 |
| MME-RealWorld | `data/adapters/mme_realworld.py` | MME 遥感数据接入 |
| XLRS | `data/adapters/xlrs.py` | XLRS-Bench 与 lite 接入 |
| XLRS 图片缓存 | `data/adapters/xlrs_image_cache.py` | 图片定位/缓存辅助，不改变样本契约 |
| Manifest Draft | `data/adapters/manifest.py` | 无最终 task 数据的显式 draft 入口 |
| 数据集注册表 | `data/registry.py` | `DatasetRegistry`、内建 adapter 显式注册与 alias |
| 样本选择 | `data/selection.py` | 稳定选择和过滤相关逻辑 |
| 数据验证 | `data/validation.py` | 数据根目录与样本问题审计 |
| 显式下载 | `data/downloader.py` | 仅由下载命令调用 |
| 便利加载 | `data/loader.py` | 受契约约束的数据加载门面 |

## 4. 模型与模型身份

| 功能 | 代码位置 | 说明 |
|---|---|---|
| 模型协议 | `models/base.py` | Agent/Workflow 所依赖的抽象客户端契约 |
| 统一模型 factory | `models/entry.py` | `create_model(...)`、惰性 builder 注册 |
| Qwen 通用实现 | `models/qwen_transformers.py` | Transformers 主模型适配 |
| Qwen3-VL baseline | `models/qwen3_vl/baseline.py` | baseline 客户端 |
| Qwen3.5 | `models/qwen3_5/model.py` | Qwen3.5 客户端实现 |
| Qwen3.5 多 LoRA | `models/qwen3_5/multi_adapter.py` | 单基座、多命名 adapter 绑定 |
| 模型缓存 | `models/cache.py` | 请求缓存与逻辑模型身份 |
| 模型设置 | `models/settings.py` | 模型侧设置类型 |
| 图像输入工具 | `models/images.py` | 图像规范化、摘要与模型输入辅助 |
| SegFormer 客户端 | `models/segformer_transformers.py` | 可选语义分割/特征协议实现 |
| ChangeHead runtime | `models/change_head/runtime.py` | 运行时只读 checkpoint 消费 |
| ChangeHead 身份与门禁 | `models/change_head/manifest.py`、`fingerprint.py`、`checkpoint.py`、`calibration.py` | 资产身份、加载与校准 |

具体模型只能由 `application/bootstrap.py` 选择。领域 Agent 不应 import `models.entry` 或具体 Qwen 实现。

## 5. 任务规划、路由与编排

| 功能 | 代码位置 | 关键对象/职责 |
|---|---|---|
| 视觉任务规划 | `workflows/visual_planner.py` | `VisualTaskPlanner`、v5 计划校验、fail-closed |
| 路由策略表 | `routing/policies.py` | 每个公开 task 的 primary/fallback agent 与 tiling 属性 |
| 确定性 Router | `routing/router.py` | `TaskRouter`，不读取 question、不调用模型 |
| 路由 Schema | `routing/schema.py` | `RoutePolicy`、`RoutingDecision`、能力描述 |
| Agent 注册表 | `agents/registry.py` | Agent 注册、覆盖和查找 |
| 调用预算 | `workflows/call_budget.py` | Planner、Agent fallback、Judge 的样本级硬上限 |
| 单样本执行 | `workflows/sample_runner.py` | 规划、物化、路由、attempt、评测、Judge、状态和 trace |
| 数据集编排 | `workflows/dataset_runner.py` | probe、选择、分片、并发、resume、fail-fast、汇总 |
| 运行 Schema | `workflows/schema.py` | `SampleRunStatus`、`DatasetRunOptions`、`RunRequest`、冻结身份 |
| Run 管理 | `workflows/run_store.py` | manifest、配置/Prompt 快照、run ID 与目录创建 |
| Artifact 写入 | `workflows/artifact_writer.py` | 原子 JSON/JSONL、固定文件名和 result path 校验 |
| 事件流 | `workflows/events.py` | 稳定事件记录、进程内并发写保护和敏感信息拒绝 |
| Judge 编排 | `workflows/judge_service.py` | 可选 Judge 调用与失败隔离 |

## 6. Agent 与领域流水线

### 6.1 公共 Agent 契约

| 功能 | 代码位置 |
|---|---|
| Agent 协议 | `agents/base.py` |
| `AgentExecution` 等 Schema | `agents/schema.py` |
| 稳定错误类型 | `agents/errors.py` |
| 通用视觉 Agent 基类 | `agents/visual_base.py` |
| 视觉证据 catalog | `agents/evidence_catalog.py`、`agents/evidence_catalog.json` |

### 6.2 General VQA

| 功能 | 代码位置 |
|---|---|
| 主 Agent、payload 与答案约束 | `agents/general_vqa/agent.py` |
| 证据执行 | `agents/general_vqa/evidence/executor.py` |
| ROI/坐标几何 | `agents/general_vqa/evidence/geometry.py` |
| mask、检测框和 ROI 渲染 | `agents/general_vqa/evidence/rendering.py` |
| 证据数据结构 | `agents/general_vqa/evidence/schema.py` |

### 6.3 Caption 与 Grounding

| 功能 | 代码位置 |
|---|---|
| CaptionAgent | `agents/caption/agent.py` |
| GroundingAgent | `agents/grounding/agent.py` |
| Grounding 候选、ROI 与确定性回映 | `agents/grounding/evidence.py` |

### 6.4 Counting

| 功能 | 代码位置 |
|---|---|
| CountingAgent 门面 | `agents/counting/agent.py` |
| 计数结果 Schema | `agents/counting/schema.py` |
| backend 执行与 fallback | `agents/counting/executor.py` |
| backend 协议/注册/选择 | `agents/counting/backends/base.py`、`registry.py`、`selector.py` |
| Qwen 点计数 | `agents/counting/backends/qwen_point.py` |
| 数量建议 | `agents/counting/backends/quantity_proposal.py` |
| 语义分割计数 | `agents/counting/backends/semantic_segmentation.py` |
| YOLO OBB | `agents/counting/backends/yolo_obb.py`、`yolov5_obb_onnx.py` |
| YOLO 模型惰性存储 | `agents/counting/backends/yolo_model_store.py` |
| 点管线、tile/seam 几何 | `agents/counting/point_pipeline.py`、`geometry.py` |
| 目标提示校验 | `agents/counting/target_parser.py` |
| ExpertCatalog | `agents/counting/expert_catalog.py`、`expert_catalog.json` |
| 计数设置 | `agents/counting/settings.py` |

### 6.5 Change

| 阶段 | 代码位置 |
|---|---|
| 主流程编排 | `agents/change/agent.py` |
| 时相图对校验 | `agents/change/pair_validator.py` |
| 保守全局配准 | `agents/change/registration.py` |
| 辐射一致化 | `agents/change/harmonizer.py` |
| 基础感知 | `agents/change/perception.py` |
| 低层差异 proposal | `agents/change/difference_proposal.py` |
| 特征残差 | `agents/change/feature_residual.py` |
| 语义差异/迁移 | `agents/change/semantic_difference.py`、`semantic_transition.py` |
| 多源 proposal 融合 | `agents/change/proposal_fusion.py` |
| 输入预处理 | `agents/change/preprocess.py` |
| Prompt 契约 | `agents/change/prompt_contract.py` |
| 结果复核 | `agents/change/reviewer.py` |
| Change Schema/设置 | `agents/change/schema.py`、`settings.py` |

## 7. 评测与 Judge

| 功能 | 代码位置 | 说明 |
|---|---|---|
| 统一评测记录 | `evaluation/records.py` | runtime task → metric family 和 artifact 名的单一映射 |
| Counting 指标 | `evaluation/metrics/counting.py` | 精确匹配与误差指标 |
| VQA 指标 | `evaluation/metrics/vqa.py` | 严格 exact match |
| Grounding 指标 | `evaluation/metrics/grounding.py` | 坐标兼容时的 IoU |
| Caption 记录 | `evaluation/metrics/caption.py` | candidate/references |
| 聚合指标 | `evaluation/metrics/aggregate.py` | 各评测族汇总与 caption 语料级适配 |
| Judge 协议 | `evaluation/judges/base.py` | 可选 Judge 抽象 |
| DeepSeek Judge | `evaluation/judges/deepseek.py` | 凭据由 application 注入 |
| 外部标准评测 seam | `evaluation/standard/adapter.py` | 独立 `external_standard` namespace |
| VRSBench 评测适配 | `evaluation/datasets/vrsbench.py` | 数据集标准评测映射 |

修改 `evaluation/` 前应先确认指标定义、Ground Truth、坐标系和历史结果可比性；不得用 Judge 覆盖确定性指标。

## 8. 报告与导出

| 功能 | 代码位置 | 说明 |
|---|---|---|
| 报告 Schema | `reporting/schema.py` | 报告、样本、汇总视图 |
| Artifact 只读适配 | `reporting/adapters.py` | 安全读取持久化记录 |
| 报告构建 | `reporting/builder.py` | 当前索引、样本详情、指标、失败和性能聚合 |
| HTML 输出 | `reporting/html.py` | 审计 dashboard |
| JSON/CSV/JSONL 导出 | `reporting/exporters.py` | 报告 bundle 和样本导出 |
| 计数可视化 | `reporting/visualization.py` | 从结果生成视觉 overlay |

报告层不调用模型、不执行 Agent、不修改原始运行产物，也不信任任意绝对 `result_path`。

## 9. Prompt、配置与资产

| 内容 | 位置 | 说明 |
|---|---|---|
| 版本化 Prompt | `prompts/*.md` | Planner/Agent/Judge 使用的冻结文本 |
| 默认与本地配置 | `configs/default.yaml`、`configs/local.yaml` | 应用行为和本地模型配置 |
| 模型配置示例 | `configs/models.example.yaml` | 本地模型身份与路径示例 |
| YOLO 配置示例 | `configs/yolo.example.yaml` | detector 配置 |
| Change 配置 | `configs/change_v3.example.yaml`、`configs/change_ablations/` | Change V3 与消融配置 |
| Import DAG | `architecture/import_rules.json` | 包边界机器规则 |

模型权重、checkpoint、数据集、缓存和 outputs 属于本地资产，不应作为通用代码入口或逻辑模型身份。

## 10. 离线训练与数据工具

训练代码与在线 Agent runtime 分离：

| 领域 | 代码位置 | 说明 |
|---|---|---|
| 多模态 SFT 契约 | `training/multimodal_sft/contracts.py` | 训练输入输出与参数契约 |
| 数据与 profile | `training/multimodal_sft/data.py`、`profiles/` | 数据物化和任务 profile |
| 模型 adapter | `training/multimodal_sft/adapters/` | Qwen3-VL/Qwen3.5 等训练适配 |
| 优化与训练核心 | `training/multimodal_sft/optimizer.py`、`trainer_core.py` | 训练执行 |
| checkpoint/export | `training/multimodal_sft/checkpoint.py`、`exporter.py` | 恢复与导出 |
| Change 语料 | `training/multimodal_sft/change_corpus.py` 等 | 变化任务语料、迁移和审计 |
| ChangeHead 训练 | `training/change_head/` | dataset、loss、trainer、evaluator、release gate |
| 可执行离线工具 | `scripts/` | 数据准备、微调、评估、导出、审计、可视化 |

训练运行手册位于 `docs/train/` 和 `docs/training/`。`agents/` runtime 不得 import `training/`。

## 11. 测试定位

| 修改区域 | 优先测试目录/文件 |
|---|---|
| 数据 Schema/adapter | `tests/data/`、`tests/test_schema.py` |
| VisualTaskPlanner | `tests/workflows/` 中 planner 相关测试、`tests/test_refine_visual_planner_dataset.py` |
| Router | `tests/routing/` |
| Agent | `tests/agents/` 与相应顶层专项测试 |
| Counting | `tests/agents/counting/`、`tests/test_*count*`、`tests/test_*yolo*` |
| Change | `tests/agents/change/`、`tests/test_change*` |
| Run/resume/artifact | `tests/workflows/`、`tests/entry/` |
| Evaluation | `tests/evaluation/`、标准评测专项测试 |
| Reporting | `tests/reporting/`、报告专项测试 |
| 配置/application/CLI | `tests/application/`、`tests/entry/`、`tests/test_main.py` |
| 架构边界 | `tests/architecture/` |
| 训练 | `tests/test_multimodal_sft_*`、`tests/test_finetune_*`、ChangeHead 专项测试 |

## 12. 常见改动路径

### 新增一个数据集适配器

1. 在 `data/adapters/` 实现 `DatasetAdapter`。
2. 只读转换到 `UnifiedSample` 或 `SampleDraft`。
3. 在 `data/registry.py` 显式注册名称和 alias。
4. 增加 adapter、selection、validation 测试。
5. 更新 `DETAILS.md`、README 和本索引中的数据集列表。

### 新增一个公开 task 或 Agent

1. 更新 `data.schema.TaskName` 及相关 Schema 校验。
2. 实现符合 `agents/base.py` 的 Agent，并声明 `supported_tasks`。
3. 更新 `routing/policies.py` 和 `agents/registry.py` 覆盖。
4. 在 `application/bootstrap.py` 组装依赖。
5. 明确 deterministic evaluation family，必要时更新 `evaluation/records.py`。
6. 增加 planner、router、Agent、workflow、resume 和架构测试。
7. 同步 `DETAILS.md`、README、项目说明书和本索引。

### 新增主流程模型

1. 在 `models/` 实现 `models/base.py` 协议。
2. 在 `models/entry.py` 注册惰性 builder。
3. 只在 `application/bootstrap.py` 选择和注入具体实现。
4. 定义安全的逻辑模型身份、revision 和缓存 hash 输入。
5. 增加 entry 惰性加载、单次组装、缓存与 resume identity 测试。

### 修改持久化或 resume

依次检查 `workflows/schema.py`、`run_store.py`、`artifact_writer.py`、`sample_runner.py`、`dataset_runner.py`、`application/runtime.py` 和 `reporting/`。任何语义变化都需要专门测试，并保证旧绝对路径不被重新信任、`predictions.jsonl` 不被覆盖、成功样本不重复推理。

