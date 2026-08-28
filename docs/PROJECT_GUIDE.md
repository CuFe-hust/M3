# M3 项目说明书

> 文档定位：面向项目交付、演示、部署与二次开发人员，说明 M3 当前已经实现的目标、功能、架构、运行方式和质量边界。内部接口的机器级细节以 [`DETAILS.md`](../DETAILS.md) 为准，编码约束以 [`AGENTS.md`](../AGENTS.md) 为准。

## 1. 项目概述

M3 是“面向太空智算的多模态遥感大模型应用探索”项目的当前架构实现。系统以本地多模态大模型为核心，将遥感图像理解任务组织为统一的数据接入、视觉任务规划、确定性路由、领域 Agent 执行、可恢复运行、确定性评测、可选 Judge 和只读报告流水线。

项目重点不只是完成一次模型问答，还包括：统一不同遥感数据集的样本表达，隔离模型与业务模块，保存可复现运行身份，对失败和恢复过程进行审计，以及在不重新推理的情况下生成评测与报告。

## 2. 建设目标

- 支持单图、多图和双时相遥感任务的统一执行。
- 通过 `UnifiedSample` 消除数据集私有结构对业务层的渗透。
- 由 VisualTaskPlanner 对每条 fresh 样本进行一次视觉任务规划。
- 将任务识别与 Agent 路由分离，保证路由同步、确定且不调用模型。
- 通过专用 Agent 承载计数、变化理解、定位、描述和 VQA 工作流。
- 形成可恢复、可追踪、路径安全的运行产物体系。
- 同时支持确定性指标和可选语义 Judge，且 Judge 不覆盖确定性结果。
- 对已持久化结果生成 HTML、JSON、CSV 和审计导出，不在报告阶段重新推理。

## 3. 功能范围

### 3.1 公开任务

| 任务名 | 功能 | 执行 Agent |
|---|---|---|
| `counting` | 通用目标计数 | CountingAgent |
| `fine_grained_counting` | 细粒度目标计数 | CountingAgent |
| `change_caption` | 双时相变化描述 | ChangeAgent |
| `change_qa` | 双时相变化问答 | ChangeAgent，有限回退到 GeneralVQAAgent |
| `grounding` | 文本目标定位 | GroundingAgent |
| `spatial_relation` | 空间关系理解 | GeneralVQAAgent |
| `scene_classification` | 遥感场景分类 | GeneralVQAAgent |
| `general_vqa` | 通用遥感视觉问答 | GeneralVQAAgent |
| `caption` | 遥感图像描述 | CaptionAgent |
| `multiple_choice_vqa` | 多项选择视觉问答 | GeneralVQAAgent |

### 3.2 数据集接入

内建适配器包括 VRSBench、LEVIR-CC、MME-RealWorld、XLRS-Bench 和 XLRS-Bench-lite。系统还支持显式 manifest 驱动的 `SampleDraft` 路径，适合原始记录尚未给出最终 task 的数据。

数据层只负责探测、选择、验证和只读转换，不调用模型，不写运行产物，也不根据问题文本自行选择 Agent。普通加载默认离线；数据下载只能通过显式 `download-data` 命令触发。

### 3.3 任务规划与执行

fresh 样本的核心链路如下：

```text
数据集记录或本地图像
  -> UnifiedSample / SampleDraft
  -> VisualTaskPlanner（有序预览图 + 原始问题）
  -> materialize/rebuild UnifiedSample
  -> TaskRouter（确定性）
  -> 领域 Agent
  -> 确定性评测 + 可选 Judge
  -> 持久化运行产物
  -> 只读报告
```

VisualTaskPlanner 的模型输出决定 fresh execution 的 task；CLI task、数据集 task 和 adapter task 仅用于运行命名空间或审计，不能覆盖规划结果。Planner 失败时稳定关闭，不猜测 `general_vqa`。TaskRouter 只接受已经确定的 task，不读取 question，也不调用模型。

### 3.4 领域能力

- Counting：支持 Qwen 点计数、数量建议、语义分割和 YOLO OBB 等独立 backend，由 selector 规划、executor 执行有限 fallback，并保留 tile、seam 和尝试审计。
- Change：按时间顺序处理 `t1`、`t2`，依次进行图对校验、保守全局配准、辐射一致化、多源变化感知、proposal 融合、Qwen 推理和结果复核。
- Grounding：支持 direct 路径和受规划控制的视觉证据路径；ROI 局部坐标由确定性逻辑映射回整图。
- General VQA：覆盖通用问答、场景分类、多选题和空间关系；可按规划结果使用有界的检测/分割 ROI 证据。
- Caption：按稳定图像顺序生成图像描述，并保留逐样本 candidate 与 references 供语料级指标汇总。

### 3.5 运行管理与恢复

DatasetRunner 支持稳定样本顺序、`start-index`、样本 ID 过滤、稳定 SHA-256 分片、单进程 asyncio 并发、fail-fast 和 resume。所有入选样本最终必须进入 `succeeded`、`partial`、`failed` 或 `skipped`，汇总满足：

```text
total = succeeded + partial + failed + skipped
```

resume 以持久化的 `run_request.json` 为调用参数权威来源。成功样本默认不重复推理；允许时只补缺失或损坏的确定性评测/Judge。模型、adapter、证据预处理等冻结身份不一致时拒绝静默续跑。

### 3.6 评测与报告

确定性评测族包括：

| 评测族 | 主要内容 |
|---|---|
| counting | 预测数、真实数、精确匹配、绝对/相对误差、平滑误差分数 |
| general_vqa | 严格 exact match |
| grounding | 受坐标契约保护的轴对齐 IoU 与 IoU@0.5 |
| caption | 逐样本 candidate/references；报告阶段可聚合 BLEU、METEOR、ROUGE-L、CIDEr |

DeepSeek Judge 是可选的语义审计信号，只与确定性指标并列保存。报告层从 `predictions.jsonl` 和安全推导的样本目录读取产物，可导出 HTML、JSON、CSV、JSONL 及外部标准评测结果。

## 4. 系统架构

| 层 | 主要职责 | 关键边界 |
|---|---|---|
| `data` | Schema、adapter、选择、验证、下载 | 不调用模型，不执行 Agent |
| `models` | 模型协议、缓存、图像工具、模型 factory | 不依赖具体数据集 |
| `agents` | 单任务领域工作流 | 只依赖模型协议，不选择具体模型 |
| `routing` | 已知 task 的确定性路由 | 不读问题，不调用模型 |
| `workflows` | 规划、预算、单样本/数据集编排、产物 | 不成为模型 composition root |
| `evaluation` | 确定性指标、Judge seam、标准评测适配 | Judge 不覆盖确定性指标 |
| `reporting` | 只读聚合与导出 | 不重新推理，不修改执行产物 |
| `application` | 配置、模型选择和组件组装 | 唯一 composition root |
| `main.py` | CLI 参数解析与命令委托 | 唯一公开 CLI surface |

机器可检查的包依赖规则位于 `architecture/import_rules.json`，由 `tests/architecture/` 下的测试保护。

## 5. 核心数据契约

### 5.1 UnifiedSample

跨模块的 canonical sample 是 `data.schema.UnifiedSample`，核心字段为：

```text
sample_id, dataset, split, task, images,
question, ground_truth, metadata, normalization
```

`UnifiedSample.task` 始终必填。图像路径必须相对 dataset root，禁止绝对路径和 `..` 逃逸。变化任务图像角色为 `t1, t2, [context...]`，其他任务为 `image, [context...]`。metadata、Ground Truth 原始值、normalization 和 trace 必须严格 JSON-safe。

### 5.2 SampleDraft

`SampleDraft` 用于最终 task 和图像角色尚未确定的预任务样本。它经过 VisualTaskPlanner 后由 `materialize_sample(...)` 转换为 `UnifiedSample`，不能以 `task=None` 的形式冒充正式样本。

### 5.3 Agent 契约

Agent 统一实现：

```python
async def run(sample: UnifiedSample, context: AgentContext) -> AgentExecution:
    ...
```

`AgentContext` 只携带单样本所需轻量依赖；`AgentExecution` 的主结果名和附加结果名必须是跨平台安全的纯 basename，trace 不得保存凭据、Base64 图像或敏感异常正文。

## 6. 部署与运行

### 6.1 环境

- Python 3.11 或更高版本。
- 基础安装：`python -m pip install -e .`
- 开发测试：`python -m pip install -e ".[dev]"`
- 模型、CUDA、ONNX、YOLO 和 SegFormer 依赖按部署目标选择安装。

普通 import 和离线测试不会主动下载模型。配置入口为 `application.settings.AppSettings`，示例配置位于 `configs/`。本地 checkpoint 作为主模型时需要提供与机器路径无关的 `cache_model_id`。

### 6.2 常用命令

```bash
# 查看命令
python main.py --help

# 查看数据集注册表
python main.py list-datasets

# 对本地图像执行一次请求
python main.py --config configs/local.yaml ask \
  --images-dir /path/to/images \
  --question "图中有多少架飞机？" \
  --task auto

# 批量运行数据集
python main.py --config configs/local.yaml run-dataset \
  --dataset VRSBench \
  --root /path/to/VRSBench \
  --split test \
  --run-id vrsbench-test

# 恢复已存在运行
python main.py --config configs/local.yaml resume-run --run-id vrsbench-test

# 运行后确定性评测
python main.py --config configs/local.yaml evaluate-run --run-id vrsbench-test
```

详细参数、模型准备、HTTP 接口和专项命令见项目根目录 [`README.md`](../README.md)。

## 7. 运行产物

默认运行根目录是 `outputs/runs`，典型结构为：

```text
outputs/runs/<run_id>/
├── manifest.json
├── config.snapshot.json
├── run_request.json
├── prompts.snapshot/
├── events.jsonl
├── predictions.jsonl
├── report/
└── tasks/<run_task>/
    ├── dataset_probe.json
    ├── dataset_summary.json
    └── samples/<storage_key>/
        ├── sample.json
        ├── status.json
        ├── routing_decision.json
        ├── visual_task_plan.json
        ├── agent_result.json 或 counting_result.json
        ├── <task>_evaluation.json
        └── agent_trace.json
```

`predictions.jsonl` 是 append-only 执行索引；同一 `(run_task, sample_id)` 的最后一行代表当前状态。报告层不会把索引中的任意路径当作磁盘读取权限，而是使用冻结身份推导安全样本目录。

## 8. 安全、可靠性与可复现性

- 默认离线，不隐式下载模型或数据集，不隐式调用云 API。
- API key 只由 composition root 从指定环境变量读取，不进入配置快照、trace 或报告。
- 结构化 JSON 使用统一原子写入；JSONL 当前只承诺单 Python 进程内并发安全。
- run ID、sample storage key 和 result path 均进行跨 POSIX/Windows 的路径安全校验。
- request hash 覆盖模型逻辑身份、生成参数、prompt、消息、图像摘要、响应 schema 和版本等结果相关输入。
- Ground Truth 只读保留；不通过过滤失败样本、猜测坐标系或修改 fixture 美化结果。

## 9. 测试与验收建议

基础验收：

```bash
python -m pytest -q
```

架构边界重点检查：

```bash
python -m pytest -q \
  tests/architecture/test_repository_hygiene.py \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_no_new_to_legacy_imports.py
```

live Qwen、DeepSeek、真实数据集与特定硬件能力需要显式环境和授权，不能由离线测试结果替代。验收报告应分别记录离线契约测试、模型 live smoke、数据集质量测试和目标设备性能测试。

## 10. 当前限制

- 实际推理能力取决于本地 Qwen checkpoint、adapter 和硬件环境是否就绪。
- YOLO、ONNX Runtime、SegFormer、ChangeHead 等专家能力是可选能力，缺少相应依赖或已验证资产时不能宣称可用。
- DeepSeek Judge 需要显式配置 API key；无 key 时系统保持纯确定性评测。
- Grounding 对不明确坐标系、source-pixel 未转换坐标和 polygon 输入严格 fail-closed。
- Caption 的完整语料级指标依赖可选 `pycocoevalcap`，METEOR 还依赖 Java；缺失时报告会明确标记未计算或部分完成。
- 当前运行并发语义是单进程 asyncio，不承诺多进程同时追加同一 run。

## 11. 相关文档

- 使用说明：[`README.md`](../README.md)
- 当前架构与接口事实：[`DETAILS.md`](../DETAILS.md)
- 功能代码索引：[`CODE_INDEX.md`](CODE_INDEX.md)
- 编码代理规则：[`AGENTS.md`](../AGENTS.md)
- 架构决策：[`docs/architecture/`](architecture/)
- 迁移与 parity：[`docs/migration/`](migration/)
- 训练说明：[`docs/train/`](train/)、[`docs/training/`](training/)

