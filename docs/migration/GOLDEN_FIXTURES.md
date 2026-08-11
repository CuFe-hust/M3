# GOLDEN_FIXTURES.md — 迁移 Golden Fixture 说明

> Task 01 / 03.5 产物。本文件说明 `tests/fixtures/migration/` 下每个 Golden
> fixture 锁定的行为、生成方式与允许变化字段。Golden 由锁定参考提交的**旧实现
> 离线生成**，后续迁移必须以这些文件为行为与产物形状契约。

## 1. 目的与原则

- 用最小离线 fixture 锁定：数据适配输出（sample）、路由决策、Agent/Counting 结果
  形状、运行产物文件名、轨迹、报告输入记录（report record）、原始数据集适配器
  输入/输出，以及 VRSBench 任务规范化。
- 迁移完成后只比较**稳定字段与文件存在性**，不比较 Python 类 identity。
- Golden 内容已剥离：绝对机器路径、时间戳、推理耗时、随机 request id / UUID。
- fixture 完全离线：图片为 4×4 纯色或 32×32 固定种子纹理的极小 PNG，不包含模型
  Base64、密钥或大权重。

## 2. 生成方式（可复现）

Golden 由仓库内生成器 `scripts/generate_migration_fixtures.py` 离线生成：

```bash
python scripts/generate_migration_fixtures.py \
  --reference-root <try_yolo checkout at the locked commit> \
  --expected-commit ec962eb87c3ad0b8c1502efcbd08db0daec48868
```

要求与行为：

- `--reference-root` 的 HEAD 必须逐字等于锁定提交（生成器用 `git rev-parse` 校验）；
- 完全离线：不下载模型/数据集、不调用 API，不使用密钥；
- 生成器把参考 checkout 加入 `sys.path`（仅迁移工具，不进入 wheel，不被架构守卫扫描）；
- **连续生成两次并逐字节比较**，不一致即中止（`STABILITY: two runs are byte-identical`）；
- 输出去除绝对路径、时间戳、UUID、推理耗时、机器目录；不保存 Base64/密钥/大权重；
- 输出文件排序稳定，UTF-8 与固定 JSON 格式（`indent=2` + 换行；`expected_samples.jsonl`
  为每行一个 JSON 对象）。

注意：`try_yolo` 分支可能随时间前进；锁定参考 `ec962eb` 需要 worktree/checkout
方式提供，例如 `git worktree add <tmp> ec962eb`。

## 3. Fixture 一览与锁定行为

### 3.1 运行时场景（`<case>/`，9 个）

| 目录 | 数据集 | 任务 | 状态 | 锁定行为 |
|---|---|---|---|---|
| `vrsbench_vqa` | VRSBench | general_vqa | succeeded | VRSBench 语义路由；general VQA 输出；DeepSeek Judge 成功 |
| `levir_cc` | LEVIR-CC | change_caption | succeeded | LEVIR-CC 归一化；双图 t1/t2；ChangeAgent 双路径成功 |
| `mme_realworld` | MME-RealWorld | general_vqa | succeeded | MME 归一化；general VQA + Judge |
| `xlrs` | XLRS-Bench-lite | general_vqa | succeeded | XLRS 归一化；general VQA + Judge |
| `counting` | parity | counting | succeeded | CountingAgent 整图计数（4×4 图 4 个 owner core）；`final_count` 来自 accepted points；后端 `qwen_point` |
| `spatial` | parity | spatial_relation | succeeded | SpatialAgent 输出（evidence_items、geometry） |
| `change` | parity | change_caption | succeeded | ChangeAgent 双路径（非数据集依赖版本） |
| `counting_partial` | parity | counting | partial | 指定 tile 失败 → `status=partial`、`failed_tiles` 非空 |
| `failed` | parity | general_vqa | failed | 主 Agent 调用抛错 → `failed_sample_status`；无 result/trace/evaluation |

每个目录固定包含：`sample.json`、`routing_decision.json`、`status.json`、
`agent_result.json` 或 `counting_result.json`、`agent_trace.json`、
`vqa_evaluation.json`（仅 general_vqa 域）、`report_record.json` 与极小 PNG 图片。

### 3.2 原始适配器场景（`adapters/<dataset>/`，4 个数据集）

每个数据集（`vrsbench`、`levir_cc`、`mme_realworld`、`xlrs`）包含：

- `raw/success/` — 最小成功布局（官方标注或 `spacers_adapter.json` + samples + 图片）
- `raw/missing_image/` — 标注引用不存在的图片 → 失败代码 `DatasetProbeError`
- `raw/missing_field/` — 行缺失必需字段 → 失败代码 `DatasetProbeError`
- `raw/duplicate_candidates/` — 重复候选（VRSBench：两个标注候选 → 失败；
  manifest 数据集：decoy samples 文件 → 参考实现忽略，记录 `behavior` 说明）
- `expected_samples.jsonl` — 每场景一行：成功场景为参考适配器**实际运行**产出的
  expected 稳定字段（dataset/task/split/question/sample_id/image_roles/image_paths/
  ground_truth/metadata）；失败场景记录 `failure` 代码。

### 3.3 VRSBench 任务规范化（`vrsbench_normalization.json`，6 条）

锁定问题 → 标准任务映射（由 try_yolo 行为审计得出，使用 Adapter normalization 语义）：

| 问题 | normalized_task | semantic_subtype | reason_codes |
|---|---|---|---|
| How many small vehicles are in the image? | counting | counting | quantity_question |
| What category is the topmost vehicle? | spatial_relation | extreme_category | extreme_category_question |
| Where is the large vehicle located in the image? | spatial_relation | grid_position | grid_position_question |
| Are there any small vehicles? | general_vqa | existence | existence_question |
| What color is the building? | general_vqa | color | color_question |
| Describe the scene. | general_vqa | general | general_question |

每条含 `source_task="vrsbench_vqa"`、`normalizer="vrsbench_task_normalizer"`、
`version="1"`、`confidence=1.0`、结构化 `spatial_query`/`answer_constraints`/
`count_target_hint`。

## 4. 已剥离 / 允许变化的字段

| 类别 | 字段/模式 | 说明 |
|---|---|---|
| 时间戳 | `updated_at` | 生成时删除 |
| 硬件耗时 | `inference_seconds`、`model_load_seconds` | 生成时删除 |
| 绝对路径 | `C:`、`/Users`、`Desktop` 等片段 | 生成时扫描，出现即中止；图片 path 改写为相对文件名，`result_path` 改写为 `<RUN_ROOT>/...` 占位 |
| 随机标识 | UUID 模式 | Golden 中不允许出现 |
| 允许变化 | `report_record.result` 证据细节、`trace` 的 `geometry`/`route` 文本 | 测试只断言键存在与稳定值 |

## 5. 必须保持 / 有意变化

### 必须保持

- 数据集原始字段转换后的关键事实（question、ground truth、图片顺序与角色）；
- sample ID 稳定性（源 ID 优先，否则稳定摘要）；
- 图片顺序（change 的 t1/t2、单图 image）；
- 最终任务专家选择（同一问题规范化到同一标准任务，路由到同一 Agent）；
- 运行产物文件名与指标稳定字段（status/trace/evaluation/report record 形状）；
- 三种样本状态（succeeded / partial / failed）的持久化语义。

### 有意变化

- **VRSBench 语义任务判断从运行时 Router 前移到 Adapter**：
  `router_source` 由 `vrsbench_semantic_rule` 变为 `normalized_task_policy`；
  `routing_decision.json` 的 `router_source` 字段不再作为最终稳定契约锁定
  （parity 测试只断言其为非空字符串）。
- **`spatial_relation` 由 `general_vqa_agent` 接管**：`agents/spatial/` 已删除，
  `spatial_relation` 路由到 GeneralVQAAgent，单次 general_vqa_v2 Prompt 调用并
  输出 `agent_result.json`。行为差异：旧实现最多两次 Qwen 调用（候选 + review）
  且可能做确定性几何改写；现为单次调用、无候选完成、无确定性几何重写。冻结的
  `tests/fixtures/migration/spatial/` 保留为历史事实（`spatial_agent` 路由、
  `evidence_items`/`geometry` 输出），与当前运行结果不直接可比。任务名
  `spatial_relation`、`TaskNormalization.spatial_query`、VRSBench normalization
  与 VQA 评测族保持不变。
- `CanonicalSample`/`CanonicalPrediction` 不再是内部主 Schema（保留为外部
  兼容/报告记录）。
- 新代码不再保留 `spacers_agent` 兼容层；本分支从零重建，旧包目录永久禁止。
- `stable_sample_id` 签名升级为多图版（含 dataset/split/有序图片路径），
  不安全源 ID 不再原样返回，改由稳定摘要替代（原始值由适配器存入 metadata）。

## 6. 测试与验证

```bash
python -m pytest -q tests/parity/test_baseline_golden_fixtures.py
python -m pytest -q tests/contracts/test_data_schema_contract.py
```

测试只读 Golden JSON 与 PNG，不 import `spacers_agent`/`eval` 等旧包。覆盖：
运行时场景产物与稳定字段、三态覆盖、无易变值扫描、适配器场景完整性/失败代码、
规范化映射、新 Schema 无损往返。
