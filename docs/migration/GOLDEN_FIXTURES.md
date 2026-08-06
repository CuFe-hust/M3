# GOLDEN_FIXTURES.md — 迁移 Golden Fixture 说明

> Task 01 产物。本文件说明 `tests/fixtures/migration/` 下每个 Golden fixture
> 锁定的行为、生成方式与允许变化字段。Golden 由锁定参考提交的**旧实现离线生成**，
> 后续迁移必须以这些文件为行为与产物形状契约。

## 1. 目的与原则

- 用最小离线 fixture 锁定：数据适配输出（sample）、路由决策、Agent/Counting 结果形状、
  运行产物文件名、轨迹与报告输入记录（report record）。
- 迁移完成后只比较**稳定字段与文件存在性**，不比较 Python 类 identity。
- Golden 内容已剥离：绝对机器路径、时间戳、推理耗时、随机 request id / UUID。
- fixture 完全离线：图片为 4×4 纯色或 32×32 固定种子纹理的极小 PNG，不包含模型
  Base64、密钥或大权重。

## 2. 生成方式（可稳定重跑）

Golden 由旧仓库（`try_yolo` @ `ec962eb87c3ad0b8c1502efcbd08db0daec48868`）的
确定性离线 harness 生成：

1. 构造 `UnifiedSample`（dataset/task/question/双图 t1/t2 等稳定输入）；
2. 用 `RecordingFakeQwen` / `RecordingFakeDeepSeek`（旧测试确定性客户端）+ `assemble_runtime`
   执行 `SampleRunner.run_one(..., judge_policy="all")`，完全不调用模型/网络；
3. 失败路径复刻 `DatasetRunner._run_sample`（`failed_sample_status` + `write_final_status`）；
4. 产物经净化（见第 4 节）后写入 `tests/fixtures/migration/<case>/`；
5. **连续两次生成输出必须逐字节一致**（生成器内置稳定性校验，不一致即中止）。

生成脚本为本地工具（不入库）：`C:/Users/TZDEZACR/Desktop/spacers-agent/code/tmp/generate_golden.py`，
运行环境 Python 3.11（m3）。重跑命令：

```bash
cd C:/Users/TZDEZACR/Desktop/spacers-agent/code
python tmp/generate_golden.py   # 需要 m3 环境；两次运行结果一致才写盘
```

## 3. Fixture 一览与锁定行为

| 目录 | 数据集 | 任务 | 状态 | 结果文件 | 锁定行为 |
|---|---|---|---|---|---|
| `vrsbench_vqa` | VRSBench | general_vqa | succeeded | agent_result.json | VRSBench 语义路由（`router_source=vrsbench_semantic_rule`，reason_codes 含 `vrsbench_type_unspecified`/`vrsbench_semantic_general_vqa`）；general VQA Agent 输出；DeepSeek VQA Judge 成功 |
| `levir_cc` | LEVIR-CC | change_caption | succeeded | agent_result.json | LEVIR-CC 数据集归一化；双图 t1/t2；ChangeAgent 双路径（32×32 纹理图，PIF/LAB 一致化成功） |
| `mme_realworld` | MME-RealWorld | general_vqa | succeeded | agent_result.json | MME-RealWorld 归一化；general VQA + Judge |
| `xlrs` | XLRS-Bench-lite | general_vqa | succeeded | agent_result.json | XLRS-Bench-lite 归一化；general VQA + Judge |
| `counting` | parity | counting | succeeded | counting_result.json | CountingAgent 整图计数（4×4 图 4 个 owner core）；`final_count` 来自 accepted points；后端 `qwen_point`；trace 含 `backend/primary_backend/attempted_backends` |
| `spatial` | parity | spatial_relation | succeeded | agent_result.json | SpatialAgent 输出（evidence_items、geometry） |
| `change` | parity | change_caption | succeeded | agent_result.json | ChangeAgent 双路径（非数据集依赖版本） |
| `counting_partial` | parity | counting | partial | counting_result.json | 指定 tile 失败 → `status=partial`、`failed_tiles` 非空、`final_count` 仅计成功 tile；sample_id=`counting_one_failed_tile` |
| `failed` | parity | general_vqa | failed | 无 | 主 Agent 调用抛错 → `failed_sample_status`（`error_code=RuntimeError`）；无 result/trace/evaluation；sample_id=`primary_qwen_failure` |

每个目录固定包含：

- `sample.json` — UnifiedSample 规范化序列化（images[].path 为相对图片名）
- `routing_decision.json` — task/primary_agent/execution_mode/reason_codes/router_source
- `status.json` — state/sample_id/task/result_path（`<RUN_ROOT>/samples/<case>/<file>` 占位）/error_code
- `agent_result.json` 或 `counting_result.json` — Agent/Counting 结果载荷
- `agent_trace.json` — 轨迹（VQA/空间/变化含 `prompt_version`；计数含 `backend` 族键）
- `vqa_evaluation.json` — 仅 general_vqa 任务（VRSBench/MME/XLRS）存在，含
  `exact_match`、`judge_status`、`judge_parsed`
- `report_record.json` — 报告输入记录（sample/task/dataset/state/route/agent/
  result_file/result/trace/evaluation/errors），是迁移后 `reporting.schema.ReportRecord`
  的契约雏形
- 图片文件 — `image.png`（单图）或 `image_t1.png` + `image_t2.png`（双时相）

## 4. 已剥离 / 允许变化的字段

| 类别 | 字段/模式 | 说明 |
|---|---|---|
| 时间戳 | `updated_at` | 生成时删除 |
| 硬件耗时 | `inference_seconds`、`model_load_seconds` | 生成时删除 |
| 绝对路径 | `C:`、`/Users`、`Desktop` 等片段 | 生成时扫描，出现即中止；图片 path 改写为相对文件名，`result_path` 改写为 `<RUN_ROOT>/...` 占位 |
| 随机标识 | UUID 模式 | 生成时校验，Golden 中不允许出现 |
| 允许变化 | `report_record.result` 中的证据细节、`trace` 中的 `geometry`/`route` 文本 | 迁移后这些字段内容可能合理变化；测试只断言键存在与稳定值（state/task/agent/status 等） |

## 5. 测试与验证

```bash
python -m pytest -q tests/parity/test_baseline_golden_fixtures.py
```

测试只读 Golden JSON 与 PNG，不 import `spacers_agent`/`eval` 等旧包。覆盖：

- 每个场景的必需产物与图片存在性（含 PNG 头校验）
- sample / routing / status / result / trace / report_record 的稳定字段
- 三态覆盖（succeeded / partial / failed）
- 全量扫描：无易变键、无绝对路径、无 UUID、无机器路径
