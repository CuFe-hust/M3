# M3-RS 标准化评测系统

面向太空智算的多模态遥感大模型标准化评测工具。一键式运行 LEVIR-CC、VRSBench、XLRS-Bench 和 MME-RealWorld-RS 四个固定基准评测，自动保存不可变运行证据，生成历史对比数据和核心《评测表》。

## 快速路径

五条标准命令：`doctor` | `run --mode smoke` | `run --mode full` | `rebuild-table` | `prepare-report`

在 `标准化评测/` 目录下依次执行以下五条命令，完成从环境检查到报告生成的完整流程：

### 1. 环境检查

```bash
python -m m3rs_eval doctor --config configs/server.yaml
```

检查配置、命令、数据路径、磁盘空间、Python 依赖和 GPU 工具。正式评测前必须返回退出码 0。

### 2. 冒烟测试

```bash
python -m m3rs_eval run --config configs/server.yaml --mode smoke --limit 2
```

每数据集任务只跑前 2 条样本，快速验证系统连通性和 JSONL 契约。冒烟结果不进入正式历史排名。

### 3. 全量正式评测

```bash
python -m m3rs_eval run --config configs/server.yaml --mode full
```

运行完整的四数据集评测流程：请求生成、系统调用、预测校验、指标计算、资源采集、运行注册和历史汇总。结果写入不可变运行包 `runs/<run_id>/`，并更新 `history/runs.csv`、`metrics_long.csv` 和 `coverage.csv`。

如需断点恢复：

```bash
python -m m3rs_eval run --config configs/server.yaml --mode full --resume --run-id <RUN_ID>
```

### 4. 重建评测表

```bash
python -m m3rs_eval rebuild-table
```

从历史 CSV/JSONL 重建 `评测表.xlsx`（Python openpyxl）。工作簿包含核心宽表、指标长表、运行元数据、数据覆盖、对比看板、指标字典和协议与质检八个工作表。

### 5. 生成报告上下文

```bash
python -m m3rs_eval prepare-report --run-id <RUN_ID>
# 或自动选择最新兼容运行：
python -m m3rs_eval prepare-report --latest-compatible
```

生成结构化 JSON 报告上下文（`reports/report_context_<run_id>.json`），随后使用 `prompts/生成评测对比报告提示词.md` 驱动 LLM 生成固定十节的 Markdown 评测对比报告。

---

## 目录地图

```text
标准化评测/
├── README.md                              # 本文件
├── 测试说明.md                             # 完整操作手册（12 节）
├── pyproject.toml                          # Python 包配置
├── configs/
│   ├── server.example.yaml                 # 服务器配置模板 → 复制为 server.local.yaml
│   └── fixture.yaml                        # 假系统测试配置（无需 GPU 即可验证）
├── protocols/
│   └── official_full_v1.yaml               # 锁定正式全量评测协议
├── schemas/
│   ├── request.schema.json                 # 标准请求 JSON Schema
│   ├── prediction.schema.json              # 标准预测 JSON Schema
│   ├── metric_record.schema.json           # 指标记录 JSON Schema
│   └── run_manifest.schema.json            # 运行清单 JSON Schema
├── registry/
│   └── metrics.yaml                        # 194 项统一指标注册表
├── src/m3rs_eval/                          # Python 评测框架
│   ├── cli.py                              # CLI 入口（doctor / run / rebuild-table / prepare-report）
│   ├── config.py                           # 配置加载
│   ├── contracts.py                        # JSONL 契约
│   ├── preflight.py                        # 环境预检（doctor）
│   ├── orchestrator.py                     # 运行编排
│   ├── command_adapter.py                  # 命令行适配器
│   ├── metadata.py                         # 自动元数据采集
│   ├── resources.py                        # 资源指标采集
│   ├── registry.py                         # 指标注册表
│   ├── history.py                          # 历史记录
│   ├── reporting.py                        # 报告上下文生成
│   ├── evaluation.py                       # 指标计算
│   └── datasets/                           # 四数据集适配器
├── tools/
│   ├── fake_system.py                      # 假系统（测试用，--behavior ok|missing|duplicate|malformed|error|nonzero|timeout）
│   └── build_workbook.py                  # Excel 工作簿构建器（Python）
├── prompts/
│   └── 生成评测对比报告提示词.md              # LLM 报告生成提示词
├── tests/                                  # pytest 自动测试
├── test_fixtures/                          # 测试夹具（微型数据与标注）
├── history/
│   ├── runs.csv                            # 历史运行注册表（CSV，事实来源）
│   ├── metrics_long.csv                    # 历史指标长表（CSV，事实来源）
│   └── coverage.csv                        # 历史覆盖数据（CSV，事实来源）
├── runs/                                   # 不可变运行包（Git 忽略）
├── reports/                                # 报告输出目录
└── 评测表.xlsx                              # 核心评测工作簿（可从历史 CSV 重建）
```

---

## 不可变运行规则

1. 已完成运行（`runs/<run_id>/`）的内容永不原地修改。重跑产生新 `run_id`。
2. 历史 CSV（`runs.csv`、`metrics_long.csv`、`coverage.csv`）是事实来源，由评测脚本写入，不得手动编辑。
3. `评测表.xlsx` 是派生展示层，每次 `rebuild-table` 从历史 CSV 重建，不应手动修改数值。
4. 缺失、重复、畸形和错误的预测均计入失败（`n_failures`），不得静默删除。
5. 只有 `mode=full`、`status=complete` 且协议检查通过的运行才能进入正式历史比较和排名。
6. 冒烟运行（`--mode smoke`）保存但标记为 `eligible_for_history=false`，不参与正式历史。

---

## 重要声明

当前版本中，`configs/server.yaml` 使用 `replace-with-*` 占位符。真实评测需要：

1. 获得服务器环境和 GPU 资源。
2. 下载 LEVIR-CC、VRSBench、XLRS-Bench 和 MME-RealWorld-RS 官方完整数据集。
3. 配置被测系统的启动命令、模型权重和数据集路径（见 `configs/server.example.yaml` 和 `测试说明.md` 第 3 节）。
4. 安装各数据集官方评测仓库并配置 `official_scorer_command`。

**在提供有效路径和官方评测命令前，系统可在夹具模式下使用以下命令完整验证流程：**

```bash
# 夹具模式验证（无需 GPU，无需真实数据）
python -m m3rs_eval doctor --config configs/fixture.yaml
python -m m3rs_eval run --config configs/fixture.yaml --mode smoke --limit 2
python -m m3rs_eval run --config configs/fixture.yaml --mode full
```

---

## 更多信息

- 完整操作说明、字段查找方法、JSONL 示例和故障排除请阅读 **[测试说明.md](./测试说明.md)**。
- 评测报告生成请阅读 **[prompts/生成评测对比报告提示词.md](./prompts/生成评测对比报告提示词.md)**。
- 设计文档、协议细节和验收标准见 `docs/superpowers/specs/` 和 `docs/superpowers/plans/`。
