# M3 — New Architecture (new_structure)

本仓库正处于**新架构重建阶段**。行为参考是 `try_yolo` 分支的锁定提交
`ec962eb87c3ad0b8c1502efcbd08db0daec48868`（只读，不合并、不修改）。

## 当前状态（Task 00–03 完成）

- 迁移基线文档：`docs/migration/BASELINE_INVENTORY.md`、`BASELINE_COMMANDS.txt`
- Golden fixtures（离线行为契约）：`tests/fixtures/migration/`
- 架构守卫（文件白名单 / import 依赖 DAG / 旧包禁止）：`tests/architecture/`
- 数据层统一样本契约：`data/schema.py`（TaskName、ImageRef、GroundTruth、
  TaskNormalization、UnifiedSample、ValidationIssue、stable_sample_id）

**尚未实现**：模型、Agent、Router、Workflow、Evaluation、Reporting、
Application 与 CLI（`main.py`）。请勿将其当作可用功能使用。

## 安装与测试

```bash
python -m pip install -e ".[dev]"
python -m compileall data tests scripts/generate_migration_fixtures.py
python -m pytest -q tests/architecture
python -m pytest -q tests/contracts/test_data_schema_contract.py
python -m pytest -q tests/parity/test_baseline_golden_fixtures.py
python -m pytest -q
```

GitHub Actions（`.github/workflows/offline-tests.yml`，Ubuntu/Python 3.11）
执行上述 Foundation tests；不运行 live 模型、真实数据集或密钥相关测试。

## 目录职责（已实现部分）

- `architecture/`：文件白名单、import 规则、实施状态清单
- `data/`：统一样本契约（仅此一层已实现）
- `tests/`：架构守卫 / 契约 / Golden parity 测试
- `docs/migration/`：迁移基线、Golden 说明
- `scripts/`：Golden fixture 生成器（离线可复现）
