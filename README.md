# M3 — New Architecture (new_structure)

本仓库正处于**新架构重建阶段**。行为参考是 `try_yolo` 分支的锁定提交
`ec962eb87c3ad0b8c1502efcbd08db0daec48868`（只读，不合并、不修改）。

## 当前状态（Task 00–15 完成）

- 迁移基线文档：`docs/migration/BASELINE_INVENTORY.md`、`BASELINE_COMMANDS.txt`
- Golden fixtures（离线行为契约）：`tests/fixtures/migration/`
- 架构守卫（文件白名单 / import 依赖 DAG / 旧包禁止）：`tests/architecture/`
- 数据层：`data/`（统一样本契约、4 个数据集 Adapter、校验/选择/审计）
- 模型层基础：`models/`（协议/缓存/图像工具/配置声明、统一 entry、本地 Transformers Qwen 客户端、Qwen3-VL 基线封装）
- Agent 通用契约：`agents/`（AgentResult/VisualEvidence、AgentContext/AgentExecution、Registry、错误类型、数据集无关 VisualAgentBase）

**尚未实现**：具体领域 Agents（counting/spatial/change/grounding/caption/general_vqa）、
routing、workflows、evaluation、reporting、application 与 CLI（`main.py`）。
请勿将其当作可用功能使用。

## 安装与测试

```bash
python -m pip install -e ".[dev,migration]"
python -m compileall data tests scripts/generate_migration_fixtures.py
python -m pytest -q tests/architecture
python -m pytest -q tests/contracts/test_data_schema_contract.py
python -m pytest -q tests/parity/test_baseline_golden_fixtures.py
python -m pytest -q
```

GitHub Actions（`.github/workflows/offline-tests.yml`，Ubuntu/Python 3.11）
执行上述 Foundation tests；不运行 live 模型、真实数据集或密钥相关测试。

## 目录职责（已实现部分）

- `architecture/allowed_python_files.txt`：**最终架构白名单**（冻结文件，普通任务
  不得修改）；白名单中尚未创建的文件是已批准的未来路径，不代表已经实现
- `architecture/implementation_status.json`：当前实际实现状态（implemented/pending）
- `architecture/ALLOWLIST_CHANGE_POLICY.md`：白名单变更政策
- `data/`：统一样本契约（仅此一层已实现）
- `tests/`：架构守卫 / 契约 / Golden parity 测试
- `docs/migration/`：迁移基线、Golden 说明
- `scripts/`：Golden fixture 生成器（离线可复现）
