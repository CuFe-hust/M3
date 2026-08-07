# M3 — New Architecture (new_structure)

本仓库正处于**新架构重建阶段**。行为参考是 `try_yolo` 分支的锁定提交
`ec962eb87c3ad0b8c1502efcbd08db0daec48868`（只读，不合并、不修改）。

## 当前状态（Task 00–25 完成，25.5/25.6 hardening 完成）

- 迁移基线文档：`docs/migration/BASELINE_INVENTORY.md`、`BASELINE_COMMANDS.txt`
- Golden fixtures（离线行为契约）：`tests/fixtures/migration/`
- 架构守卫（文件白名单 / import 依赖 DAG / 旧包禁止 / 打包发现）：`tests/architecture/`
- 数据层：`data/`（统一样本契约、4 个数据集 Adapter、校验/选择/审计）
- 模型层基础：`models/`（协议/缓存/图像工具/配置声明、统一 entry、本地 Transformers Qwen 客户端、Qwen3-VL 基线封装）
- Agent 通用契约：`agents/`（AgentResult/VisualEvidence、AgentContext/AgentExecution、Registry、错误类型、数据集无关 VisualAgentBase）
- 领域 Agents：`agents/general_vqa/`、`agents/caption/`、`agents/grounding/`（薄视觉 Agent，含 MCQ/Grounding 输出约束）
- 计数子系统：`agents/counting/`（契约/几何/证据/pipeline/backends/选择器/目标解析/CountingAgent，主输出恒为 CountingResult，后端使用显式 kind）
- 路由：`routing/`（同步确定性 Thin Router，不读 question、不调用模型）

计数后端契约：每个后端显式声明 `kind`（`qwen_point`/`quantity_proposal`/`yolo_obb`）；
只有 `yolo_obb` 进入 detector plan、zero-review 与 detector fallback；
`CountingBackendUnavailableError` 全仓唯一（`agents.errors` 权威定义，顶层导出与
backend import 为同一对象）；公共入口只抛稳定错误，trace 不含原始异常文本、
绝对路径、密钥或 Base64。

**尚未实现**：spatial、change Agents、workflows、evaluation、reporting、
application 与 CLI（`main.py`）。请勿将其当作可用功能使用。Task 26 尚未开始。

## 安装与测试

```bash
python -m pip install -e ".[dev,migration]"
python -m compileall \
  data \
  models \
  agents \
  routing \
  tests \
  scripts/generate_migration_fixtures.py
python -m pytest -q tests/architecture
python -m pytest -q tests/contracts/test_data_schema_contract.py
python -m pytest -q tests/parity/test_baseline_golden_fixtures.py
python -m pytest -q
```

GitHub Actions（`.github/workflows/offline-tests.yml`，Ubuntu/Python 3.11）
执行上述 Foundation tests；不运行 live 模型、真实数据集或密钥相关测试，
也不下载真实模型权重、不运行 live GPU inference。

## 模型身份配置说明

- 本地 checkpoint 使用路径时必须显式设置稳定的 `cache_model_id`
- `cache_model_id` 必须是逻辑标识符，不能是本地路径（POSIX 绝对、
  Windows drive、UNC）或 `file://` URI；它用于 request hash 与 Agent trace，
  物理 checkpoint 路径只传给 `from_pretrained`

## 目录职责（已实现部分）

- `architecture/allowed_python_files.txt`：**最终架构白名单**（冻结文件，普通任务
  不得修改）；白名单中尚未创建的文件是已批准的未来路径，不代表已经实现
- `architecture/implementation_status.json`：当前实际实现状态（implemented/pending）
- `architecture/ALLOWLIST_CHANGE_POLICY.md`：白名单变更政策
- `data/`：统一样本契约、Adapter、校验/选择/审计
- `models/`：模型协议与请求哈希、响应缓存、图像工具、纯声明配置、
  统一模型入口、本地 Transformers Qwen 客户端、Qwen3-VL 基线封装
- `agents/`：AgentResult/AgentExecution 契约、安全校验、Registry、
  错误类型、数据集无关 VisualAgentBase、general_vqa/caption/grounding
  薄视觉 Agent、计数子系统（契约/几何/证据/pipeline/backends/选择器/Agent）
- `routing/`：同步确定性 Thin Router（task→policy 表，不读 question）
- `tests/`：架构守卫 / 契约 / Golden parity 测试
- `docs/migration/`：迁移基线、Golden 说明
- `scripts/`：Golden fixture 生成器（离线可复现）
