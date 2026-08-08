# M3 — New Architecture (new_structure)

本仓库正处于**新架构重建阶段**。行为参考是 `try_yolo` 分支的锁定提交
`ec962eb87c3ad0b8c1502efcbd08db0daec48868`（只读，不合并、不修改）。

## 当前状态（Task 00–33 完成，25.5–25.7 / 33.5 / 33.6 / 33.7 hardening 完成；新计划 Task 01 审计、Task 02 白名单、Task 03 Judge 层完成）

- 迁移基线文档：`docs/migration/BASELINE_INVENTORY.md`、`BASELINE_COMMANDS.txt`
- Golden fixtures（离线行为契约）：`tests/fixtures/migration/`
- 架构守卫（文件白名单 / import 依赖 DAG / 旧包禁止 / 打包发现）：`tests/architecture/`。import DAG 边界：领域层（agents/workflows/evaluation.metrics）只依赖模型协议（`models.base`/`models.images`），具体模型实现（`models.entry`/`models.qwen_transformers`/`models.qwen3_*`）只允许 composition root（application）选择；`routing` 不依赖 models；`evaluation.judges` 经 path rule 显式批准后可依赖模型契约/配置（path rule 优先于 package 规则）
- 数据层：`data/`（统一样本契约、4 个数据集 Adapter、校验/选择/审计）
- 模型层基础：`models/`（协议/缓存/图像工具/配置声明、统一 entry、本地 Transformers Qwen 客户端、Qwen3-VL 基线封装）
- Agent 通用契约：`agents/`（AgentResult/VisualEvidence、AgentContext/AgentExecution、Registry、错误类型、数据集无关 VisualAgentBase）
- 领域 Agents：`agents/general_vqa/`、`agents/caption/`、`agents/grounding/`（薄视觉 Agent，含 MCQ/Grounding 输出约束）
- 计数子系统：`agents/counting/`（契约/几何/证据/pipeline/backends/选择器/目标解析/CountingAgent/CountingPlanExecutor，主输出恒为 CountingResult，后端使用显式 kind；Executor 承担 primary 执行、运行时 fallback 与 zero review）
- 空间子系统：`agents/spatial/`（通用 SpatialQuerySpec、几何规则、候选复核、证据合并与 SpatialAgent；候选复核使用真实内容 MIME，canonical label 不做词形猜测）
- 变化子系统：`agents/change/`（PairValidator/Harmonizer/DifferenceProposal/Preprocess/Reviewer/双路径 ChangeAgent；cv2 与 numpy 为可选依赖 `[change]` extra，base 导入不触发；无效时相图对在模型调用前稳定失败）
- 路由：`routing/`（同步确定性 Thin Router，不读 question、不调用模型）
- 工作流：`workflows/`（CallBudget、EventWriter/RunStore、ArtifactWriter、TaskResolver、运行契约；JSONL 写入进程内并发安全，跨进程并发追加不受当前工作流层支持）
- 任务解析：`TaskResolver`（仅缺失 task 时可调用本地模型；明确 task 直接通过；空问题仅 caption/change_caption 两条确定性规则；低置信度只返回结构化候选，不执行 Agent；`TaskRouter` 保持同步确定性、不读 question、不调用模型）
- 评估：`evaluation/`（统一 EvaluationRecord 与确定性指标：counting/VQA/grounding/caption；corpus 级 caption 指标依赖可选 pycocoevalcap；judge 永不覆盖确定性指标）
- Judge 层：`evaluation/judges/`（文本与结构化证据 Schema、JudgeClient 协议、纯载荷/哈希构建、标准库 HTTP 的 DeepSeekJudgeClient——api_key 由 composition root 注入、绝不读环境变量、错误只含稳定 code）+ `workflows/judge_service.py`（策略 none/errors-only/all、仅真正发起 Judge 时 `reserve_deepseek`、失败保留确定性记录、resume 补判不重复；judge 异常以稳定类型名记录，绝不保存原始异常文本；`sample_dir/deepseek_vqa_judge/` 与 `samples/<id>/deepseek/` 产物：request_meta/raw_response/validation/parsed）

计数后端契约：每个后端显式声明 `kind`（`qwen_point`/`quantity_proposal`/`yolo_obb`）；
只有 `yolo_obb` 进入 detector plan、zero-review 与 detector fallback；所有 YOLO tile
均失败时 backend 抛出稳定错误并由 `CountingPlanExecutor` 执行显式 fallback
（`CountingAgent` 只负责计划与打包）；tile warning 不保存原始异常文本；
`CountingBackendUnavailableError` 全仓唯一（`agents.errors` 权威定义，顶层导出与
backend import 为同一对象）；公共入口只抛稳定错误，trace 不含原始异常文本、
绝对路径、密钥或 Base64。

**尚未实现**：reporting、application 与 CLI（`main.py`）。Task 34 尚未开始；
`SampleRunner`/`DatasetRunner` 尚未实现（`TaskResolver` 尚未被 dataset runner
使用）。请勿将其当作可用功能使用。

## 安装与测试

```bash
python -m pip install -e ".[dev,migration,change]"
python -m compileall \
  data \
  models \
  agents \
  routing \
  workflows \
  evaluation \
  tests \
  scripts/generate_migration_fixtures.py
python -m pytest -q tests/architecture
python -m pytest -q tests/contracts/test_data_schema_contract.py
python -m pytest -q tests/parity/test_baseline_golden_fixtures.py
python -m pytest -q tests/workflows
python -m pytest -q tests/evaluation
python -m pytest -q
```

GitHub Actions（`.github/workflows/offline-tests.yml`，Ubuntu/Python 3.11）
执行上述 Foundation tests；不运行 live 模型、真实数据集或密钥相关测试，
也不下载真实模型权重、不运行 live GPU inference。

## 运行时边界说明

- **Change 可选依赖**：base wheel 的 `import agents.change` 不要求 cv2/numpy
  （惰性加载）；运行变化预处理/一致化/提议需要 `pip install "m3[change]"`
  （`numpy` + `opencv-python-headless`）。缺少时相关函数抛出
  `OptionalDependencyMissingError`。
- **JSONL 并发边界**：`events.jsonl` / `predictions.jsonl` 的写入在单 Python
  进程内对并发 writer 安全（按解析后路径身份的进程内锁 + 原子替换 + 有限
  重试吸收 Windows 瞬态文件锁）；当前工作流层不支持跨进程并发追加。
- **Caption 指标**：corpus 级 BLEU/METEOR/ROUGE/CIDEr 依赖可选
  `pycocoevalcap`，缺少时 `evaluate_caption` 抛出明确 `RuntimeError`。
- **Workflow 安全**：`run_id` 是经过校验的跨平台 plain identifier（拒绝
  遍历/绝对路径/drive/UNC/控制字符、Windows 保留设备名（CON/PRN/AUX/NUL/
  COM1-9/LPT1-9，含带扩展名形式）与尾点/尾空格，任何文件写入前失败）；
  事件与配置的密钥扫描递归处理任意 Mapping/Sequence/Set（含 tuple、set、
  frozenset 组合）嵌套。
- **Evaluation 不变式**：`EvaluationRecord` 强制 task ↔ deterministic metric
  类型一致（构造时失败，聚合器同样 fail-closed）；VQA canonical merge 返回
  统一 `EvaluationRecord`（旧 `VQAEvaluationRecord` 仅作 legacy 兼容，可经
  `to_evaluation_record` 显式转换）；Grounding IoU 阈值每条记录只判定一次，
  聚合复用存储标志；task↔metrics 映射为内部不可变注册表，不可通过公共
  顶层 API 修改。

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
