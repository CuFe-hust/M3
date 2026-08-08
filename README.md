# M3 — New Architecture (new_structure)

本仓库正处于**新架构重建阶段**。行为参考是 `try_yolo` 分支的锁定提交
`ec962eb87c3ad0b8c1502efcbd08db0daec48868`（只读，不合并、不修改）。

## 当前状态（Task 00–33 完成，25.5–25.7 / 33.5 / 33.6 / 33.7 hardening 完成；新计划 Task 01–09 完成（06.5/06.6/06.6.1 hardening 含内）——核心运行时链路已就绪：Judge → SampleRunner → DatasetRunner → auto-task seam → Reporting → Application/Bootstrap → `main.py run-dataset`）

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
- 工作流：`workflows/`（CallBudget、EventWriter/RunStore、ArtifactWriter、TaskResolver、SampleRunner、运行契约；JSONL 写入进程内并发安全，跨进程并发追加不受当前工作流层支持）
- 单样本执行：`SampleRunner`（路由→attempt plan→Agent→产物→确定性评估→可选 judge→trace→状态；低置信度 TaskResolution 按最多 3 个候选构建去重 attempt plan，绝不跑所有 Agent；候选样本 model_copy 重建、不兼容稳定跳过；共享逐样本 CallBudget（可外部注入，贯穿 resolver 与 attempts）；确定性评估经**共享 dispatch**（`build_deterministic_evaluation`，fresh 与 resume 同一 helper）：general_vqa/multiple_choice_vqa/scene_classification→VQA exact-match（vqa_evaluation.json）、counting/fine_grained_counting→计数（counting_evaluation.json，非 CountingResult 载荷 fail-closed）、grounding→轴对齐 IoU（grounding_evaluation.json，**仅 prediction 与 GT 同为 normalized_0_999_top_left 且均为 4-value xyxy 时产出**；source_pixels_top_left 与 8-point polygon 当前 fail-closed，等待 official evaluator / 显式坐标转换）、caption→逐样本候选+参考（caption_evaluation.json）；spatial_relation/change_qa/change_caption 无样本级指标；仅 general_vqa 走 judge 且经 asyncio.to_thread 不阻塞事件循环）
- 可移植产物路径：`SampleRunStatus.result_path` = sample-relative 结果 basename（如 `agent_result.json`/`counting_result.json`），**schema 级强制 plain basename**（拒绝 absolute/drive/UNC/dot-dot/嵌套/控制字符；旧版绝对路径 status 在 resume 时视为无效并重新执行样本）；`predictions.jsonl` 行内 `result_path` = run-relative（由实际 sample 目录推导，如 `tasks/auto/samples/<key>/agent_result.json`，绝不根据 status.task 拼目录）；机器绝对路径绝不进入任何产物；resume 确定性评估以 `status.task` 作为 execution task 显式传给共享 dispatch（候选兜底后不会因 canonical resolved sample.task 生成错误指标族）
- 候选兜底一致性：`sample.json` 保存 canonical resolved sample（task=解析任务，绝不因 fallback 覆盖）；`status.json` 保存最终执行 task；`agent_trace.json` 显式记录 `resolved_task` 与 `execution_task`（`task_type` 固定等于 resolved_task）；`routing_decision.json` 最终写成功 execution task 的决策；resume 按 `status.task`（执行任务）决定补判类型，绝不按解析任务误补；`fallback_used` 涵盖 routing fallback / partial fallback / 候选任务兜底（candidate index > 0）
- 报告：`reporting/`（只读层：schema/adapters/builder/html/exporters/visualization）
  —— 从执行索引与样本产物构建 `Report`（逐样本行 + 每 task 汇总：状态计数、
  fallback 率、agent 使用、judge 状态、离线确定性指标聚合；caption 只计记录
  数，语料级指标留给可选 pycocoevalcap）；HTML 完全离线（无 CDN/Base64，用户
  与模型文本全部转义，只输出稳定 code 与 run 相对路径）；CSV utf-8-sig；
  counting overlay（源图 + CountingResult，尺寸不匹配稳定失败）；绝不调用
  模型、绝不重新计算结果
- 应用层：`application/`（唯一 composition root）——settings（YAML + 环境变量
  覆盖，密钥值绝不进入 snapshot/repr/artifact，只声明环境变量名）、prompts
  （17 个逻辑键现役绑定，构造时一次性加载，缺失明确报错，快照路径去重）、
  bootstrap（唯一创建 Qwen 客户端——一次组装恰好一次；DeepSeek 仅在注入
  api_key 时创建，无 key 即 judge 禁用；BackendRegistry/AgentRegistry/
  TaskResolver/JudgeService/SampleRunner/DatasetRunner factory/Reporting
  服务全部组装；路由覆盖校验）、runtime（高层用例：run_dataset 委托
  DatasetRunner + build_report + `build_dataset_run_options`；不做 ask/serve/CLI）
- 公开入口：`main.py run-dataset`（唯一 CLI；不实现 serve/ask/health/run-init/
  resume-run/evaluate-run/judge-vqa-run/inspect-data/render-count/smoke-qwen 等）——
  极薄接线：解析（`--dataset/--root/--split/--task/--auto-task/--run-id/--resume/
  --evaluate/--judge-policy/--max-samples/--start-index/--shard-index/--shard-count/
  --sample-concurrency/--fail-fast/--config`；`--task` 与 `--auto-task` 互斥；
  evaluate 默认开、judge-policy 默认 none（offline by default））→ 配置 → 运行时
  （Qwen 一次加载，多 task/样本复用）→ `build_dataset_run_options`（架构规则禁止
  main 导入 workflows，构造在 application）→ `Runtime.run_dataset` → 汇总 JSON 与
  run_dir → 退出码（0/1/2/130）；公共错误只输出稳定类型名，绝无原始异常/密钥
- 数据集执行：`DatasetRunner`（selection 固定顺序 adapter 稳定序→start_index→shard→sample_ids→limit（SHA256 分片）、resume 只跳过 succeeded 且按 `status.task` 执行任务经**同一共享 dispatch** 补判缺失确定性评估（general_vqa/multiple_choice_vqa/scene_classification/counting/fine_grained_counting/caption/grounding 兼容坐标时；不兼容坐标系绝不伪造指标）与缺失或失败 VQA judge（仅 general_vqa）、单进程 asyncio 并发、fail-fast 取消不遗留 running、未启动样本记 `FAIL_FAST_NOT_STARTED`、`DatasetRunSummary` 计数强制闭合；目录 `runs/<run_id>/tasks/<task>/samples/<sha256(sample_id)[:24]>`；`task=None` 是内部显式 auto-task mode（未来外部入口经 `DatasetRunOptions.auto_task=True` 选择）；`predictions.jsonl` 是 **append-only execution index**——`(run_task, sample_id)` 最后一行代表当前状态，行字段 sample_id/run_task/task/status/result_path/updated_at；数据集 probe 经 `ArtifactWriter.write_dataset_probe` 按 task 目录独立写 `tasks/<task>/dataset_probe.json`（`sample_file` dataset-relative，root 外稳定失败）——manifest.json 保持 RunManifest schema 可解析、绝不动态扩 schema）
- 任务解析：`TaskResolver`（仅缺失 task 时可调用本地模型；明确 task 直接通过；空问题仅 caption/change_caption 两条确定性规则；低置信度只返回结构化候选，不执行 Agent；`TaskRouter` 保持同步确定性、不读 question、不调用模型）
- 无 task 数据集 seam：`SampleDraft`（data/schema.py，pre-sample 契约、无角色校验）+ `DraftDatasetAdapter` 协议 + `data/adapters/manifest.py` 的 manifest 驱动 draft 适配器（`spacers_adapter.json` 显式字段映射、JSON/JSONL、task 列可选、不猜字段、不调模型、不 import workflows/models；`samples_file` 经 `resolve_dataset_relative_path` 严格限制在 dataset root 内，拒绝遍历/绝对路径/UNC）+ `materialize_sample`（draft→UnifiedSample，任务解析后重建图像角色）+ DatasetRunner draft 模式（共享默认 CallBudget 贯穿 resolver 与 agent attempts，未知 task 绝不冒充 general_vqa，预 task 失败以稳定 code + 诚实 `unknown` 标签记录，单样本异常隔离不炸整批；`DatasetRunOptions.auto_task` 显式契约：auto_task=True 要求 tasks 为空、False 要求非空）
- 评估：`evaluation/`（统一 EvaluationRecord 与确定性指标：counting/VQA/grounding/caption；corpus 级 caption 指标依赖可选 pycocoevalcap；judge 永不覆盖确定性指标）
- Judge 层：`evaluation/judges/`（文本与结构化证据 Schema、JudgeClient 协议、纯载荷/稳定哈希——canonical hash 覆盖 model/prompt 文本与版本/sample_id/完整 payload/response schema、标准库 HTTP 的 DeepSeekJudgeClient——api_key 由 composition root 注入、绝不读环境变量、错误只含稳定 code）+ `workflows/judge_service.py`（策略 none/errors-only/all、仅真正发起 Judge 时 `reserve_deepseek`、失败保留确定性记录、resume 补判不重复；judge 异常以稳定类型名记录，绝不保存原始异常文本；`sample_dir/deepseek_vqa_judge/` 与 `samples/<id>/deepseek/` 产物：request_meta/raw_response/validation/parsed；async 边界经 `asyncio.to_thread` 不阻塞事件循环）

计数后端契约：每个后端显式声明 `kind`（`qwen_point`/`quantity_proposal`/`yolo_obb`）；
只有 `yolo_obb` 进入 detector plan、zero-review 与 detector fallback；所有 YOLO tile
均失败时 backend 抛出稳定错误并由 `CountingPlanExecutor` 执行显式 fallback
（`CountingAgent` 只负责计划与打包）；tile warning 不保存原始异常文本；
`CountingBackendUnavailableError` 全仓唯一（`agents.errors` 权威定义，顶层导出与
backend import 为同一对象）；公共入口只抛稳定错误，trace 不含原始异常文本、
绝对路径、密钥或 Base64。

**尚未实现**：
- **semantic segmentation runtime: not implemented**——SegFormer/OEM/iSAID
  是独立扩展线（见 `docs/architecture/12_SEGMENTATION_TRACK.md`），不阻塞
  core release candidate。
- `serve`/`ask`/运维 CLI 不实现（核心运行时稳定后可加薄壳）。
- 验收状态：Task 10 Windows 集成验收通过（WINDOWS_INTEGRATION_READY）；
  Task 11 Spark 真机验收被 ENVIRONMENT_BLOCKER 阻塞（本地
  Qwen3-VL-4B-Instruct checkpoint 权重缺失，就位后重跑）。

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
