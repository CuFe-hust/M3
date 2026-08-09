# M3 — New Architecture (new_structure)

本仓库正处于**新架构重建阶段**。行为参考是 `try_yolo` 分支的锁定提交
`ec962eb87c3ad0b8c1502efcbd08db0daec48868`（只读，不合并、不修改）。

## 当前状态（Task 00–33 完成，25.5–25.7 / 33.5 / 33.6 / 33.7 hardening 完成；新计划 Task 01–09 完成（06.5/06.6/06.6.1 hardening 含内）——核心运行时链路已就绪：Judge → SampleRunner → DatasetRunner → auto-task seam → Reporting → Application/Bootstrap → `main.py run-dataset`；Task 11A 手动 `ask` 路径已恢复：PublicAnswer/`Runtime.ask`/`main.py ask`；Task 11B 本地 HTTP 服务已恢复：`main.py serve`（隐式默认）/`GET /health`/`POST /ask`；Task 11C 核心运维命令已恢复：`run-init`/`health`/`list-datasets`/`smoke-qwen`/`resume-run`/`inspect-data`；Task 11C2 run-dataset 运维面已恢复：adapter default 任务模式/`--sample-ids`/`--num-shards` 别名/`--render-errors`/确定性 `--judge-sample-rate`（持久化，resume 一致）；Task 11D 计数维护工具已恢复：`count-image`/`render-count`/`summarize-evaluations`（基于当前 CountingAgent/SampleRunner/Reporting/EvaluationRecord）；Task 11E 离线评估运维已恢复：`evaluate-run`/`judge-vqa-run`（零 Qwen 构造/调用，共享确定性 dispatch 按执行任务定键，可选 DeepSeek 仅 Judge）；Task 11F 标准与数据集评估 seam 已实施：`evaluation/standard`（外部评估器适配器 + `standard-evaluate` CLI）与 `evaluation/datasets`（VRSBench 官方评估 seam）；Task 11G 基准/报告导出已恢复：`reporting/exporters` 的 samples.jsonl/deepseek_audit.jsonl/元数据 JSON/external_standard 命名空间/MME 官方提交导出（源只读））

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
  DatasetRunner + build_report + `build_dataset_run_options` + 手动 `ask`——
  PublicAnswer/`Runtime.ask`：单主 Agent、无 Judge/评测/fallback；显式任务零
  resolver，auto 经 TaskResolver（空问题确定性规则，有问即一次模型调用）；
  手动图片目录第一层自然排序收集（jpg/jpeg/png/webp/tif/tiff/bmp、忽略隐藏、
  损坏/超 8 张失败），`ImageRef.path` 相对图片目录且 `AgentContext.data_root`
  指向它，请求产物只含 `manual://input` + 相对路径、`artifact_dir` run-root
  相对）、commands（`ask` CLI 薄接线 + `serve` 串行本地 HTTP 服务——仅
  `GET /health`（就绪元数据，无模型/Judge 调用）与 `POST /ask`（委托
  `Runtime.ask`，source=http_service）；1 MiB 请求体上限（413 + 排空）、
  坏 JSON/非对象/缺 image_dir→400、非法请求→400 固定稳定错误、内部异常→500
  稳定类型名；未知路径 404 JSON；端口校验 1..65535；服务进程一次
  `Runtime.create()`（Qwen 一次、无 DeepSeek），handler 绝不构造模型客户端；
  仅 stdlib http.server；运维命令——`run-init`（RunStore.create_run，快照与
  fresh Runtime 一致，重复显式 id 稳定失败）、`health qwen|deepseek [--live]`
  （正常模式只输出元数据与 env 名、绝不输出密钥值；live 注入 fake 客户端或
  构造真实客户端后恰好一次探测）、`list-datasets`（DatasetRegistry.names()）、
  `smoke-qwen --image --question`（直接 VisionLanguageClient 一次请求，不经
  Agent 路由）、`resume-run --run-id`（读 RunManifest + config 快照 + run 目录
  task 命名空间重建 DatasetRunOptions 委托 Runtime resume=True；缺失/损坏/
  不足/不匹配稳定失败，绝不复制 DatasetRunner 循环）、`inspect-data --root
  --output [--scan-mode quick|full]`（audit_dataset_root 只读审计）、
  `count-image --image --question [--target-spec] [--run-id] [--evaluate]
  [--render] [--resume] [--force] [--no-seam-verify] [--max-qwen-calls]
  [--max-deepseek-calls]`（当前 CountingAgent + 当前 run/sample 存储；
  resume succeeded 零 Qwen、force 重跑、旧版绝对 status 无效重跑、budget/
  seam 覆盖仅请求局部；target-spec 经 CountTargetSpec 校验）、
  `render-count --image --result --output`（当前 overlay；tile 调试仅在
  有足够几何时，否则 tile_overlay=not_available，绝不猜测）、
  `summarize-evaluations --run-id`（解析全部当前 EvaluationRecord，损坏
  稳定失败，按 task 确定性聚合，无模型）、`evaluate-run --run-id
  [--deepseek] [--only-missing] [--force-judge]`（离线确定性评估，零 Qwen
  构造/调用；与 fresh/resume 共用共享 dispatch 且按执行任务 status.task
  定键；不支持任务/不兼容几何 → not_applicable 绝不伪造指标；--deepseek
  仅 Judge，失败保留确定性记录；--only-missing 只补缺失；输出刷新报告）、
  `judge-vqa-run --run-id [--force]`（零 Qwen；仅执行任务 general_vqa；
  succeeded judge 默认跳过、--force 重判；失败保留确定性记录）、
  `standard-evaluate --result [--tool-dir] [--output] [--python]`（外部团队
  标准评估器 seam：canonical result → evaluate.py → *.standard.json 校验
  JSON 对象；shell=False；失败稳定；仅当结果关联当前 run 时刷新报告；
  绝不复活旧 eval.audit_report）、`evaluation/datasets/`（VRSBench 官方
  评估 seam：仅答案归一化/官方输入映射/封闭词汇元数据，绝不选 Agent/调
  模型/改任务/重复通用指标，任务语义保留在数据层 normalizer））、报告导出
  （`reporting/exporters`：write_samples_jsonl/deepseek_audit/元数据/
  external_standard 命名空间/MME 官方提交——源数据只读、未关联字段原样
  保留、无主机绝对路径、无 auth/原始 secret；新统一 Reporting 权威，
  不重建旧 HTML builder）
- 公开入口：`main.py run-dataset` + `main.py ask` + `main.py serve`（无子命令
  隐式 serve）+ 运维命令（run-init/health/list-datasets/smoke-qwen/resume-run/
  inspect-data/count-image/render-count/summarize-evaluations/evaluate-run/
  judge-vqa-run/standard-evaluate 等）——
  极薄接线：解析（`--dataset/--root/--split/--task/--auto-task/--sample-ids/
  --run-id/--resume/--evaluate/--judge-policy/--judge-sample-rate/
  --render-errors/--max-samples/--start-index/--shard-index/--shard-count
  （别名 --num-shards）/--sample-concurrency/--fail-fast/--config`；
  任务选择模式：`--task` 显式 / `--auto-task` 逐样本 resolver / 两者都不给 →
  `adapter.supported_tasks`（不调 TaskResolver）/ 两者都给 → 参数错误；
  `--sample-ids` 文件空白分隔 ID 喂 selection 管线（先于执行与 limit）；
  evaluate 默认开、judge-policy 默认 none（offline by default））→ 配置 →
  运行时（Qwen 一次加载，多 task/样本复用）→ `build_dataset_run_options`
  （架构规则禁止 main 导入 workflows，构造在 application）→
  `Runtime.run_dataset` → 汇总 JSON 与 run_dir → 退出码（0/1/2/130）；
  `--judge-sample-rate` 由 SHA256(run_id:sample_id) 确定性抽样并持久化于
  summary（resume 恢复同一策略）；`--render-errors` 执行后渲染失败样本
  counting overlay（无模型调用，不支持转稳定 note）；公共错误只输出稳定
  类型名，绝无原始异常/密钥；run-dataset 命令实现在
  `application/commands/run_dataset.py`
- 数据集执行：`DatasetRunner`（selection 固定顺序 adapter 稳定序→start_index→shard→sample_ids→limit（SHA256 分片）、resume 只跳过 succeeded 且按 `status.task` 执行任务经**同一共享 dispatch** 补判缺失确定性评估（general_vqa/multiple_choice_vqa/scene_classification/counting/fine_grained_counting/caption/grounding 兼容坐标时；不兼容坐标系绝不伪造指标）与缺失或失败 VQA judge（仅 general_vqa，受确定性抽样率约束）、单进程 asyncio 并发、fail-fast 取消不遗留 running、未启动样本记 `FAIL_FAST_NOT_STARTED`、`DatasetRunSummary` 计数强制闭合（含持久化 `judge_sample_rate`；resume 无显式率时恢复持久化值）、确定性 judge 抽样 `_judge_policy_for`（SHA256(run_id:sample_id) 模 10000 与率比较，绝不随机）；目录 `runs/<run_id>/tasks/<task>/samples/<sha256(sample_id)[:24]>`；`task=None` 是内部显式 auto-task mode（外部入口经 `DatasetRunOptions.auto_task=True` 选择）、`tasks=None` 是 adapter default 模式（运行全部 supported_tasks，不调 resolver）；`predictions.jsonl` 是 **append-only execution index**——`(run_task, sample_id)` 最后一行代表当前状态，行字段 sample_id/run_task/task/status/result_path/updated_at；数据集 probe 经 `ArtifactWriter.write_dataset_probe` 按 task 目录独立写 `tasks/<task>/dataset_probe.json`（`sample_file` dataset-relative，root 外稳定失败）——manifest.json 保持 RunManifest schema 可解析、绝不动态扩 schema）
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
- 运维 CLI 其余命令不实现（核心运行时稳定后可加薄壳）。
- 验收状态：Task 10 Windows 集成验收通过（WINDOWS_INTEGRATION_READY，
  审计记录见 `docs/architecture/10_WINDOWS_INTEGRATION_GATE.md`）；
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
