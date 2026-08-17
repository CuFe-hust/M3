# 18 — 删除 VisualTaskPlan confidence 的实施计划

> Status: implemented; fresh planner and historical resume semantics are now wired.
>
> 状态：已实施；fresh planner 与历史 resume 语义已完成接线。
>
> Baseline inspected: `2191ac6aa60eff4aab5a4a5381622681ed81afdb` plus the
> uncommitted doc-16/doc-17 implementation present in the workspace on 2026-08-17.
> The implementation agent must re-run preflight checks and preserve all unrelated
> user changes in the dirty worktree.

## 1. 给实施代理的上文

当前唯一 fresh planning 链路是 `VisualTaskPlanner`：

```text
thumbnail-or-full-image block(s) + raw question
  -> exactly one planning Qwen call
  -> VisualTaskPlan
  -> deterministic MaterializedVisualView(s)
  -> planned task materialization
  -> deterministic TaskRouter
  -> selected Agent
```

第一次 Qwen 的 user content 只能包含有序图像块与原始问题，不得附加 source task、
metadata、Ground Truth、image id/role、尺寸、backend 或 checkpoint。冻结图像规则保持不变：

1. 仅当最长边大于 `1080` 时等比例缩到 `1080`，否则发送整图，永不放大；
2. 大图采用方案 A：`width > 1024 and height > 1024`；
3. 只有“大图 + 问题明确指定区域”才物化一个 `1024 x 1024` 固定 ROI；
4. 其他情况默认全图；
5. direct 与 evidence 路径消费同一组 `MaterializedVisualView`，不加 halo。

当前 `VisualTaskPlan` 仍要求模型返回：

```json
{
  "version": "visual-task-plan-v2",
  "task": "general_vqa",
  "needs_visual_assistance": false,
  "object_categories": [],
  "region_request": {
    "explicit": false,
    "image_index": null,
    "focus_xy_norm": null
  },
  "confidence": 0.93,
  "reason_codes": []
}
```

`VisualTaskPlanner` 再将该分数与配置的 `confidence_threshold=0.70` 比较，低于阈值时抛出
`VisualTaskPlanError("LOW_CONFIDENCE")`。用户现在要求删除这一整套 planner confidence
语义。

## 2. 本任务的准确语义

删除后，现役 planner 输出应为：

```json
{
  "version": "visual-task-plan-v3",
  "task": "general_vqa",
  "needs_visual_assistance": false,
  "object_categories": [],
  "region_request": {
    "explicit": false,
    "image_index": null,
    "focus_xy_norm": null
  },
  "reason_codes": []
}
```

必须同时满足：

1. `VisualTaskPlan` 不再声明、序列化或持久化 `confidence`；
2. planner prompt 不再要求模型估计或返回 confidence/probability/certainty；
3. `VisualTaskPlanner` 不再接受 `confidence_threshold`，不再保存阈值，也不再产生
   planner 的 `LOW_CONFIDENCE` 失败；
4. schema 合法、task/category/image index 合法且 capability 可执行的计划直接被接受；
5. 不得用 `is_confident`、`uncertain`、logprobs、candidate tasks、reason code 或其他字段
   重新实现同一阈值门；
6. `reason_codes` 只保留为决策事实的安全审计标签，不能成为隐式 confidence 数值或
   accept/reject 信号；
7. schema/类别/capability/图像索引/图像解码/预算/客户端错误仍按现有稳定错误码
   fail closed；
8. 规划调用次数、模型身份、request hash、图像/ROI 语义和下游执行不变。

没有 confidence 后，模型仍必须输出一个合法 task 和完整规划意图。系统不会因为模型主观
表示“不确定”而再跑其他 task、第二个 planner 或旧 resolver。

## 3. 为什么必须升级到 v3

这不是向后兼容的可选字段调整：`confidence` 当前是 v2 response schema 的必填字段，且
低分会改变样本终态。原地保留 `visual-task-plan-v2` 名称会导致同一个版本字符串对应两种
JSON shape 和两种执行语义，破坏 artifact 审计、cache 隔离与 resume 可复现性。

因此实施必须冻结为新版本：

```text
schema version       visual-task-plan-v3
prompt asset         prompts/visual_task_plan_v3.md
prompt catalog       visual_task_plan -> v3
planning mode        visual-task-plan-v3
prompt snapshot      visual_task_plan_v3.runtime.md
artifact basename    visual_task_plan.json          # 保持不变
```

保留 `visual_task_plan.json` basename 是有意行为：它表示 artifact 的逻辑种类，payload 内的
`version` 区分 v2/v3。不得另建平行 writer 或复制 reporting 流程。

prompt version 和 `VisualTaskPlan.model_json_schema()` 都进入 request hash，因此 v3 请求不会
命中 v2 cache。不得通过忽略旧 response 中的 confidence 来复用 v2 cache；v3 schema 继续
`extra="forbid"`，带 `confidence` 的 v3 response 必须判为 `SCHEMA_INVALID`。

## 4. 删除与保留边界

### 4.1 必须删除

- `agents.schema.VisualTaskPlan.confidence`；
- `prompts/visual_task_plan_v2.md` 中的 confidence 输出字段及任何置信度要求；
- `VisualTaskPlanner.__init__(confidence_threshold=...)`、范围校验、成员变量；
- `_post_validate()` 中的分数比较和 planner `LOW_CONFIDENCE` 分支；
- `VisualPlannerSettings.confidence_threshold`；
- bootstrap 对 planner threshold 的读取和注入；
- v2 planner fake response/fixture 中的 confidence；
- “低 confidence 会拒绝 planner plan”的测试与当前事实文档；
- v3 新 artifact、trace、request 或报告中的 planner confidence。

### 4.2 必须保留的其他 confidence

本任务只删除**视觉任务规划器输出的 confidence**。以下字段/规则不得顺手删除或改义：

- `agents.schema.VisualEvidence.confidence`；
- counting point、seam、detector、YOLO/SegFormer 内部 confidence；
- `VisualDetectorSettings.confidence_threshold` 和 grounding/VQA evidence detector policy；
- `CountingSettings.min_confidence`；
- Judge 或其他业务结果中合法、已有契约的 confidence；
- 数据 normalizer、Golden fixture 和 parity 记录中的 confidence；
- counting 的 `LOW_CONFIDENCE` rejection reason；
- `RouterSettings.confidence_threshold` 即使当前可能已无消费者，也不在本任务中处理；若要
  清理，应先独立确认职责和兼容影响。

全仓搜索 `confidence` 会命中大量合法领域逻辑，不能进行全局机械替换。

### 4.3 其他必须保持不变的契约

- `UnifiedSample` / `SampleDraft` 字段和 materialization；
- source task 仅审计、planned task 权威；
- `TaskRouter` 确定性与 Agent fallback；
- `needs_visual_assistance`、`object_categories`、`RegionRequest`、
  `MaterializedVisualView`；
- 1080 shrink-only、方案 A、固定 1024 ROI、无明确区域全图；
- model/checkpoint/processor/tokenizer、cache logical identity；
- CallBudget 计数和第一次规划调用恰好一次；
- GT、deterministic evaluation、Judge 解耦、split 和样本纳入规则；
- artifact 原子写入、路径安全、predictions append-only 和 summary 闭合；
- reporting 对历史 artifact 的只读能力。

## 5. schema、prompt 与 planner 修改

### 5.1 `agents/schema.py`

1. 将 `VISUAL_TASK_PLAN_SCHEMA_VERSION` 改为 `visual-task-plan-v3`；
2. 从 `VisualTaskPlan` 删除 `confidence` 字段；
3. 保持 `extra="forbid"`，确保旧 shape 不能伪装成 v3；
4. 保持 assistance/category、region linkage 和 reason-code 安全校验不变；
5. 不添加 optional/default confidence，也不添加替代字段。

至少增加以下 schema 断言：

- 不带 confidence 的 v3 plan 校验成功；
- v3 response schema 的 `properties` 和 `required` 均无 `confidence`；
- 显式携带 `confidence` 因 extra field 稳定失败；
- v2 version 不能作为 v3 plan 校验通过；
- 其他 linkage 校验保持原行为。

### 5.2 prompt 资产

新增 `prompts/visual_task_plan_v3.md`，输出示例不含 confidence，并明确：

```text
Do not return a confidence, probability, certainty score, uncertainty flag,
candidate task list, or any substitute for those fields.
```

在 `application/prompts.py` 将 `visual_task_plan` 绑定改为 v3。新 prompt 经现有
`PromptCatalog` 进入 run prompt snapshot 和 manifest hash。

v3 接线和测试完成后，删除仓库中的未绑定 `prompts/visual_task_plan_v2.md`，避免它被误当作
现役资产。历史 v2 run 已在自己的 `prompts.snapshot/` 中保存实际 prompt；不得修改已有 run
或伪造历史 snapshot。doc 16/migration 文档可以继续提到 v2 作为历史版本。

### 5.3 `workflows/visual_planner.py`

1. 默认 `prompt_version` 改为 `v3`；
2. 删除构造参数 `confidence_threshold`、阈值范围校验和 `_confidence_threshold`；
3. `_post_validate()` 只执行 category/capability/image-index 校验，不读取主观分数；
4. 删除 planner `LOW_CONFIDENCE` 错误路径；
5. `planning_parameters["planning_mode"]` 改为 `visual-task-plan-v3`；
6. prompt snapshot basename 改为 `visual_task_plan_v3.runtime.md`；
7. module/class/docstring 不再将 confidence 描述为 planner policy；
8. 维持图像 user content、response schema hash、budget 和 materialized view 逻辑逐行为等价。

注意：`request_hash` 已覆盖 prompt version、messages 和 response schema。不要删减任何 hash
输入，也不要为了兼容 v2 手工删除 response 的 confidence 后再校验。

## 6. settings 与 composition root

在 `application/settings.py`：

- 删除 `VisualPlannerSettings.confidence_threshold`；
- `task_prompt_version` 默认改为 `v3`；
- `planning_mode` 唯一 fresh 值改为 `visual-task-plan-v3`；
- 保留 catalog version、preview/ROI/large-image policy；evidence catalog 未改变，不要仅因
  planner schema 升级而改写 `first-qwen-evidence-catalog-v2` 身份；
- 因 `extra="forbid"`，用户 YAML 若继续在 visual planner 下提供
  `confidence_threshold`，应明确配置校验失败，不要静默忽略。

在 `application/bootstrap.py`：

- 删除 `confidence_threshold=planner_settings.confidence_threshold`；
- 更新 DatasetRunner factory 的 current planning-mode 默认值；
- 不影响 detector/evidence service 的 threshold 注入。

配置 snapshot 中旧 v2 planner threshold 只作为历史 JSON 被 reporting 查看；不能被读取后
映射为 v3 隐式策略。

## 7. run identity 与 resume

### 7.1 planning mode

`DatasetRunOptions` 和 `RunRequest` 需要同时理解：

```text
visual-task-plan-v3   current fresh mode
visual-task-plan-v2   historical confidence-bearing mode
legacy                pre-v2 historical tombstone
```

默认值和所有 fresh 检查改为 v3。v2/legacy 仅用于解析已持久化 run request 和保护 resume，
不能用于创建新 run。

### 7.2 resume 规则

必须保持：

- v3 succeeded resume：零 planning Qwen，可补缺失的持久化评测/Judge；
- v3 partial/failed/running/pending：按现有规则使用冻结 v3 identity 重跑；
- v2 succeeded resume：仍允许零模型补评测/Judge；旧 plan artifact 不需要按 v3 schema
  重新校验；
- v2 任何需要重新推理的样本：稳定写失败并保持零 Qwen，不能用无-confidence v3 语义重跑；
- legacy run 同样只允许 model-free supplement，需要重推理时稳定拒绝。

可以沿用现有 `LEGACY_PLANNING_RESUME_UNSUPPORTED` stable code 处理所有非 current planning
mode，避免无必要改变公共失败码；若实现代理认为需要新 code，必须先补专门 resume 测试并在
`DETAILS.md` 明确兼容影响，不能静默改变。

### 7.3 需要修改的接线点

- `workflows/schema.py`：planning-mode Literals/defaults；
- `application/runtime.py`：option defaults、fresh guard、resume match、planner identity、manual
  request audit；
- `application/bootstrap.py`：DatasetRunner factory default；
- `workflows/dataset_runner.py`：fresh v3 guard；resume 时任何非 v3 mode 走历史重推理拒绝；
- `workflows/sample_runner.py`：fresh trace/resolution source 写 v3；
- `workflows/artifact_writer.py`：文档从“v2 plan”改成“current versioned plan”，basename 不变；
- `reporting/builder.py`：v2 和 v3 trace 都展示为 `VisualTaskPlanner`，不能把 v3 误判成无
  planner 路径。

`workflows/run_store.py` 将完全缺少 `planning_mode` 的早期 run 映射为 `legacy` 的逻辑保留。
不得把缺失 mode 猜成 v2 或 v3。

## 8. artifact、trace 与 reporting

新 `visual_task_plan.json` 必须：

- `version == "visual-task-plan-v3"`；
- 完全不含 planner confidence；
- 继续包含安全的 `materialized_views`；
- 不含 raw response、图像 bytes、绝对路径、secret；
- 保持同一个纯 basename 和原子写入路径。

新 trace：

- `planning_mode == "visual-task-plan-v3"`；
- `visual_task_plan_version == "visual-task-plan-v3"`；
- 不新增 planner confidence 或 low-confidence 字段；
- 继续诚实记录 planned/execution task、Agent fallback 和 Judge 状态。

reporting 必须继续泛化读取：

- 历史 v2 `visual_task_plan.json`，其中可以存在 confidence；
- 新 v3 `visual_task_plan.json`，其中不存在 confidence；
- 更早的 `visual_plan.json` / `joint_visual_plan.json`。

reporting 只能展示历史值，不能补写、删除或重新解释旧 confidence，也不能把它聚合成当前
metric。

## 9. 代码与文档范围

预计需要修改：

```text
agents/schema.py
application/prompts.py
application/settings.py
application/bootstrap.py
application/runtime.py
workflows/visual_planner.py
workflows/schema.py
workflows/dataset_runner.py
workflows/sample_runner.py
workflows/artifact_writer.py
reporting/builder.py
prompts/visual_task_plan_v3.md          # new non-Python asset
prompts/visual_task_plan_v2.md          # remove after v3 binding
README.md
DETAILS.md
docs/architecture/16_VISUAL_ONLY_PLANNER_REPLACEMENT_PLAN.md
docs/architecture/17_REMOVE_LEGACY_TASK_RESOLVER_VISUAL_GATE_PLAN.md
docs/migration/JOINT_TASK_VISUAL_PLANNER.md
相关 tests
```

doc 16 和 migration 文档保留 v2 历史正文，只需增加“v3 删除 confidence 后 superseded”的
注记。doc 17 中“保留 `confidence_threshold`”和“v2 low confidence fail closed”的指令会与
本方案冲突，必须改为由 doc 18 supersede，避免后续代理重新加回 confidence。

不新增 Python 文件，因此不得修改 `architecture/allowed_python_files.txt`；也不需要为本任务
修改 `architecture/implementation_status.json`。若实际实施同时包含 doc 17 尚未完成的 Python
文件删除，应只按 doc 17 的要求更新 status，不能把两个任务的架构状态混写。

## 10. 实施顺序

### 阶段 A：基线与契约测试

1. 运行 `git status --short`、`git rev-parse HEAD`，保护当前未提交改动；
2. 确认 doc 17 删除工作已完成或明确分离其剩余 diff；
3. 运行当前 v2 planner、runtime、resume、ROI 测试并记录基线；
4. 先新增 v3 schema 测试：无 confidence 成功、带 confidence 失败；
5. 新增 scoped architecture/contract guard，禁止 planner schema、prompt、planner class、
   planner settings 再引入 confidence，但不能禁止其他领域的合法 confidence。

### 阶段 B：schema、prompt 与 planner

1. 创建并绑定 v3 prompt；
2. 升级 plan schema version，删除字段；
3. 删除 planner threshold 和 LOW_CONFIDENCE gate；
4. 更新 fake client/plan constructors；
5. 验证 request meta、response schema、cache hash 和单次 budget 调用。

### 阶段 C：identity、runtime 与 resume

1. 将 fresh planning mode、snapshot 和 trace 更新为 v3；
2. 保留 v2/legacy 的历史解析；
3. 阻止 v2/legacy inference rerun；
4. 验证 v2/v3 succeeded model-free supplement；
5. 验证 fresh v2 在 run 创建和模型调用前被拒绝。

### 阶段 D：artifact、reporting 与文档

1. 断言新 artifact 无 confidence；
2. reporting 同时读取 v2/v3；
3. 删除未绑定 v2 prompt 资产；
4. 更新 README、DETAILS 和 docs 16/17/migration 的版本说明；
5. 运行完整离线验证，不修改历史 run/fixture 掩盖问题。

## 11. 测试计划

### 11.1 planner/schema

- v3 plan 不带 confidence 可验证；
- v3 plan 带 confidence 因 `extra="forbid"` 拒绝；
- prompt JSON 示例没有 `"confidence":`；
- prompt 明确禁止 confidence/probability/certainty 及替代字段；
- planner constructor 不再接受 threshold；
- 任意 schema-valid plan 不经数值门直接进入其余 post-validation；
- category、capability、image index、region linkage 错误仍稳定失败；
- planner 源码和测试中不再出现 planner `LOW_CONFIDENCE`；
- request hash 使用 v3 prompt version 和无-confidence response schema；
- 每条 sample 仍恰好一次 planning Qwen call和一次 budget reserve。

### 11.2 图像与 ROI 回归

- 最长边 `1079/1080/1081`；
- `1024xN`、`Nx1024` 非大图，`1025x1025` 大图；
- 大图 + 明确区域为固定 `1024x1024`；
- 无明确区域为全图；
- direct/evidence crop 一致；
- confidence 删除不改变任何 preview digest、focus clamp 或 crop geometry。

### 11.3 runtime/artifact

- manual ask 和 dataset explicit/default/auto 使用 v3；
- source task 不发送给 planner；
- request/config/prompt snapshot 记录 v3；
- 新 `visual_task_plan.json` 无 confidence；
- trace 无 planner confidence 且版本为 v3；
- v3 plan/task mismatch 和缺 materialized views 仍稳定失败；
- Router/Agent fallback 不变。

### 11.4 resume/reporting

- v3 succeeded resume 零模型补评测；
- v3 非终态按冻结 v3 identity 重跑；
- v2 succeeded resume 零模型补评测；
- v2/legacy 非终态稳定拒绝且 Qwen 调用为零；
- v2 prompt snapshot 和 confidence-bearing plan artifact 不被修改；
- reporting 同时显示 v2/v3 planner path；
- v3 report 不要求 confidence，历史 v2 confidence 只作原样只读展示。

### 11.5 scope regression

现有 detector、counting、VisualEvidence confidence tests 必须继续通过，证明没有做全仓误删。
尤其不能删除 counting 的 `LOW_CONFIDENCE` acceptance policy 或 detector threshold 校准测试。

## 12. 建议验证命令

目标测试：

```bash
pytest -q \
  tests/workflows/test_visual_planner.py \
  tests/agents/general_vqa/evidence/test_schema.py \
  tests/agents/general_vqa/evidence/test_executor.py \
  tests/workflows/test_artifact_writer.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/workflows/test_run_store.py \
  tests/integration/test_auto_task_dataset_vertical_slice.py \
  tests/integration/test_run_dataset_vertical_slice.py \
  tests/integration/test_dataset_runner_resume.py \
  tests/application/test_prompts.py \
  tests/application/test_settings.py \
  tests/application/test_bootstrap.py \
  tests/application/test_runtime.py \
  tests/reporting/test_builder.py \
  tests/reporting/test_html.py
```

证明其他 confidence 未受影响的代表性测试：

```bash
pytest -q \
  tests/agents/counting/test_point_pipeline.py \
  tests/agents/counting/test_yolo_runtime.py \
  tests/agents/grounding/test_evidence.py \
  tests/data/test_vrsbench_task_normalizer.py \
  tests/parity/test_baseline_golden_fixtures.py
```

架构门：

```bash
pytest -q \
  tests/architecture/test_allowed_python_files.py \
  tests/architecture/test_implementation_status.py \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_no_new_to_legacy_imports.py
```

scoped 静态审计：

```bash
rg -n 'plan\.confidence|_confidence_threshold|confidence_threshold.*planner|LOW_CONFIDENCE' \
  agents/schema.py workflows/visual_planner.py application/bootstrap.py \
  tests/workflows/test_visual_planner.py

rg -n '"confidence"\s*:' prompts/visual_task_plan_v3.md

rg -n 'visual-task-plan-v2|visual_task_plan_v2' \
  application agents routing workflows reporting tests README.md DETAILS.md
```

第一个 planner-scoped 搜索和第二个 JSON-key 搜索预期无命中。第三个 v2 搜索只允许命中：

- v2/legacy resume 兼容 Literals 与专门测试；
- reporting 对历史 v2 trace/artifact 的只读识别；
- 明确标注为 historical/superseded 的文档或 fixture。

不要用全仓 `rg confidence` 的“零命中”作为验收，因为其他领域必须继续使用 confidence。

最终离线门：

```bash
python -m compileall -q application data models agents routing workflows evaluation reporting
pytest -q
git diff --check
git status --short
```

默认离线，不运行真实 Qwen live call。若未运行 live gate，最终汇报必须明确写出，不能将 fake
client 测试描述为 live 验证。

## 13. 最终验收清单

- [ ] `VisualTaskPlan` v3 schema 不含 confidence，且 extra confidence 被拒绝；
- [ ] v3 prompt 不要求 confidence，并禁止等价替代字段；
- [ ] `VisualTaskPlanner` 无 threshold 参数、成员和 LOW_CONFIDENCE 分支；
- [ ] planner settings/bootstrap 无 confidence threshold；
- [ ] schema/prompt/planning mode/snapshot/trace 一致升级到 v3；
- [ ] request hash 与 cache 通过 v3 prompt + response schema 与 v2 隔离；
- [ ] 新 `visual_task_plan.json` 无 confidence，basename 与原子写入不变；
- [ ] schema 合法计划不再被主观分数拒绝；其他严格校验保持；
- [ ] 1080、方案 A、1024 ROI 和默认全图规则无漂移；
- [ ] v2/legacy 历史 run 只读或 model-free supplement，需要重推理时稳定拒绝；
- [ ] reporting 同时支持 v2/v3，不修改历史 confidence；
- [ ] detector、counting、VisualEvidence、Judge 等其他 confidence 未被删除；
- [ ] UnifiedSample、routing、model interface、evaluation、report、CLI、resume 安全契约无未经授权变化；
- [ ] README、DETAILS、docs 16/17/migration 与当前事实一致；
- [ ] 目标测试、代表性 scope tests、architecture tests、完整 pytest、compileall、
      `git diff --check` 的真实结果已记录；
- [ ] 未运行的 live/平台验证及原因已明确记录。

## 14. 实施代理最终汇报要求

最终回复至少说明：

1. 删除了哪些 planner confidence schema、prompt、threshold 和错误分支；
2. 为什么升级 v3，以及 v2 cache/artifact/resume 如何隔离；
3. 修改和删除的文件；
4. 新 artifact/trace 是否确认无 planner confidence；
5. 哪些其他领域 confidence 明确保留；
6. 实际运行的测试/检查及结果；
7. 未运行的验证及原因；
8. 对 UnifiedSample、task/routing、model interface、evaluation、report、CLI、resume 的影响；
9. 剩余风险。不得声称不存在尚未验证的问题。
