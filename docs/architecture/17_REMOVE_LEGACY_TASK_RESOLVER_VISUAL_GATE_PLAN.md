# 17 — 删除旧 TaskResolver / VisualPlanningGate 的实施计划

> Status: implemented; the executable legacy resolver/gate/joint paths, assets, and
> current tests have been removed. The remaining live-Qwen/platform gates are not
> part of the default offline execution.
>
> 状态：已实施；可执行的旧 resolver/gate/joint 路径、资产与当前测试均已删除。
> 真实 Qwen/平台 live gate 不属于默认离线执行，结果单独记录。
>
> Baseline inspected: `2191ac6aa60eff4aab5a4a5381622681ed81afdb` plus the
> uncommitted doc-16 visual-only planner implementation present in the workspace on
> 2026-08-17. The implementation agent must treat the current dirty worktree as user
> work, run the preflight checks again, and never reset or overwrite unrelated changes.

## 1. 给实施代理的上文

doc 16 已经把 fresh inference 的规范入口改为唯一的
`workflows.visual_planner.VisualTaskPlanner`。第一次 Qwen 调用的 user content 必须只有：

```text
ordered thumbnail-or-full-image block(s)
+ raw question text
```

不得向这次调用附加 source task、image id/role、metadata、normalization、Ground Truth、
answer constraints、backend 或 checkpoint。合法 task、输出 schema 和能力闭集属于 system
prompt/response schema，而不是额外 user payload。

图像与 ROI 的冻结规则是：

1. 只有 `max(width, height) > 1080` 才等比例缩小，最长边缩到 `1080`；否则发送整图，
   永不放大；
2. “大图”采用方案 A：`width > 1024 and height > 1024`；任一维等于或小于 `1024`
   都不是大图；
3. 只有“大图 + 问题明确给出区域”才物化一个原图坐标系中的 `1024 x 1024` 固定 ROI；
4. 大图但问题没有明确区域、小图即使带区域描述、或无法确定目标图时，都使用全图；
5. ROI 不加 halo，边界使用确定性 clamp，最终 Agent 与 evidence executor 必须消费同一组
   `MaterializedVisualView`；
6. 规划输出只决定 task、`needs_visual_assistance`、对象类别和可选区域请求，不决定最终答案
   或具体 detector/segmenter/backend。

规范 fresh 链路应当只有这一条：

```text
SampleDraft or UnifiedSample
  -> deterministic preview preprocessing
  -> VisualTaskPlanner.plan_with_views(...)       # exactly one planning Qwen call
  -> VisualTaskPlan + MaterializedVisualView(s)
  -> materialize/rebuild UnifiedSample with planned task
  -> TaskRouter.route(planned_task)                # deterministic, model-free
  -> AgentContext(visual_task_plan, visual_views, visual_bindings)
  -> selected Agent and its declared Agent fallback(s)
  -> deterministic evaluation
  -> optional Judge
```

这里的“恰好一次”指每条 fresh 样本恰好一次**规划调用**；最终业务 Agent 和被明确请求的
evidence 流程仍可按各自预算调用模型。`TaskRouter` 继续只接收已知 task，不读 question，
不调用模型。

## 2. 为什么仍需本次删除

当前 composition root 已经不给旧对象接线，但旧实现仍是可执行、可直接注入的代码，不能
算“已经删除”：

- `workflows/task_resolver.py` 仍定义完整的文本 `TaskResolver`、
  `TaskResolutionRequest` 消费路径、低置信度跨 task 候选逻辑和模型调用；
- `workflows/visual_planner.py` 仍定义 `VisualPlanner`、`VisualPlanningGate`，以及 doc 15 的
  `JointVisualPlanner`；
- `DatasetRunner` 仍接受 `task_resolver` / `joint_planner`，并保留 resolver、joint 和无规划
  直通分支；
- `SampleRunner` 仍接受 `visual_planning`、`resolution`、`joint_plan`，可在 runner 内再次
  调用旧 planner；
- `AgentContext.visual_plan` 和 VQA/Grounding evidence 仍允许
  `FirstQwenVisualPlan`，v2 计划还会被适配回旧 `RoiRegion`；
- `ArtifactWriter` 仍能新写 `visual_plan.json` / `joint_visual_plan.json`；
- 旧 prompt、`scripts/classify_tasks.py`、旧设置字段和专门测试仍在仓库中。

因此，本任务不是再改一次 bootstrap 的开关，而是删除所有**可执行旧路径**，使旧对象既
不能由 production 组装，也不能由测试或直接构造重新启用。

## 3. 完成定义

完成后应同时成立：

1. 仓库中不存在可实例化或可调用的 `TaskResolver`、`VisualPlanner`、
   `VisualPlanningGate`、`JointVisualPlanner`；
2. fresh manual ask、dataset explicit/default/auto 都只能先走 v2
   `VisualTaskPlanner`；显式/source task 仍只作审计；
3. `DatasetRunner` 不提供 resolver/joint/no-plan fallback；
4. `SampleRunner` 不调用任何 planner，只消费已经物化的 v2 plan/views；
5. VQA/Grounding evidence 原生消费 `VisualTaskPlan` 与 `MaterializedVisualView`，不再通过
   v1 plan/normalized ROI schema 过桥；
6. 新运行只写 `visual_task_plan.json`，不能再写旧两个 plan artifact；
7. TaskRouter 声明的 primary/fallback Agent 行为保留；TaskResolver 的低置信度跨 task
   candidate fallback 删除；
8. 旧 run 仍能被 reporting 只读展示，succeeded resume 仍可零模型补评测；需要重新推理的
   legacy run 必须稳定拒绝，不能套用 v2 语义静默重跑；
9. 不改变 `UnifiedSample` 字段、GT、metrics、Judge 关系、结果路径或 append-only index。

## 4. 删除与保留边界

### 4.1 必须删除

| 区域 | 删除内容 |
|---|---|
| 文本任务解析 | `TaskResolver`、`TaskResolutionError`、私有模型 response schema、`TaskResolutionRequest`、低置信度跨 task candidates、resolver prompt 与独立分类脚本 |
| v1 视觉规划 | `VisualPlanner`、`VisualPlanningGate`、`VisualPlanError`、兼容矩阵 gate |
| doc 15 联合规划 | `JointVisualPlanner`、`JointPlanError`、`JointQwenVisualPlan`、`joint_plan_to_resolution` 及 runner 分支 |
| runner 注入面 | `task_resolver`、`joint_planner`、`visual_planning`、`resolution`、`joint_plan` 参数与条件分支 |
| 旧下游契约 | `AgentContext.visual_plan`、`FirstQwenVisualPlan`、`RoiPlan`、`ObjectEvidenceRequest`、旧 execution family；在 v2 原生化后删除只服务旧计划的 `RoiRegion`/halo mapping |
| 新写入能力 | `VISUAL_PLAN_FILENAME`、`JOINT_VISUAL_PLAN_FILENAME`、`write_visual_plan`、`write_joint_visual_plan` |
| 误导配置 | `visual_planning.enabled` 以及只服务 v1 的 `prompt_version`、`max_rois`、`halo_ratio`、`VisualPlannerFailureSettings/failure` |
| 旧资产 | `prompts/task_resolver_v1.md`、`prompts/first_qwen_visual_plan_v1.md`、`prompts/joint_qwen_task_visual_plan_v1.md` |
| 旧工具/测试 | `scripts/classify_tasks.py`；纯 TaskResolver 测试；gate/joint/v1 planner 专用测试与 fake |

`JointVisualPlanner` 虽未在需求标题中单独点名，但它仍是 v2 之外的另一套 task + visual
planner。若保留它，系统仍不是“只使用视觉规划 v2”，所以它属于本次删除范围。

### 4.2 必须保留

- `VisualTaskPlanner`、`VisualTaskPlanError`、`VisualTaskPlan`、`RegionRequest`、
  `MaterializedVisualView`；
- 1080 shrink-only preview、方案 A 大图判定、固定 `1024 x 1024` ROI 和无明确区域全图逻辑；
- `TaskRouter`、固定 task → Agent policy，以及同一 route 内声明的 Agent fallback；
- `SampleDraft -> UnifiedSample` 的确定性 materialization 及变化任务图像 role 校验；
- 共享 `CallBudget`、模型 cache identity、request hash、prompt snapshot；
- `visual_task_plan.json` 的原子、安全写入；
- `RunRequest.planning_mode="legacy"` 这一**墓碑值**和旧 run 的 fail-closed resume 判断；它们
  只阻止旧 run 被新语义重推理，不是旧 planner 执行逻辑；
- reporting 中对字符串 `visual_plan.json`、`joint_visual_plan.json` 以及旧 trace 路径名的
  allowlisted 只读识别；
- `docs/architecture/14*`、`15*` 和 migration 文档的历史事实。可以加 superseded 注记，
  不应把历史记录改写成从未存在；
- 所有 evaluation、GT、split、selection、resume index 和路径安全契约。

### 4.3 不在本任务中做

- 不迁移、重写或删除用户已有 run artifact；
- 不修改 Golden fixtures 来适应代码删除；
- 不改模型、processor/tokenizer、checkpoint 或依赖版本；
- 不改 detector/segmenter 选择和已校准类别目录；
- 不改变 ROI、缩略图或大图定义；本任务只移除旧实现；
- 不顺手重构无关 Agent、metric、reporting 或 CLI；
- 不下载模型/数据集，不做联网 live call。

## 5. 目标文件布局与职责

### 5.1 样本物化

`workflows/task_resolver.py` 当前同时放着旧 resolver 和仍在使用的
`materialize_sample`。为真正删除该旧模块，实施时应：

1. 将 `SampleMaterializationError` 与 `materialize_sample` 移到现有
   `data/schema.py`；它们只依赖 data contract，不能调用模型、路由或写 artifact；
2. 更新 `application/runtime.py` 和 `workflows/dataset_runner.py` 的 import；
3. 将物化测试并入现有 `tests/contracts/test_data_schema_contract.py`；
4. 删除 `workflows/task_resolver.py` 和 `tests/workflows/test_task_resolver.py`；
5. 从 `architecture/implementation_status.json` 的 `implemented_files` 删除已不存在的生产
   文件记录；不要修改 `architecture/allowed_python_files.txt`，因为其规则允许批准路径不存在，
   且普通清理任务无权改白名单。

物化行为不得变化：未知 task、单图变化任务、schema 违规仍需稳定失败；输入 draft 和 GT
不得被原地修改。

### 5.2 routing schema

删除 `routing.schema.TaskResolutionRequest`。同时删除只为旧 resolver/joint/candidate
fallback 服务的 `TaskResolution`、`ResolutionSource`、`joint_plan_to_resolution` 和
`visual_task_plan_to_resolution`，让 `VisualTaskPlan.task` 直接成为 `TaskRouter.route(...)`
的输入。

不要把 TaskResolver 逻辑移动进 Router。Router 仍不得读取 question 或调用模型。

若需要保持新 trace 的审计字段，直接从 `VisualTaskPlan` 写入：例如
`planning_mode="visual-task-plan-v2"`、`planned_task`、`plan_version`。不得为了保留旧字段而
继续构造一个假的 `TaskResolution`。reporting 对旧 trace 的读取是泛化、只读的，不要求
fresh runtime 继续生成 `candidate_tasks`、`low_confidence` 或 `joint_plan`。

### 5.3 visual planner 模块

保留 `workflows/visual_planner.py`，但把它收缩为 v2 实现：

- 删除从文件开头到 doc-15 joint planner 结束的所有 v1/joint class 和 helper；
- 清理 `FirstQwenVisualPlan`、`JointQwenVisualPlan`、`ObjectEvidenceRequest`、`RoiPlan` 等
  imports；
- 文件 module docstring 只描述当前 v2，不再声称旧类为 migration-test compatibility；
- 保持 v2 user content、preview digest、schema validation、cache identity、budget 和
  materialized views 行为不变。

### 5.4 DatasetRunner

`DatasetRunner` 的 fresh path 必须结构上只有 v2：

1. 删除构造参数和成员 `task_resolver`、`joint_planner`；
2. 将 `visual_task_planner`、`call_budget_factory`、`data_root` 变成 fresh execution 所需的
   明确依赖，不再用 `None` 表示回退到旧路径；
3. `_run_sample` 在非 succeeded resume 时只调用 `_run_sample_visual`；
4. `_run_draft` 在非 succeeded resume 时只调用 `_run_draft_visual`，即使 draft 带
   `explicit_task` 也不得绕过第一次视觉规划；
5. 删除 `_run_sample_joint`、`_run_draft_joint`、`TaskResolutionRequest` 分支和无 planner
   的 `SampleRunner.run_one(sample, ...)` fresh fallback；
6. fresh `planning_mode != "visual-task-plan-v2"` 必须在任何 sample/model 调用前稳定失败；
7. 保留 legacy resume 墓碑：succeeded 样本可只补持久化评测；任何需要重新推理的 legacy
   样本写 `LEGACY_PLANNING_RESUME_UNSUPPORTED`，Qwen 调用数必须为零。

### 5.5 SampleRunner

`SampleRunner` 不再拥有 planner：

1. 删除构造参数/成员 `visual_planning`，把 `joint_bindings` 重命名为当前含义明确的
   `visual_bindings`；
2. 删除 `run_one(..., resolution=..., joint_plan=...)`；只允许可选的 v2
   `visual_task_plan` + `visual_views` 输入；
3. 删除 runner 内的 `plan_sample()` 调用、旧 artifact 写入、joint-plan task 转换和相互排斥
   分支；
4. 有 v2 plan 时强制 `sample.task == plan.task` 且 views 非空；
5. dataset/manual fresh 上游保证 plan 存在。`count-image` 等明确 task 的非规划专用工具若仍
   直接调用 SampleRunner，可继续使用 sample.task，但不得获得重新启用旧 planner 的参数；
6. 删除 TaskResolver 的跨 task candidate attempts、`MAX_ATTEMPTS` 和相应 trace 逻辑；
7. 保留同一 `RoutingDecision` 内 primary/fallback Agent 的执行、
   `_ROUTING_FALLBACK_TASK_REMAP` 和执行 task 的诚实记录。这是 Router policy，不是
   TaskResolver fallback。

### 5.6 AgentContext 与 evidence 原生 v2 化

这是删除 Gate 最容易漏掉的部分，必须先完成消费者迁移，再删 schema：

1. 从 `AgentContext` 删除 `visual_plan`，只保留 `visual_task_plan`、`visual_views`、
   `visual_bindings`；
2. 将 `VqaEvidenceService` / `GroundingEvidenceService` protocol 收窄为
   `VisualTaskPlan` + `MaterializedVisualView`；删除 union 和动态 `getattr` 兼容调用；
3. `GeneralVQAAgent` 删除 `context.visual_plan`、`FirstQwenVisualPlan` 和
   `execution_family` 分支：`needs_visual_assistance=False` 走 direct，`True` 才走 v2
   evidence；
4. `GroundingAgent` 同样只根据 v2 plan 选择 direct/evidence；
5. VQA/Grounding evidence executor 直接以 `MaterializedVisualView.crop_xyxy`、
   `source_size`、`crop_size` 建立现有 evidence record，不要先还原成 normalized
   `RoiRegion`；
6. rendering/crop 直接消费像素 `crop_xyxy`，保证 direct 与 evidence 得到同一裁片；不要
   恢复旧 halo、normalized xyxy 或多 ROI fallback；
7. 待所有引用清零后，从 `agents/schema.py` 删除 `FirstQwenVisualPlan`、
   `JointQwenVisualPlan`、`RoiPlan`、`ObjectEvidenceRequest`、旧版本常量和 execution
   family；`RoiRegion` 若已无非旧用途也一并删除。

这一阶段不得更改 `VisualTaskPlan`、`RegionRequest` 或 `MaterializedVisualView` 的已冻结
字段和几何含义。

### 5.7 composition root、settings、prompts 与 artifact writer

在 `application/bootstrap.py`：

- 从 `RuntimeComponents` 删除 `task_resolver`、`joint_planner` 历史槽位；
- `visual_task_planner` 改为现役必需组件，不再以 `None` 表示功能关闭；
- 删除 `visual_planning = None` / `joint_planner = None` 局部变量和条件 bindings；
- `SampleRunner` 直接接收 `visual_bindings`，`DatasetRunner` 直接接收 v2 planner。

在 `application/settings.py`：

- 删除 `VisualPlanningSettings.enabled`；视觉规划没有关闭开关；
- 删除只服务 v1 的 planner 字段与未被 v2 读取的 failure policy；
- 保留并验证 v2 的 `task_prompt_version`、`catalog_version`、
  `confidence_threshold`、`preview_max_side=1080`、`roi_size=1024`、方案 A policy；
- 因 `extra="forbid"`，用户 YAML 再提供已删除字段应明确校验失败。不要加入静默忽略旧字段
  的兼容层；旧 `config.snapshot.json` 由 reporting 只读，不通过 AppSettings 重放。

在 prompts/artifacts：

- 删除三个未绑定旧 prompt 文件和相应测试 fixture 列表；
- `PromptCatalog` 只绑定 `visual_task_plan_v2.md`；
- `ArtifactWriter` 只保留 `VISUAL_TASK_PLAN_FILENAME` 和 `write_visual_task_plan`；
- reporting 的旧 artifact filename 字符串必须留在 `reporting/adapters.py`，不要重新 import
  已删除的 schema 或 writer constant。

### 5.8 旧独立脚本

删除 `scripts/classify_tasks.py`。该脚本会自行构造 Qwen client 和旧 TaskResolver，是另一条
可执行 composition root。若仍需要批量任务分类，应使用现有
`python main.py run-dataset --auto-task ...` 的规范 v2 链路；不要在本删除任务中复制一套
新的 planner CLI。

## 6. 实施顺序

### 阶段 A：冻结现状与加回归门

1. 运行 `git status --short`、`git rev-parse HEAD`，记录并保护用户现有修改；
2. 阅读 `AGENTS.md`、`DETAILS.md`、doc 16 和相关测试；
3. 用 `rg` 重新生成旧符号引用清单；
4. 先在现有 architecture test 中增加 AST 级守卫：production/test code 不得定义或 import
   旧 resolver/planner/schema。守卫只看 AST symbol，不应误伤 reporting 中的历史字符串和
   docs；
5. 先运行当前 v2 planner/ROI 测试，保存基线，避免清理时改变新逻辑。

### 阶段 B：让下游原生消费 v2

1. 收窄 AgentContext 和 evidence protocols；
2. 改 GeneralVQA/Grounding agent；
3. 改 evidence geometry/rendering/executor，直接消费 materialized pixel views；
4. 更新对应 Agent/evidence 测试，只使用 v2 schema；
5. 验证 direct/evidence 裁片、坐标转换和 artifact 与清理前 v2 基线一致。

### 阶段 C：收口 runner 与 composition root

1. 删除 DatasetRunner 的 resolver/joint/no-plan 分支；
2. 删除 SampleRunner 的 gate/joint/resolution 输入和跨 task candidate fallback；
3. 删除 RuntimeComponents 的历史槽位；
4. 更新 manual/dataset/count-image 调用点和 fake factories；
5. 单独验证 legacy resume 墓碑与 succeeded model-free supplement。

### 阶段 D：删除定义、文件与资产

1. 移动 sample materialization 后删除 `workflows/task_resolver.py`；
2. 从 `workflows/visual_planner.py` 删除 v1 gate 与 joint planner；
3. 从 routing/agent schema 删除旧 contract 与 export；
4. 删除旧 writer 能力、settings 字段、prompt 文件、脚本和旧测试；
5. 更新 `architecture/implementation_status.json`，不改 Python allowlist。

### 阶段 E：文档与审计兼容

1. 更新 `DETAILS.md` 为唯一 v2 当前事实，删除“旧类仍留在源码”的描述；
2. 更新 README，删除 `visual_planning.enabled` 示例和可关闭/双模式措辞；
3. 更新 adapter/workflow 当前注释，以及 `docs/architecture/99_FINAL_LIVE_GATE_RUNBOOK.md`
   中“auto-task 使用 TaskResolver”的过期说明；
4. 在 doc 14/15 或 migration 文档顶部加 superseded/removal 注记即可，不重写历史；
5. reporting 继续只读展示旧 artifact 与旧 trace 模块字符串，并增加测试证明 fresh v2
   report 不声称调用旧 resolver/planner。

### 阶段 F：完整验证

按第 8 节运行目标测试、架构测试和完整离线门。修复实现，不得删除/skip/放宽失败测试，也
不得修改 Golden fixture 掩盖行为漂移。

## 7. 必须更新的测试

### 7.1 删除或改写旧测试

- 删除 `tests/workflows/test_task_resolver.py`，把 materialization 契约测试移到
  `tests/contracts/test_data_schema_contract.py`；
- `tests/workflows/test_visual_planner.py` 删除 v1 VisualPlanner/Gate/Joint 部分，只保留并
  加强 v2 输入、1080、方案 A、1024 ROI、cache/budget 测试；
- `tests/workflows/test_sample_runner.py` 删除 TaskResolution candidate、gate、joint tests，
  新增 v2 task/view 校验与 Router Agent fallback 保留测试；
- `tests/workflows/test_dataset_runner.py` 删除 fake resolver/joint planner，新增
  explicit/default/auto 都恰好一次 v2 planning call；
- `tests/workflows/test_artifact_writer.py` 删除旧两个 writer 的测试，只断言新 writer；
- `tests/integration/test_auto_task_dataset_vertical_slice.py` 改为 v2 planner vertical slice，
  或在覆盖完全重复时删除并由 `test_run_dataset_vertical_slice.py` 承担；
- routing contract/router tests 删除 `TaskResolution*` 和 joint conversion；
- GeneralVQA/Grounding/evidence tests 全部改用 v2 plan/views；
- bootstrap/settings/prompts tests 删除 dead slots、enabled flag 和旧 prompt 文件假设。

### 7.2 必须新增或保留的断言

- 第一次规划 user content 只有有序 image block + raw question；
- 每条 fresh manual/dataset 样本恰好一次 planning call；
- 显式/source task 不绕过 planner、不发送给 planner、不覆盖 model task；
- `DatasetRunner` 无法注入 resolver/joint，`SampleRunner` 无法注入 gate/joint/resolution；
- v2 plan/task 冲突稳定失败，缺 views 稳定失败；
- Router primary/fallback 仍可执行，且 Router 零模型调用；
- v2 low confidence 按 `VisualTaskPlanError` fail closed，不跑跨 task candidates；
- fresh run 只产生 `visual_task_plan.json`；
- reporting 能读一个包含旧 `visual_plan.json` / `joint_visual_plan.json` 的历史 fixture；
- succeeded legacy resume 零模型补判；需要重推理的 legacy resume 稳定失败且零 Qwen；
- 旧设置字段被 `extra="forbid"` 拒绝；
- AST architecture guard 不允许旧 class/import 回归，但允许 reporting 的历史字符串。

## 8. 验证命令

实施代理应按风险分层执行，实际结果如实记录。

目标单元/集成测试：

```bash
pytest -q \
  tests/contracts/test_data_schema_contract.py \
  tests/contracts/test_routing_contract.py \
  tests/routing/test_router.py \
  tests/workflows/test_visual_planner.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/workflows/test_artifact_writer.py \
  tests/integration/test_auto_task_dataset_vertical_slice.py \
  tests/integration/test_run_dataset_vertical_slice.py \
  tests/integration/test_dataset_runner_resume.py \
  tests/application/test_bootstrap.py \
  tests/application/test_runtime.py \
  tests/application/test_settings.py \
  tests/application/test_prompts.py \
  tests/agents/test_visual_base.py \
  tests/agents/general_vqa/test_agent.py \
  tests/agents/general_vqa/evidence/test_geometry.py \
  tests/agents/general_vqa/evidence/test_rendering.py \
  tests/agents/general_vqa/evidence/test_executor.py \
  tests/agents/grounding/test_agent.py \
  tests/agents/grounding/test_evidence.py \
  tests/reporting/test_html.py
```

若某个已删除测试文件不再存在，从命令中移除它，并在最终汇报说明其职责已迁移到哪个测试。

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

旧可执行符号审计：

```bash
rg -n '\b(class TaskResolver|TaskResolutionRequest|class VisualPlanner|VisualPlanningGate|class JointVisualPlanner|FirstQwenVisualPlan|JointQwenVisualPlan)\b' \
  application data models agents routing workflows scripts tests

rg -n '\b(write_visual_plan|write_joint_visual_plan|VISUAL_PLAN_FILENAME|JOINT_VISUAL_PLAN_FILENAME)\b' \
  application data models agents routing workflows scripts tests

rg -n '\b(task_resolver=|joint_planner=|visual_planning=|resolution=|joint_plan=)\b' \
  application agents routing workflows scripts tests
```

上述 production/test 扫描预期无旧执行符号。允许的残留仅限：

- `reporting/adapters.py` 中两个历史 artifact basename 字符串；
- `reporting/builder.py` 中旧 trace 的只读展示字符串；
- docs/migration、旧 architecture docs 和历史 run fixture 的文本。

最终离线门：

```bash
python -m compileall -q application data models agents routing workflows evaluation reporting
pytest -q
git diff --check
git status --short
```

真实 Qwen live gate 不属于本次默认离线删除任务。若未运行，必须明确写“未运行”，不能用 fake
测试代替 live 结论。

## 9. 实施结果与最终验收清单

本次执行结果：

- 核心离线门（agents/contracts/routing/workflows/integration/application/reporting，排除
  HTTP socket 用例）：`1228 passed, 39 deselected`；
- architecture implementation/import/init/package/no-legacy 门：`33 passed`；
- `compileall` 与 `git diff --check`：通过；
- 全离线集合（排除缺失 `safetensors` 导出的收集错误）：`1953 passed, 33 failed,
  1 skipped`。失败包含既有 allowlist 漂移、`peft`/`transformers`/`safetensors` 缺失、
  HTTP sandbox/oversized-body 环境问题和模型 import 测试顺序污染，不能宣称全仓全绿；
- 沙箱外 HTTP serve 门：`10 passed, 1 failed`；唯一失败为超大请求发送阶段 timeout，
  未改变其既有 HTTP 语义；
- 真实 Qwen、真实数据集、DeepSeek、GPU/ONNX live gate：未运行。

- [x] fresh runtime 只有 `VisualTaskPlanner` 一套规划实现；
- [x] TaskResolver、VisualPlanningGate、v1 VisualPlanner、JointVisualPlanner 的定义和调用均不存在；
- [x] 第一次 Qwen 输入仍是缩略图/整图 + 原始问题，未重新加入 JSON metadata；
- [x] 1080 shrink-only、方案 A、固定 1024 ROI、默认全图语义无漂移；
- [x] `VisualTaskPlan`/`MaterializedVisualView` 是 planning 与 Agent/evidence 之间唯一契约；
- [x] direct 与 evidence 使用相同 materialized crop，且不再适配成旧 normalized ROI；
- [x] DatasetRunner/SampleRunner/composition root 没有可重新启用旧逻辑的参数；
- [x] Router 确定性和 Agent fallback 保留，跨 task candidate fallback 已删除；
- [x] 新运行不会写 `visual_plan.json` 或 `joint_visual_plan.json`；
- [x] 历史 reporting 与 legacy resume fail-closed 行为有测试；
- [x] `UnifiedSample`、GT、evaluation、report、CLI、resume 和路径安全没有有意的语义变化；
- [x] `DETAILS.md`、README、当前注释与实际代码一致；
- [x] 目标测试、可运行的 architecture tests、compileall、`git diff --check` 的真实结果已记录；
- [x] 未运行的 live/平台测试及环境失败已明确列出。

## 10. 实施代理最终汇报要求

最终回复至少列出：

1. 删除了哪些旧 class、branch、schema、prompt、script 和 tests；
2. v2 evidence 如何改为直接消费 `MaterializedVisualView`；
3. materialization helper 移到哪里，行为是否保持；
4. legacy reporting/resume 如何保留且保证零旧模型执行；
5. 修改/删除的文件清单；
6. 实际运行的每条测试/检查及结果；
7. 未运行的验证及原因；
8. 对 UnifiedSample、task/routing、model interface、evaluation、report、CLI、resume 的影响；
9. 剩余风险。不得声称不存在尚未验证的问题。
