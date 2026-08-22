# 24 — VQA Agent 视觉证据辅助范围扩展计划

> Status: Implemented / 已实施（2026-08-21）
>
> 范围：让 `GeneralVQAAgent` 管辖的全部 VQA task 都以
> `VisualTaskPlan.needs_visual_assistance` 作为 direct/evidence 路径的唯一开关。
> 本计划不修改模型、评测、Ground Truth、task 路由或结果 schema。

## 1. 结论

当前并不是全系统只有 `general_vqa` 能使用视觉证据：

- `counting`、`fine_grained_counting` 使用 Counting 自己的专家证据流水线；
- `grounding` 使用 Grounding 自己的证据流水线；
- 但在 `GeneralVQAAgent` 覆盖的四个 task 中，当前确实只有 `general_vqa`
  能进入 `object_evidence_vqa`。

`GeneralVQAAgent` 当前覆盖：

```text
general_vqa
scene_classification
multiple_choice_vqa
spatial_relation
```

现有 `GeneralVQAAgent.run()` 在看到
`VisualTaskPlan.needs_visual_assistance == true` 后，又执行一次
`sample.task == "general_vqa"` 判断；其余三个 task 稳定失败。这是对 planner
决定的重复 task gate。

目标行为应改为：

```text
plan.needs_visual_assistance == false
  -> direct_vqa

plan.needs_visual_assistance == true
  -> object_evidence_vqa
```

Agent 不再用 `sample.task == "general_vqa"` 二次否决 planner 的视觉辅助决定。
`sample.task` 仍用于路由、Prompt/answer constraint、结果语义和评测 dispatch，
但不再作为 GeneralVQAAgent 内部 direct/evidence 分支的 blanket gate。

## 2. 当前限制不只存在于 Agent

仅删除 `GeneralVQAAgent` 中的一条判断并不能完成目标。当前至少有四层约束：

| 层 | 当前行为 | 需要的调整 |
|---|---|---|
| `GeneralVQAAgent` | 只允许 `general_vqa + assistance` | 全部 supported task 消费 planner 开关 |
| `VisualTaskPlanner` | 视觉能力 task 集合只含 counting、fine counting、general VQA、grounding | 将四个 GeneralVQAAgent task 都暴露为 VQA 证据可规划 task |
| `EvidenceCatalog` 调用方式 | 按 `plan.task` 查找能力，目录只登记 `general_vqa` 这一 VQA capability owner | 三个新增 VQA task 复用同一 VQA capability owner，不复制类别表 |
| `application.bootstrap` | 只为 `general_vqa` 计算运行时可执行类别 | 为四个 VQA task 注入同一份、按当前模型可用性过滤后的类别集合 |

如果只改 Agent，上游 planner 对
`scene_classification`、`multiple_choice_vqa`、`spatial_relation` 返回
`needs_visual_assistance=true` 时仍会在 category/capability post-validation 阶段
fail closed，实际路径不会被启用。

## 3. 设计原则

### 3.1 单一决策来源

是否启用 VQA 视觉证据只由已校验的以下字段决定：

```text
VisualTaskPlan.needs_visual_assistance
VisualTaskPlan.object_categories
VisualTaskPlan.region_request
```

职责分离保持为：

```text
VisualTaskPlanner
  -> 决定 task、是否需要辅助、类别和可选 ROI

planner post-validation
  -> 验证类别属于 VQA evidence capability 且当前 runtime 可执行

SampleRunner
  -> 验证 plan.task 与 materialized sample.task 一致

TaskRouter
  -> 根据已知 task 确定 GeneralVQAAgent

GeneralVQAAgent
  -> 只按 needs_visual_assistance 选择 direct_vqa/object_evidence_vqa
```

Agent 不重新识别 task，不读取 question 判断是否需要证据，也不选择 backend。

### 3.2 保留的 fail-closed 边界

本修改不应删除以下校验：

1. `sample.task` 必须属于 `GeneralVQAAgent.supported_tasks`；
2. `SampleRunner` 继续要求 `visual_task_plan.task == sample.task`；
3. assistance 为 true 时必须有非空、规范、无重复的 canonical leaf；
4. planner 只能接受当前 runtime 真正可执行的类别；
5. assistance 为 true 但未注入 `VqaEvidenceService` 时稳定失败；
6. evidence executor 继续只消费 plan、materialized views 和注入模型协议；
7. `multiple_choice_vqa` 在 evidence 路径后仍执行 choice constraint postprocess；
8. actual execution task 继续决定 deterministic evaluation family。

这些校验负责契约一致性和真实能力，不属于需要移除的重复 task gate。

### 3.3 共享 capability family，不复制类别表

四个 VQA task 都由同一个 `GeneralVQAAgent` 和同一个
`VqaEvidenceService` 执行，应共享 `general_vqa` 对应的 evidence capability
owner。不要在 `agents/evidence_catalog.json` 中复制四份完全相同的 leaf 列表，
否则类别增删容易漂移。

建议建立一个共享、公开且命名明确的 Agent task 集合，例如：

```text
GENERAL_VQA_AGENT_TASKS = {
  general_vqa,
  scene_classification,
  multiple_choice_vqa,
  spatial_relation,
}
```

它表达“由哪个 Agent/证据协议拥有这些 task”，不表达“这些 task 必须启用证据”。
是否启用仍只看 planner 的 `needs_visual_assistance`。

planner 在类别校验时把上述 task 映射到同一 `general_vqa` capability owner；
对外生成的 `task_executable_categories` binding 仍按真实 task 分别列出，使 planner
能为每个 task 生成严格合法的计划。

## 4. 目标行为矩阵

| planned task | assistance=false | assistance=true | 额外约束 |
|---|---|---|---|
| `general_vqa` | `direct_vqa` | `object_evidence_vqa` | 保持现状 |
| `scene_classification` | `direct_vqa` | `object_evidence_vqa` | 最终答案仍是 scene classification 语义 |
| `multiple_choice_vqa` | `direct_vqa` | `object_evidence_vqa` | evidence final Qwen 后继续 choices 校验 |
| `spatial_relation` | `direct_vqa` | `object_evidence_vqa` | 不新增专用几何规则，仍由 VQA 协议回答 |

以下组合保持不变：

- `counting` / `fine_grained_counting` 仍由 `CountingAgent` 拥有；
- `grounding` 仍由 `GroundingAgent` 拥有；
- `caption`、`change_caption`、`change_qa` 不借此接入 VQA evidence；
- routing fallback 不得改写 persisted resolved task 或评测 task。

## 5. 实施步骤

### 5.1 统一 GeneralVQAAgent task 集合

修改：

```text
agents/schema.py
agents/general_vqa/agent.py
```

1. 在 Agent 公共契约层声明 `GENERAL_VQA_AGENT_TASKS`；
2. `GeneralVQAAgent.supported_tasks` 直接复用该集合，避免 planner、bootstrap 与
   Agent 各自维护一份四 task 列表；
3. 不在 `data.schema` 中引入 evidence 概念，保持数据契约与执行能力解耦。

### 5.2 移除 Agent 内的 general_vqa-only gate

修改：

```text
agents/general_vqa/agent.py
```

将 `run()` 收敛为：

```text
unsupported sample.task
  -> AgentTaskMismatchError

no visual plan or assistance=false
  -> existing direct path

assistance=true and service unavailable
  -> stable vqa_evidence_service_unavailable failure

assistance=true and service available
  -> existing _run_object_evidence(...)
```

删除：

- `_DIRECT_ONLY_TASKS` 及其过期注释；
- `sample.task != "general_vqa"` 的拒绝分支；
- `visual_assistance_forbidden_for_task:*` 这一仅由旧矩阵产生的错误路径。

不要改动 `_run_object_evidence()` 的模型调用次数、request hash、图像协议、artifact
basename 或 additional result 结构。

### 5.3 扩展 planner 的 VQA capability binding

修改：

```text
workflows/visual_planner.py
```

1. 将 `GENERAL_VQA_AGENT_TASKS` 纳入可声明视觉能力的 task 集合；
2. 对这四个 task 的 catalog leaf 校验统一解析到 `general_vqa` capability owner；
3. `_runtime_executable_by_task` 和 system prompt 的
   `task_executable_categories` 必须以四个真实 task 为 key；
4. 继续保持 planner 只做 schema/category/runtime capability 校验，不根据
   task 额外改写 `needs_visual_assistance`；
5. counting 的 `count_target` 精确展开规则和 grounding 的独立能力保持不变。

这里的 task-to-capability-owner 映射只解决“用哪份可执行类别目录校验”，不能成为
新的 assistance true/false 决策表。

### 5.4 扩展 composition root 的运行时能力注入

修改：

```text
application/bootstrap.py
```

1. 只计算一次 `_vqa_executable_leaves(...)`；
2. 将同一个不可变结果注入四个 GeneralVQAAgent task；
3. `VqaEvidenceService` 不可用时，四个 task 的可执行类别都必须为空；
4. 不因某个具体 task 重复构造 YOLO、SegFormer、Qwen 或 evidence executor；
5. 不添加 dataset-specific 分支。

### 5.5 冻结新的 assistance scope，保护 resume

修改：

```text
workflows/schema.py
application/runtime.py
workflows/visual_planner.py
workflows/dataset_runner.py
```

这次变化会使三个原本 direct-only 的 VQA task 在重新规划时可能进入 evidence 路径，
属于影响模型调用和答案的运行语义。不能让旧的 partial/failed/pending run 在 resume
时静默采用新行为。

建议新增一个独立、JSON-safe 的冻结身份字段，例如：

```text
vqa_assistance_scope = "general-vqa-agent-tasks-v1"
```

要求：

1. fresh run 将该值写入 planner `planning_parameters` 和 `run_request.json`；
2. manual ask 的 request identity 同样记录该值；
3. 新 run 的 planner identity 比较覆盖该字段；
4. 历史 run 缺失该字段时解析为 `None`，不得从当前默认值回填；
5. 历史 succeeded 样本仍允许零模型补 evaluation/Judge/report；
6. 历史非终态 VQA 样本如果需要重新规划或重跑 evidence，使用稳定错误码
   `LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED`，不得悄悄切到新 scope；
7. counting、grounding、caption/change 的 resume 行为不受影响；
8. 不复用或污染 `EvidencePreprocessingIdentity`，task scope 与 tile preprocessing
   是两个独立身份。

`VisualTaskPlan` 的字段和 JSON schema 本身没有变化，因此不需要仅为本修改升级
`visual-task-plan-v5` schema。静态 prompt 规则仍是“只输出所选 task 声明的可执行
类别”；真正变化的是 capability binding，必须由 system prompt snapshot、request
hash 和上述 run scope identity 共同冻结。

### 5.6 文档与迁移记录

修改：

```text
DETAILS.md
docs/migration/<VQA assistance scope 说明>.md
```

更新内容：

- GeneralVQAAgent 四个 task 的统一 evidence 开关；
- 删除 `scene_classification + object evidence` 属于禁止组合的旧描述；
- planner capability binding 与 VQA capability owner 的关系；
- 新旧 run 的 resume gate；
- 新旧结果可比性：历史结果不改写，但新 run 的三个 task 可能多调用证据模型并产生
  不同答案，不能把行为差异误写成 metric 改变。

`architecture/implementation_status.json` 无需因本计划新增生产 Python 文件；如果实施时
没有新增生产文件，也不修改其文件清单。

## 6. 测试计划

### 6.1 GeneralVQAAgent 单元测试

更新：

```text
tests/agents/general_vqa/test_agent.py
```

覆盖：

1. 参数化四个 supported task，`assistance=true` 时都调用一次 evidence service；
2. 四个 task 都只执行一次 final Qwen；
3. 四个 task 都持久化安全 basename `vqa_evidence.json`；
4. `assistance=false` 时四个 task 都保持 direct path，evidence service 零调用；
5. evidence service 缺失时四个 task 都稳定失败且 final Qwen 零调用；
6. `multiple_choice_vqa` evidence 路径仍执行单选/多选合法化；
7. unsupported task 仍抛 `AgentTaskMismatchError`；
8. 删除旧的“非 general_vqa assistance 必须拒绝”断言。

### 6.2 Planner 与 catalog binding 测试

更新：

```text
tests/workflows/test_visual_planner.py
tests/agents/test_evidence_catalog.py
tests/application/test_bootstrap.py
tests/application/test_prompts.py
```

覆盖：

1. system binding 为四个 VQA task 暴露相同的 runtime executable leaves；
2. 四个 task 的合法 assistance plan 均通过 post-validation；
3. 四个 task 的未知、未启用或非 canonical leaf 均 fail closed；
4. VQA service 不可用时四个 task 的 binding 均为空，planner 返回 assistance 时得到
   `CAPABILITY_UNAVAILABLE`；
5. counting/fine counting 的 capability 仍互相独立；
6. grounding 不复用 VQA service；
7. planner request 仍不包含 GT、路径、backend、checkpoint 或 answer constraints。

### 6.3 Workflow、runtime 与 resume 测试

更新：

```text
tests/workflows/test_sample_runner.py
tests/workflows/test_dataset_runner.py
tests/application/test_runtime.py
tests/integration/test_auto_task_dataset_vertical_slice.py
tests/integration/test_dataset_runner_resume.py
```

覆盖：

1. `plan.task != sample.task` 仍在 SampleRunner 边界拒绝；
2. 至少选择一个非 `general_vqa` task 做 fresh vertical slice，验证
   planner -> router -> evidence -> result -> evaluation 全链；
3. 对 `multiple_choice_vqa` 增加端到端 choice constraint 断言；
4. fresh `run_request.json` 持久化新 scope；
5. 同 scope resume 保持 succeeded 样本零 planner/YOLO/SegFormer/final-Qwen 调用；
6. 历史缺 scope 的 succeeded run 只补缺失的后处理产物；
7. 历史缺 scope 的非终态 VQA run 稳定拒绝，不产生新 evidence artifact；
8. summary 继续满足
   `total == succeeded + partial + failed + skipped`；
9. `predictions.jsonl` 继续 append-only，result path 仍为安全相对路径。

### 6.4 回归与架构门禁

实施后至少运行：

```bash
pytest -q \
  tests/agents/general_vqa/test_agent.py \
  tests/agents/general_vqa/evidence \
  tests/agents/test_evidence_catalog.py \
  tests/workflows/test_visual_planner.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/application/test_bootstrap.py \
  tests/application/test_runtime.py \
  tests/integration/test_auto_task_dataset_vertical_slice.py \
  tests/integration/test_dataset_runner_resume.py \
  tests/evaluation/test_vqa_metrics.py

pytest -q \
  tests/architecture/test_implementation_status.py \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_no_new_to_legacy_imports.py

git diff --check
git status --short
```

## 7. 明确不改的契约

本计划不修改：

- `UnifiedSample` / `SampleDraft` 字段或图像角色；
- `TaskName` 集合；
- task -> Agent 路由表；
- 主 Qwen、processor/tokenizer、checkpoint 加载和 cache model identity；
- `VisualTaskPlan` 字段、ROI 坐标制式或 materialization 几何；
- VQA evidence executor 的 detector/segmenter 选择与阈值；
- `AgentResult`、`VqaEvidenceBundle`、artifact basename；
- deterministic VQA metric、GT 解释、split、样本纳入规则；
- Judge 与 deterministic metric 的解耦；
- reporting 的只读语义；
- CLI 参数和公开命令。

## 8. 风险与验收标准

### 8.1 主要风险

1. planner binding 未同步扩展，导致 Agent 已放开但计划仍无法通过；
2. 四份类别列表被复制后漂移，出现 task 间能力不一致；
3. multiple-choice evidence 路径漏掉原 postprocess；
4. 旧非终态 run 静默采用新 assistance scope；
5. evidence service 关闭时 planner 仍宣称类别可执行；
6. 误把 task scope 扩展写成 evaluation family 或 routing 变化。

### 8.2 完成定义

只有同时满足以下条件才算实施完成：

- GeneralVQAAgent 的四个 supported task 都只按 planner assistance 字段选择路径；
- Agent 内不存在 `sample.task == "general_vqa"` 的视觉辅助专属判断；
- planner 能为四个 task 生成并校验同一 VQA capability family 的合法 evidence plan；
- runtime 可用性关闭时四个 task 一致 fail closed；
- multiple-choice、evaluation、artifact 和调用预算契约保持不变；
- 新 scope 已持久化，旧非终态 resume 不会静默改变行为；
- 目标测试、架构测试和 `git diff --check` 实际通过；
- `DETAILS.md` 与迁移说明已同步。
