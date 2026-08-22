# VQA assistance scope 扩展迁移说明（doc 24）

> 日期：2026-08-21
>
> 范围：GeneralVQAAgent 的四个 supported task 统一按
> `VisualTaskPlan.needs_visual_assistance` 决定 direct/evidence 路径；planner、
> 组合根与 resume 冻结身份同步扩展。

## 1. 旧行为基线

- `GeneralVQAAgent.run()` 在 `needs_visual_assistance == true` 后还要求
  `sample.task == "general_vqa"`，其余三个 task（`scene_classification`、
  `multiple_choice_vqa`、`spatial_relation`）以
  `visual_assistance_forbidden_for_task:<task>` 稳定失败；
- `VisualTaskPlanner` 的视觉能力 task 集合只含 `counting`、
  `fine_grained_counting`、`general_vqa`、`grounding`；
- 组合根只为 `general_vqa` 计算运行时可执行类别；
- 因此旧行为下只有 `general_vqa` 能进入 `object_evidence_vqa`，其余三个
  GeneralVQAAgent task 恒为 direct-only。

## 2. 有意行为差异

| 维度 | 旧行为 | 新行为 |
|---|---|---|
| direct/evidence 开关 | `general_vqa + assistance` 专属 | 四个 GeneralVQAAgent task 统一由 `needs_visual_assistance` 决定 |
| Agent 内部 task gate | `sample.task == "general_vqa"` 二次否决 | 无；`sample.task` 仍用于路由/answer constraint/评测 dispatch |
| planner 能力绑定 | 只有 `general_vqa` 有 VQA capability | 四个 task 共享 `general_vqa` capability owner，`task_executable_categories` 按四个真实 task 分别列出同一份叶子 |
| 组合根注入 | 只算 `general_vqa` 一份 | 计算一次 `_vqa_executable_leaves` 注入四个 task；服务不可用时四个 task 一致为空 |
| 运行语义 | 三个 task 不可能调用 evidence 模型 | 新 run 的四个 task 都可能调用证据模型并产生不同答案 |

### 2.1 明确不变

- `counting` / `fine_grained_counting` 仍由 CountingAgent 拥有；
- `grounding` 仍由 GroundingAgent 拥有；
- `caption`、`change_caption`、`change_qa` 不借此接入 VQA evidence；
- routing fallback 不得改写 persisted resolved task 或评测 task；
- `VisualTaskPlan` 字段与 schema 版本（`visual-task-plan-v5`）不变；
- VQA evidence executor、`VqaEvidenceBundle`、artifact basename、deterministic
  metric、GT 解释与 Judge 解耦均不变。

## 3. 冻结身份与 resume 门禁

新运行把 `vqa_assistance_scope = "general-vqa-agent-tasks-v1"` 写入 planner
`planning_parameters`（进入 system prompt 绑定、prompt snapshot 与 request
hash）、`run_request.json` 与手动 ask 的 `request.json`。planner identity 比较
覆盖该字段。

- 历史 run request 缺失该字段解析为 None，绝不从当前默认值回填；
- 历史 succeeded VQA 样本 resume 仍只补缺失的确定性评测/Judge/report，零
  模型调用；
- 历史非终态 VQA 样本（四个 GeneralVQAAgent task 之一）需要重新规划/重跑
  evidence 时以 `LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED` 稳定失败，绝不
  静默采用新 scope；
- 两个 legacy 门禁都以**持久化 `status.json` 的 execution task** 为权威，
  而不是 adapter 本次返回的 source task——planner 可能改写 source（如
  source=caption 被改写为 general_vqa），只看 source 会漏判并静默重规划；
  仅当没有任何持久化状态（缺失/损坏状态按文档契约视为缺席重跑）时才回退
  source task；
- 持久化 execution task 为诚实哨兵 `unknown`（预 task 失败）时无法证明重
  规划会留在 VQA 族之外，按 fail-closed 处理（返回
  `LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED`），draft 路径同样适用；
- counting、grounding、caption/change 的 resume 行为不受影响（持久化执行
  task 非 VQA 时仍按文档契约重规划）；
- 不复用或污染 `EvidencePreprocessingIdentity`：task scope 与 tile
  preprocessing 是两个独立冻结身份，各自有独立 legacy gate。

## 4. 历史结果可比性

- 历史 run 的执行产物、评测与报告一律不改写；
- 新 run 的三个 task（`scene_classification`、`multiple_choice_vqa`、
  `spatial_relation`）在 planner 请求辅助时可能多调用证据模型并产生不同
  答案——这是运行语义差异，不是 metric 定义或 GT 解释变化，不得把两者
  混淆；
- 行为差异通过 `vqa_assistance_scope` 身份与逐样本 resume gate 冻结，任何
  历史非终态样本都不会悄悄切换到新行为。

## 5. Golden fixtures / 测试影响

- `tests/agents/general_vqa/test_agent.py`：删除“非 general_vqa assistance
  必须拒绝”断言，新增四 task 参数化 evidence/direct/service-missing 覆盖；
- `tests/workflows/test_visual_planner.py`：`_EXECUTABLE_BY_TASK` 扩展为七个
  task，新增共享 capability owner 绑定与 fail-closed 覆盖；
- `tests/application/test_bootstrap.py`：新增四 task 共享绑定与 scope 冻结
  断言；
- `tests/application/test_runtime.py`：新增 fresh scope 持久化、resume 重建、
  scope 冲突拒绝、历史缺 scope succeeded 补判测试；
- `tests/integration/test_dataset_runner_resume.py`：`_setup` 默认注入新
  scope，新增历史缺 scope 非终态 VQA 稳定拒绝测试，以及评审回归：source
  task 非 VQA 而持久化 execution task 为 VQA 时两个 legacy 门禁均以
  persisted task 判罚（`test_legacy_scope_gate_uses_persisted_task_not_source_task`、
  `test_legacy_preprocessing_gate_uses_persisted_task_not_source_task`）；
  persisted `unknown` 哨兵在 sample 与 draft 路径都 fail-closed；持久化
  非 VQA task（caption）保持可重规划不被过度拦截；
- `tests/integration/test_auto_task_dataset_vertical_slice.py`：新增
  scene_classification 与 multiple_choice_vqa 的 evidence 垂直切片。
