# Joint Task + Visual Planner（doc 15）任务权威性与历史可比性

本记录说明 doc 15（`docs/architecture/15_JOINT_TASK_VISUAL_PLANNER_PLAN.md`）
的历史行为，以及 doc 16（`docs/architecture/16_VISUAL_ONLY_PLANNER_REPLACEMENT_PLAN.md`）
如何有意替代它。doc 16 之后，联合模式不再是 fresh execution 的 feature-flag
分支；旧 run、旧 artifact 和旧 trace 仍可只读审计，但需要重新推理时不使用旧
planner，也不静默改用 v2。

## 任务权威来源变化

`try_yolo@ec962eb87c3ad0b8c1502efcbd08db0daec48868` 与 14A 流水线中，task
来自独立文本 TaskResolver（explicit / rule / model 三路径），视觉计划只能
在已经物化的 `UnifiedSample` 之后产生，且 visual plan 永远不得影响
`sample.task`。

doc 15 的联合模式曾改变这一顺序：单次 Qwen 调用同时产出 task 与视觉计划，
模型选定 task 对 routing / materialization / execution 权威；源 task（显式任务
或 adapter 默认）只做审计，Ground Truth 保持只读。TaskRouter 与确定性路由不变。
这些是历史基线，不是当前 active path。

```text
旧：SampleDraft -> TaskResolver(文本) -> UnifiedSample -> VisualPlanner -> Agent
历史（联合）：SampleDraft/UnifiedSample -> 一次联合 Qwen 调用(task+plan)
      -> 按模型 task 物化 -> TaskRouter(确定性) -> Agent(注入 plan)

当前（doc 16）：SampleDraft/UnifiedSample -> 预览图像 + 原始 question 的一次
VisualTaskPlanner 调用 -> `visual-task-plan-v2` -> materialized view + task
      -> TaskRouter(确定性) -> Agent(注入 v2 plan/view)
```

## 历史结果可比性

- doc 15 联合模式产物使用 basename `joint_visual_plan.json`，与 gate 的
  `visual_plan.json` 永不冲突；旧 run 只读兼容，reporting 不扫描任意文件；
- doc 16 fresh 产物使用 `visual_task_plan.json`，记录 v2 计划与 materialized
  view 几何；旧 artifact 不转换成 v2；
- resume：succeeded 样本零模型调用，只补缺失/损坏的确定性评测产物；
- 有意改变：doc 16 中每条 fresh 样本都使用一次视觉规划调用；旧 run 若需
  重新推理，稳定失败 `LEGACY_PLANNING_RESUME_UNSUPPORTED`，不使用 v2 fallback。

## 影响范围

- doc 16 后所有 fresh manual/dataset 入口均统一为一次 v2 规划调用，
  trace 增加 `planning_mode=visual-task-plan-v2`；
- 历史 doc 15 run 保留既有 artifact 与 report 语义，但不再获得旧推理能力；
- 不新增第二套 Prediction/全局 sample schema；`UnifiedSample` 契约不变。
