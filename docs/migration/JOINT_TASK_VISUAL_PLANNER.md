# Joint Task + Visual Planner（doc 15）任务权威性与历史可比性

本记录说明 doc 15（`docs/architecture/15_JOINT_TASK_VISUAL_PLANNER_PLAN.md`）
落地后，任务（task）权威来源与历史结果可比性如何变化。联合模式只在新
feature flag 开启时生效；默认关闭路径与 14A/14B 的 gate 流水线逐字节一致，
因此历史 run 的可比性不受影响。

## 任务权威来源变化

`try_yolo@ec962eb87c3ad0b8c1502efcbd08db0daec48868` 与 14A 流水线中，task
来自独立文本 TaskResolver（explicit / rule / model 三路径），视觉计划只能
在已经物化的 `UnifiedSample` 之后产生，且 visual plan 永远不得影响
`sample.task`。

联合模式（`visual_planning.enabled=True`）改变这一顺序：单次 Qwen 调用
同时产出 task 与视觉计划，模型选定 task 对 routing / materialization /
execution 权威；源 task（显式任务或 adapter 默认）只做审计，Ground Truth
保持只读。TaskRouter 与确定性路由不变，低置信度候选 fallback 语义不变。

```text
旧：SampleDraft -> TaskResolver(文本) -> UnifiedSample -> VisualPlanner -> Agent
新（联合）：SampleDraft/UnifiedSample -> 一次联合 Qwen 调用(task+plan)
      -> 按模型 task 物化 -> TaskRouter(确定性) -> Agent(注入 plan)
```

## 历史结果可比性

- flag 默认 `False`，关闭时 gate 路径、产物、trace、调用次数与启用前逐字节
  一致（14A3 C9 保持）；
- 联合模式产物使用新 basename `joint_visual_plan.json`，与 gate 的
  `visual_plan.json` 永不冲突；旧 run 只读兼容，reporting 不扫描任意文件；
- resume：succeeded 样本零模型调用，只补缺失/损坏的确定性评测产物；
- 有意改变：仅在显式开启联合模式时，模型 task 取代文本 TaskResolver 模型
  解析路径（§23.1 旁路）；开启时任务权威来源即模型选定 task，该语义变化
  在 `DETAILS.md` §79.9 记录。

## 影响范围

- 默认配置下无任何可观察行为变化；
- 开启联合模式后：每条样本一次联合调用（替代 gate 一次调用），trace 增加
  `joint_plan` 审计字段，task 权威来自模型；
- 不新增第二套 Prediction/全局 sample schema；`UnifiedSample` 契约不变。
