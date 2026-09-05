# MME-RealWorld question-with-choices v2 intentional difference

MME-RealWorld adapter 从 `official-v1` 升级为
`official-v2-question-with-choices`。新样本将原始 `Text` 和
`Answer choices` 按源顺序以换行符合并到
`UnifiedSample.question`，例如：

```text
What color is the roof?
(A) Yellow
(B) Blue
(C) Gray
(D) White
```

这是经明确授权的 Planner/Agent 输入协议变更：

- fresh VisualTaskPlanner 现在能看到 MME 完整题面；
- `TaskNormalization.choices/allow_multiple` 仍为 VQA 答案约束权威；
- Ground Truth 不进入 question，评测指标与答案解释不变；
- split、样本筛选、样本顺序与图片解析不变；
- question 参与 sample ID 与 request hash，因此 v2 与 v1 的 sample
  identity、Planner/Qwen cache 及历史 run 不可混用。

本变更不修改 Golden fixtures；历史 fixture 继续用于解释锁定的
legacy manifest-adapter 行为。
