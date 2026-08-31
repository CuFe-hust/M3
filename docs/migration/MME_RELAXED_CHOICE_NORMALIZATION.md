# MME relaxed choice normalization intentional difference

选择题 Agent postprocess 不再将无法映射到 `choices` 的模型文本标记为
`partial` / `answer_constraint_violation`。新行为是宽松的尽力规范化：

- 带标签选项的字母、括号字母、完整选项和选项文本统一为源标签；
- 无标签选项保留 canonical 选项文本；
- 可识别多选答案去重并按源选项顺序输出；
- 无法匹配的自由文本原样保留，样本状态不降级；
- 确定性 exact match 仍决定正误，Judge 仍不覆盖确定性指标。

MME Ground Truth 使用选项标签，因此 `D`、`(D)`、`(D) White` 与
`White` 会在评测前统一为 `D`。这改变 Agent 结果文本和样本状态语义，
与历史严格约束 run 分开解释；Ground Truth、split、样本纳入和
exact-match 指标定义本身不变。
