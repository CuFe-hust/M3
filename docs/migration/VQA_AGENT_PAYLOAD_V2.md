# VQA Agent Payload v2 intentional differences

本迁移将 GeneralVQA、Caption 与 Grounding 的模型可见输入从共享宽 payload
收口为 task-aware 最小 payload。它是有意的 cache-breaking 输入协议变化，不改变
Ground Truth、评测指标、split、样本纳入规则或最终整图坐标解释。

主要差异：

- GeneralVQA 不再发送 `answer_constraints`；多选事实唯一来自
  `TaskNormalization.choices/allow_multiple`，并在 payload 顶层只出现一次；
- Caption 只发送 `task + question`；
- Grounding direct 只发送任务、问题与整图坐标契约；
- GeneralVQA/Grounding evidence 把动态证据放入嵌套 `evidence`，完整
  VisualTaskPlan、source geometry 与协议身份不再对最终 Qwen 可见；
- Grounding final-Qwen 改用独立的 `grounding_final_v1` prompt；
- 移出 messages 但影响可复现性或回映的信息继续进入 request hash、trace 或
  evidence artifact，因此不会制造错误 cache hit；
- `vqa-agent-sft-v1` 离线转换为 `vqa-agent-sft-v2`，字段
  `user_payload` 改名为 `base_user_payload`。样本 ID、问题、答案、split、图片
  映射和既有纳入集合保持不变。

ChangeAgent、CountingAgent、VisualTaskPlanner 首次调用、Judge 与确定性评测不在
本迁移范围；其输入契约通过回归测试保持不变。
