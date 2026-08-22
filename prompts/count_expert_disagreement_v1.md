<!-- name: count_expert_disagreement; version: v1; schema: DisagreementReview -->

Review only the requested detector disagreements for the supplied object-counting
target. Each crop is annotated with letter markers (A/B/C...) and the structured
context maps each marker to its exact candidate_id. Decide only among the supplied
candidate_ids: accept exactly one, accept multiple supplied candidates, reject all,
or mark uncertain. Do not invent a candidate, delete a candidate, recount consensus
objects outside these crops, or infer a different target class. Prefer visible
geometry over detector confidence. Return only the requested structured decisions.

仅复核给定的检测器分歧区域。每个 crop 都有 A/B/C 等字母标记，结构化上下文
会把标记映射到精确的 candidate_id。只能在给定 candidate_id 中选择：接受一个、
接受多个给定候选、全部拒绝或 uncertain；不能新增或删除候选，不能重新统计 crop
外已达成共识的目标，也不能改变目标类别。优先依据可见几何而不是检测置信度。
只返回请求的结构化决策。
