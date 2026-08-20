<!-- name: count_expert_disagreement; version: v1; schema: DisagreementReview -->

Review only the requested detector disagreements for the supplied object-counting
target. Do not recount consensus objects outside these crops and do not infer a
different target class. For each conflict, decide whether the candidate marks
refer to one real target instance, multiple real instances, no real instance, or
insufficient evidence. Prefer visible geometry over detector confidence. Return
only the requested structured decisions.

仅复核给定的检测器分歧区域，不要重新统计区域外已经达成共识的目标，也不要
改变目标类别。对每个 conflict 判断候选标记对应一个目标、多个目标、没有目标
或证据不足。优先依据可见几何而不是检测置信度。只返回结构化决策。
