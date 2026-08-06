<!-- name: missing_point_review; version: v3; schema: TileCountResponse -->

Independently rescan the complete owner core of one remote-sensing crop. Do not assume that an earlier empty result was correct. Use the supplied aliases and inclusion/exclusion rules, and return one point per visually supported target whose centre belongs to the owner core.

Use a top-to-bottom, left-to-right sweep and explicitly check low-contrast, small, edge-adjacent, and elongated candidates. If a possible target remains too small or ambiguous, set `needs_split=true` and include `zero_unconfirmed` in `uncertainty`; do not certify zero. Use an empty point list with `confirmed_absent` only after the complete core is clear and contains no plausible candidate. Return JSON only and preserve the supplied target and tile ID.

独立重新扫描一个遥感裁剪图的完整 owner core，不要假设先前的空结果正确。依据给定的别名及纳入/排除规则，为中心属于 owner core 的每个有视觉依据的目标返回一个点。

按从上到下、从左到右扫描，并检查低对比度、小尺寸、靠近边缘和细长的候选。若可能目标仍过小或含糊，设置 `needs_split=true`，并在 `uncertainty` 中加入 `zero_unconfirmed`，不得确认零。只有完整核心区域清晰且不存在任何合理候选时，才返回空点并加入 `confirmed_absent`。仅返回 JSON，并保持给定的 target 和 tile ID。
