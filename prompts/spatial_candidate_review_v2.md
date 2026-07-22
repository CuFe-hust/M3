<!-- name: spatial_candidate_review; version: v2 -->

Perform an independent candidate-enumeration pass for the supplied remote-sensing question. No first-pass evidence is provided: inspect the entire image from scratch to avoid anchoring. Use `semantic_subtype` and the supplied answer vocabulary only to determine which evidence is required.

Return every relevant visible instance as a separate tight box. For vehicle questions, label instances only as `small-vehicle` or `large-vehicle`; do not return a group box. For proximity, return both the target and reference region. Use whole-image `0..999` coordinates with top-left origin, x right, and y down. Never emit zero-area boxes. Return JSON only and do not include hidden reasoning.

对给定遥感问题执行独立的候选枚举。输入不提供首轮证据；必须从整图重新检查，以避免锚定偏差。仅使用 `semantic_subtype` 和答案词表判断需要哪些证据。

将每个相关可见实例作为独立紧致框返回。车辆问题中的实例只能标为 `small-vehicle` 或 `large-vehicle`，不得返回群组框。邻近问题同时返回目标和参照区域。使用整图 `0..999` 坐标，原点在左上角，x 向右，y 向下。不得输出零面积框。仅返回 JSON，不包含隐藏推理。
