<!-- name: spatial_candidate_review; version: v3 -->

Perform an independent evidence-localization pass for the supplied remote-sensing question. No first-pass evidence or first-pass answer is provided: inspect the entire image from scratch to avoid anchoring. Use `semantic_subtype` only to determine which physical evidence is required.

For a singular grid-position question, return the tight box of the one physical target referred to by the question. A spatially isolated instance may be the singular target when other same-class objects form a separate cluster. Never return a quadrant, grid cell, image corner, or generic answer region as evidence. If the target is genuinely ambiguous, return each plausible physical candidate separately and set status to `partial`.

For extreme-category and arrangement questions, return every relevant visible instance as a separate tight box. For vehicle questions, label instances only as `small-vehicle` or `large-vehicle`; do not return a group box. For proximity, return both the target and reference region. Use whole-image `0..999` coordinates with top-left origin, x right, and y down. Never emit zero-area boxes. Return JSON only and do not include hidden reasoning.

对给定遥感问题执行独立的证据定位。输入不提供首轮证据或首轮答案；必须从整图重新检查以避免锚定。仅使用 `semantic_subtype` 判断需要哪些物理证据。

对于单数形式的九宫格位置问题，返回问题实际指代的那个物理目标的紧致框。当其他同类物体形成另一组聚集时，空间上孤立的实例可以是该单数目标。不得把象限、九宫格单元、图像角落或通用答案区域作为证据。若目标确实有歧义，则分别返回每个合理的物理候选并将 status 设为 `partial`。

极值类别和排列问题应将每个相关可见实例作为独立紧致框返回。车辆问题中的实例只能标为 `small-vehicle` 或 `large-vehicle`，不得返回群组框。邻近问题同时返回目标和参照区域。使用整图 `0..999` 坐标，原点在左上角，x 向右，y 向下。不得输出零面积框。仅返回 JSON，不包含隐藏推理。
