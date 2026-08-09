<!-- name: seam_review; version: v2; schema: SeamDecision -->

You are reviewing exactly two boundary observations from neighbouring image
tiles. Inspect only the supplied local seam crop and decide whether both points
mark the same physical instance, mark different physical instances, or cannot
be resolved from the crop. Never count or rescan the full image.

Return JSON only, with exactly one field and one of these values:

```json
{"decision":"same_instance"}
```

Allowed decisions are `same_instance`, `different_instances`, and `uncertain`.

你只需复核来自相邻图像切片的两个边界观测。仅检查提供的局部 seam 裁剪图，判断
两个点是否指向同一物理实例、不同物理实例，或无法根据该裁剪图确定。禁止重新计数
或扫描整张图。

只返回 JSON，且只能包含 `decision` 字段；取值必须是 `same_instance`、
`different_instances` 或 `uncertain`。
