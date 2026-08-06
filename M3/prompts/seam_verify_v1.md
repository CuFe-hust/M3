<!-- name: seam_verify; version: v1; schema: SeamDecision -->

Two points from neighbouring tiles are shown in one local seam crop. Decide only whether they indicate the same instance, different instances, or remain uncertain. Do not recount the full image. Return JSON only.

局部 seam 裁剪图中显示了来自相邻 tile 的两个点。只判断它们是同一实例、不同实例还是不确定。不要重新计数整图。只返回 JSON。

```json
{"decision":"same_instance","canonical_point":[0,0],"confidence":0.0,"short_reason":"string"}
```
