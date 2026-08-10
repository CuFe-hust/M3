<!-- name: spatial_candidate_review; version: v4 -->

Independently inspect the entire remote-sensing image and enumerate every physical instance relevant to the supplied question. No first-pass answer or evidence is provided. For vehicle questions, label each tight instance box only as `small-vehicle` or `large-vehicle`; never return a group box. Use whole-image `0..999` coordinates with top-left origin. Set `complete` to false if any relevant instance may be missing or ambiguous.

Return exactly this compact JSON shape: `{"boxes":[["small-vehicle",x1,y1,x2,y2]],"complete":true}`. Each box array is `[label,x1,y1,x2,y2]`. Do not return an answer, prose evidence, confidence, points, geometry, status, Markdown, or hidden reasoning.

独立检查整幅遥感图像，并枚举与问题相关的每个物理实例。输入不提供首轮答案或证据。车辆问题中，每个紧致实例框只能标为 `small-vehicle` 或 `large-vehicle`，不得返回群组框。使用左上角为原点的整图 `0..999` 坐标。若可能遗漏实例或存在歧义，将 `complete` 设为 false。

只返回上述紧凑 JSON，不返回答案、文字证据、置信度、点、geometry、status、Markdown 或隐藏推理。
