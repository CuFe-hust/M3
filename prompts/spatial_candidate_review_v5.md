<!-- name: spatial_candidate_review; version: v5 -->

Independently localize the one physical target named by the grid-position question. No first-pass answer or evidence is provided. Return its tight box, never a quadrant, grid cell, image corner, generic answer region, or group box. Use the explicit vehicle class as the label; otherwise use `position-target`. If the target is ambiguous, return every plausible physical candidate and set `complete` to false. Use whole-image `0..999` coordinates with top-left origin.

Return exactly this compact JSON shape: `{"boxes":[["large-vehicle",x1,y1,x2,y2]],"complete":true}`. Each box array is `[label,x1,y1,x2,y2]`. Do not return an answer, prose evidence, confidence, points, geometry, status, Markdown, or hidden reasoning.

独立定位九宫格位置问题所指的唯一物理目标。输入不提供首轮答案或证据。返回目标的紧致框，不得返回象限、网格单元、图像角落、通用答案区域或群组框。标签使用问题明确指定的车辆类别，否则使用 `position-target`。若目标有歧义，返回所有合理物理候选并将 `complete` 设为 false。使用左上角为原点的整图 `0..999` 坐标。

只返回上述紧凑 JSON，不返回答案、文字证据、置信度、点、geometry、status、Markdown 或隐藏推理。
