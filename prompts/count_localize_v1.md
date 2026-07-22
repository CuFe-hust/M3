<!-- name: count_localize; version: v1; schema: ExpertResult -->

Independently verify a proposed whole-image vehicle count using only visible evidence in the image.
Return one tight box in `evidence_items` for every supported in-scope instance, copy those boxes into `boxes`, and put the verified integer in `answer`.

The supplied proposal is a hypothesis, not ground truth. Correct it when the visible instances disagree.
Use integer coordinates in the normalized `0..999` image frame with the origin at the top-left, x increasing rightward, and y increasing downward.
Treat cars and motorcycles as small vehicles. Treat trucks, buses, trailers, and semi-trucks as large vehicles. A generic vehicle target includes both groups.
Exclude tiny boundary fragments whose visible centre remains at the image edge because their object centre is outside the image.
Set `expert` to `counting_localizer`, keep `evidence` concise, and set `status` to `completed` when enumeration is complete.
Return valid JSON only. Do not include hidden reasoning.

仅依据图像中的可见证据，独立核验整图车辆数量提议。
对每个有证据支持且属于计数范围的实例，在 `evidence_items` 中返回一个紧框，将这些框复制到 `boxes`，并把核验后的整数写入 `answer`。

输入的数量提议只是待核验假设，不是标准答案；当可见实例不一致时应予以纠正。
坐标必须是整数，使用左上角为原点、x 向右增大、y 向下增大的 `0..999` 归一化图像坐标系。
轿车和摩托车属于小型车辆；卡车、公交车、拖车和半挂车属于大型车辆；泛指车辆时包括两类。
排除可见中心仍贴近图像边缘、因而物体中心位于图外的微小边界残片。
将 `expert` 设为 `counting_localizer`，`evidence` 保持简洁，枚举完成时将 `status` 设为 `completed`。
只返回合法 JSON，不要包含隐藏推理过程。
