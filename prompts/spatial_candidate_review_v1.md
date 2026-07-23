<!-- name: spatial_candidate_review; version: v1 -->

Perform a completeness review of visual candidates for the supplied remote-sensing question. The first pass evidence is included in the user message. Return all relevant instances visible in the whole image, including the first-pass instances, as separate tight evidence boxes. Do not return a group box.

For vehicle questions, label every instance only as `small-vehicle` or `large-vehicle`. Cars and motorcycles are small vehicles; trucks, buses, trailers, and semi-trucks are large vehicles. Use whole-image `0..999` raster coordinates with top-left origin, x positive right, and y positive down. Return valid JSON only and do not include hidden reasoning.

对给定遥感问题执行候选完整性复查。用户消息包含首轮证据。返回整张图中所有相关可见实例，包括首轮实例；每个实例使用独立紧致框，不得返回群组框。

车辆问题的每个实例只能标为 `small-vehicle` 或 `large-vehicle`。小汽车和摩托车属于小型车辆；卡车、公交车、拖车和半挂卡车属于大型车辆。使用整图 `0..999` 栅格坐标，原点在左上角，x 向右为正，y 向下为正。只输出合法 JSON，不要包含隐藏推理过程。
