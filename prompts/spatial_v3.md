<!-- name: spatial; version: v3 -->

First localize the visual evidence needed by the question, then answer concisely. Return each relevant instance as one tight labeled `evidence_items` box or point and copy all boxes into `boxes` in the same order. Use whole-image `0..999` raster coordinates with top-left origin, x positive right, and y positive down.

For top-most or bottom-most questions, enumerate every visible candidate before answering; do not return only the candidate you believe is extreme. Label vehicle instances only as `small-vehicle` or `large-vehicle`, where cars and motorcycles are small vehicles and trucks, buses, trailers, and semi-trucks are large vehicles. For arrangement questions, return separate tight instance boxes rather than one group box. For grid-position questions, return the target box. For proximity questions, return both the target and reference-region boxes. For cardinal-direction questions, do not invent north-up metadata.

If the required candidate set cannot be localized completely, keep the best supported evidence, set status to `partial`, and state a concise answer without claiming programmatic verification. Do not include hidden reasoning.

先定位回答问题所需的视觉证据，再给出简洁答案。每个相关实例使用一个紧致且带标签的 `evidence_items` 框或点，并按相同顺序把框复制到 `boxes`。使用整图 `0..999` 栅格坐标，原点在左上角，x 向右为正，y 向下为正。

对于最上方或最下方问题，回答前必须枚举所有可见候选，不能只返回认为是极值的单个候选。车辆实例只能标为 `small-vehicle` 或 `large-vehicle`：小汽车和摩托车属于小型车辆，卡车、公交车、拖车和半挂卡车属于大型车辆。排列问题必须返回逐实例紧致框，不能只返回一个群组框。九宫格位置问题返回目标框；邻近问题同时返回目标与参照区域框；地理方向问题不得虚构 north-up 元数据。

如果无法完整定位所需候选，保留已有视觉依据，将状态设为 `partial`，并给出简短答案，但不得声称已经通过程序几何验证。不要输出隐藏推理过程。
