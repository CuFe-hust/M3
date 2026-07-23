<!-- name: spatial; version: v4 -->

Answer the supplied remote-sensing question using its `semantic_subtype` and, when non-empty, exactly the supplied `answer_vocabulary`. The semantic subtype takes precedence over a coarse dataset type. For example, an orientation question remains orientation even if the dataset type says position.

First enumerate the visual evidence needed by the subtype. Use separate tight boxes for real object extents and points only when a reliable extent cannot be located. Never encode a line or point as a zero-area box. Use whole-image `0..999` raster coordinates with top-left origin, x right, and y down.

For extreme-category and arrangement questions, enumerate every relevant vehicle as `small-vehicle` or `large-vehicle`. For grid position, enumerate all candidates of the named class; do not choose a convenient first object. For proximity, include both target and reference-region evidence. For orientation, answer in the requested vocabulary based on the visible dominant image axis; do not answer with endpoints such as “top-right” or with a relation such as “parallel”. For arrangement, use the supplied closed vocabulary rather than a descriptive sentence.

If required evidence is incomplete, keep the supported observations and set status to `partial`. Return JSON only and do not include hidden reasoning.

依据输入的 `semantic_subtype` 回答遥感问题；当 `answer_vocabulary` 非空时，答案必须严格取自该词表。语义子类型优先于粗粒度数据集类型，例如方向问题即使被数据集标为位置，也仍按方向处理。

先枚举该子类型所需的视觉证据。真实物体范围使用独立紧致框；只有无法可靠定位范围时才使用点。不得用零面积框表示线或点。使用整图 `0..999` 栅格坐标，原点在左上角，x 向右，y 向下。

极值类别和排列问题必须枚举所有相关车辆，并仅标为 `small-vehicle` 或 `large-vehicle`。九宫格位置问题先枚举指定类别的全部候选，不能直接选择第一个方便的物体。邻近问题同时提供目标和参照区域证据。方向问题依据图像中的主要可见轴从指定词表作答，不得回答“右上”等端点位置，也不得用“平行”等关系代替方向。排列问题必须使用给定封闭词表，而不是描述性长句。

若所需证据不完整，保留有依据的观测并将 status 设为 `partial`。仅返回 JSON，不包含隐藏推理。
