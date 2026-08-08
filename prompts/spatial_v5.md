<!-- name: spatial; version: v5 -->

Answer the supplied remote-sensing question using its `semantic_subtype`. When `answer_vocabulary` is non-empty, use it exactly. The semantic subtype takes precedence over a coarse dataset type. For example, an orientation question remains orientation even if the dataset type says position.

First localize the physical visual evidence needed by the subtype. Use separate tight boxes that follow actual object pixels, and use points only when a reliable extent cannot be located. Never encode a direction, quadrant, grid cell, image corner, or other answer region as a box. Never encode a line or point as a zero-area box. Use whole-image `0..999` raster coordinates with top-left origin, x right, and y down.

For a singular grid-position question, locate the one physical target referred to by the question and return its tight object box. A spatially isolated instance may be the singular target when the remaining same-class objects form a separate cluster. Do not draw the named grid cell and do not invent convenient corner coordinates. If the singular target is genuinely ambiguous, return the plausible physical candidates as separate boxes and set status to `partial`; program logic will derive the final grid label from evidence geometry.

For extreme-category and arrangement questions, enumerate every relevant vehicle as `small-vehicle` or `large-vehicle`. For proximity, include both target and reference-region evidence. For orientation, answer in the requested vocabulary based on the visible dominant image axis; do not answer with endpoints such as “top-right” or with a relation such as “parallel”. For arrangement, use the supplied closed vocabulary rather than a descriptive sentence.

If required evidence is incomplete, keep the supported observations and set status to `partial`. Return JSON only and do not include hidden reasoning.

依据输入的 `semantic_subtype` 回答遥感问题；当 `answer_vocabulary` 非空时，答案必须严格取自该词表。语义子类型优先于粗粒度数据集类型，例如方向问题即使被数据集标为位置，也仍按方向处理。

先定位该子类型所需的物理视觉证据。真实物体范围使用贴合实际物体像素的独立紧致框；只有无法可靠定位范围时才使用点。不得把方向、象限、九宫格单元、图像角落或其他答案区域当作物体框，也不得用零面积框表示线或点。使用整图 `0..999` 栅格坐标，原点在左上角，x 向右，y 向下。

对于单数形式的九宫格位置问题，定位问题实际指代的那个物理目标，并返回贴合该物体的紧致框。当其余同类物体形成另一组聚集时，空间上孤立的实例可以是该单数目标。不得绘制答案所在的九宫格单元，也不得编造方便的角落坐标。若单数目标确实有歧义，则把合理候选作为独立物体框返回并将 status 设为 `partial`；程序将依据证据几何计算最终位置标签。

极值类别和排列问题必须枚举所有相关车辆，并仅标为 `small-vehicle` 或 `large-vehicle`。邻近问题同时提供目标和参照区域证据。方向问题依据图像中的主要可见轴从指定词表作答，不得回答“右上”等端点位置，也不得用“平行”等关系代替方向。排列问题必须使用给定封闭词表，而不是描述性长句。

若所需证据不完整，保留有依据的观测并将 status 设为 `partial`。仅返回 JSON，不包含隐藏推理。
