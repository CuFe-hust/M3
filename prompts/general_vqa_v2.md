<!-- name: general_vqa; version: v2 -->

Answer the question concisely from the image. Preserve up to four representative relevant localized objects as labeled `evidence_items`; copy all evidence-item boxes into `boxes` in the same order. Coordinates are whole-image `0..999` raster coordinates with the origin at the top-left, positive x to the right, and positive y downward. A box is one flat array `[x1,y1,x2,y2]`, never a pair of corner arrays. Use an empty evidence list only when the answer genuinely has no localizable visual support. Do not include hidden reasoning.

根据图像简洁回答问题。将最多四个有代表性的相关可定位目标保留为带标签的 `evidence_items`，并按相同顺序把其中的框复制到 `boxes`。坐标使用整图 `0..999` 栅格坐标：原点在左上角，x 向右为正，y 向下为正；一个框必须是单个扁平数组 `[x1,y1,x2,y2]`，不能写成两个角点数组。只有当答案确实没有可定位视觉依据时才使用空证据列表。不要输出隐藏思维过程。
