<!-- name: general_vqa; version: v3 -->

Answer the question concisely from the image. Preserve up to four representative relevant localized objects as labeled `evidence_items`; copy all evidence-item boxes into `boxes` in the same order. All boxes use integer whole-image `0..999` top-left `xyxy` coordinates and must be serialized as JSON flat arrays `[x1,y1,x2,y2]`. A box is never a pair of corner arrays. Use an empty evidence list only when the answer genuinely has no localizable visual support. Do not include hidden reasoning.

根据图像简洁回答问题。将最多四个有代表性的相关可定位目标保留为带标签的 `evidence_items`，并按相同顺序把其中的框复制到 `boxes`。所有框都使用整图左上角原点的 `0..999` 整数 `xyxy` 坐标，并在 JSON 中序列化为扁平数组 `[x1,y1,x2,y2]`；不能写成两个角点数组。只有当答案确实没有可定位视觉依据时才使用空证据列表。不要输出隐藏思维过程。
