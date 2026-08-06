# Modification Note: Fix SegFormer iSAID Class Label Mapping - 2026-08-06 19:58:14

## Modification Time

2026-08-06 19:58:14 CST

## Modifier

Cooper (crj31415926@gmail.com)

## Modification Goal

修正 `segformer_mitb2_isaid` 模型的类别标签映射。模型 `config.json` 中仅有 `LABEL_0~15` 通用标签，经混淆矩阵验证，模型学到的类别顺序为本地 iSAID_coco category id 顺序（与 iSAID 官方 instance 标注顺序不同）。新增 `classes.json` 固化该映射，供推理、渲染、评估统一使用。

## Modified Files

- `models/segformer_mitb2_isaid/classes.json`（新增）

## Core Changes

1. 用训练集语义掩膜（`P0000` / `P0025` / `P0111`）对模型做混淆矩阵实验，确认预测索引 j 与真值掩膜像素值 j 对角线一致（覆盖类别 1,2,3,4,6,7,8,9,10,11,13，其余类别同属本地语义掩膜像素值体系）。
2. 确认类别顺序为：`0=background, 1=storage_tank, 2=Large_Vehicle, 3=Small_Vehicle, 4=plane, 5=ship, 6=Swimming_pool, 7=Harbor, 8=tennis_court, 9=Ground_Track_Field, 10=Soccer_ball_field, 11=baseball_diamond, 12=Bridge, 13=basketball_court, 14=Roundabout, 15=Helicopter`。
3. 新增 `classes.json`，包含 `id2name` / `name2id` 双向映射与验证说明，作为该模型的权威类别定义。

## Whether the Canonical Sample Format Was Changed

否

## Whether the Model Interface Was Changed

否（仅新增类别元数据文件，不改动模型权重、加载接口或推理路径）

## Whether the Configuration Was Changed

否

## Whether Evaluation Was Affected

否（不修改任何评估逻辑；`metrics.json` 中官方 val JSON 的 category id 与 train 不一致需重映射的说明保持不变）

## Whether Deployment Was Affected

否

## Whether pytest Was Updated

否（纯数据文件，无代码逻辑变更）

## Whether .gitignore Was Updated

否

## Validation Method

- 混淆矩阵验证：`P0000`（类别 1-4）、`P0025` / `P0111`（类别 2,3,5,6,7,8,9,10,11,13）三张训练图，预测索引与真值像素值对角线对应
- 桌面上两张多伦多图的渲染用修正后标签重渲染：`idx3=Small_Vehicle`（小汽车）区域显示黄色标签，与目视场景一致

## Risks and Follow-up TODOs

- 官方 iSAID instance 标注的 category id 与本地语义掩膜像素值体系不同（val 尤甚），使用其他来源标注时需先核对 id 映射
- 后续推理/渲染/评估脚本应读取 `classes.json` 作为类别定义，避免硬编码类别名
