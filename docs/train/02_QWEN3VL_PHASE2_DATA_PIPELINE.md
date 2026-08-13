# Phase 2 Dataset、在线增强与 Collator 实现任务

## 1. 任务目标

实现：

```text
scripts/qwen3vl_phase2_data.py
tests/test_qwen3vl_phase2_data.py
```

该文件只消费 `01_PREPARE_PHASE2_SFT_DATA.md` 定义的 canonical Episode JSONL，
负责：

```text
读取 Episode
-> 安全解析图片路径
-> 在线、任务感知的几何增强
-> 同步变换 input_boxes/target_boxes
-> 渲染统一 Qwen 对话
-> processor 编码
-> 构造 labels
-> collate batch
```

它不解析 VRSBench/GeoChat 原始 annotation，不加载主模型、不挂 LoRA、不创建 optimizer、
不保存训练 checkpoint。

## 2. 开始实现前必须确认

1. 阅读根目录 `AGENTS.md`、`DETAILS.md` 和
   `docs/train/01_PREPARE_PHASE2_SFT_DATA.md`；
2. 执行 `git status --short`、`git rev-parse HEAD`；
3. 检查第一轮输出 schema 和测试，不靠本文示例猜字段；
4. 确认目标 Python 路径已经在架构白名单中；
5. 检查当前固定 Transformers 版本下 Qwen3-VL processor 的真实返回键；
6. 不修改第一轮已经冻结的数据语义和 40% 选择结果。

当前训练协议：

```text
Grounding assistant 输出框。
VQA 的框只作为部分 user 输入的 annotation context。
VQA assistant 只输出原始标准答案。
```

## 3. 对外接口

该模块至少提供：

```python
Phase2EpisodeDataset
Phase2DataCollator
AugmentationConfig
DatasetRootConfig
```

训练脚本通过这些公开对象构建 train/validation dataset，不复制内部预处理逻辑。

Dataset 需要能接收当前 epoch，以便在线增强 seed 由 epoch 决定。可以提供显式
`set_epoch(epoch)`，并要求训练侧在每个 epoch 开始时调用；不要依赖不可控的全局随机状态。

## 4. 图片路径安全

Episode 只保存相对路径和 `image_source`。训练 CLI 显式提供类似映射：

```text
vrsbench=/data/VRSBench-full
geochat=/data/GeoChat/images
```

解析规则：

- 拒绝绝对路径、Windows drive、UNC、`.`、`..` 和 nested escape；
- `resolve()` 后确认文件仍在声明 root 内；
- 图片缺失、不可读或格式错误时使用稳定错误类型；
- 错误信息不得把机器绝对路径写入持久化 artifact；
- PIL 图像只存在于运行时，不进入 Episode、trace 或 manifest；
- train 与 validation 使用同一安全解析实现。

## 5. 数据增强总体规则

增强只在 train dataset 开启。validation 必须始终使用 identity transform。

首版支持：

```text
离散旋转：90° / 180° / 270°
小角度仿射旋转：默认范围 -5°..+5°
缩放：默认 0.95..1.05
平移：默认不超过宽高 2%
轻度透视：默认角点扰动不超过宽高 1%..2%
恶劣成像质量模拟：亮度、对比度、模糊、噪声、JPEG 压缩、暗角，参数见第 8 节
```

不实现 elastic deformation，也不实现桶形/枕形等会移动像素坐标的几何镜头畸变。
恶劣成像质量模拟只改变图像的可见质量，不改变任何像素坐标或框。增强概率和幅度全部由
`AugmentationConfig` 显式配置，不要在 `__getitem__` 内散落魔法数字。

所有被采样的几何增强必须由同一个、可记录的几何变换同步作用于：

```text
image
turns[].input_boxes
turns[].target_boxes
```

绝不允许只做旋转、仿射或透视图像而不更新框。恶劣成像质量模拟只作用于经过几何阶段的
图像，禁止改变框、输出尺寸或像素坐标网格。

## 6. 方向和空间语义保护

第一轮会给出：

```text
augmentation_policy.geometry = orientation_locked | geometry_safe
```

初版行为固定为：

```text
orientation_locked -> 几何阶段使用 identity；仍可做恶劣成像质量模拟
geometry_safe      -> 可按配置采样几何增强和恶劣成像质量模拟
```

本文件不得自动改写 `left/right/top/bottom/north/south/east/west` 等自然语言。
如果第一轮漏标了明显方向敏感文本，本文件可以进行额外保守检查并降级为 identity，但不能
把 `orientation_locked` 升级为可增强。

## 7. 框的几何变换

输入框坐标是相对于模型实际看到的整图的 `0..999 xyxy`。增强内部应转为浮点像素空间：

```text
xyxy_999
-> 原始图像 pixel xyxy
-> 四个角点
-> affine/homography matrix
-> 变换后四边形
-> enclosing axis-aligned box
-> 与增强图边界相交/裁剪
-> 质量门禁
-> round 到 0..999 xyxy
```

不能只变换 `(x1, y1)` 和 `(x2, y2)`，因为旋转/透视后它们不再是新 AABB 的对角点。

每个框至少检查：

```text
finite coordinates
x1 < x2 and y1 < y2
与图像有非空交集
可见面积比例 >= 配置阈值
enclosing AABB 面积膨胀 <= 配置阈值
最终坐标在 0..999
```

如果任意必需 input/target box 未通过门禁，则整条 Episode 回退 identity transform：

- 不删除某一个框；
- 不丢弃该 Episode；
- 不把有框 VQA 改成无框 VQA；
- 不把 Grounding 多框目标改成部分框目标。

运行时可返回结构化 augmentation metadata，但默认不把每个 step 的大规模 metadata 持久化。
调试模式下只记录 seed、变换类型、矩阵和稳定 fallback code。

## 8. 恶劣成像质量模拟

### 8.1 定义与边界

该阶段的目标是模拟低照度、失焦、抖动、传感器噪声、低动态范围、传输压缩和镜头边缘
衰减等恶劣成像质量，让模型在视觉信息退化时仍能识别目标和完成 VQA/Grounding。

它必须保持坐标不变：输出图像的宽高和像素网格与输入完全一致，任何源像素位置不发生
几何位移。因此：

```text
退化前 input_boxes  == 退化后 input_boxes
退化前 target_boxes == 退化后 target_boxes
```

本阶段明确不包含：

```text
桶形/枕形径向畸变
切向镜头畸变
rolling shutter 几何形变
局部 elastic deformation
随机 crop/resize/translation
任何会移动目标像素坐标的操作
```

这些属于几何变换；如果未来加入，必须同步变换框，不能混入本坐标保持管线。

### 8.2 组合策略

恶劣成像质量模拟在几何阶段成功或回退后执行。先用独立随机子流判断是否启用，然后从
候选退化中不重复选择 `1..max_degradations_per_sample` 个，建议初始最多 3 个。

建议总配置：

```text
degradation_probability = 0.45
min_degradations_per_sample = 1
max_degradations_per_sample = 3
randomize_degradation_order = false
```

初版采用固定顺序，只有被选中的步骤生效：

```text
低对比度
亮度减弱
暗角
失焦/运动模糊（二选一）
传感器噪声
JPEG 压缩
```

固定顺序使实现和测试更可复现，也更接近“场景光照/镜头 -> 采集模糊/噪声 -> 编码压缩”的
成像链路。不要让一个样本同时叠加失焦和运动模糊，也不要在默认配置下一次叠加全部退化。

### 8.3 亮度减弱

```text
I_out = clamp(I_in * brightness_factor)
```

建议初始参数：

```text
brightness_weight = 1.0
brightness_factor_min = 0.55
brightness_factor_max = 0.90
```

要求：

- factor 始终小于等于 1，只减暗、不提高亮度；
- RGB 三通道乘相同系数，保留色相和相对饱和度；
- 在 RGB uint8 或定义明确的 `[0,1]` 浮点空间执行，不能在 processor normalization 后乘；
- 最小 factor 必须通过小目标可辨识度的可视化 QA。

### 8.4 低对比度

围绕图像或每通道均值收缩动态范围：

```text
I_out = mean + contrast_factor * (I_in - mean)
```

建议：

```text
contrast_weight = 0.6
contrast_factor_min = 0.55
contrast_factor_max = 0.90
```

factor 不得大于 1。均值定义（全 RGB 标量或逐通道）必须固定并由测试覆盖，建议逐通道均值，
避免额外色偏。

### 8.5 失焦模糊

使用轻度 Gaussian blur 模拟失焦：

```text
defocus_blur_weight = 0.6
kernel_size in {3, 5}
sigma_min = 0.4
sigma_max = 1.2
```

kernel 必须为奇数；不允许使用足以让遥感小目标完全消失的大核。

### 8.6 轻度运动模糊

使用中心对齐、归一化的一维线性卷积核，随机角度但不平移输出网格：

```text
motion_blur_weight = 0.4
kernel_size in {3, 5, 7}
angle_degrees = 0..180
```

卷积边界策略必须固定，建议 reflect padding。卷积会降低局部清晰度，但输出像素坐标和尺寸
保持不变，所以框无需调整。

### 8.7 传感器噪声

首版使用零均值 Gaussian noise：

```text
sensor_noise_weight = 0.7
noise_sigma_min = 0.005
noise_sigma_max = 0.03       # 对 [0,1] RGB
```

噪声使用独立确定性子种子。加噪后 clip 到合法范围。初版不模拟复杂条带噪声、坏点或泊松
光子噪声，后续只有在目标传感器分布明确时再增加。

### 8.8 JPEG 压缩

通过内存 buffer 做一次 JPEG encode/decode：

```text
jpeg_weight = 0.5
jpeg_quality_min = 55
jpeg_quality_max = 90
```

不得写临时图片到训练数据目录。encode/decode 后恢复 RGB，宽高必须完全不变。源图含 alpha
时先按明确规则转换为 RGB。

### 8.9 暗角

用以图像中心为中心的平滑径向 mask 衰减边缘亮度，不移动像素：

```text
vignette_weight = 0.4
vignette_edge_factor_min = 0.55
vignette_edge_factor_max = 0.85
vignette_power_min = 1.5
vignette_power_max = 3.0
```

mask 中心值固定为 1，向边缘单调下降。暗角只是亮度 mask，不使用 OpenCV camera remap，
不产生桶形/枕形几何变化。

### 8.10 统一安全规则

所有退化都必须：

- 保持图像宽、高和像素坐标网格不变；
- 不修改 `input_boxes`、`target_boxes`、question、answer 或空间描述；
- train-only，validation 全部关闭；
- 使用确定性 seed；
- paired 有框/无框 VQA 共享同一选择集合、顺序和全部参数；
- 不使用训练时全局随机状态；
- 对 dtype/range 做显式检查，输出有限且可转回 RGB；
- 将实际选择和参数放入轻量 augmentation metadata；
- 失败时使用稳定错误码并将整条 degradation pipeline 回退到几何阶段的输出，不改样本类型；
- 通过抽样 montage 做视觉 QA，确认组合后仍保留可学习的视觉信息。

几何增强与恶劣成像质量模拟使用独立随机子流。启用/禁用或调整某个几何操作不应改变同一
Episode 的 degradation 选择；新增一个退化类型也不应改变其他已选择类型的参数子种子。

## 9. 成对有框/无框 VQA 的增强一致性

`vqa_box_assisted` 和对应 `vqa_self_attention` 共享 `parent_episode_id`。同一 epoch
必须看到完全相同的增强图像，唯一差异是 user prompt 中是否出现 annotation boxes。

增强 seed：

```text
sha256(global_seed, epoch, parent_episode_id)
```

其他 Episode 使用自身 `episode_id` 作为 group id。禁止使用 Python `hash()`。

要求：

- 相同 seed、epoch、group id 和配置产生相同变换；
- 不同 DataLoader worker 不改变结果；
- resume 到相同 epoch/step 时不发生增强漂移；
- paired view 不共享可变 PIL/tensor 对象，避免 worker 间状态污染。

paired view 必须共享：

```text
同一几何族和几何参数
同一恶劣成像退化步骤集合和固定顺序
每个退化步骤的全部参数
```

## 10. 对话渲染协议

结构化框必须在增强完成后才渲染为文本。

### 10.1 Grounding

User：

```text
<image>
Locate the region described below.
Description: <原 referring expression>
Return the bounding boxes as JSON in 0..999 xyxy coordinates.
```

Assistant：

```json
{"boxes":[{"xyxy":[380,200,450,260]}]}
```

多框按源标注稳定顺序输出。禁止输出隐藏推理过程。

### 10.2 VQA 有框版

User：

```text
<image>
Question: How many small vehicles are visible?
Available annotated regions:
- vehicle: [380, 200, 450, 260] — The small vehicle ...
```

Assistant：

```text
1
```

`Available annotated regions` 是图像级 annotation context，不宣称框完整覆盖问题证据。

### 10.3 VQA 自主注意力版/天然无框版

User：

```text
<image>
Question: How many small vehicles are visible?
```

Assistant 与有框 parent 完全相同。prompt 中不能出现框、坐标、annotation description
或暗示已知区域的占位文本。

### 10.4 GeoChat `[identify]`

User 使用统一 `0..999 xyxy` annotation-region 格式，assistant 保留区域描述文本。

### 10.5 普通 GeoChat 多轮

- 第一轮包含一个 `<image>`；
- 后续轮复用同一图像上下文，不重复注入图片；
- 角色顺序保持；
- GeoChat 原私有框语法不得漏入最终 prompt/answer；
- 所有合法 assistant turn 都参与监督。

## 11. Processor、tokenization 和 loss mask

必须优先使用当前 Qwen3-VL `AutoProcessor` 和 `apply_chat_template` 的真实行为，不手写
猜测的 Qwen 特殊 token 序列。

要求：

- `add_generation_prompt=False` 用于完整训练对话；
- user/system/image/padding token 对应 label 全部为 `IGNORE_INDEX=-100`；
- 每个 assistant 内容 span 参与 loss；
- 是否监督 assistant role/header/end token 必须统一并用测试固定；
- image/video placeholder 数量必须与视觉输入一致；
- 保留 processor 返回的 Qwen3-VL 必需张量，例如实际版本中的
  `pixel_values`、`image_grid_thw`、`mm_token_type_ids`；
- `remove_unused_columns=False` 由训练脚本设置；
- 不对 validation 做随机 resize/crop 或几何增强。

不要把整段文本编码一次后靠字符串长度猜 assistant mask。应使用 processor/tokenizer 能
提供的 chat-template assistant mask；若固定版本确实不提供，则实现逐 turn 边界编码并用
测试验证特殊 token、Unicode 和重复文本场景。

## 12. 截断策略

不能简单执行 `input_ids[:max_seq_length]`，因为这可能留下只有 user prompt、没有 assistant
目标的坏样本，也可能切断视觉 token span。

规则：

1. 图片 token 不可被截断；
2. 以完整 user/assistant turn pair 为截断边界；
3. 多轮过长时优先保留图片第一轮上下文，并从末尾删除完整的后续 turn pair；
4. 至少保留一个有非空监督 token 的 assistant turn；
5. 单个完整 pair 已超过限制时抛出稳定 `episode_too_long`，不要生成全 `-100` labels；
6. 训练脚本启动前应能通过预检统计过长 Episode 数量。

## 13. Collator

Collator 应：

- 右侧 padding `input_ids`、`attention_mask`、`labels` 和必要文本侧张量；
- label padding 使用 `-100`；
- 按当前 Qwen3-VL processor 的约定拼接视觉张量；
- 支持 batch 内不同原图尺寸经 processor 得到的不同视觉 token 数；
- 拒绝缺少必要 key 的 feature；
- 不把 Episode metadata 传入模型 forward；
- 可选保留轻量 `episode_ids` 供错误定位，但必须在 forward 前移除。

## 14. 依赖和性能边界

- 优先使用仓库已有 PIL、torch、torchvision 和 OpenCV；当前仓库已声明
  `opencv-python>=4.8`，不得再引入 Albumentations/Kornia 等新依赖；
- 模糊、JPEG 和暗角实现必须固定 interpolation、padding、色彩空间、dtype 和 rounding 语义；
- 如果当前运行环境缺少某个已声明依赖，应明确失败或使用经过同等测试的现有依赖路径，
  不能静默关闭某类退化而继续声称使用了完整配置；
- Dataset 应 lazy load 图片和 tokenization，不预加载完整图像集；
- 不在 worker 中调用网络；
- 图像打开后及时关闭文件句柄；
- 保持默认离线。

## 15. 测试与验收

单元测试使用临时小图、fake processor 或轻量 tokenizer，不加载 8B 模型。至少覆盖：

1. dataset root 路径安全及逃逸拒绝；
2. 90°、180°、270° 的精确框变换；
3. 仿射旋转、缩放、平移和透视的四角点变换；
4. 每种成像退化前后图像尺寸和全部框逐值不变；
5. 亮度只减弱、RGB 通道比例保持；
6. 对比度 factor 不大于 1，均值定义与实现一致；
7. 失焦和运动模糊 kernel 归一化、输出尺寸不变；
8. Gaussian noise seed 可复现且输出范围合法；
9. JPEG encode/decode 不写磁盘且保持 RGB/尺寸；
10. 暗角中心不变、边缘单调衰减且不移动像素；
11. 同一样本不会同时选择失焦和运动模糊；
12. 每次选择 1..配置上限个退化，不默认堆叠全部效果；
13. 几何与成像退化使用独立随机子流；
14. degradation 失败回退到几何阶段输出，框和样本类型不变；
15. enclosing AABB 与 `0..999` round 规则；
16. 几何框失败触发整条 geometry identity fallback；
17. `orientation_locked` 永不做几何增强，但允许恶劣成像质量模拟；
18. validation 的几何和成像退化全部关闭；
19. paired VQA 在同 epoch 得到相同几何与成像退化参数；
20. 不同 worker/重建 dataset 后增强可复现；
21. 有框与无框 prompt 的唯一区别符合契约；
22. Grounding 和 GeoChat identify 框方向正确；
23. 多轮中所有 assistant 内容有监督；
24. user/image/padding token 均为 `-100`；
25. turn-aware truncation 不产生全 `-100` labels；
26. batch 内不同文本长度和视觉 token 数正确 collate；
27. Episode metadata 不进入 model kwargs。

完成后至少运行：

```text
python -m pytest -q tests/test_qwen3vl_phase2_data.py
python -m compileall -q scripts/qwen3vl_phase2_data.py
git diff --check
git status --short
```

## 16. 交给下一轮的接口

`scripts/finetune_qwen3vl_phase2.py` 只需要传入：

```text
Episode JSONL
image root 映射
processor
augmentation config
max sequence length
seed/epoch
```

它不应自己打开 annotation、变换框、渲染 prompt 或构造 assistant loss mask。
