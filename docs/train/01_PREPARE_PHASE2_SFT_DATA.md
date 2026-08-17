# Phase 2 训练数据准备器实现任务

## 1. 任务目标

实现：

```text
scripts/prepare_qwen3vl_phase2_sft.py
tests/test_prepare_qwen3vl_phase2_sft.py
```

该脚本只读解析：

```text
data/phase2-train/VRSBench/
data/phase2-train/GeoChat/GeoChat_Instruct.json
```

并导出与 Transformers、PEFT 和具体 Qwen chat template 解耦的 canonical
training episodes。后续数据集、在线增强和训练脚本只消费这里冻结的 Episode
契约，不重新解释原始数据集。

本任务只处理数据语义和确定性导出，不加载模型、不调用网络、不做图像增强，
也不覆盖原始标注。

## 2. 开始实现前必须确认

编码代理开始前必须：

1. 阅读根目录 `AGENTS.md` 和 `DETAILS.md`；
2. 执行 `git status --short` 和 `git rev-parse HEAD`；
3. 阅读 `data/phase2-train/VRSBench/README.md`、`manifest.json` 和少量真实记录；
4. 抽样检查 GeoChat 的普通对话、`[refer]`、`[identify]` 和多轮记录；
5. 确认目标 Python 路径已经过架构白名单批准；如果尚未批准，停止创建 Python
   文件，先提交独立的 allowlist 变更请求；
6. 保留工作树中已有的未跟踪文件和用户修改，不顺带修改它们。

现有 `docs/training/VQA_STAGE_DATA_REQUIREMENTS.md` 包含旧的探索性口径，其中
“VQA 输出证据框”的描述不适用于本实现。当前冻结协议是：

```text
Grounding：图像 + 指代描述 -> 框
有框 VQA：图像 + 问题 + annotation context -> 答案
无框 VQA：图像 + 相同问题 -> 相同答案
```

VQA assistant 不需要先输出证据框。

## 3. 输入事实

### 3.1 VRSBench

当前目录包含：

```text
VRSBench_train.jsonl
VRSBench_val.jsonl
VRSBench_test_raw.jsonl
manifest.json
README.md
```

train/val 是图片级记录，每张图包含：

```text
objects[]
qa_pairs[]
```

当前数据规模约为：

```text
train images:       18,237
validation images:   2,027
grounding objects:  36,313
VQA pairs:          85,813
```

`box_999` 是已经派生的 `0..999 xyxy` 训练视图；`box_valid=false` 的原始框不得
在本脚本中钳制或修复。

关键限制：VRSBench 的 `qa_pairs` 没有显式 `ques_id -> obj_id` 绑定，而且
`objects` 不保证完整覆盖某个问题涉及的全部实例。因此本任务不得使用模糊文本匹配
伪造问题级 evidence binding，也不得声称提供的框是问题的完整 Ground Truth。

### 3.2 GeoChat

`GeoChat_Instruct.json` 是 JSON array，本地副本约 308,861 条记录，包含普通
单轮/多轮对话、`[refer]` 和 `[identify]`。

典型 `[refer]`：

```text
user:      [refer] where is <p>silver airplane</p>?
assistant: {<16><55><24><63>|<11>}
```

典型 `[identify]`：

```text
user:      [identify] {<51><23><63><35>|<15>}
assistant: <p>1 boeing737 airplane at the top</p>
```

GeoChat 框使用 `0..100` 整数坐标及私有 class id；训练视图统一转换为
`0..999 xyxy`。class id 保存在 provenance/audit 字段，不作为不透明的模型输出目标。

## 4. 输出文件

CLI 至少接收显式输入和输出路径，建议输出：

```text
<output_dir>/
├── train.jsonl
├── validation.jsonl
├── manifest.json
└── rejected.jsonl
```

输出目录不能默认写入源数据目录。所有输出路径必须是可配置的，不得硬编码个人机器
绝对路径。

`rejected.jsonl` 只记录稳定错误码和必要的源记录身份，不复制包含大量原始数据的完整
异常对象，也不写原始异常全文。

## 5. Canonical Episode 契约

建议使用以下 JSON-safe 结构；实现可以用 dataclass/Pydantic 或显式校验函数，但输出
字段语义必须稳定：

```json
{
  "schema_version": 1,
  "episode_id": "vrsbench/train/P1419_0005.png/qa/2/box_assisted",
  "parent_episode_id": "vrsbench/train/P1419_0005.png/qa/2",
  "dataset": "VRSBench",
  "split": "train",
  "image_source": "vrsbench",
  "image": "Images_train/P1419_0005.png",
  "task_kind": "vqa_box_assisted",
  "source_task": "counting",
  "turns": [
    {
      "user_text": "How many small vehicles are visible?",
      "assistant_text": "1",
      "input_boxes": [
        {
          "xyxy_999": [380, 200, 450, 260],
          "label": "vehicle",
          "description": "The small vehicle with a white color...",
          "source_object_id": 0
        }
      ],
      "target_boxes": []
    }
  ],
  "augmentation_policy": {
    "geometry": "orientation_locked",
    "reason": "spatial_language"
  },
  "provenance": {
    "source_record_id": "vrsbench/train/P1419_0005.png",
    "question_id": 2,
    "view": "box_assisted"
  }
}
```

要求：

- `episode_id` 全局唯一且跨机器稳定；
- 成对有框/无框 VQA 共享 `parent_episode_id`；
- 图片路径始终相对于对应 dataset root；
- 不输出 `Path`、PIL image、bytes、NaN/Infinity 或任意不可 JSON 序列化对象；
- 原始路径、原始 annotation 和原始回答保持只读；
- 框暂时保持结构化字段，不提前渲染进最终 prompt，以便在线增强同步更新坐标。

允许的核心 `task_kind`：

```text
vrsbench_grounding
vqa_box_assisted
vqa_self_attention
vqa_naturally_unboxed
geochat_refer
geochat_identify
geochat_conversation
```

## 6. VRSBench 转换规则

### 6.1 Grounding

每个 `box_valid=true` 且 `box_999` 合法的 object 生成一条独立 Episode：

```text
image + referring_sentence -> target_boxes
```

规则：

- `input_boxes=[]`；
- `target_boxes` 包含对应 object 的框；
- 保留 `obj_id`、`obj_cls` 和原始 referring sentence；
- 非法框只计入审计，不修复、不钳制、不生成训练目标；
- 不改变源标注或 `manifest.json`。

### 6.2 VQA 有框主视图

当前数据只能可靠提供“同图 annotation context”，不能可靠提供问题级 object
binding。因此初版采用保守且确定性的规则：

1. 收集该图片所有 `box_valid=true` 的 objects；
2. 至少存在一个合法框时，每个 QA 都生成一份 `vqa_box_assisted`；
3. `input_boxes` 包含该图所有合法 annotation boxes；
4. prompt 渲染时必须称为 `Available annotated regions`，不得称为
   `all relevant objects`、`ground-truth evidence` 或其他暗示完整问题绑定的名称；
5. 不通过 question/answer 字符串匹配筛框；
6. 保存每个框的类别、描述和 source object id 供模型使用及审计。

如果未来要实现 question-specific binding，应作为独立的数据规则变更，并带有人工验证
或明确的源标注关系；不能在本任务中临时发明启发式规则。

### 6.3 40% 自主注意力增广

40% 是增广比例，不是将有框集合二选一划分：

```text
所有可生成有框视图的 VQA：100% 保留有框版
上述 parent 中稳定选择 40%：额外生成一份无框版
```

例如有 `N` 个有框 parent，最终生成：

```text
N 个 vqa_box_assisted
约 0.4N 个 vqa_self_attention
```

无框副本规则：

- question 和 answer 与 parent 完全相同；
- `turns[].input_boxes=[]`；
- provenance/audit 中仍可保存原框摘要，但这些坐标不得进入 user prompt；
- `parent_episode_id` 与有框版相同；
- `episode_id` 使用 `/self_attention` 后缀；
- 只对 train split 生成该增广；validation 不做随机/复制增广；
- 按 `source_task` 分层；
- 每层用 `sha256(seed + parent_episode_id)` 排序；
- 每层取前 `round(0.4 * N)` 条；
- seed 由 CLI 显式配置并进入 manifest；
- 不使用 Python `hash()` 或运行时随机抽样。

### 6.4 天然无框 VQA

图片没有任何合法 annotation box 时，每个 QA 仍生成一份
`vqa_naturally_unboxed`：

```text
image + question -> answer
```

它们不进入 40% 分母，且不得伪造空框为“已绑定的负样本”。

### 6.5 一图一问题

VRSBench 每个 QA 是独立 Episode。不要将同图多个 QA 组合为一条多轮记录，否则前一问
提供的框会泄漏到本应无框的后续问题。

## 7. GeoChat 全量使用规则

“全量使用”表示所有结构和坐标合法的记录都进入训练，而不是只保留 `[refer]`。

### 7.1 `[refer]`

语义：

```text
image + referring expression -> target_boxes
```

要求：

- 从 assistant 文本解析一个或多个框；
- 将每个坐标 `c` 确定性转换为 `round(c * 999 / 100)`；
- 转换前后验证 `x1 < x2`、`y1 < y2` 以及范围；
- assistant 最终由第二个文件按统一 JSON box 协议重新渲染；
- GeoChat 私有 class id 仅保存在 provenance 中。

### 7.2 `[identify]`

语义：

```text
image + input_boxes -> region description
```

它不是 Grounding，但能训练模型利用给定区域识别目标，因此必须保留。解析 user 中的框到
`input_boxes`，assistant description 保持为文本答案。

### 7.3 普通 VQA、分类、推理和多轮对话

- 保留原始 user/assistant turn 顺序；
- 一条源记录对应一条 Episode；
- 不再把同一多轮记录全量拆成重复单轮副本；
- 如果普通对话中出现可解析坐标，也必须结构化保存，不能让旧坐标协议直接漏进最终 prompt；
- 每个 assistant turn 后续都参与 loss，但具体 token mask 由第二个文件实现。

### 7.4 允许拒绝的客观条件

只有以下情况可以拒绝：

```text
missing_image_field
invalid_conversation_type
invalid_role_order
missing_turn_text
image_token_count_mismatch
unparseable_box
non_finite_box
out_of_range_box
degenerate_box
```

所有拒绝必须进入 `rejected.jsonl` 和 manifest 统计，不能静默过滤。

本脚本只验证路径字段和对话/坐标结构；是否实际读取图像由第二个文件负责。

## 8. 增强策略元数据

第一文件不执行增强，但应为第二文件提供保守策略：

```text
geometry=orientation_locked
geometry=geometry_safe
```

该策略只控制旋转、仿射和透视等几何增强。恶劣成像质量模拟是独立的、坐标保持不变的
退化管线，不改变框或方向语义，因此 `orientation_locked` Episode 仍可按第二文件的配置
执行亮度减弱、模糊、噪声、低对比度、JPEG 压缩和暗角等成像退化。

以下情况至少应标记为 `orientation_locked`：

- VRSBench 的 position/direction/spatial 类型；
- user 或 assistant 中包含 top/bottom/left/right 及组合方向；
- north/south/east/west 等地理方向；
- relative position、left-most、top-most 等关系描述；
- 任何无法确定几何变换后文本语义仍成立的对话。

初版宁可少做几何增强，也不能让图像、框和文本空间语义互相矛盾。

## 9. Manifest 契约

`manifest.json` 至少包含：

```text
schema_version
生成参数与 seed
输入文件及 sha256
输出文件及 sha256
dataset × split × task_kind 数量
VRSBench 每层 40% 分母、选择数和实际比例
GeoChat refer/identify/conversation/多轮数量
坐标转换规则和版本
VRSBench 无效框统计
GeoChat 拒绝原因统计
总输入、输出、拒绝数量闭合检查
```

不得写机器绝对路径。

## 10. CLI 和实现边界

CLI 至少支持：

```text
--vrsbench-dir
--geochat-file
--output-dir
--self-attention-ratio  # 默认 0.40
--seed
```

默认离线。不得增加第三方依赖；优先标准库流式处理，避免把 256 MB GeoChat 和全部
派生 Episode 同时复制多份到内存。输出使用临时文件后原子替换。

不要：

- 创建 `utils.py`/`helpers.py`；
- 修改原始数据；
- 下载图片或数据集；
- 调用模型判断任务或绑定框；
- 改写 question/answer；
- 把数据准备逻辑放进训练脚本。

## 11. 测试与验收

单元测试必须使用小型临时 fixture，不读取完整数据集。至少覆盖：

1. VRSBench pretty-printed JSON blocks 和标准单行 JSONL；
2. GeoChat JSON array；
3. VRSBench Grounding 展开；
4. 有合法图像级框时所有 QA 都有有框版；
5. 40% 是额外副本，不替代有框版；
6. 40% 只作用于 train，按 task 分层且可复现；
7. 天然无框 QA 不进入 40% 分母；
8. GeoChat `[refer]` 多框解析与 `0..100 -> 0..999` 转换；
9. GeoChat `[identify]` 输入框解析；
10. 普通多轮角色和文本保持；
11. 非法框和错误对话写入 rejected 并闭合计数；
12. 输出没有绝对路径，严格 JSON-safe；
13. 相同输入、seed 和参数生成字节级稳定输出；
14. 原始输入文件 checksum 在运行前后不变。

完成后至少运行：

```text
python -m pytest -q tests/test_prepare_qwen3vl_phase2_sft.py
python -m compileall -q scripts/prepare_qwen3vl_phase2_sft.py
git diff --check
git status --short
```

如果本轮批准或修改了 Python 路径，还要运行 AGENTS.md 列出的架构测试。

## 12. 交给下一轮的接口

下一轮 `scripts/qwen3vl_phase2_data.py` 只能依赖：

```text
train.jsonl / validation.jsonl
manifest.json
显式 image_source -> dataset root 映射
```

它不应再次读取 VRSBench 或 GeoChat 的原始 annotation，也不应重新执行 40% 选择。
