# Qwen3.5-9B Visual Planner LoRA 微调

## 1. 训练目标

本训练只适配 visual planner，不训练答案生成 Agent，也不改变运行时评测定义。

```text
base: Qwen/Qwen3.5-9B（远端本地 checkpoint）
data: data/phase2-train-visualplanning-refined-v4/training/{train,val}.jsonl
method: multimodal supervised fine-tuning + LLM LoRA
vision encoder: frozen
language-model base: frozen
trainable: LLM LoRA A/B only
network: disabled by default
```

数据集当前规模：

| task | train |
|---|---:|
| caption | 804 |
| change_caption | 800 |
| change_qa | 800 |
| counting | 305 |
| fine_grained_counting | 800 |
| general_vqa | 2651 |
| grounding | 800 |
| multiple_choice_vqa | 800 |
| scene_classification | 419 |
| spatial_relation | 372 |
| 合计 | 8551 |

其中 6951 条是单图，1600 条是双图；split 与样本纳入规则完全沿用现有
`training/train.jsonl` 和 `training/val.jsonl`，训练脚本不会重新划分或过滤。
train 中 `region_request.explicit=true` 为 2962 条，validation 中为 312 条；ROI 坐标与
其他 VisualTaskPlan 字段一样作为 assistant JSON token 接受语言模型交叉熵监督。

## 2. 模型实际输入

单条样本按数据集顺序构造成：

```text
system:
  数据集 messages[0].content 中冻结的 visual-task-plan-v5 协议与 planner binding

user:
  image block 1
  [image block 2，仅 change 样本]
  数据集原始 question 文本

assistant generation prefix:
  Qwen3.5 chat template 在 enable_thinking=False 下生成的空 thinking 前缀
```

processor 产生并传给模型的 tensor 为：

```text
input_ids
attention_mask
mm_token_type_ids
pixel_values
image_grid_thw
labels
```

`pixel_values` 与 `image_grid_thw` 来自 Qwen3.5 checkpoint 自带的
`Qwen3VLProcessor`。脚本不自行 resize/crop 图像，不改变 `training_images/` 中已经确定的
planner preview，也不自定义 position IDs；Qwen3.5 Transformers forward 根据
`input_ids`、`image_grid_thw` 与 `mm_token_type_ids` 走其自身 position/rope 实现。

训练使用 teacher forcing，因此完整 assistant 目标也存在于 `input_ids` 后缀中；它在第
`t` 个位置的 token 由前面的图像、system、question 和目标前缀 token 预测。这不是把
Ground Truth 暴露给推理：推理时只提供相同的 system/user 与 generation prefix。

## 3. 模型监督输出

监督输出不重新生成、不归一化，也不从 `datasets/*.jsonl` 的 provenance 猜测，而是逐字使用：

```text
training/*.jsonl
  -> messages[2].content[0].text
```

该文本是一个 `visual-task-plan-v5` JSON，例如：

```json
{
  "count_target": null,
  "needs_visual_assistance": true,
  "object_categories": ["developed-space", "road", "water", "building"],
  "reason_codes": ["general_question"],
  "region_request": {
    "explicit": false,
    "image_index": null,
    "roi_xyxy": null
  },
  "task": "general_vqa",
  "version": "visual-task-plan-v5"
}
```

实际文件使用紧凑 JSON。训练目标还包含 chat template 的 `<|im_end|>` 结束 token；system、
user、视觉 token 和 generation prefix 的 label 全部为 `-100`，不参与 loss。

## 4. 不同任务的 loss

所有任务共享原模型的 causal language-modeling head，不增加 ROI regression head。
语言模型 loss 是 assistant 监督 token 的平均负对数似然：

```text
L_τ = - 1 / N_τ * Σ(i 属于 τ) Σ(t 属于 assistant_i)
      log pθ(y_i,t | images_i, system_i, question_i, y_i,<t)
```

其中 `N_τ` 是该任务 assistant 监督 token 总数。`roi_xyxy` 的字段名、标点和普通数字 token
均包含在同一个 `L_JSON` 中；不计算 L1、IoU 或 GIoU，也不增加额外模型分支。

各任务只是目标 JSON 的字段内容不同：

| 任务族 | 主要被监督的差异字段 | loss |
|---|---|---|
| counting / fine_grained_counting | `task`、精确 `count_target`、可执行类别与可选 ROI | assistant JSON token CE |
| grounding | `task`、视觉辅助类别、显式区域请求（若有） | assistant JSON token CE |
| change_caption / change_qa | 双图条件下的 `task`、辅助类别与区域请求 | assistant JSON token CE |
| caption | 空问题/单图对应的 `task` 及计划字段 | assistant JSON token CE |
| general_vqa / multiple_choice_vqa / scene_classification / spatial_relation | 对应 `task`、辅助类别、reason/region 字段 | assistant JSON token CE |

默认不做 inverse-frequency task weighting，也不重采样 `general_vqa`。因此每条原始训练记录
每 epoch 被访问一次，较多样本或较长 JSON 的任务对全局 loss 贡献更大。这是标准 SFT 的
token-mean 语义。若以后要引入 task weighting，
应作为独立实验并保存权重配置，不能把结果与当前基线混称。

## 5. LoRA target

Qwen3.5-9B 是 32 层 hybrid decoder：24 层 linear attention，8 层 full attention。脚本按
真实模块结构逐层枚举完整路径，而不是用模糊的短名称全局匹配。

```text
linear-attention layer:
  linear_attn.in_proj_qkv
  linear_attn.in_proj_z
  linear_attn.in_proj_a
  linear_attn.in_proj_b
  linear_attn.out_proj
  mlp.gate_proj
  mlp.up_proj
  mlp.down_proj

full-attention layer:
  self_attn.q_proj
  self_attn.k_proj
  self_attn.v_proj
  self_attn.o_proj
  mlp.gate_proj
  mlp.up_proj
  mlp.down_proj
```

远端 checkpoint 共枚举 248 个 LoRA target。启动时 `parameter_audit.json` 必须确认：

- 248 个 target 全部挂载；
- 所有 base 参数冻结；
- vision 全冻结；
- `lm_head` 和 embedding 冻结；
- 唯一 trainable 参数是上述语言模块的 LoRA A/B；不存在 auxiliary ROI head。

PEFT adapter config 使用由 248 个完整路径组成的正则字符串，而不是可被 PEFT 缩短成
`q_proj` 等叶子名的列表。这样保存后的 adapter 重新加载时仍只能命中语言塔，不能误命中
视觉塔中的同名 projection。PEFT `modules_to_save` 为空。

默认 `rank=32, alpha=64, dropout=0.05`。这是一张 48 GB GPU 上相对保守的 BF16 LoRA
起点，并非声称已经完成超参数最优性验证。

## 6. 远端运行

先做纯数据审计，不加载模型：

```bash
cd ~/M3
bash scripts/run_qwen35_9b_visual_planner_lora.sh --inspect-only
```

正式训练：

```bash
cd ~/M3
bash scripts/run_qwen35_9b_visual_planner_lora.sh
```

等价的显式命令：

```bash
conda run --no-capture-output -n m3 \
  python scripts/finetune_qwen35_9b_visual_planner_lora.py \
  --model-path models/Qwen3.5-9B \
  --base-model-id Qwen/Qwen3.5-9B \
  --dataset-root data/phase2-train-visualplanning-refined-v4 \
  --output-dir outputs/finetune/qwen35-9b-visual-planner-lora \
  --local-files-only \
  --torch-dtype bfloat16 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --epochs 2
```

单步真实 smoke run：

```bash
bash scripts/run_qwen35_9b_visual_planner_lora.sh \
  --output-dir outputs/finetune/qwen35-9b-smoke \
  --max-train-samples 1 \
  --max-eval-samples 1 \
  --max-steps 1 \
  --gradient-accumulation-steps 1 \
  --eval-steps 1 \
  --save-steps 1
```

继续最近 checkpoint：

```bash
bash scripts/run_qwen35_9b_visual_planner_lora.sh --resume-from-checkpoint auto
```

## 7. 产物

```text
output_dir/
├── parameter_audit.json
├── qwen35_visual_planner_training_manifest.json
├── checkpoint-N/                  # Trainer/PEFT adapter + optimizer/scheduler state
└── final_adapter/                 # 最终 LoRA adapter + processor/tokenizer
```

manifest 保存数据 checksum、逻辑 base model identity、LoRA 完整 target、训练参数、依赖版本
和最终 step/loss。实际本机 checkpoint 绝对路径不作为逻辑模型身份写入 manifest。
resume 只接受同一 output directory 下完整写成的 `checkpoint-N`，并逐项比较 base config
checksum、数据 checksum/选择规模、LoRA 与所有影响权重的优化参数；冲突稳定拒绝，不使用
当前默认值猜测旧请求。`status=completed` 的 run 不再 resume；该入口只恢复中断的
`status=running` 训练。

## 8. 环境与已知限制

远端核对环境为 Python 3.11、PyTorch 2.13.0+cu130、Transformers 5.14.1、PEFT 0.18.1，
模型加载入口为 `AutoModelForImageTextToText`。当前环境未安装
`flash-linear-attention`/`causal-conv1d`，Transformers 会使用 PyTorch fallback；这不改变
监督语义，但可能显著降低 hybrid linear-attention 层的训练速度。脚本不会自动联网安装
依赖，也不会自动下载模型。

`max_seq_length=6144` 超限时稳定失败，不从 assistant 尾部截断；应先通过 preflight 查明
样本，再明确提高长度或调整实验参数，不能静默丢掉目标 token。
默认 preflight 从每个 task 选择首条记录，十条覆盖全部任务族及单双图 processor 路径。

当前生产 runtime 仍以生成的 JSON 为计划权威；本任务只实现训练与 adapter 持久化，不改变
runtime 模型接口，也不存在需要额外接入的 ROI head 推理路径。
