# Qwen3-VL-8B Phase 2 微调脚本实现任务

## 1. 任务目标

实现：

```text
scripts/finetune_qwen3vl_phase2.py
tests/test_finetune_qwen3vl_phase2.py
```

训练参数策略固定为：

```text
Qwen3-VL-8B
├── Vision Encoder：冻结
├── visual.merger：全部参数解冻
├── visual.deepstack_merger_list.*：全部参数解冻
└── LLM：原始参数冻结，在 Attention + MLP projection 上挂 LoRA
```

Merger 与 LLM LoRA 使用不同学习率。训练脚本消费
`scripts/qwen3vl_phase2_data.py` 提供的 Dataset/Collator，不复制数据语义、增强或 prompt
逻辑。

训练唯一产物是可 resume 的复合 checkpoint；完整部署模型由第四轮独立 exporter 生成。

## 2. 开始实现前必须确认

1. 阅读 `AGENTS.md`、`DETAILS.md` 和 `docs/train/01_*`、`02_*`；
2. 执行 `git status --short`、`git rev-parse HEAD`；
3. 阅读历史参考脚本：

   ```text
   git show 157af713e2c5947214f5ce3ed04b414e62872e39:scripts/finetune_qwen3vl_merger_lora.py
   ```

   它只能作为行为参考，不能从旧分支 import；旧脚本是 merger-only LoRA，与本任务的
   “全量 Merger + LLM LoRA”不同；
4. 检查当前本地 Qwen3-VL 配置和真实模块路径；
5. 确认新脚本和测试路径已经过架构白名单批准；
6. 默认离线，不触发 Hugging Face 自动下载。

## 3. CLI 参数分组

建议用 `HfArgumentParser` dataclasses 或等价清晰结构，至少分为：

```text
ModelArguments
DataArguments
LoRAArguments
OptimizationArguments / TrainingArguments
CheckpointArguments
```

至少支持：

```text
--model-id
--merger-lora-adapter             # 可选：phase1 merger LoRA adapter 目录，
                                  # 训练前合并进 base 权重（默认启动配置使用）
--local-files-only                 # 默认 true
--torch-dtype                      # 默认 bfloat16
--attn-implementation
--train-file
--eval-file
--image-root source=path           # 可重复
--max-seq-length
--image-min-pixels
--image-max-pixels
--augmentation-*                   # 几何增强和恶劣成像质量模拟，交给第二文件
--lora-rank                        # 建议初始默认 64
--lora-alpha                       # 建议初始默认 128
--lora-dropout                     # 建议初始默认 0.05
--lora-lr                          # 建议初始默认 1e-4
--merger-lr                        # 建议初始默认 1e-5
--weight-decay
--output-dir
--resume-from-checkpoint
--max-train-samples                # smoke test
--max-eval-samples                 # smoke test
```

GPU 数量、per-device batch、gradient accumulation、DeepSpeed/FSDP 配置不得硬编码个人
机器参数。

## 4. 模型加载

要求：

- `import scripts.finetune_qwen3vl_phase2` 不加载权重；
- torch/transformers/peft 的重依赖尽可能在执行路径惰性导入；
- 使用 `AutoConfig` 先验证 `model_type == "qwen3_vl"`；
- 使用 `AutoModelForImageTextToText` 和同一 checkpoint 的 `AutoProcessor`；
- 默认 `local_files_only=True`；
- 验证当前 Transformers 版本具有 Qwen3-VL deepstack fusion 路径；
- processor/tokenizer 不单独选择其他模型；
- 不在每个 Dataset worker 或样本中重复构造模型/processor。

## 5. 参数冻结与 LoRA 注入

### 5.1 总体顺序

```text
加载原始模型
-> 全部 requires_grad=False
-> 精确识别语言模型 Attention/MLP Linear
-> 给这些语言模块挂 LoRA
-> 全量解冻主 Merger 与 DeepStack Mergers 的 base 参数
-> 执行参数审计
```

### 5.2 Vision Encoder

除下面明确列出的 merger 外，整个 Vision Encoder 保持冻结。

不得通过模糊字符串 `"visual" not in name` 猜测。应从真实模型结构定位语言模型根和
视觉根，并用单元测试固定允许集合。

### 5.3 全量 Merger

需要全量训练：

```text
visual.merger
visual.deepstack_merger_list.*
```

“全量”包含这些模块中实际存在的：

```text
Linear weight/bias
LayerNorm weight/bias
其他属于 merger 子树的浮点参数
```

Merger 不挂 LoRA。注入 PEFT 后应通过 unwrap/base-model 逻辑重新定位并解冻 merger
base 参数，避免 PEFT 包装改变名称导致遗漏。

### 5.4 LLM LoRA

目标语义是每个语言层的：

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

必须枚举完整模块路径并断言它们都位于语言模型子树。显式排除：

```text
visual.*
*.merger.*
lm_head
token embeddings
```

不要只把 `target_modules=["q_proj", ...]` 交给 PEFT 后假设结果正确；必须在注入前后审计
完整命中列表。LoRA bias 初始使用 `none`，除非用户后续明确改变。

### 5.5 启动时硬断言

训练开始前生成并保存参数审计：

```text
LoRA target module 全路径
trainable LoRA parameter 全路径/shape/dtype/count
trainable merger parameter 全路径/shape/dtype/count
frozen vision parameter count
frozen LLM base parameter count
total/trainable parameter count
```

必须断言：

- 至少找到主 merger 和配置声明数量的 deepstack mergers；
- 每个预期 LLM layer 的七个 projection 都命中；
- 没有视觉 projection 被挂 LoRA；
- 没有 merger LoRA 参数；
- 没有非 merger Vision Encoder 参数可训练；
- 没有 LLM base 参数可训练；
- 所有 `requires_grad=True` 参数被精确分类为 `merger_base` 或 `llm_lora`；
- 两类集合不重叠且没有未分类参数。

## 6. 双学习率 optimizer

不能依赖 Trainer 默认的单学习率 optimizer。显式创建四组：

```text
1. merger_base + decay
   lr = merger_lr
   weight_decay = configured weight_decay

2. merger_base + no_decay
   lr = merger_lr
   weight_decay = 0

3. llm_lora + decay
   lr = lora_lr
   weight_decay = configured weight_decay

4. llm_lora + no_decay
   lr = lora_lr
   weight_decay = 0
```

no-decay 至少覆盖 bias 和 normalization 参数；LoRA A/B matrix 是否 decay 必须采用一种
明确、测试固定的策略，不能靠名字碰巧分类。

每个 trainable parameter 必须且只能进入一个组。启动时保存组名、参数量、初始 LR 和
weight decay，但不要在公开日志打印巨大完整 state dict。

scheduler 建议：

```text
cosine
warmup_ratio = 0.03
```

scheduler 应按每个参数组各自初始 LR 做相同比例缩放，不能把两套 LR 归一成一个值。

## 7. 建议训练默认值

这些是可覆盖的初始实验值，不属于不可变数据契约：

```text
dtype:                    bfloat16
LoRA rank:                64
LoRA alpha:               128
LoRA dropout:             0.05
LLM LoRA LR:              1e-4
Merger LR:                1e-5
epochs:                   2
scheduler:                cosine
warmup ratio:             0.03
gradient checkpointing:   true
gradient clipping:        1.0
```

如使用 gradient checkpointing，必须调用适合当前 PEFT/Qwen3-VL 版本的 input-gradient
启用方法，并用反向传播 smoke test 验证 LoRA 和 merger 都实际获得非零梯度。

不要默认启用 4-bit QLoRA：本方案是 bf16 base + LLM LoRA + 全量 merger。量化训练会改变
参数和导出契约，应另行讨论。

## 8. 数据遍历与采样

数据来自：

```text
VRSBench Grounding
VRSBench VQA 有框主视图
40% VRSBench VQA 自主注意力额外视图
VRSBench 天然无框 VQA
GeoChat 全量合法对话（refer/identify/普通/多轮）
```

初版默认不得丢弃或下采样某一整类数据。一个 epoch 至少遍历导出的全部 train Episode
一次。可以通过显式、确定性的 group repeat weight 重复少数类，但：

- 默认 repeat weight 为 1；
- 配置和实际 group 数量进入 manifest；
- 不使用有放回随机采样导致部分 GeoChat 样本一个 epoch 内完全看不到；
- distributed sampler 需要 `set_epoch`；
- Dataset 的 augmentation epoch 也必须同步更新。

如果后续需要按 token 或 task 平衡，应作为可审计的独立实验配置，不修改源 Episode。

## 9. 复合 checkpoint 格式

PEFT 默认 adapter 保存不能替代全量 merger 保存。每个可恢复 checkpoint 至少包含：

```text
checkpoint-N/
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── merger_model.safetensors
├── processor/
├── phase2_training_manifest.json
├── trainer_state.json
├── optimizer.pt
├── scheduler.pt
└── rng_state...
```

### 9.1 Adapter

只保存 LLM LoRA adapter。保存后检查 adapter keys 不包含视觉或 merger LoRA。

### 9.2 Merger state

`merger_model.safetensors` 保存主 merger 和所有 deepstack merger 的完整 state：

- 使用 unwrap 后 base model 的稳定逻辑 key；
- 包含参数和必要 persistent buffer；
- manifest 保存 key、shape、dtype 和文件 sha256；
- 保存后读回并核对；
- 禁止用 pickle 作为唯一长期权重格式。

### 9.3 Training manifest

至少记录：

```text
schema version
base model logical identity/revision
processor identity
train/eval Episode 文件 checksum
上游数据 manifest checksum
LoRA config 和完整 target module list
merger module/parameter list
optimizer group 摘要及两套 LR
augmentation config 和 seed
训练参数
git HEAD
transformers/torch/peft 版本
checkpoint step/epoch
```

本地 checkpoint 绝对路径不得作为可持久化逻辑身份。

## 10. Resume 契约

resume 顺序：

```text
读取 checkpoint manifest
-> 与当前显式请求做兼容性校验
-> 加载同一 base model
-> 构造完全相同 LoRA target/config
-> 加载 adapter
-> 严格加载 merger state
-> 构造四 optimizer groups
-> 恢复 optimizer/scheduler/trainer/RNG
-> 恢复 sampler 和 augmentation epoch
-> 继续训练
```

以下冲突必须稳定拒绝，不得猜测继续：

```text
base model logical identity/revision
processor identity
train data checksum
LoRA rank/alpha/target set
merger parameter set/shape
optimizer group topology
augmentation seed/config
max sequence/image pixel settings
```

新的 `output_dir` 已有不兼容 checkpoint 时不能覆盖。

## 11. Trainer 集成

可以子类化 `transformers.Trainer` 或使用明确 callback，但必须保证：

- 自定义 optimizer 在 scheduler 前创建；
- checkpoint save 同时保存 adapter 和 merger；
- checkpoint rotation 不留下“只有 adapter、没有 merger”的可见成功目录；
- rank 0 负责落盘，其他 rank 正确同步；
- `remove_unused_columns=False`；
- eval 关闭增强；
- 训练中断后的最后完整 checkpoint 可恢复；
- 保存失败使用稳定错误状态，不伪造完成标记。

## 12. 测试与验收

单元测试构造微型 fake Qwen 模型树和 fake PEFT seam，不加载 8B 权重。至少覆盖：

1. import 脚本不触发权重加载；
2. 只给语言 Attention/MLP projection 挂 LoRA；
3. 视觉同名 projection 不被命中；
4. 主 merger 和全部 deepstack merger base 参数可训练；
5. Vision Encoder 其余参数冻结；
6. LLM base、embedding、lm_head 冻结；
7. trainable 参数分类闭合；
8. 四个 optimizer group 无重复、无遗漏；
9. merger LR 与 LoRA LR 不同且 scheduler 保持比例；
10. checkpoint 同时包含 adapter、merger、manifest 和 trainer state；
11. merger save/read-back key、shape、dtype 一致；
12. resume 对不兼容 data checksum、LoRA config、merger topology 稳定失败；
13. resume 恢复 epoch 后 augmentation seed 不漂移；
14. 一次小型 forward/backward 后 LoRA 与 merger 均有梯度，冻结参数无梯度；
15. 默认 `local_files_only=True`。

完成后至少运行：

```text
python -m pytest -q tests/test_finetune_qwen3vl_phase2.py
python -m pytest -q tests/test_qwen3vl_phase2_data.py
python -m compileall -q scripts/finetune_qwen3vl_phase2.py
git diff --check
git status --short
```

真实 Qwen3-VL-8B、真实数据和目标 GPU smoke test 必须单独报告；不能用 fake-model 单测替代。

## 13. 交给 exporter 的接口

第四轮只依赖完整 checkpoint 中：

```text
adapter/
merger_model.safetensors
processor/
phase2_training_manifest.json
```

Exporter 不依赖 Trainer Python 对象，不重新训练，也不从当前 CLI 默认值猜训练时配置。
