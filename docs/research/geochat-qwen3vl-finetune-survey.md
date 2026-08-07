# GeoChat 官方规格 + Qwen3-VL 微调实践 + 第三阶段 RL 趋势调研

# GeoChat Official Spec + Qwen3-VL Fine-Tuning Practice + Stage-3 RL Trends Survey

**Date / 日期:** 2026-08-07

> 本文件为调研记录（research note），不改变任何代码行为。
> This file is a research note; it changes no code behavior.

## 1. GeoChat 数据集（官方）/ GeoChat Dataset (Official)

- **论文 / Paper:** Kuckreja et al., "GeoChat: Grounded Large Vision-Language Model for Remote
  Sensing", CVPR 2024, MBZUAI. arXiv:2311.15826
- **官方发布 / Official release:** HuggingFace `MBZUAI/GeoChat_Instruct`
- **规模 / Size:** 约 318k 指令对（本地副本为 308,861 条，存在版本差异，以官方为准）；
  完整数据约 102 GB；仅 train split。
- **许可 / License:** Apache 2.0
- **组成 / Composition:** LRBEN（二值场景属性 VQA，约 57k）、NWPU-RESISC45 场景分类
  （31.5k，闭集类别列表提问）、floodnet（约 4k）、场景级推理/对话（Google Earth 风格裁剪图，
  约 178k）、DOTA 系裁剪图的目标级 QA 与多轮对话（约 75k，P 前缀与 train_/valid_ 前缀）。
  多轮对话样本约 139k。`[grounding]` 前缀的详描指令约 18k。
  注意：官方 Instruct 文件中目标级回答以文本空间关系为主，坐标框格式需在转换时确认。
- **GeoChat 模型训练配方 / GeoChat model recipe:** LLaVA-1.5 + LoRA；CLIP ViT-L/14 输入提升到
  504×504；3× A100 40GB，约 25 小时；global batch 144，lr 2e-5，1 epoch，seq 2048，
  DeepSpeed ZeRO-3。对齐阶段沿用 LLaVA-1.5 的 558K projector 预训练。
- **评测基准 / Eval benchmark:** GeoChat-Bench（MBZUAI/GeoChat-Bench）。
- **来源 / Sources:**
  - https://arxiv.org/abs/2311.15826
  - https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct
  - https://github.com/mbzuai-oryx/geochat
  - https://mbzuai-oryx.github.io/GeoChat/

## 2. Qwen3-VL 微调实践 / Qwen3-VL Fine-Tuning Practice

### 2.1 架构要点 / Architecture notes

- ViT（depth 27, hidden 1152）→ merger（MLP PatchMerger）→ LLM；
  `deepstack_visual_indexes = [8, 16, 24]` 的中间层特征经三个 deepstack merger
  以残差方式注入 LLM 前几层。
- patch size 16，spatial merge 2 → 每个视觉 token 对应 32×32 像素；
  `num_tokens = (H/32) × (W/32)`。min/max pixels 必须为 32 的倍数
  （Qwen2-VL/2.5-VL 是 28，不可混用）。
- Interleaved-MRoPE；原生 256K 上下文。
- 来源：arXiv:2511.21631（Qwen3-VL Technical Report）、HF model_doc/qwen3_vl、qwen.ai blog。

### 2.2 社区共识配方 / Community consensus recipe

| 部件 / Component | 建议 / Recommendation |
|---|---|
| ViT | 冻结（默认）；仅域差极大时解冻 |
| merger / deepstack mergers | 默认冻结；对齐需要域适配时可训 |
| LLM | LoRA 或全参训（LoRA 常用 r=64/alpha=128, dropout 0.05，目标模块 q/k/v/o/gate/up/down） |
| 学习率 / LR | LoRA 1e-4；全参 5e-6..1e-5 |
| 其他 / Others | cosine + 3% warmup，bf16，gradient checkpointing，2-3 epochs |

- 显存 / Memory: Qwen3-VL-8B LoRA bf16 单卡约 30-37 GB（48G 卡可行）；QLoRA 4bit 约 15-20 GB。
- 多图对话 SFT 原生支持：每个 `<image>` 占位符必须与图片数严格一致；图片标签放句首防截断。
- 来源：
  - https://medium.com/@aminfadaeinejad.edu/fine-tuning-qwen3-vl-a-practical-guide-for-vision-language-model-adaptation-d66d3f61e888
  - https://kaitchup.substack.com/p/qwen3-vl-fine-tuning-on-your-computer

### 2.3 已知坑 / Known pitfalls

- LLaMA-Factory 曾未把 `deepstack_merger_list` 归入 `vision_model_keys`，导致 deepstack merger
  被误训/误冻（已在后续 commit 修复）；使用外部框架前必须核对本仓库固定的
  transformers==5.14.1 兼容性。
- LoRA + gradient checkpointing 必须调用 `enable_input_require_grads()`，否则梯度到不了 adapter。
- `remove_unused_columns` 必须为 False，防止图片字段被 Trainer 丢弃。
- `<|image_pad|>` 必须从 labels 中屏蔽（-100）。
- 多图/视频场景优先 Flash Attention 2。
- 训练 collator 用 `add_generation_prompt=False`。

## 3. 第三阶段 RL / 偏好微调趋势（截至 2026-08）/ Stage-3 RL & Preference Tuning Trends

### 3.1 遥感领域 GRPO/RLVR 先例 / RS-domain GRPO/RLVR precedents

- **MilChat**（arXiv:2505.07984）：Qwen2-VL-2B；72B 教师生成 caption → GPT-4o 改写 CoT →
  SFT → GRPO（关键词奖励 + 格式奖励）；8×A100 RL，lr 1e-6，每图 4 采样。
- **TinyRS-R1**（arXiv:2505.12099）：首批 2B 级 GRPO 对齐 CoT 遥感 VLM，在分类/VQA/grounding
  上追平或超过 7B 遥感模型。
- **GeoVLM-R1**（arXiv:2509.25026）：GRPO + 双目标奖励（格式 + 正确性），用于
  referring detection / region captioning / grounding description；代码 GeoVLM-R1-Toolkit。
- **GEO-R1**：少样本地理空间指代表达理解的后训练 RL。

### 3.2 通用 VLM RL 生态 / General VLM RL ecosystem

- Skywork-R1V 系列（R1V→R1V4，arXiv:2504.05599 / 2504.16656 / 2507.06167 / 2512.02395）、
  om-ai-lab/VLM-R1（arXiv:2504.07615）、R1-VL（ICCV 2025）、Reason-RFT（NeurIPS 2025）。
- 工具链：ms-swift GRPO 多模态训练文档、TRL GRPOTrainer、Unsloth Qwen3-VL GRPO notebook。

### 3.3 蒸馏自改进主流管线 / Dominant distillation/self-improvement pipeline

"大教师模型生成数据 → 学生 SFT → RL 精调"是主流模式（MilChat 为典型代表）。
自蒸馏方向：SDPO、Teacher-Guided Policy Optimization、OPSD 等。

## 4. 未验证项 / Unverified items

- GeoChat SOTA/SIOR/FAST 组件的确切来源与分项样本数（官方只给出 318k 总量）。
- "RemoteGRPO" 未检索到正式命名，可能为非正式称呼。
