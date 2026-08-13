# Phase 2 完整模型导出器实现任务

## 1. 任务目标

实现独立导出器：

```text
scripts/export_qwen3vl_phase2_checkpoint.py
tests/test_export_qwen3vl_phase2_checkpoint.py
```

它将第三轮产生的复合训练 checkpoint：

```text
base model
+ LLM LoRA adapter
+ 全量训练后的主 Merger/DeepStack Merger state
```

导出为可由 `AutoModelForImageTextToText.from_pretrained()` 和项目 Qwen3-VL 主流程直接
加载的完整 checkpoint。

Exporter 只恢复、合并、保存和验证模型；不读取训练集、不执行训练、不重新解释数据配置。

## 2. 开始实现前必须确认

1. 阅读 `AGENTS.md`、`DETAILS.md` 和 `docs/train/03_FINETUNE_QWEN3VL_PHASE2.md`；
2. 执行 `git status --short`、`git rev-parse HEAD`；
3. 阅读现有 `scripts/merge_qwen3vl_merger_lora.py`，只复用已经稳定且适用的加载、processor
   和辅助文件思路，不把它直接当作当前复合 checkpoint exporter；
4. 检查第三轮真实 checkpoint manifest 和权重 key，不靠本文示例猜名称；
5. 确认新脚本与测试路径已经过架构白名单批准；
6. 默认离线，不触发模型下载。

用户已经明确：导出器必须独立实现，不能隐含在训练结束回调中。

## 3. CLI

至少支持：

```text
--model-id             # 原始 Qwen3-VL-8B base checkpoint
--checkpoint-path      # Phase 2 复合 checkpoint
--output-path          # 完整导出目录
--torch-dtype          # 默认 bfloat16，可选 cpu float32 等
--device               # 默认 cpu，允许显式 cuda:0
--local-files-only     # 默认 true
--verify-forward       # 可选真实最小前向验证
```

`output_path` 已存在时必须拒绝，不能覆盖或与旧文件混合。

## 4. 输入 checkpoint 契约

至少要求：

```text
checkpoint-path/
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── merger_model.safetensors
├── processor/
└── phase2_training_manifest.json
```

在加载任何大权重前先验证：

- 文件存在且类型正确；
- manifest schema version 受支持；
- adapter、merger 文件 sha256 与 manifest 一致；
- base logical identity/revision 与 CLI 指向的 checkpoint metadata 一致；
- adapter config 是预期 LoRA 类型；
- adapter target modules 只属于 LLM；
- manifest 声明主 merger 和全部 deepstack merger；
- manifest 不包含危险绝对 result path 或 secret。

本地 base 路径只是定位方式，不能替代 manifest 中跨机器稳定的 logical identity。

## 5. 固定导出顺序

实现顺序必须是：

```text
1. 读取并验证 training manifest
2. 加载原始 base model
3. 从 base model 枚举预期 merger state keys
4. strict 加载 merger_model.safetensors
5. 将 LLM LoRA adapter 挂到已更新 merger 的 base model
6. 验证 PEFT adapter keys/targets
7. merge_and_unload LLM LoRA
8. 再次审计最终模型参数和配置
9. 保存完整模型和 processor
10. 复制 save_pretrained 未产出的必要辅助配置
11. 从导出目录重新离线加载
12. 完成结构/checksum/可选 forward 验证
13. 原子发布最终目录
```

不能先 merge 到错误 base 后再尝试覆盖 merger，也不能遗漏 deepstack merger。

## 6. Merger strict load

加载前从真实 base model 获取主 merger和所有 deepstack merger 的完整预期 state key、shape、
dtype。与 checkpoint manifest 和 safetensors 内容三方比较：

```text
missing key      -> fail
unexpected key   -> fail
shape mismatch   -> fail
非安全 dtype     -> fail
merger 数量不符  -> fail
```

允许按目标 dtype 做显式、记录在 manifest 中的浮点转换，但不能忽略 shape/key 冲突。

加载后逐 key 验证 base model 中的 tensor 与源 merger state 一致，再挂 LoRA。

## 7. LoRA merge

通过 PEFT 官方加载接口挂 adapter，然后：

- 验证 adapter 声明的 base identity 与 training manifest 一致；
- 验证 target module set 与 training manifest 完全一致；
- 验证所有 adapter tensor 都被消费；
- 禁止出现视觉或 merger LoRA target；
- 调用 `merge_and_unload()`；
- 最终 state dict 不得残留 `lora_A`、`lora_B` 或 PEFT wrapper-only keys；
- 最终模型仍包含已经加载的训练后 merger 参数。

## 8. 输出目录内容

完整导出目录至少应包含：

```text
config.json
model*.safetensors
model.safetensors.index.json       # 分片时
generation_config.json             # base 中存在时
processor/tokenizer files
chat_template.json                 # 当前模型需要时
preprocessor_config.json
video_preprocessor_config.json     # base 中存在且适用时
phase2_export_manifest.json
```

使用 `save_pretrained(..., safe_serialization=True)`。辅助文件仅在标准 save 方法没有产出时
从经过验证的 base/processor 来源复制；不得覆盖已经生成的新文件。

不要把 adapter、optimizer、scheduler 或训练数据复制进完整部署模型目录。

## 9. 原子和不可覆盖写入

为避免中断留下貌似完整的 checkpoint：

1. 要求最终 `output_path` 不存在；
2. 在同一父目录创建名称明确的临时目录；
3. 所有保存和 reload 验证都在临时目录完成；
4. 验证成功后用同文件系统原子 rename 发布；
5. 失败时不创建最终目录；
6. 临时目录清理由代码安全处理，不能删除宽泛目录或未解析路径。

禁止使用 `rm -rf` 处理用户给定路径。不得覆盖已有导出结果。

## 10. Export manifest

`phase2_export_manifest.json` 至少记录：

```text
schema version
base logical identity/revision
source training checkpoint logical id
source training manifest sha256
adapter sha256
merger sha256
LoRA config 和 target module set
merger module/key 摘要
LoRA merged=true
输出 dtype
transformers/torch/peft 版本
导出文件列表、size 和 sha256
reload validation 结果
optional forward validation 结果
git HEAD
```

不得记录：

```text
API key
环境变量完整 dump
机器绝对路径作为 logical identity
原始异常全文
图片 Base64
```

## 11. 验证门禁

### 11.1 必须执行的离线 reload 验证

从临时导出目录以 `local_files_only=True` 重新加载：

```text
AutoConfig
AutoProcessor
AutoModelForImageTextToText
```

验证：

- `model_type == qwen3_vl`；
- config 中关键视觉/语言结构与 base 一致；
- 主 merger 和 deepstack merger 数量正确；
- 不存在 PEFT/LoRA 残留模块；
- processor 能渲染最小 image+text chat template；
- 权重索引引用的所有 shard 都存在；
- 所有输出文件 checksum 可读回。

### 11.2 可选真实前向验证

`--verify-forward` 可以使用程序生成的小型 RGB 图像和固定短 prompt 做一次无梯度 forward
或极短 generation：

- 不读取训练图片；
- 固定 seed；
- 不联网；
- 只验证有限 logits/output 和接口可用，不宣称模型质量；
- 失败则不发布最终目录。

## 12. 错误与资源处理

- 公共 stderr 只输出稳定阶段和异常类型，不传播可能包含路径、HTTP body 或 token 的原始
  异常全文；
- manifest 只记录稳定失败码；
- CPU 导出允许较慢但显存安全；
- CUDA 导出后释放引用并 `empty_cache()`；
- 不声称 CPU 内存或目标 GPU 需求已经验证，除非实际运行；
- `KeyboardInterrupt` 返回 130，并且不发布最终目录。

## 13. 测试与验收

单元测试使用小型 fake model/PEFT seam 和临时目录，不加载 8B 权重。至少覆盖：

1. import/`--help` 不加载大模型；
2. 缺 adapter、merger 或 manifest 时失败；
3. checksum 不匹配时在加载大权重前失败；
4. output 已存在时拒绝；
5. merger missing/unexpected/shape mismatch 均失败；
6. 固定顺序是 merger load 后 LoRA load/merge；
7. LoRA target 与 manifest 不符时失败；
8. merge 后无 LoRA state 残留；
9. merge 后 merger tensor 保持训练 checkpoint 值；
10. processor 和必要辅助文件齐全；
11. 输出文件 manifest checksum 正确；
12. reload 失败时不发布最终 output；
13. 中断或异常时最终 output 不存在；
14. 默认 `local_files_only=True`；
15. public error 不包含模拟 secret 或绝对内部路径。

完成后至少运行：

```text
python -m pytest -q tests/test_export_qwen3vl_phase2_checkpoint.py
python -m compileall -q scripts/export_qwen3vl_phase2_checkpoint.py
git diff --check
git status --short
```

真实 8B 导出、真实 reload 和 forward gate 必须在资源满足时单独执行并如实汇报。未执行时要
列出命令、原因和剩余风险。

## 14. 完成交付标准

只有同时满足以下条件，才可以把最终目录称为“完整可部署 checkpoint”：

```text
Merger strict load 通过
LoRA target 校验和 merge 通过
模型/processor save 完整
离线 reload 通过
输出 checksum manifest 完整
最终目录由临时目录原子发布
```

训练完成或 adapter 保存成功本身不等于导出完成。
