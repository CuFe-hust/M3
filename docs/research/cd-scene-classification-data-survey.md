# 变化检测 MLLM 与场景分类指令数据调研

# Change-Detection MLLM & Scene-Classification Instruction Data Survey

**Date / 日期:** 2026-08-07

> 本文件为调研记录（research note），不改变任何代码行为。
> This file is a research note; it changes no code behavior.
> 标注 [unverified] 的数字来自二手检索，落地前需以官方发布为准。
> Numbers marked [unverified] come from secondary sources; verify against official releases before use.

## 1. 变化检测多模态大模型方法 / Change-Detection MLLM Methods

| 方法 / Method | 基座 / Base | 图像对输入 / Pair input | 训练数据 / Data | 训练部件 / Modules | 来源 / Source |
|---|---|---|---|---|---|
| ChangeChat (2024) | LLaVA 系 [unverified] | 双分支视觉编码器处理双时相图 + LLM | LEVIR-CC + CD 数据集 [unverified] | projector + LLM，ViT 冻结 | arXiv:2409.08582 |
| DeltaVLM (2026) | ChangeChat 扩展 | 多轮指令式双时相探索 | LEVIR-CC + 扩展 | 同族 | MDPI RS 18(4):541 |
| TEOChat (ICLR 2025) | LLaVA-1.5 13B | 变长时序图像序列，一图一个 `<image>` token | 时序 QA + CD + caption（xBD、LEVIR-CC、fMoW 等）[unverified] | MLP projector 训练 + LLM LoRA，ViT-L 冻结 | arXiv:2410.06234 |
| Change-Agent (TGRS 2024) | MCI 视觉模型 + LLM 工具调用 | MCI 消费双时相对，LLM 规划工具 | LEVIR-CD / LEVIR-CC | 视觉 MCI + LLM prompt/tools（非端到端微调） | arXiv:2403.19646 |
| GeoLLaVA (2024) | LLaVA + LoRA | 双时相图像对 | LEVIR-CC（caption）；LEVIR-CD 等 | LLM LoRA + projector，ViT 冻结 | arXiv:2410.19552；LEVIR-CC 上 BERTScore 0.864 / ROUGE-1 0.576 |
| ChangeCLIP (ISPRS 2024) | CLIP（非生成式） | 双时相对 + 语义 prompt | LEVIR-CD / WHU-CD / DSIFN-CD / CDD | 视觉侧微调，输出变化掩膜而非文本 | github.com/dyzy41/ChangeCLIP |
| UniChange (2025) | MLLM | 统一 CD 范式 + 语言先验 | 多个 CD 基准 | projector + LLM LoRA [unverified] | arXiv:2511.02607 |
| CCExpert (2024) | MLLM | 变化 captioning | LEVIR-CC、SECOND-CC [unverified] | [unverified] | arXiv:2411.11360 |

**结论 / Takeaway:** 生成式 CD 方法的主流做法 = 冻结 ViT，训 projector + LLM（LoRA 为主），
双图以多 `<image>` token 的同一对话输入（TEOChat/GeoLLaVA 路线与本仓库 LEVIR-CC
双 `<image>` ShareGPT 格式一致）。

## 2. LEVIR-CC 数据集事实 / LEVIR-CC Facts

- 论文：Chen et al., "LEVIR-CC: A New Benchmark for Remote Sensing Change Captioning",
  IEEE TGRS 2022；GitHub: Chen-Yang-Liu/LEVIR-CC-Dataset；官网 levir.buaa.edu.cn/datasets。
- 图像 256×256、Google Earth 0.5 m、双时相对齐。
- 规模数字冲突，**以官方发布为准 / official release is authoritative**：
  - 检索到 [unverified] 说法：7,064 对，train/val/test = 4,478/1,020/1,566，每对 5 条 caption。
  - 本仓库 2026-08-03 的本地副本约 1,000 对（merged 集：train 4,000 条 / val 500 条 /
    test 500 条，约每对 5 caption）。
- 许可：官网未给出标准开源许可；影像来自 Google Earth，实际按学术研究用途使用，再分发受限。
- 扩展版本：LEVIR-CC+（规模未确认）；Band-Aided LEVIR-CC（扰动鲁棒变体，见 SECOND-CC
  arXiv:2501.10075）；其他可混数据：SECOND-CC（GRSL 2025）、RSCC（NeurIPS 2025 D&B，
  arXiv:2509.01907）、S2Looking（Sentinel-2 中分辨率，域差较大）。

### 小样本扩充常用手段 / Common augmentation tricks for small CD caption data

1. 时相交换（T2,T1）并把 caption 改为反向表述（样本×2，学方向敏感性）；
2. 无变化负样本对（同地 T1-T1 或相邻不重叠裁剪 + "无显著变化"）；
3. 基于 caption 的 GPT 生成 QA（SkyEye-968k、MMRS-1M 均使用）；
4. 单图对多轮 QA / 对话化（ChangeChat、DeltaVLM、SkyEyeGPT）；
5. 跨数据集混合（SECOND-CC、RSCC、S2Looking）；
6. caption 改写增强多样性；
7. 双时相同步空间增广（翻转/旋转保持对应关系）。

## 3. 场景分类指令数据 / Scene-Classification Instruction Data

### 3.1 业界做法 / Field practice

- EarthGPT/MMRS-1M：AID、EuroSAT、NWPU-RESISC45、UCMerced、WHU-RS19 统一转 VQA 指令格式。
- SkyEyeGPT/SkyEye-968k：以 AID 为主，任务前缀（如 `[cls]`）+ "What is the scene category
  of this remote sensing image?" → 类别词；另构造单图多任务多轮对话样本。
- 通用模式：分类一律转 QA 格式；闭集类别列表可随机抽样放入 prompt（GeoChat NWPU 子集即此做法）。

### 3.2 数据集许可 / Licenses

| 数据集 / Dataset | 类别/图像 | 影像来源 | 许可 / License |
|---|---|---|---|
| AID | 30 类 / 10,000 张 600×600 | Google Earth | 无正式开源许可，研究用途 |
| UCMerced | 21 类 / 2,100 张 256×256 | USGS 航拍 | 研究用途 |
| NWPU-RESISC45 | 45 类 / 31,500 张 256×256 | Google Earth | 研究用途公开 |
| EuroSAT | 10 类 / 约 27,000 张 | Sentinel-2 | Copernicus 开放数据政策，最安全 |

学术研究使用四者均为惯例；商用部署仅 EuroSAT 明确安全。

### 3.3 RSVQA 系列 / RSVQA family

| 数据集 | 影像 | 规模 | 问题类型 | 许可 |
|---|---|---|---|---|
| RSVQA-LR (TGRS 2020) | Sentinel-2 10m | 772 tile / 77,232 QA | Presence、Comparison、Rural/Urban | Sentinel-2 开放；OSM ODbL |
| RSVQA-HR | USGS 航拍 15cm | 约 955k tile / 约 895k QA | Count、Presence、Comparison、Area | USGS 公有领域；OSM ODbL |
| RSVQA-B | 未检索到独立发布 [unverified] | — | — | — |

来源：arXiv:2003.07333，rsvqa.sylvainlobry.com。
注意 GeoChat 的 LRBEN 子集与 RSVQA-LRBEN 同源，混用时去重。

## 4. 主要来源 / Key Sources

- ChangeChat https://arxiv.org/abs/2409.08582 · TEOChat https://arxiv.org/abs/2410.06234
- Change-Agent https://arxiv.org/abs/2403.19646 · GeoLLaVA https://arxiv.org/abs/2410.19552
- UniChange https://arxiv.org/abs/2511.02607 · CCExpert https://arxiv.org/abs/2411.11360
- SECOND-CC https://arxiv.org/abs/2501.10075 · RSCC https://arxiv.org/abs/2509.01907
- LEVIR-CC https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset · https://levir.buaa.edu.cn/datasets/index.html
- EarthGPT/MMRS-1M https://arxiv.org/abs/2401.16822 · SkyEyeGPT https://arxiv.org/abs/2401.09712
  · SkyEye-968k https://huggingface.co/datasets/ZhanYang-nwpu/SkyEye-968k
- RSVQA https://arxiv.org/abs/2003.07333 · AID https://captain-whu.github.io/AID/
  · UCMerced https://vision.ucmerced.edu/datasets/ · NWPU-RESISC45
  https://figshare.com/articles/NWPU-RESISC45/19166525 · EuroSAT https://github.com/phelber/eurosat
- Awesome-RS-SFT-Data https://github.com/zytx121/Awesome-RS-SFT-Data
