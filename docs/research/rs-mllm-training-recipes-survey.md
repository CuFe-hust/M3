# Remote-Sensing Multimodal LLM Training Recipes Survey

**Date:** 2026-08-07
**Scope:** RS multimodal large language models, especially Qwen-VL / Qwen2-VL / Qwen2.5-VL / Qwen3-VL and LLaVA-family bases. Focus on training-stage design, module freezing/unfreezing, datasets, and stated lessons.

---

## 1. Comparison Table

| Model | Base | Stages | Modules Trained per Stage | Data per Stage | Notable Lessons |
|---|---|---|---|---|---|
| **EarthGPT** (2024) | LLaMA-2 13B + DINOv2-ViT-L/14 + CLIP-ConvNeXt-L | 3 stages | S1: visual-enhanced perception (encoders frozen); S2: cross-modal comprehension (LLM self-attn + RMSNorm unfrozen, projector trained); S3: unified multi-task tuning (only bias β + shift α in LLM, rest frozen) | S2: LAION-400M + COCO Caption; S3: MMRS-1M (1,005,842 samples, 34 RS datasets) | Vision encoders always frozen; only lightweight LLM params updated in final stage to avoid catastrophic forgetting |
| **EarthGPT-X** (2025) | LLaMA-2 13B + DINOv2 + CLIP-ConvNeXt + SAM | 1 stage ("all-in-one") | Only LLM self-attention layers + added cross-attention module unfrozen; visual encoders + projection layers frozen | M-RSVP: 650k+ image-visual-prompt-text triples; tasks span captioning, detection, SAR, IR | One-stage training is viable if visual prompts are well-designed; full-model unfreezing unnecessary |
| **SkyEyeGPT** (2025) | LLaVA-style (LLM + frozen pretrained RS encoder + alignment layer) | 2 stages | S1: alignment layer (projector) only, encoder + LLM frozen; S2: alignment layer + LLM unfrozen | S1: RS image-caption pairs; S2: multi-task RS instruction dataset (single-task + multi-task conversations) | Vision encoder frozen throughout; LLM directly fine-tuned (not LoRA) in S2 |
| **LHRS-Bot** (ECCV 2024) | LLaVA-style + CLIP ViT-L/14 | 2 stages | S1: bridge layer (projector) only, encoder + LLM frozen; S2: instruction tuning | S1: LHRS-Align (VGI-derived RS image-text); S2: LHRS-Instruct (caption + VQA) | Multi-level vision-language alignment with learnable queries + vision perceiver |
| **LHRS-Bot-Nova** (2024) | Enhanced LHRS-Bot + MoE vision perceiver | 3 stages (curriculum learning) | S1: pre-training (vision encoder + perceiver trained); S2: multi-task instruction tuning; S3: SFT | S1: LHRS-Bot-Recap (~1.1M re-captioned images); S2+S3: expanded instruction data | Unfreezing the vision encoder in pre-training improves RS domain adaptation vs. keeping it frozen |
| **AdaptLLM / RS-Qwen2-VL-2B** (EMNLP 2025) | Qwen2-VL-2B-Instruct | 1 stage (main recipe) or 2 stages (variant) | Full model unfrozen (vision encoder + projector + LLM all trained) | 40k RS image-caption pairs + 15k synthetic instruction pairs (LLaVA-v1.6-8B synthesizer + Llama-3-8B filtering) | For domain adaptation with small data, full-model unfreezing with low LR (1e-5) works; synthesizer quality matters more than quantity |
| **SkySenseGPT** (2024) | CLIP-ViT-L/14 (336×336) + MLP projector + LLM | 1 described stage (instruction tuning) | Vision encoder frozen; projector fine-tuned; LLM fine-tuned with LoRA | FIT-RS: 1,800,851 instruction samples (~1.4M training) | LoRA on LLM + frozen encoder is a practical, parameter-efficient recipe for RS |
| **GeoLLaVA-8K** (NeurIPS 2025) | LLaVA framework + RS-adapted encoder (contrastive pre-trained) | SFT training | Encoder adapted via contrastive pre-training; then SFT with background token compression | SuperRS-VQA (avg 8376×8376) + HighRS-VQA (avg 2000×1912) | Contrastive pre-training of the encoder on RS data before SFT is key for ultra-high-resolution; background token compression addresses token explosion |
| **GeoLLaVA** (change detection, 2024) | Video-LLaVA / LLaVA | 1 stage (LoRA/QLoRA fine-tuning) | LoRA / QLoRA on LLM; vision encoder implicitly frozen | Annotated video frame-pair dataset for change detection | LoRA/QLoRA enables efficient domain adaptation for narrow tasks |
| **RS-LLaVA** (2024) | LLaVA | 2 stages (standard LLaVA recipe) | S1: projector trained, encoder + LLM frozen; S2: projector + LLM trained, encoder frozen | RS-instructions dataset (combined captioning + VQA) | Standard LLaVA 2-stage recipe transfers directly to RS with curated data |
| **RemoteCLIP** (TGRS 2024) | OpenCLIP (ViT-B/32) | 1 stage (contrastive pre-training) | Full CLIP model fine-tuned with InfoNCE loss | ~1.36M RS image-text pairs (aggregated + deduplicated RET-3 subset) | First RS vision-language foundation model; contrastive pre-training on RS data yields strong zero-shot and few-shot downstream |
| **GeoRSCLIP** (TGRS 2024) | OpenAI CLIP (ViT-B/32) | 1 stage (CLIP fine-tuning or PEFT) | Compared: full FT, CoOp, CoCoOp, MaPLe, adapter, LoRA on RS5M | RS5M: 5M RS images with English descriptions | Full fine-tuning outperforms PEFT methods for CLIP-scale RS adaptation, but LoRA/adapters offer strong cost-performance tradeoff |
| **SkyScript / SkyCLIP** (AAAI 2024) | CLIP | 1 stage (continual pre-training) | Full CLIP fine-tuned | 5.2M image-text pairs from OSM geo-tags, 44k semantic categories | Large-scale semantically diverse RS data improves open-vocabulary classification |
| **FLAVARS** (Microsoft, 2025) | FLAVA framework (ViT-B/16) + SatCLIP location encoder | Multi-modal pre-training | Contrastive learning + masked modeling + geospatial alignment; location encoder initialized from SatCLIP | Top-30% subset of SkyScript (5M pairs) + SkyScript-Grounded (GPT-4V) | Combining contrastive, masked, and geospatial alignment objectives yields best KNN and segmentation performance among RS CLIPs |
| **GeoLLM** (ICLR 2024) | GPT-4 / text-only LLM | 1 stage (text-only, no vision) | LLM fine-tuned on RS knowledge | RS caption data converted to text knowledge | Demonstrates RS knowledge extraction without a vision encoder; text-only baseline |
| **GeoChat** (CVPR 2024) | LLaVA + grounding module | Standard LLaVA 2-stage + grounding head | S1: projector; S2: projector + LLM + grounding head | RS VQA + grounding data | Adding a grounding head on top of LLaVA enables spatial referencing in RS |
| **Qwen3-VL few-shot detection** (MDPI RS 2026) | Qwen3-VL | 2 stages (PEFT + hierarchical prompting) | LoRA modules in both vision encoder and LLM; in novel-FT stage, vision LoRA frozen, only text LoRA trained | Few-shot RS object detection data | Two-stage LoRA with selective freezing of vision LoRA in stage 2 improves few-shot detection |
| **EarthMind** (2025) | Multi-sensor EO LMM | Multi-stage | Projector/fusion modules tuned across stages | 1M EO-specific multimodal samples, multi-sensor | Multi-granular, multi-sensor approach with staged fusion |
| **TerraScope** (CVPR 2026) | Vision-language reasoning model | Multi-stage | Segmentation masks ground each reasoning step | Multi-temporal EO data | Pixel-grounded reasoning improves faithfulness for EO change reasoning |

---

## 2. Per-Model Details

### 2.1 EarthGPT (2024)

- **Paper:** "EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Image Comprehension in Remote Sensing Domain" — W. Zhang et al., IEEE TGRS 2024, ~456 citations.
- **arXiv:** [2401.16822](https://arxiv.org/abs/2401.16822) | **GitHub:** [wivizhang/EarthGPT](https://github.com/wivizhang/EarthGPT)
- **Base model:** LLaMA-2 13B; vision encoders: DINOv2-ViT-L/14 + CLIP-ConvNeXt-L; projector: randomly initialized MLPs; segmentation branch uses pre-trained SAM encoder/decoder.
- **Training stages (3):**
  1. **Visual-enhanced perception stage:** Vision encoders frozen. Projector trained. No separate LR given.
  2. **Cross-modal mutual comprehension stage:** Vision encoders frozen. LLM: self-attention + RMSNorm layers unfrozen. Projector trained. Data: LAION-400M + COCO Caption (counts not stated).
  3. **Unified multi-task tuning:** All weights from stage 2 frozen. Only bias β and shift α added to LLaMA-2 are trained. Data: MMRS-1M (1,005,842 samples from 34 RS datasets; breakdown: captioning 198,883, VQA 141,264, classification 28,705, detection 575,350, visual grounding 30,820, region-level captioning 30,820).
- **Optimizer:** AdamW, peak LR 2×10⁻⁵, β₁=0.9, β₂=0.95.
- **Key lesson:** Progressive freezing protects pre-trained knowledge; only tiny LLM parameter additions in the final stage prevent catastrophic forgetting.

### 2.2 EarthGPT-X (2025)

- **Paper:** "EarthGPT-X: A Spatial MLLM for Multi-level Multi-source Remote Sensing Imagery Understanding with Visual Prompting" — W. Zhang et al., 2025.
- **arXiv:** [2504.12795](https://arxiv.org/abs/2504.12795) | **GitHub:** [wivizhang/EarthGPT-X](https://github.com/wivizhang/EarthGPT-X)
- **Base model:** LLaMA-2 13B + DINOv2-ViT-L/14 + CLIP-ConvNeXt + SAM encoder.
- **Training strategy:** Explicitly **one-stage all-in-one training** (not multi-stage).
- **Modules trained:** Only LLM self-attention layers + added cross-attention module unfrozen. Visual encoders, projection layers, etc. are kept frozen.
- **Optimizer:** AdamW, LR 2×10⁻⁵, weight decay 0.01.
- **Data:** M-RSVP dataset: 650k+ image-visual-prompt-text triples. Captioning: RSICD 24.3k, UCM-Captions 10k. SAR: SARDet 106k, SAR-Ship 33.7k, MSAR 32.5k, etc. Infrared: Aerial-mancar 31.6k, Infrared-security 9.6k.
- **Compute:** 8× A100 80GB, ~310 hours.
- **Key lesson:** One-stage training is viable when the architecture supports dual text + visual prompts; full-model unfreezing is unnecessary.

### 2.3 SkyEyeGPT (2025)

- **Paper:** "SkyEyeGPT: Unifying Remote Sensing Vision-Language Tasks via Instruction Tuning with Large Language Model" — Y. Zhan et al., ISPRS J. Photogrammetry & Remote Sensing 2025, ~298 citations.
- **arXiv:** [2401.09712](https://arxiv.org/abs/2401.09712) | **GitHub:** [ZhanYang-nwpu/SkyEyeGPT](https://github.com/ZhanYang-nwpu/SkyEyeGPT)
- **Base model:** LLaVA-style architecture (pretrained RS visual encoder + alignment layer + LLM decoder).
- **Training stages (2):**
  1. **Alignment stage:** Only alignment layer (projector) trained. Vision encoder and LLM frozen. Data: RS image-caption pairs.
  2. **Instruction tuning stage:** Alignment layer + LLM unfrozen. Vision encoder frozen throughout. Data: multi-task RS instruction dataset (single-task + multi-task conversation instructions).
- **Key detail:** LLM is fine-tuned directly (not via LoRA). Vision encoder frozen throughout all training.
- **Key lesson:** A frozen, RS-pretrained vision encoder combined with direct LLM fine-tuning achieves strong multi-task RS performance without LoRA.

### 2.4 LHRS-Bot (ECCV 2024) and LHRS-Bot-Nova (2024)

- **Papers:**
  - "LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model" — D. Muhtar et al., ECCV 2024, ~242 citations. [arXiv:2402.02544](https://arxiv.org/abs/2402.02544) | [GitHub](https://github.com/NJU-LHRS/LHRS-Bot)
  - "LHRS-Bot-Nova: Improved Multimodal Large Language Model for Remote Sensing" — Nov 2024. [arXiv:2411.09301](https://arxiv.org/abs/2411.09301)
- **LHRS-Bot (2 stages):**
  1. **Alignment stage:** Bridge layer trained on LHRS-Align (VGI-derived RS image-text pairs). Vision encoder (CLIP ViT-L/14, 224×224) and LLM frozen.
  2. **Instruction tuning:** Fine-tune on LHRS-Instruct data. Multi-level vision-language alignment with learnable queries + vision perceiver.
- **LHRS-Bot-Nova (3 stages, curriculum learning):**
  1. **Pre-training:** Vision encoder + MoE vision perceiver trained on LHRS-Bot-Recap (~1.1M re-captioned images with OSM features). **Vision encoder is unfrozen** in this stage — a departure from the original.
  2. **Multi-task instruction tuning:** Projector + LLM trained on expanded instruction data.
  3. **Supervised fine-tuning:** Final task-specific tuning.
- **Key lesson:** Nova demonstrates that unfreezing the vision encoder during a pre-training stage improves RS domain adaptation. The MoE vision perceiver adds capacity without full encoder retraining.

### 2.5 AdaptLLM / remote-sensing-Qwen2-VL-2B (EMNLP 2025)

- **Paper:** "On Domain-Adaptive Post-Training for Multimodal Large Language Models" — Cheng, Huang et al., EMNLP 2025 Findings.
- **arXiv:** [2411.19930](https://arxiv.org/abs/2411.19930) | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.17.pdf) | [GitHub](https://github.com/bigai-ai/QA-Synthesizer)
- **Models:** `AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct`, `AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct`
- **Training recipe (Qwen2-VL-2B):**
  - **Single-stage post-training (main recipe):** Full model unfrozen. LR 1e-5 for both language/projector and image encoder. 1 epoch, batch size 128, max seq length 6144. Cosine schedule, weight decay 0.1, warmup ratio 0.1. Loss restricted to response tokens (next-token prediction).
  - **Two-stage variant:** Stage 1 (caption alignment): full model trainable, same hyperparameters. Stage 2 (visual instruction): full model trainable, same hyperparameters.
- **Data:** 40K RS image-caption pairs from NWPU-Captions, RSICD, RSITMD, Sydney-Captions, UCM-Captions. 15K synthetic instruction-response pairs generated by LLaVA-v1.6-8B-based synthesizer checked with Llama-3-8B consistency filtering.
- **Synthesizer training:** LLaVA-v1.6-8B, full model trainable, 2 epochs, batch 128, max length 6144, LR 2e-5 (language/projector) and 2e-6 (image encoder), cosine schedule.
- **Key results:** CLRS 55.0 vs. 48.9 (original), UC Merced 61.8 vs. 61.0, FloodNet 62.2 vs. 55.7, NWPU 56.0 vs. 26.0.
- **Key lesson:** For domain adaptation with relatively small data (55k total), full-model unfreezing with a uniform low learning rate (1e-5) is effective. Data synthesis quality (LLM-based generation + consistency filtering) matters more than sheer volume.

### 2.6 SkySenseGPT (2024)

- **Paper:** "SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding" — J. Luo et al., Jun 2024, ~142 citations.
- **arXiv:** [2406.10100](https://arxiv.org/abs/2406.10100) | [GitHub](https://github.com/Luo-Z13/SkySenseGPT)
- **Base model:** CLIP-ViT-L/14 (336×336) + MLP projector + LLM.
- **Training (instruction tuning stage):**
  - Vision encoder: **frozen**.
  - Projector: **fine-tuned**.
  - LLM: **fine-tuned with LoRA**.
- **Data:** FIT-RS dataset: 1,800,851 instruction samples (~1.4M for training). Split 6:2:2 (train:val:test). Covers general interpretation + complex comprehension tasks (scene graph generation at region and image level). FIT-RSFG: derived fine-grained benchmark.
- **Key lesson:** LoRA on the LLM + frozen encoder + tuned projector is a practical and parameter-efficient recipe for RS instruction tuning. The FIT-RS dataset introduces scene-graph-level structured understanding as a training objective.

### 2.7 GeoLLaVA-8K (NeurIPS 2025) and GeoLLaVA (change detection, 2024)

- **GeoLLaVA-8K:**
  - **Paper:** "GeoLLaVA-8K: Scaling Remote-Sensing Multimodal Large Language Models to 8K Resolution" — F. Wang et al., NeurIPS 2025. [arXiv:2505.21375](https://arxiv.org/abs/2505.21375)
  - **Base:** LLaVA framework with RS-adapted image encoder.
  - **Training:** Image encoder first adapted to RS via **contrastive pre-training**. Then SFT on SuperRS-VQA (avg 8376×8376) + HighRS-VQA (avg 2000×1912). Uses **Background Token Compression** via adaptive token clustering to handle ultra-high-resolution token explosion.
  - **Key lesson:** Contrastive pre-training of the vision encoder on RS data before SFT is critical for ultra-high-resolution understanding. Token compression makes 8K feasible.
- **GeoLLaVA (change detection, 2024):**
  - **Paper:** "GeoLLaVA: Efficient Fine-Tuned Vision-Language Models for Temporal Change Detection" — H. Elgendy et al. [arXiv:2410.19552](https://arxiv.org/abs/2410.19552) | [GitHub](https://github.com/HosamGen/GeoLLaVA)
  - **Training:** LoRA / QLoRA fine-tuning on Video-LLaVA and LLaVA. Vision encoder implicitly frozen (LoRA applied to LLM only). Data: annotated video frame-pair dataset.
  - **Key lesson:** LoRA/QLoRA suffices for narrow RS tasks like change detection without full model retraining.

### 2.8 RS-LLaVA (2024)

- **Paper:** "RS-LLaVA: A Large Vision-Language Model for Joint Captioning and Question Answering in Remote Sensing Imagery" — Y. Bazi et al., MDPI Remote Sensing 2024, ~182 citations.
- **Link:** [MDPI](https://www.mdpi.com/2072-4292/16/9/1477) | [GitHub](https://github.com/BigData-KSU/RS-LLaVA)
- **Base:** LLaVA.
- **Training:** Standard LLaVA 2-stage recipe. S1: projector trained, encoder + LLM frozen. S2: projector + LLM trained, encoder frozen.
- **Data:** RS-instructions dataset (combined from 4 captioning + VQA datasets).
- **Key lesson:** Standard LLaVA training recipe transfers directly to RS with curated instruction data; no architectural changes needed.

### 2.9 RemoteCLIP (TGRS 2024)

- **Paper:** "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing" — F. Liu, D. Chen et al., IEEE TGRS 2024, ~939 citations.
- **arXiv:** [2306.11029](https://arxiv.org/abs/2306.11029) | [GitHub](https://github.com/ChenDelong1999/RemoteCLIP)
- **Base:** OpenCLIP (ViT-B/32).
- **Training:** 1 stage — contrastive language-image pre-training optimizing InfoNCE loss. Full CLIP model fine-tuned.
- **Data:** ~1.36M RS image-text pairs aggregated from multiple RS datasets. Deduplicated RET-3 subset: 68,565 pairs from RSITMD, RSICD, UCM.
- **Downstream:** Zero-shot classification, linear probing, k-NN, few-shot learning, image-text retrieval.
- **Key lesson:** First RS VL foundation model. Contrastive pre-training on aggregated RS data yields strong transfer to downstream tasks. Data aggregation and deduplication are critical.

### 2.10 GeoRSCLIP + RS5M (TGRS 2024)

- **Paper:** "RS5M and GeoRSCLIP: A Large Scale Vision-Language Dataset and A Large Vision-Language Model for Remote Sensing" — Z. Zhang et al., IEEE TGRS 2024, ~340 citations.
- **arXiv:** [2306.11300](https://arxiv.org/abs/2306.11300) | [GitHub](https://github.com/om-ai-lab/RS5M)
- **Base:** OpenAI CLIP (ViT-B/32).
- **Training:** 1 stage — CLIP fine-tuning or PEFT on RS5M. Compared: full fine-tuning, prompt learning (CoOp, CoCoOp, MaPLe), adapter, LoRA.
- **Data:** RS5M: 5M RS images with English descriptions (first large-scale RS image-text dataset at this scale).
- **Key lesson:** Full fine-tuning outperforms PEFT methods for CLIP-scale RS adaptation, but LoRA/adapters offer strong cost-performance tradeoffs. Dataset scale (5M) drives performance.

### 2.11 SkyScript / SkyCLIP (AAAI 2024)

- **Paper:** "SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing" — Z. Wang et al., AAAI 2024, ~233 citations.
- **Link:** [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28393) | [GitHub](https://github.com/wangzhecheng/skyscript)
- **Base:** CLIP.
- **Training:** Continual pre-training — full CLIP fine-tuned on SkyScript.
- **Data:** 5.2M image-text pairs from OSM geo-tags, covering 44k distinct semantic categories.
- **Key lesson:** Semantic diversity (44k categories) matters more than raw scale for open-vocabulary RS classification. OSM tags provide a scalable source of noisy but diverse captions.

### 2.12 FLAVARS (Microsoft Research, 2025)

- **Paper:** "FLAVARS: A Multimodal Foundational Language and Vision Alignment Model for Remote Sensing" — I. Corley et al., Microsoft Research, Jan 2025.
- **arXiv:** [2501.08490](https://arxiv.org/abs/2501.08490) | [MSR page](https://www.microsoft.com/en-us/research/publication/flavars-a-multimodal-foundational-language-and-vision-alignment-model-for-remote-sensing/)
- **Base:** FLAVA pretraining framework (ViT-B/16 backbone). Location encoder initialized from SatCLIP weights.
- **Training:** Multi-modal pre-training combining contrastive learning + masked modeling + geospatial alignment. Location encoder continually pre-trained with coordinates.
- **Data:** Top-30% scoring subset of SkyScript (5M pairs). SkyScript-Grounded (GPT-4V, performance left for future work). Evaluation: KNN uses frozen vision encoder; segmentation uses pretrained encoder + UperNet decoder on SpaceNet1.
- **Key lesson:** Combining contrastive, masked, and geospatial alignment objectives yields the best KNN classification and semantic segmentation among RS CLIP variants. Geospatial coordinate encoding adds spatial awareness.

### 2.13 GeoLLM (ICLR 2024)

- **Paper:** "GeoLLM: Extracting Geospatial Knowledge from Remote Sensing Data" — R. Manvi et al., ICLR 2024, ~256 citations.
- **Link:** [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/a87f2df7c4ab0213c6ea228e7b7f0a4d-Paper-Conference.pdf)
- **Note:** This is a **text-only** approach — RS captions are converted to structured geospatial knowledge for LLM pre-training. No vision encoder. Not directly comparable to vision-language models above, but relevant as a text-baseline.

### 2.14 GeoChat (CVPR 2024)

- **Paper:** "GeoChat: Grounded Large Vision-Language Model for Remote Sensing" — Kuckreja et al., CVPR 2024.
- **Link:** [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html) | [GitHub](https://github.com/mbzuai-oryx/geochat)
- **Base:** LLaVA + grounding module.
- **Training:** Standard LLaVA 2-stage + grounding head training. S1: projector. S2: projector + LLM + grounding head.
- **Data:** RS VQA + grounding data.
- **Key lesson:** Adding a spatial grounding head on top of LLaVA enables coordinate-based spatial referencing without changing the core training recipe.

### 2.15 Qwen3-VL Two-Stage Fine-Tuning (MDPI Remote Sensing, 2026)

- **Paper:** "Two-Stage Fine-Tuning of Large Vision-Language Models with Hierarchical Prompting for Few-Shot Object Detection in Remote Sensing Images" — Y. Shi et al., MDPI Remote Sensing 2026.
- **Link:** [MDPI](https://www.mdpi.com/2072-4292/18/2/266)
- **Base:** Qwen3-VL.
- **Training (2 stages, PEFT):**
  1. LoRA modules inserted in both vision encoder and LLM.
  2. In the novel fine-tuning stage, **LoRA modules in the vision encoder are frozen**, while only the LoRA parameters in the text/LLM side are trained.
- **Key lesson:** For few-shot RS detection, selectively freezing vision-encoder LoRA in stage 2 prevents overfitting on limited data while retaining domain adaptation from stage 1.

### 2.16 EarthMind (2025) and TerraScope (CVPR 2026)

- **EarthMind:** "Towards Multi-Granular and Multi-Sensor Earth Observation LMMs" — [arXiv:2506.01667](https://arxiv.org/abs/2506.01667) | [GitHub](https://github.com/shuyansy/EarthMind). 1M EO-specific multimodal samples, multi-sensor. Multi-stage training of projector/fusion modules.
- **TerraScope:** "Pixel-Grounded Visual Reasoning for Earth Observation" — Y. Shu et al., CVPR 2026. Grounds each reasoning step in precise segmentation masks for interpretable spatial analysis. Multi-temporal change reasoning.

---

## 3. Models Not Found or Unverified

The following models from the original query could not be verified through extensive searching:

- **QEarth:** No model specifically named "QEarth" built on Qwen2-VL for earth observation was found in any search results. The closest match is AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct. The name may be misremembered or unpublished.
- **EarthGPT-2:** No distinct model officially called "EarthGPT-2" exists. The EarthGPT series consists of EarthGPT (2024) and EarthGPT-X (2025). A Scribd curriculum document mentions "EarthGPT 2" but this appears to be a secondary/imprecise reference.
- **SatChat:** Two different projects use this name: (1) TelePIX's commercial SatCHAT geospatial AI platform (no published training recipe paper found); (2) Karunarathne et al.'s text-only conversational agent for satellite manoeuvre queries (not a multimodal VLM). Neither matches the "multimodal LLM training strategy" description.
- **RS-MLLM (as a specific paper name):** No single paper titled "RS-MLLM" with detailed training-stage/LoRA/vision-encoder specifics was found. The term appears generically in survey papers to refer to "remote sensing multimodal large language models" as a category.

---

## 4. Synthesis: Field Consensus on Which Modules to Train at the Instruction-Tuning Stage for RS Domain Adaptation

Based on the surveyed models, the following patterns emerge as the field's rough consensus for the instruction-tuning stage:

### 4.1 Vision Encoder: Mostly Frozen, But Unfreezing Helps for Domain Shift

- **Default (majority of models):** The vision encoder is **frozen** during instruction tuning. This includes EarthGPT, EarthGPT-X, SkyEyeGPT, LHRS-Bot (original), SkySenseGPT, RS-LLaVA, GeoChat, and GeoLLaVA (change detection).
- **Rationale:** Pre-trained vision encoders (CLIP, DINOv2, SigLIP) already encode strong visual features. Unfreezing risks catastrophic forgetting of general visual knowledge, especially with limited RS instruction data.
- **Exception — when to unfreeze:** LHRS-Bot-Nova demonstrates that unfreezing the vision encoder during a **pre-training stage** (before instruction tuning) improves RS domain adaptation. GeoLLaVA-8K shows that **contrastive pre-training of the encoder on RS data** before SFT is critical for ultra-high-resolution tasks. GeoRSCLIP and RemoteCLIP show that full fine-tuning of CLIP on large RS datasets (5M, 1.36M pairs respectively) outperforms PEFT methods.
- **Emerging pattern:** Unfreeze the encoder during a **data-rich pre-training/alignment stage** (millions of image-text pairs), then freeze it during the **instruction-tuning stage** (thousands to hundreds of thousands of instruction samples).

### 4.2 Projector/Connector: Almost Always Trained

- The projector (MLP or bridge layer) is **trained in nearly every stage** across all models. It serves as the cross-modal alignment module and is the cheapest component to train.
- The only exception is EarthGPT's stage 3, where everything from stage 2 is frozen and only tiny bias/shift parameters are added to the LLM.

### 4.3 LLM: Full Fine-Tuning vs. LoRA

- **Full fine-tuning (majority):** SkyEyeGPT, AdaptLLM, EarthGPT (partial), LHRS-Bot, GeoLLaVA-8K, RS-LLaVA all fine-tune the LLM (or its self-attention layers) directly during instruction tuning.
- **LoRA (growing trend):** SkySenseGPT, GeoLLaVA (change detection), Qwen3-VL few-shot detection paper use LoRA on the LLM. LoRA is preferred when:
  - The instruction dataset is relatively small (prevents overfitting).
  - Compute resources are constrained.
  - Multiple task-specific adapters need to be maintained.
- **Learning rate ratios:** AdaptLLM uses a uniform LR (1e-5) for all modules. The AdaptLLM synthesizer uses 10× lower LR for the image encoder (2e-6) than for the language/projector (2e-5) — a common pattern in multi-stage training. GeoRSCLIP's full FT outperforms LoRA at scale but LoRA offers better cost-performance.

### 4.4 Data Mixing and Scale

- **Pre-training/alignment stage:** Typically uses millions of image-text pairs (RS5M: 5M, SkyScript: 5.2M, MMRS-1M: 1M, RemoteCLIP: 1.36M, LHRS-Bot-Recap: 1.1M). These are often noisy, web-crawled, or VGI-derived.
- **Instruction tuning stage:** Typically uses tens of thousands to ~1.8M curated instruction samples. The trend is toward higher-quality, synthetically generated instruction data (AdaptLLM's LLM-based synthesizer, SkySenseGPT's scene-graph instructions).
- **Data mixing:** Most models mix multiple RS datasets covering diverse tasks (captioning, VQA, classification, detection, visual grounding). EarthGPT's MMRS-1M covers 34 datasets across 6 tasks. The balance between tasks is important but no consensus ratio exists.

### 4.5 Catastrophic Forgetting Mitigation

- **Progressive freezing (EarthGPT):** Freeze more modules in later stages; only add tiny trainable parameters.
- **LoRA (SkySenseGPT, Qwen3-VL):** Low-rank updates inherently limit forgetting.
- **Curriculum learning (LHRS-Bot-Nova):** Gradually increase task difficulty.
- **Contrastive pre-training before SFT (GeoLLaVA-8K):** Ensures the encoder retains general RS features.
- **Consistency filtering of synthetic data (AdaptLLM):** Prevents noisy instructions from corrupting the model.

### 4.6 One-Stage vs. Multi-Stage

- **Multi-stage (majority):** Most models use 2 or 3 stages (alignment/pre-training → instruction tuning → optional SFT). This follows the LLaVA convention.
- **One-stage (EarthGPT-X):** Explicitly uses one-stage all-in-one training. Argues that when the architecture supports diverse visual prompts, multi-stage training is unnecessary.
- **Trend:** The field is converging toward 2 stages minimum: (1) data-rich alignment/pre-training with encoder unfrozen (if data allows), and (2) instruction tuning with encoder frozen.

---

## 5. Source URLs

All factual claims above are sourced from the following URLs. Each model's claims are linked inline in sections 2.1–2.16.

- EarthGPT: https://arxiv.org/abs/2401.16822 · https://arxiv.org/html/2401.16822v3 · https://github.com/wivizhang/EarthGPT
- EarthGPT-X: https://arxiv.org/abs/2504.12795 · https://arxiv.org/html/2504.12795v3 · https://github.com/wivizhang/EarthGPT-X
- SkyEyeGPT: https://arxiv.org/abs/2401.09712 · https://www.sciencedirect.com/science/article/pii/S0924271625000206 · https://github.com/ZhanYang-nwpu/SkyEyeGPT
- LHRS-Bot: https://arxiv.org/abs/2402.02544 · https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09511.pdf · https://github.com/NJU-LHRS/LHRS-Bot
- LHRS-Bot-Nova: https://arxiv.org/abs/2411.09301 · https://arxiv.org/html/2411.09301v1
- AdaptLLM: https://arxiv.org/abs/2411.19930 · https://arxiv.org/html/2411.19930v4 · https://aclanthology.org/2025.findings-emnlp.17.pdf · https://huggingface.co/AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct · https://github.com/bigai-ai/QA-Synthesizer
- SkySenseGPT: https://arxiv.org/abs/2406.10100 · https://arxiv.org/html/2406.10100v2 · https://github.com/Luo-Z13/SkySenseGPT
- GeoLLaVA-8K: https://arxiv.org/abs/2505.21375 · https://arxiv.org/html/2505.21375v2 · https://papers.nips.cc/paper_files/paper/2025/file/e95ef5c6ed4096c609e0b8b47ffaeb9b-Paper-Conference.pdf
- GeoLLaVA (change detection): https://arxiv.org/abs/2410.19552 · https://github.com/HosamGen/GeoLLaVA
- RS-LLaVA: https://www.mdpi.com/2072-4292/16/9/1477 · https://github.com/BigData-KSU/RS-LLaVA
- RemoteCLIP: https://arxiv.org/abs/2306.11029 · https://github.com/ChenDelong1999/RemoteCLIP
- GeoRSCLIP/RS5M: https://arxiv.org/abs/2306.11300 · https://github.com/om-ai-lab/RS5M
- SkyScript: https://ojs.aaai.org/index.php/AAAI/article/view/28393 · https://github.com/wangzhecheng/skyscript
- FLAVARS: https://arxiv.org/abs/2501.08490 · https://www.microsoft.com/en-us/research/publication/flavars-a-multimodal-foundational-language-and-vision-alignment-model-for-remote-sensing/
- GeoLLM: https://proceedings.iclr.cc/paper_files/paper/2024/file/a87f2df7c4ab0213c6ea228e7b7f0a4d-Paper-Conference.pdf
- GeoChat: https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html · https://github.com/mbzuai-oryx/geochat
- Qwen3-VL two-stage: https://www.mdpi.com/2072-4292/18/2/266
- EarthMind: https://arxiv.org/abs/2506.01667 · https://github.com/shuyansy/EarthMind
- TerraScope: https://openaccess.thecvf.com/content/CVPR2026/papers/Shu_TerraScope_Pixel-Grounded_Visual_Reasoning_for_Earth_Observation_CVPR_2026_paper.pdf
- Awesome RS MLLM list: https://github.com/ZhanYang-nwpu/Awesome-Remote-Sensing-Multimodal-Large-Language-Model
- RS VLM survey: https://www.mdpi.com/2072-4292/17/1/162
- EO MLLM survey: https://radars.ac.cn/en/article/doi/10.12000/JR25088
- RS foundation models survey: https://arxiv.org/abs/2410.16602

---

## 6. Practical Takeaways for a Qwen2-VL / Qwen2.5-VL / Qwen3-VL RS Fine-Tuning Project

Based on the synthesis above, a practical training recipe for adapting a Qwen-VL family model to remote sensing would be:

1. **Stage 1 — RS alignment/pre-training (if data is available):**
   - Unfreeze the vision encoder (or at minimum the merger/connector).
   - Train on a large RS image-caption dataset (100k–1M+ pairs).
   - Use a lower LR for the vision encoder (e.g., 2e-6) than for the projector/LLM (e.g., 2e-5).

2. **Stage 2 — Instruction tuning:**
   - Freeze the vision encoder.
   - Train the merger/projector + LLM (full fine-tuning or LoRA, depending on data size and compute).
   - Use curated RS instruction data (VQA, captioning, classification, detection, grounding).
   - LR ~1e-5 (AdaptLLM's successful setting for Qwen2-VL-2B).
   - If data is limited (<50k), prefer LoRA to prevent overfitting.

3. **Key decisions:**
   - **Vision encoder:** Freeze during instruction tuning; unfreeze only during data-rich pre-training.
   - **Merger/projector:** Always train.
   - **LLM:** Full fine-tuning for large instruction datasets; LoRA for small datasets or compute-constrained settings.
   - **Data quality > data quantity:** AdaptLLM shows that 15k high-quality synthetic instructions outperform larger noisy datasets.
