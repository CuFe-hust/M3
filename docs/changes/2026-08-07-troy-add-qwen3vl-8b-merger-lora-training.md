# Modification Note: Add Qwen3-VL-8B Merger-Layer LoRA Training Scripts - 2026-08-07 10:04:19 CST

## Modification Time

2026-08-07 10:04:19 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Add standalone, repository-owned training scripts for pure LoRA SFT on the four
Qwen3-VL-8B vision merger modules (`visual.merger` plus three
`visual.deepstack_merger_list.*`), using the processed VRSBench JSONL
annotations (local `datasets/vrsbench`, remote `/data/vrsbench`) and the
Qwen-VL-Series-Finetune conversation data format as reference.
新增仓库自有的训练脚本，对 Qwen3-VL-8B 四个视觉 merger 模块
（`visual.merger` 加三个 `visual.deepstack_merger_list.*`）做纯 LoRA SFT，
数据使用处理后的 VRSBench JSONL 标注（本地 `datasets/vrsbench`，远端
`/data/vrsbench`），数据格式参考 Qwen-VL-Series-Finetune 对话格式。

## Modified Files

- `scripts/prepare_vrsbench_sft.py` (new): JSONL -> SFT JSON converter.
- `scripts/finetune_qwen3vl_merger_lora.py` (new): merger-only LoRA training script.
- `scripts/merge_qwen3vl_merger_lora.py` (new): adapter merge script.
- `scripts/finetune_qwen3vl_merger_lora.sh` (new): single-4090/48G run entry.
- `requirements-finetune.txt` (new): reference training environment.
- `tests/test_prepare_vrsbench_sft.py` (new).
- `tests/test_finetune_qwen3vl_merger_lora.py` (new).
- `README.md`: added the merger-LoRA training section.
- `DETAILS.md`: added `scripts/` responsibilities and section 17 for the
  merger-LoRA training contract.

## Core Changes

- The converter reads the eight mapped VRSBench JSONL files (en/zh x caption/vqa
  x train/val), writes one SFT JSON per split in the reference conversation
  format, duplicates English train records by `--english-multiplier` (default
  2) to emphasize English, keeps the validation split balanced, and
  deterministically shuffles the train split.
- The training script attaches LoRA to every `nn.Linear` whose module path
  contains "merger" (8 linear layers for Qwen3-VL-8B), freezes all other
  weights by default (`--freeze_merger_base True`), and exposes model, data,
  LoRA, and all standard Hugging Face training arguments through the CLI.
- The training dataset tokenizes the Qwen3-VL chat scaffold manually
  (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`), masks prompt
  labels with `IGNORE_INDEX`, and passes `min_pixels`/`max_pixels` to the
  processor so Qwen3-VL image sizing follows multiples of 32.
- The runtime check refuses Transformers builds without the native Qwen3-VL
  deepstack fusion path (`Qwen3VLTextModel._deepstack_process`); the reference
  environment pins `transformers==5.14.1`.
- The shell entry uses environment variables `DATA_ROOT`, `MODEL_ID`,
  `OUTPUT_DIR`, and `PYTHON`; no local absolute path is hard-coded in Python.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No. The training scripts do not touch `models/entry.py`, existing wrappers, or
weight-loading logic; the merged checkpoint is loaded through the existing
`qwen_transformers` entry.

## Whether the Configuration Was Changed

No existing configuration field changed. Training hyperparameters are exposed
as new CLI arguments of the standalone script, and `requirements-finetune.txt`
is a new optional dependency file.

## Whether Evaluation Was Affected

No metric, split, reference-answer reading, or result post-processing rule was
changed. The prepared SFT validation set is derived from the existing local
train/val JSONL and is not used by `eval/`.

## Whether Deployment Was Affected

No deployment path changed. The merge script is additive and produces a
standard Transformers checkpoint.

## Whether pytest Was Updated

Yes. Two new test files cover the converter (record mapping, English
multiplier, deterministic shuffle, missing-file failure) and the training
script helpers (JSON/JSONL loading, merger linear target discovery, token
replacement, dtype resolution).

## Whether .gitignore Was Updated

No. New outputs go under `outputs/`, which is already ignored, and
`requirements-finetune.txt` is a source dependency file that should be tracked.

## Validation Method

- `/opt/miniconda3/envs/m3/bin/python -m pytest -q tests/test_prepare_vrsbench_sft.py tests/test_finetune_qwen3vl_merger_lora.py` passed (10 tests).
- `/opt/miniconda3/envs/m3/bin/python -m compileall -q` passed for the new Python files.
- `python scripts/prepare_vrsbench_sft.py --help`, `python scripts/finetune_qwen3vl_merger_lora.py --help`, and `python scripts/merge_qwen3vl_merger_lora.py --help` passed.
- `bash -n scripts/finetune_qwen3vl_merger_lora.sh` passed.

## Risks and Follow-up TODOs

- No real GPU training was run; the local M3 conda environment lacks `peft`
  and `torchvision`, and no Qwen3-VL-8B weights are cached locally.
- The remote `/data/vrsbench` annotations and images are not yet downloaded;
  the shell entry defaults assume the VRSBench root contains the expected
  `VRSBench_*_*.jsonl` files plus `Images_train/Images_train/...` layout.
- A dedicated finetune environment is recommended (see
  `requirements-finetune.txt`); the remote `py311` env currently conflicts with
  the pre-existing llamafactory `transformers<=5.6.0` constraint.
- `models/qwen3_vl_8b/` is still not registered in `models/entry.py`; decide
  later whether to register the 8B wrapper or evaluate the merged checkpoint
  through the existing `qwen_transformers` entry.
