# Modification Note: Add Qwen3-VL Merger LoRA Test-Set Evaluation Script - 2026-08-07 10:43:06 CST

## Modification Time

2026-08-07 10:43:06 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Add a standalone evaluation script that runs VRSBench test-set inference with
LoRA enabled (base checkpoint + adapter, no merging), after the merger-LoRA
training scripts were added.
新增独立评测脚本：微调完成后不合并权重，直接以 base + LoRA adapter 的方式
在 VRSBench 测试集上做推理（在 merger-LoRA 训练脚本之后补充）。

## Modified Files

- `scripts/evaluate_qwen3vl_merger_lora.py` (new): LoRA-enabled VRSBench
  test-set inference script.
- `tests/test_evaluate_qwen3vl_merger_lora.py` (new): pure-helper unit tests.
- `README.md`: added the LoRA test-set evaluation command.
- `DETAILS.md`: documented the evaluation script under `scripts/`
  responsibilities.

## Core Changes

- The script loads the base Qwen3-VL checkpoint with
  `AutoModelForImageTextToText`, optionally wraps it with
  `PeftModel.from_pretrained` when `--adapter-path` is given, and runs greedy
  generation per test record.
- It reads `VRSBench_test_caption.jsonl` and/or `VRSBench_test_vqa.jsonl` from
  `--data-root`, converts each record into the canonical
  `CanonicalSample`, and writes canonical `{"sample", "prediction"}` JSONL.
- The summary JSON records per-task totals, succeeded/failed counts, mean
  inference latency, and VQA exact match (after deterministic answer
  normalization); failures are persisted in the prediction meta instead of
  being silently skipped.
- Generation ceilings are exposed as `--max-new-tokens-caption` (512) and
  `--max-new-tokens-vqa` (64); `--max-samples` supports smoke runs and image
  pixel limits follow the Qwen3-VL multiple-of-32 rule.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No. The script loads the base model and adapter directly and does not modify
`models/entry.py`, wrappers, or weight-loading logic.

## Whether the Configuration Was Changed

No existing configuration field changed; the evaluation script exposes its own
CLI arguments.

## Whether Evaluation Was Affected

No evaluation metric, dataset split, or reference-answer reading rule was
modified. VQA exact match in the summary is a new lightweight reporting metric
computed inside the standalone script, separate from `eval/`.

## Whether Deployment Was Affected

No deployment path changed.

## Whether pytest Was Updated

Yes. `tests/test_evaluate_qwen3vl_merger_lora.py` covers answer normalization,
record-to-canonical-sample conversion, and message construction.

## Whether .gitignore Was Updated

No. The default output path is under `outputs/`, which is already ignored.

## Validation Method

- `/opt/miniconda3/envs/m3/bin/python -m pytest -q
  tests/test_prepare_vrsbench_sft.py tests/test_finetune_qwen3vl_merger_lora.py
  tests/test_evaluate_qwen3vl_merger_lora.py` passed (14 tests).
- `python -m compileall -q scripts/evaluate_qwen3vl_merger_lora.py` passed.
- `python scripts/evaluate_qwen3vl_merger_lora.py --help` passed.

## Risks and Follow-up TODOs

- No real model inference was run: the local environment lacks the 8B weights,
  the VRSBench test images, and `peft`.
- The script assumes the remote `/data/vrsbench` contains
  `VRSBench_test_{caption,vqa}.jsonl` and the corresponding
  `Images_test/Images_test/...` layout.
- Caption evaluation currently reports counts/latency only; official caption
  metrics such as CIDEr should be computed with the upstream evaluator or
  `eval_standard` after converting the canonical JSONL.
