# Modification Note: Phase 2 Qwen3-VL SFT finetune script - 2026-08-08

## Modification Time

2026-08-08 (task docs/train/03_FINETUNE_QWEN3VL_PHASE2.md, round 3)

## Modifier

troy (AI coding agent, per user instruction "增加许可不再需要向我确认")

## Modification Goal

Implement the Phase 2 Qwen3-VL-8B SFT training script
(scripts/finetune_qwen3vl_phase2.py): full merger training + LLM LoRA with
two learning rates, resumable composite checkpoints only.
实现 Phase 2 Qwen3-VL-8B 微调训练脚本：全量 Merger + LLM LoRA 双学习率，
唯一产物为可 resume 的复合 checkpoint。

## Modified Files

- `architecture/allowed_python_files.txt` (allowlist; independent commit):
  added `scripts/finetune_qwen3vl_phase2.py` and
  `tests/test_finetune_qwen3vl_phase2.py` (user pre-authorized the allowlist
  change; test file itself is deferred per user instruction).
- `scripts/finetune_qwen3vl_phase2.py` (new): training script.
- `tests/test_finetune_qwen3vl_phase2.py` (new, follow-up commit): unit tests
  with a fake Qwen model tree + fake processor + real peft (19 tests, all
  CPU, offline).
- `DETAILS.md`: added the script to `scripts/` responsibilities (section 85).

Timeline: the user initially deferred tests ("测试暂时不管"); the test file
was written and committed in a follow-up once the M3 conda environment was
confirmed ready, and the acceptance battery from doc section 12 was run.

## Core Changes

- Strategy: Vision Encoder frozen; `model.visual.merger` +
  `deepstack_merger_list.*` fully trainable (merger_lr); LLM base frozen with
  LoRA on all seven projections per decoder layer (lora_lr).
  策略：Vision Encoder 冻结；主 merger 与全部 deepstack merger 全量训练；
  LLM base 冻结并挂七 projection LoRA。
- Structure location is attribute-based (merger + deepstack_merger_list /
  layers + embed_tokens), verified against transformers 5.14.1
  (`model.visual` / `model.language_model`); never fuzzy string matching.
- Hard startup audit writes `parameter_audit.json`; trainable parameters are
  exactly classified into {merger_base, llm_lora} with disjoint sets.
- Explicit four-group optimizer (merger/lora x decay/no_decay); cosine +
  warmup scheduler keeps the LR ratio via one shared lambda.
- Composite checkpoint per save step: adapter/ + merger_model.safetensors
  + processor/ + phase2_training_manifest.json (completion marker written
  last) + trainer_state/optimizer/scheduler/rng from the standard Trainer
  helpers; rotation only ever sees complete dirs; save failures write a
  stable save_error.json and never fake completion.
- Resume: manifest validated field-by-field against the explicit request
  (base/processor identity, data checksums, LoRA config/targets, merger
  parameter table, optimizer group topology, augmentation seed/config,
  seq/image pixel settings); any conflict is a stable refusal; fresh starts
  over existing checkpoints are refused; incomplete dirs are never treated
  as successful.
- DeepSpeed/FSDP CLI passthrough is refused with a clear error: the Trainer
  DeepSpeed path replaces the four-group optimizer and FSDP would silently
  write sharded weights for the composite save.
- `--smoke-gradients` runs a warm-up step + verification backward (peft
  zero-initializes lora_B, so lora_A's first-step gradient is zero by design)
  and asserts non-zero gradients for every LoRA and merger parameter.
- `--image-min/max-pixels` are recorded in the manifest and applied only when
  the pinned processor declares them (5.14.1 Qwen3-VL does not; recorded as
  image_pixels_applied=false).

## Verification (M3 conda env: torch 2.13.0 / transformers 5.14.1 / peft 0.20.0)

- Scratch end-to-end run (fake model tree + fake processor + real peft, not
  committed): structure location, audit closure (28 LoRA + 18 merger params),
  four groups with no overlap, scheduler ratio 10.0 preserved, merger state
  save/read-back, adapter key verification, group-repeat determinism,
  one-step Trainer training with complete composite checkpoint, resume
  validation (data/lora conflicts detected), broken-dir refusal, gradient
  smoke check — all passed.
- `pytest tests/test_finetune_qwen3vl_phase2.py`: 19 passed (follow-up run
  with the committed test suite; also covers import-weight-freedom,
  visual same-name projection trap, epoch augmentation-seed stability,
  resume conflict refusal, composite checkpoint layout, gradient smoke).
- `pytest tests/test_qwen3vl_phase2_data.py tests/test_prepare_qwen3vl_phase2_sft.py`:
  52 passed.
- Architecture tests: 41 passed; 1 pre-existing failure
  (test_every_existing_python_file_matches_the_whitelist) caused by the
  user's own uncommitted files (scripts/qwen3vl_lora_cli.py,
  scripts/qwen3vl_lora_remote.py, scripts/prepare_vrsbench_phase2.py,
  tests/test_qwen3vl_lora_cli.py, tests/test_qwen3vl_lora_remote.py) which
  are not in the allowlist yet — not part of this task.

Bug fix included in the follow-up commit: `audit_trainable_parameters`
passed the vision root name as a plain string to `_under_any`, which
iterates over characters and silently skipped frozen-vision counting; now
wrapped as a single-element list (covered by the new test suite).

## Not Done / Risks

- `tests/test_finetune_qwen3vl_phase2.py` deferred per user instruction; the
  scratch verification is not a substitute for the committed test suite.
- Real Qwen3-VL-8B weight loading, real data and target-GPU smoke test are
  NOT run (no GPU / checkpoint locally); must be reported separately.
- image pixel bounds are not applied by the pinned processor (recorded, not
  silent).
