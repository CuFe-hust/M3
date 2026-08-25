# ChangeAgent Qwen SFT

ChangeAgent SFT uses canonical ordered `raw_full_t1`, `raw_full_t2` episodes,
the frozen production prompt, and `ChangeInitialResult` assistant targets.
Only `change_caption` and `change_qa` enter the current profile. Structured
localization and context-dependent multi-turn sources remain excluded.

## Formal corpus

Build the fail-closed mixed corpus from an explicit source specification:

```bash
python scripts/build_change_qwen_sft_corpus.py \
  --source-spec <source_spec.yaml> \
  --output-dir <corpus_dir> \
  --prompt-ref change_dual_path_v9
```

The output contract includes `train.jsonl`, `validation.jsonl`,
`manifest.json`, `pair_registry.jsonl`, `changechat_row_map.jsonl`,
`source_summary.json`, and `rejected.jsonl`. The manifest fixes official split,
pair registry, source files, exclusions, prompt, and output SHA-256 identities.

## No-weight token audit

Run the complete, untruncated audit before model training:

```bash
python scripts/audit_change_qwen_sft_tokens.py \
  --model-id /home/user/models/Qwen3.5-9B \
  --model-adapter qwen3_5 \
  --train-file <corpus_dir>/train.jsonl \
  --validation-manifest <corpus_dir>/validation.jsonl \
  --data-manifest <corpus_dir>/manifest.json \
  --image-root levir=<LEVIR_root> \
  --prompt-ref change_dual_path_v9 \
  --threshold 4096 \
  --threshold 8192 \
  --local-files-only \
  --output <audit.json>
```

The audit loads the processor but never model weights. It validates every
episode, every referenced image, ordered image placeholders, assistant-only
supervision, exact untruncated length, and visual-token expansion. Visual token
deltas are cached only after all image dimensions are resolved and recorded.

## Real gradient smoke

`--smoke-gradients-only` loads the real model, applies the selected tuning
policy, builds optimizer groups, runs one forward/backward pass, validates
finite loss and gradients, checks frozen-parameter leakage, prints a structured
result, and exits before any optimizer step or checkpoint write:

```bash
python scripts/finetune_multimodal_sft.py \
  --model-id /home/user/models/Qwen3.5-9B \
  --model-adapter qwen3_5 \
  --data-profile change_agent \
  --train-file <corpus_dir>/train.jsonl \
  --validation-manifest <corpus_dir>/validation.jsonl \
  --data-manifest <corpus_dir>/manifest.json \
  --image-root levir=<LEVIR_root> \
  --prompt-ref change_dual_path_v9 \
  --tuning-policy lora_plus_projector \
  --lora-rank 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --dtype bfloat16 \
  --device cuda:0 \
  --batch-size 1 \
  --max-seq-length <audited_length> \
  --max-train-samples 1 \
  --max-eval-samples 1 \
  --smoke-gradients-only \
  --local-files-only
```

The command must report `steps=0`, `manifest=null`, `passed=true`, zero
non-finite gradients, and zero frozen-parameter gradients. Qwen3.5 structure
discovery must continue to select exactly 248 LoRA target modules and
`model.visual.merger` as the full-train connector.

## Training and resume

Formal training uses `scripts/finetune_multimodal_sft.py` with
`--model-adapter qwen3_5 --data-profile change_agent`. The CLI passes
`--resume-from` into the generic training configuration. Resume remains
fail-closed on completion marker, base weights, processor semantic/content
identity, data contract, parameter/training plans, optimizer, scheduler,
trainer state, and RNG state.

Do not launch formal training until the corpus SHA checks, full token audit,
real gradient smoke, CLI resume tests, environment checks, and final readiness
record all pass. Runtime configuration is not a substitute for training CLI
device, sequence-length, batch, or accumulation settings.

## Export

`scripts/export_qwen3vl_phase2_checkpoint.py` remains the compatibility export
entry point for Qwen3.5 and Qwen3-VL composite checkpoints. Release requires
merge validation, processor save/reload, offline reload, synthetic ordered
two-image forward, and a real Change fixture forward.
