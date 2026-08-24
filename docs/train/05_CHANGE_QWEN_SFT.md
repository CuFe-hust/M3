# ChangeAgent Qwen SFT

`scripts/prepare_change_qwen_sft.py` creates canonical ordered ChangeAgent
episodes for `change_caption` and `change_qa`.  It is offline and read-only:
it does not download LEVIR/ChangeChat, synthesize masks/proposals, or train.

Each episode contains authoritative `raw_full_t1`, then `raw_full_t2`, an
initial-stage request payload, and a `ChangeInitialResult` JSON target.  The
`semantic_pair_v1` contract contains only real raw pairs; runtime evidence is
reserved for future `runtime_initial_v1` captures.

Prepare a source with a frozen production prompt file:

```bash
python scripts/prepare_change_qwen_sft.py \
  --source-type levir_caption --source /data/LevirCCcaptions.json \
  --output-dir /data/change-qwen-sft --prompt-file prompts/change_v9.txt
```

Train through the thin wrapper.  It shares Phase2's Qwen loading, frozen
vision encoder, merger tuning, LLM LoRA, dual learning rates, composite
checkpoint, resume and exporter.  Pair augmentation is intentionally off.

```bash
python scripts/finetune_change_qwen_sft.py \
  --train_file /data/change-qwen-sft/train.jsonl \
  --eval_file /data/change-qwen-sft/validation.jsonl \
  --image_root levir=/data/LEVIR-CC \
  --change_prompt_file prompts/change_v9.txt --output_dir outputs/change-qwen
```

The resulting training manifest has `training_profile=change_agent`, the
ordered multi-image data contract, and prompt SHA-256.  It cannot resume a
generic Phase2 checkpoint, and the existing Phase2 exporter accepts both
profiles after its unchanged checksum and offline-reload gates pass.
# Compatibility note

The model-family-agnostic entry point is now:

```text
scripts/finetune_multimodal_sft.py --model-adapter auto --data-profile change_agent
scripts/export_multimodal_sft_checkpoint.py --model-adapter auto
```

`finetune_change_qwen_sft.py`, `finetune_qwen3vl_phase2.py`, and
`export_qwen3vl_phase2_checkpoint.py` remain legacy compatibility entry points
for existing checkpoints. New model families must be implemented under
`training/multimodal_sft/adapters/` and must not add model branches to the
generic trainer or exporter.
