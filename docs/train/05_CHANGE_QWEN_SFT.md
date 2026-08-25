# ChangeAgent Qwen SFT

ChangeAgent SFT uses ordered `raw_full_t1`, `raw_full_t2` episodes, the frozen
production prompt, and canonical `ChangeInitialResult` assistant targets. Only
`change_caption` and `change_qa` enter this profile. Structured localization and
context-dependent multi-turn sources remain excluded.

## Current formal target contract

Formal rows use episode schema 2 and target contract
`change_initial_result_v2_no_legacy_evidence`. The exact `target.result` fields
are `agent_name`, `answer`, `boxes`, `evidence_items`, `geometry`, and `status`.
Formal training rejects the removed `evidence` field and the removed
`evidence_items[*].confidence` field. Runtime validators still accept these
fields when reading historical artifacts and omit them from canonical output.

The contract descriptor SHA-256 is
`1286e0a9b901bd58e2894c5e646900f3e05e4daa352a23509decc8f9601fc53e`;
the normalized contract identity SHA-256 is
`e22cecaa239934522612f8cb2a4f42a7b2a87aabaf4d206b1d8089771b67d277`.

## Final readiness

The complete Spark gate passed on qualified training-code commit
`a917c935369ab01b85aeb0cb2944377d80704834`:

```text
READY_FOR_FORMAL_POSTTRAINING=true
FORMAL_QWEN_SFT_STARTED=false
token_audit=PASS (108956/108956, p99=796, max=850, over_4096=0)
assistant_label_gate=PASS (empty_supervision=0, encoding_errors=0)
qwen_preflight=PASS (steps=0, manifest=null, LoRA targets=248)
gradient_smoke=PASS (cuda:0, BF16, max_seq_length=4096, batch=1, GAS=8)
full_test_gate=2525 passed
```

The approved settings are sequence length 4096, micro batch 1, gradient
accumulation 8, one epoch, save interval 1000, log interval 10, and save limit
4. Complete evidence and hashes are in
[`../training/CHANGEAGENT_FINAL_POSTTRAIN_READINESS_REPORT.md`](../training/CHANGEAGENT_FINAL_POSTTRAIN_READINESS_REPORT.md).

## Formal corpus

The only approved corpus is:

```text
/home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence
train SHA256      626958265486660783e7caab336a91113057c59d979b3f57b2e2e0e46f3a18a7
validation SHA256 d709c8e01f4ced952aaf198efb0652bd72014749913bb1f0fff4f13925428297
manifest SHA256   b3e891d18a6d564c843af38aac29f5a05f58f947bae38477b240d02d36786587
```

The old `sft_corpus/v1` has a `STALE_TARGET_CONTRACT_DO_NOT_TRAIN` marker and
must not be used for training or resume.

Build a new corpus from an explicit source specification into a nonexistent
directory:

```bash
python scripts/build_change_qwen_sft_corpus.py \
  --source-spec <source_spec.yaml> \
  --output-dir <new_corpus_dir> \
  --prompt-ref change_dual_path_v9
```

The builder fixes source spec, official split, pair registry, 511 exclusions,
source files, prompt, target contract, output hashes, and builder commit/tree.
It publishes atomically and never overwrites an existing directory.

Audit or compare a legacy corpus without modifying it:

```bash
python scripts/audit_change_sft_target_contract.py \
  --train <old>/train.jsonl --validation <old>/validation.jsonl \
  --manifest <old>/manifest.json --output <audit.json>

python scripts/compare_change_sft_corpora.py \
  --old-dir <old> --new-dir <rebuilt> --output <diff.json>
```

## Full token and assistant-label audit

```bash
python scripts/audit_change_qwen_sft_tokens.py \
  --model-id /home/user/models/Qwen3.5-9B \
  --model-adapter qwen3_5 \
  --train-file <corpus>/train.jsonl \
  --validation-manifest <corpus>/validation.jsonl \
  --data-manifest <corpus>/manifest.json \
  --image-root levir=<LEVIR_root> \
  --prompt-ref change_dual_path_v9 \
  --threshold 4096 --threshold 8192 \
  --local-files-only --output <audit.json>
```

This loads the real processor and every ordered image pair without model
weights or truncation. It validates the manifest/target identity, all rows and
images, ordered placeholders, assistant-only labels, and exact token lengths.

## Preflight and real gradient smoke

Use the same formal arguments for `--preflight-only` and
`--smoke-gradients-only`. The latter runs one real forward/backward pass and
exits before any optimizer step or checkpoint write:

```bash
python scripts/finetune_multimodal_sft.py \
  --model-id /home/user/models/Qwen3.5-9B \
  --model-adapter qwen3_5 --data-profile change_agent \
  --train-file <corpus>/train.jsonl \
  --validation-manifest <corpus>/validation.jsonl \
  --data-manifest <corpus>/manifest.json \
  --image-root levir=<LEVIR_root> --prompt-ref change_dual_path_v9 \
  --tuning-policy lora_plus_projector \
  --lora-rank 64 --lora-alpha 128 --lora-dropout 0.05 \
  --lora-lr 1e-4 --connector-lr 1e-5 --weight-decay 0.01 \
  --warmup-ratio 0.03 --max-grad-norm 1.0 \
  --dtype bfloat16 --device cuda:0 --batch-size 1 \
  --gradient-accumulation 8 --max-seq-length 4096 \
  --max-train-samples 1 --max-eval-samples 1 \
  --smoke-gradients-only --local-files-only \
  --output-dir <nonexistent_guard_dir>
```

A pass requires `steps=0`, `manifest=null`, finite loss and gradients, zero
frozen-parameter gradients, zero non-finite gradients, 248 LoRA targets, and
the sole full-train connector `model.visual.merger`.

## Training, resume, and export

Formal training uses `scripts/finetune_multimodal_sft.py` with
`--model-adapter qwen3_5 --data-profile change_agent`. Resume remains
fail-closed on checkpoint completion, base/processor identities, target/data
contract, parameter/training plans, optimizer, scheduler, trainer state, and
RNG state. Never resume a V2 run from a V1 corpus checkpoint.

`scripts/export_qwen3vl_phase2_checkpoint.py` is the compatibility export
entry point. Release requires LoRA merge validation, connector preservation,
processor save/reload, offline reload, a synthetic ordered two-image forward,
and a real Change fixture forward. Formal training and deployment were not
started by the readiness work.
