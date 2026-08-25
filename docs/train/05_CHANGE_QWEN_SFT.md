# ChangeAgent Qwen SFT

ChangeAgent SFT consumes ordered `raw_full_t1`, `raw_full_t2` episodes, the
frozen `change_dual_path_v9` prompt, and canonical `ChangeInitialResult`
assistant targets. Only `change_caption` and `change_qa` enter this profile.

## Formal readiness

```text
QWEN_READY_FOR_FORMAL_TRAINING=true
FORMAL_QWEN_TRAINING_STARTED=false
DEPLOYMENT_READY=false
```

The qualified implementation commit is
`b321241868be24983c2aac814d52fc4399b458dc`. Its R1 gates passed on Spark:

```text
deterministic mixed train order=PASS
repeat build byte identity=PASS
old/new episode set and content-by-ID equality=PASS
validation byte identity=PASS
target contract audit=PASS
token/assistant-label audit=PASS (108956/108956)
real CUDA gradient smoke=PASS (steps=0, no checkpoint)
generic Qwen3.5 export and two-image forwards=PASS
compileall=PASS
full pytest=2531 passed, 6 warnings
```

## Formal corpus and target contract

The only approved corpus is:

```text
/home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence_mixed_v1
train SHA256      6a52aacf6aa384f736fb51eef732fbb704b737c430a6ced4a9b4753ef4dcc67d
validation SHA256 d709c8e01f4ced952aaf198efb0652bd72014749913bb1f0fff4f13925428297
manifest SHA256   56fae448cdab494ea1f7647e2ebb3637a52305d35521e7c012eb30886b0ab61a
```

The prior `v2_agent_result_no_evidence` directory is retained only for audit
and is marked `STALE_SOURCE_BLOCK_ORDER_DO_NOT_TRAIN`. It must not be used for
training or resume.

The manifest records the formal train ordering contract:

```text
policy=sha256_episode_id_v1
seed=1234
scope=train_only
validation_order=builder_source_order_v1
```

Sorting is applied after source ingestion, split enforcement, exclusion, and
deduplication. It changes only train row order. The old and new train sets both
contain 102,758 episodes, with identical episode IDs and identical content by
ID. Validation remains byte-identical. The largest contiguous same-source run
falls from 32,274 to 11.

Formal rows use schema 2 and target contract
`change_initial_result_v2_no_legacy_evidence`:

```text
descriptor SHA256 1286e0a9b901bd58e2894c5e646900f3e05e4daa352a23509decc8f9601fc53e
identity SHA256   e22cecaa239934522612f8cb2a4f42a7b2a87aabaf4d206b1d8089771b67d277
```

Legacy `evidence` and `evidence_items[*].confidence` fields are rejected from
formal training targets.

## Build and order audit

```bash
python scripts/build_change_qwen_sft_corpus.py \
  --source-spec /home/user/cooper/posttrain_formal_prep/pair_registry/source_spec.yaml \
  --output-dir <new_nonexistent_directory> \
  --prompt-ref change_dual_path_v9

python scripts/audit_change_sft_train_order.py \
  --old-dir /home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence \
  --new-dir <new_directory> \
  --output <audit.json>
```

The builder publishes atomically and does not overwrite an existing directory.

## Token, label, and gradient gates

The complete token audit uses the real processor and both real ordered images
without truncation. A pass requires 108,956 checked episodes, zero episodes over
4096 tokens, zero empty assistant supervision, and zero encoding errors.

Run the backward-only gate with the formal arguments plus:

```text
--max-train-samples 1
--max-eval-samples 1
--smoke-gradients-only
```

A pass requires `steps=0`, `manifest=null`, finite loss and gradients, 248 LoRA
targets, `model.visual.merger` gradients, zero frozen-parameter gradients, zero
non-finite gradients, and no output directory.

## Formal training and resume

Use `scripts/finetune_multimodal_sft.py` with
`--model-adapter qwen3_5 --data-profile change_agent`, the approved mixed corpus,
BF16, sequence length 4096, micro batch 1, gradient accumulation 8, one epoch,
save interval 1000, log interval 10, and save limit 4.

Run inside tmux. Construct the command once as a Bash array, archive that same
array with `printf '%q ' "${CMD[@]}"` to `launch_command.txt`, and execute
`"${CMD[@]}"`. Record `.ordering` from the corpus manifest in
`run_identity.txt`.

For resume, append `--resume-from <complete_checkpoint>` to the same fixed
argument set and write a new `resume_command_<timestamp>.txt`. Never overwrite
the original launch command. Resume fails closed on checkpoint completeness,
base/processor identity, data/target contract, parameter/training plan,
optimizer, scheduler, trainer state, and RNG state.

Training-time monitoring covers global step, train loss, learning rate,
gradient norm, step time, memory, and checkpoint saves. The current trainer
evaluates the complete validation set once after all epochs. Periodic
`checkpoint-N` directories are resumable checkpoints, not validation-selected
best checkpoints.

## Canonical export fixture and exporter

Build the deterministic rendered fixture from validation through
`ChangeAgentDataProfile.render_messages()`:

```bash
python scripts/build_change_export_fixture.py \
  --validation <corpus>/validation.jsonl \
  --manifest <corpus>/manifest.json \
  --image-root levir=<LEVIR_root> \
  --prompt-ref change_dual_path_v9 \
  --output <fixture.json>
```

The approved fixture is:

```text
/home/user/cooper/posttrain_formal_prep/final/qwen_export_change_fixture.json
SHA256 8bf9c53733abed96d4d122821a1fa1f5a154f6f1e181756ff05b64ae5d7aaaa8
selection_policy lexical_min_episode_id_v1
```

Export a complete Qwen3.5 checkpoint only through the generic exporter:

```bash
python scripts/export_multimodal_sft_checkpoint.py \
  --model-id /home/user/models/Qwen3.5-9B \
  --model-adapter qwen3_5 \
  --checkpoint-dir <complete_final_checkpoint> \
  --output-dir <new_export_directory> \
  --local-files-only \
  --verify-forward \
  --change-fixture /home/user/cooper/posttrain_formal_prep/final/qwen_export_change_fixture.json
```

Release requires offline processor/model reload, synthetic ordered two-image
forward, and real Change fixture forward to pass. Formal training and deployment
were not started by the readiness work.
