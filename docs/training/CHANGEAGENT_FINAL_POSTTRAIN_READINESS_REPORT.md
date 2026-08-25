# ChangeAgent final post-training readiness report

Recorded: 2026-08-25 R1

Host: `spark-b853` (NVIDIA GB10)

Branch: `change_agent`

Qualified implementation commit: `b321241868be24983c2aac814d52fc4399b458dc`

Qualified implementation tree: `8ddaeb573294ebc675b11dac546eec1f098a5cb9`

```text
QWEN_READY_FOR_FORMAL_TRAINING=true
CHANGEHEAD_READY_FOR_FORMAL_TRAINING=true
CHANGEHEAD_READY_FOR_RELEASE=false
READY_FOR_FORMAL_POSTTRAINING=true

FORMAL_QWEN_TRAINING_STARTED=false
FORMAL_CHANGEHEAD_TRAINING_STARTED=false
DEPLOYMENT_STARTED=false
```

This report separates training readiness from release and deployment readiness.
Both training chains may be started with their fixed commands. ChangeHead is not
release-ready because checkpoint-plus-test-cache inference to probability arrays
does not yet have an independent public entry point. `learned_change.enabled`
therefore remains false.

## R1 gate matrix

```text
HR1-01 deterministic train ordering implemented          PASS
HR1-02 ordering identity recorded in manifest             PASS
HR1-03 repeat build byte-identical                        PASS
HR1-04 old/new train episode set identical                PASS
HR1-05 old/new episode content identical by ID            PASS
HR1-06 validation byte-identical                          PASS
HR1-07 pair registry unchanged                            PASS
HR1-08 source summary unchanged                           PASS
HR1-09 target contract unchanged                          PASS
HR1-10 source-block bias removed                          PASS
HR1-11 generic Qwen exporter documented                   PASS
HR1-12 old Phase2 exporter removed from Qwen3.5 docs      PASS
HR1-13 rendered Change export fixture generated           PASS
HR1-14 synthetic export forward                           PASS
HR1-15 real Change fixture export forward                 PASS
HR1-16 Qwen launch command archival documented            PASS
HR1-17 Qwen tmux procedure documented                     PASS
HR1-18 resume command archival documented                 PASS
HR1-19 validation timing described correctly              PASS
HR1-20 periodic checkpoint terminology corrected          PASS
HR1-21 new token audit                                    PASS
HR1-22 new assistant-label audit                          PASS
HR1-23 new gradient smoke                                 PASS
HR1-24 LoRA target count=248                              PASS
HR1-25 frozen gradient leakage=0                          PASS
HR1-26 full suite                                         PASS
HR1-27 ChangeHead train readiness unchanged               PASS
HR1-28 ChangeHead release blocker documented              PASS
HR1-29 learned_change remains disabled                    PASS
HR1-30 formal training not started                        PASS
```

## Formal corpus ordering

Old source-block corpus:

```text
/home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence
train SHA256 626958265486660783e7caab336a91113057c59d979b3f57b2e2e0e46f3a18a7
```

Approved mixed corpus:

```text
/home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence_mixed_v1
train SHA256      6a52aacf6aa384f736fb51eef732fbb704b737c430a6ced4a9b4753ef4dcc67d
validation SHA256 d709c8e01f4ced952aaf198efb0652bd72014749913bb1f0fff4f13925428297
manifest SHA256   56fae448cdab494ea1f7647e2ebb3637a52305d35521e7c012eb30886b0ab61a
```

Ordering identity:

```text
policy=sha256_episode_id_v1
seed=1234
scope=train_only
validation_order=builder_source_order_v1
```

The old and new train files each contain 102,758 episodes. Missing IDs, extra
IDs, and content mismatches by `episode_id` are all zero. Validation and the
pair registry, ChangeChat row map, source summary, target contract descriptor,
and rejected rows are byte-identical. The repeat build matched train,
validation, and manifest bytes.

```text
old maximum contiguous same-source run=32274
new maximum contiguous same-source run=11
old maximum contiguous same-task run=32274
new maximum contiguous same-task run=22
```

## Corpus invariants

```text
train episodes=102758
validation episodes=6198
total episodes=108956
unique train pairs=6465
unique validation pairs=1242
train/validation pair intersection=0
test leakage=0
511 exclusion leakage=0
change_caption=70746
change_qa=38210
```

The target contract remains:

```text
schema_version=2
contract=change_initial_result_v2_no_legacy_evidence
descriptor SHA256=1286e0a9b901bd58e2894c5e646900f3e05e4daa352a23509decc8f9601fc53e
identity SHA256=e22cecaa239934522612f8cb2a4f42a7b2a87aabaf4d206b1d8089771b67d277
legacy evidence key count=0
legacy confidence key count=0
```

## Token and assistant-label audit

Artifacts:

```text
/home/user/cooper/posttrain_handbook_r1/qwen_token_audit.json
SHA256 67f47b0092ea48205eabeecedcbeacbab78e9e304dcc3d4df9dde10aa3f41cbd

/home/user/cooper/posttrain_handbook_r1/assistant_label_audit.json
SHA256 46a8716b5e22b68a524435acadf8ea879a03e3bbe6c518e14c21b205a1acc1d5
```

```text
status=PASS
untruncated=true
checked=108956
train=102758
validation=6198
min/p50/p95/p99/max=733/750/767/796/850
over_4096=0
over_8192=0
empty_supervision=0
encoding_errors=0
assistant supervised tokens min/p50/p95/p99/max=31/38/50/76/127
unique images=15414
image shape=256x256 + 256x256 for every episode
```

## Real CUDA gradient smoke

Artifact:
`/home/user/cooper/posttrain_handbook_r1/qwen_gradient_smoke.json`

SHA256: `db0e5e922a6193689c2feee4cb88965f8260c0b72f00dd20b118129b75a871f9`

```text
passed=true
device=cuda:0
dtype=bfloat16
max_seq_length=4096
batch=1
gradient_accumulation=8
steps=0
manifest=null
loss=2.196577310562134
trainable parameter tensors=502
parameter tensors with gradient=502
LoRA tensors with gradient=496
connector tensors with gradient=6
frozen tensors with gradient=0
nonfinite gradient tensors=0
gradient L2 norm=19.140424181525947
gradient max abs=0.2873556613922119
LoRA target modules=248
full-train connector=model.visual.merger
guard output directory=absent
```

The smoke exited after backward and before any optimizer step or checkpoint.

## Generic Qwen3.5 export smoke

The real smoke used the complete temporary Qwen3.5 Phase1 checkpoint at
`/home/user/cooper/posttrain_phase1_final_r1/qwen35_checkpoint` and the rendered
Change fixture.

```text
script=scripts/export_multimodal_sft_checkpoint.py
fixture=/home/user/cooper/posttrain_formal_prep/final/qwen_export_change_fixture.json
fixture SHA256=8bf9c53733abed96d4d122821a1fa1f5a154f6f1e181756ff05b64ae5d7aaaa8
selection_policy=lexical_min_episode_id_v1
offline_processor_reload=PASS
offline_model_reload=PASS
synthetic_two_image_forward=PASS
change_fixture_forward=PASS
```

Artifact:
`/home/user/cooper/posttrain_handbook_r1/generic_export_smoke.json`

SHA256: `01d7b28f4a5aad9bfad276276f6e7b8ee822c77c09288fa3ec2c75c340a1c463`

## Approved Qwen configuration

```text
model=/home/user/models/Qwen3.5-9B
model_adapter=qwen3_5
data_profile=change_agent
tuning_policy=lora_plus_projector
device=cuda:0
dtype=bfloat16
max_seq_length=4096
batch_size=1
gradient_accumulation=8
epochs=1
lora_rank=64
lora_alpha=128
lora_dropout=0.05
lora_lr=1e-4
connector_lr=1e-5
weight_decay=0.01
warmup_ratio=0.03
max_grad_norm=1.0
save_steps=1000
logging_steps=10
save_total_limit=4
seed=1234
```

During training, monitor train loss, learning rate, gradient norm, step time,
memory, and checkpoint saves. The trainer computes full validation `eval_loss`
once after all epochs. Periodic checkpoints are resumable checkpoints, not
validation-selected best checkpoints.

## Regression and evidence

```text
targeted tests before corpus rebuild=20 passed, 1 skipped
core regression tests=12 passed, 1 skipped
compileall=PASS
git diff --check=PASS
Spark full pytest after merging main `a6fec51`=2546 passed, 6 warnings in 34.94s
Spark full pytest log SHA256=70c7d524605c02e800aaf0f8075b63ddf3c5b78a547cc370813f44c9814dbf99
```

Primary evidence root:

```text
/home/user/cooper/posttrain_handbook_r1
```

Formal launch command template and export command are in the operation handbook.
The launch uses a single Bash array for both archival and execution, records
ordering identity, runs inside tmux, and writes a separate timestamped command
artifact for every resume.

## Final decision

Qwen3.5 SFT and ChangeHead may begin formal training with the fixed identities
and commands. No formal optimizer run or deployment was started. ChangeHead
remains blocked from release until independent checkpoint-plus-test-cache
inference, test evaluation, and release gates are complete.
