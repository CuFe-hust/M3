# ChangeAgent final post-training readiness report

Recorded at: 2026-08-25 18:46 +08:00

Host: `spark-b853` (NVIDIA GB10)

Branch: `change_agent`

Qualified training-code commit: `a917c935369ab01b85aeb0cb2944377d80704834`

```text
READY_FOR_FORMAL_POSTTRAINING=true
CHANGEHEAD_READY=true
QWEN_SFT_READY=true
FORMAL_CHANGEHEAD_TRAINING_STARTED=false
FORMAL_QWEN_SFT_STARTED=false
DEPLOYMENT_STARTED=false
```

The repository, V2 corpus, target contract, base model, processor, CUDA
gradient path, resume contract, and host resources satisfy all pre-training
gates. The work stopped before formal training: `/home/user/cooper/posttrain_runs`
contains no formal run directory, preflight and gradient smoke both report
`steps=0` and `manifest=null`, and neither guard output directory exists.

## Final gate matrix

```text
TARGET_CONTRACT_CODE_GATE       PASS
TARGET_WRITER_GATE              PASS
TARGET_PROFILE_STRICT_GATE      PASS
TARGET_MANIFEST_IDENTITY_GATE   PASS
OLD_CORPUS_AUDIT_GATE           PASS
NONEMPTY_EVIDENCE_GATE          PASS (0 rows)
CORPUS_REBUILD_GATE             PASS
CORPUS_SEMANTIC_DIFF_GATE       PASS (0 unexpected diffs)
TARGET_FIELD_SCAN_GATE          PASS (evidence=0, confidence=0)
TOKEN_AUDIT_GATE                PASS
ASSISTANT_LABEL_GATE            PASS
QWEN_PREFLIGHT_GATE             PASS
GRADIENT_SMOKE_GATE             PASS
FULL_SUITE_GATE                 PASS (2525 passed)
```

## Approved Qwen SFT configuration

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
effective_batch_size=8
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

The resolved parameter plan contains 248 LoRA target modules and one fully
trained connector, `model.visual.merger`. The 32 decoder layers resolve to 8
full-attention and 24 linear-attention layers. The language base and vision
encoder remain frozen.

## V2 target and corpus identity

```text
target_contract_version=change_initial_result_v2_no_legacy_evidence
target_contract_identity_sha256=e22cecaa239934522612f8cb2a4f42a7b2a87aabaf4d206b1d8089771b67d277
target_contract_descriptor_sha256=1286e0a9b901bd58e2894c5e646900f3e05e4daa352a23509decc8f9601fc53e
builder_commit=a917c935369ab01b85aeb0cb2944377d80704834
builder_tree=68713f267371d7c345e309fa9cf6d603a139a1aa
builder_worktree_clean=true
source_spec_sha256=f619fe38b4f6a7ed6906e1283f9bb05416476f9171b826a127865735e42c59d6
pair_registry_sha256=a1ed5089db3d0bc875bf2622aace21257f3ed0dc38f8c65eaf849fc461096b54
exclusion_sha256=f6788a3df8f62a0bd31369966ea29128227dbf11c62af8e3686bb6254f5101ec
prompt_sha256=5ca66d02fafc722b9946734dd09a8fcdb26a15b15b451d7a104ed088eedad54f
train_sha256=626958265486660783e7caab336a91113057c59d979b3f57b2e2e0e46f3a18a7
validation_sha256=d709c8e01f4ced952aaf198efb0652bd72014749913bb1f0fff4f13925428297
manifest_sha256=b3e891d18a6d564c843af38aac29f5a05f58f947bae38477b240d02d36786587
```

Approved corpus:
`/home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence`.
The V1 directory is retained with `STALE_TARGET_CONTRACT_DO_NOT_TRAIN` and is
not an eligible training or resume source.

The V1 audit covered all 108,956 rows. Every legacy `evidence` value was empty,
non-empty evidence rows were 0, confidence fields were 0, schema/type errors
were 0, and other noncanonical differences were 0. The source rebuild matched
all episode IDs and every non-allowlisted field; missing/added episodes and
unexpected differences were 0. Pair registry, rejected rows, ChangeChat row
map, and source summary are byte-identical to V1.

## Base and processor identity

```text
base_weight_identity=7923a3e4111586dd012e38daea52ffc7315bf2836080784750a18a8cb7b44dca
processor_chat_template_sha256=12de9034d5269e8b1fe6012b3538374fcb3941bf93c95fc81b0cf5acf3f08ce4
processor_special_tokens_sha256=995cbff8f8573df19478af406e7c898a4d08af4ce468ecaea1593ff6967543aa
```

These unchanged identities were re-referenced; the target-contract migration
did not modify Qwen topology, base weights, processor files, ChangeHead assets,
pair/split/511 inputs, or Phase 1 checkpoint/export semantics.

## Complete token and assistant-label audit

Artifact:
`/home/user/cooper/posttrain_target_contract_v2/reports/qwen_token_audit.json`

SHA-256: `bf6c3ac71a39cf2c01f994351d8a3d719594a4c12ff8e9bc862e597f90c8e9a1`

```text
status=PASS
untruncated=true
checked=108956
train=102758
validation=6198
min=733
p50=750
p95=767
p99=796
max=850
over_4096=0
over_8192=0
empty_supervision=0
encoding_errors=0
unique_image_files=15414
image_shapes=256x256+256x256 for all 108956 episodes
visual_token_delta=126
recommended_max_seq_length=4096
```

The audit used the real Qwen processor and both real ordered images for every
episode. The verified assistant-mask fallback handles the model chat template's
missing generation block; no episode has empty supervision.

## Qwen preflight-only

Artifact:
`/home/user/cooper/posttrain_target_contract_v2/reports/qwen_preflight.json`

SHA-256: `c463b1a9594ec7e21031a4de8329181b9cbf6d37fd1dc35a221c63d2d681fe12`

```text
probe.passed=true
steps=0
manifest=null
lora_module_count=248
full_train_module_count=1
full_train_module_paths=[model.visual.merger]
guard_output_directory=absent
```

## Real CUDA gradient smoke

Artifact:
`/home/user/cooper/posttrain_target_contract_v2/reports/qwen_gradient_smoke.json`

SHA-256: `26137a77cd6c91a03369f0b674134372d8b3cda7741a664c192d967ac7a10cb5`

```text
passed=true
steps=0
manifest=null
loss=2.894878625869751
trainable_parameter_tensors=502
parameter_tensors_with_grad=502
lora_parameter_tensors_with_grad=496
connector_parameter_tensors_with_grad=6
frozen_parameter_tensors_with_grad=0
nonfinite_gradient_tensors=0
gradient_l2_norm=28.14313601373067
gradient_max_abs=0.5390625
maximum_resident_set_size_kbytes=19394656
smoke_output_guard_absent=true
```

The smoke used `cuda:0`, BF16, sequence length 4096, micro batch 1, and GAS 8.
It exited after backward and before optimizer step or checkpoint creation.

## Regression and evidence artifacts

```text
spark_full_pytest=2525 passed, 6 warnings in 27.64s
pytest_log_sha256=03bb909b52c50a9f7d671de705ade80035e1bdcc08658a423e89e596f16990ad
old_corpus_audit_sha256=d64f320f20937c28509f6301d962a39abe4d83f1a634cdaf9068bf5d2facecfd
v2_field_audit_sha256=efa023d0e2307826c03ff6ad1d4809e3e2398120337ba601cf0449849c8af127
corpus_diff_sha256=960f16780b23da9e2c563d89de20bf9ee082751d0b50fa6b913af155a7092db4
```

Runtime compatibility tests prove legacy `AgentResult.evidence` and
`VisualEvidence.confidence` inputs still validate and are absent from dumps.
Formal profile tests prove V1, legacy evidence, legacy confidence, missing
contract identity, and incorrect contract SHA all fail closed.

## Final decision

ChangeHead and Qwen SFT are ready to start using their fixed commands and
identities. No formal training or deployment was performed. Keep
`configs/spark.yaml` at `learned_change.enabled: false` until training,
independent evaluation, release gates, export, and joint validation are all
complete.
