# ChangeAgent final post-training readiness report

Recorded at: 2026-08-25 17:16 +08:00  
Host: `spark-b853` (NVIDIA GB10)  
Branch: `change_agent`  
Qualified training-code commit: `10f94da9b6bcf453aad15b9cc033db489b23b4e1`

```text
READY_FOR_FORMAL_POSTTRAINING=true
CHANGEHEAD_READY=true
QWEN_SFT_READY=true
FORMAL_CHANGEHEAD_TRAINING_STARTED=false
FORMAL_QWEN_SFT_STARTED=false
DEPLOYMENT_STARTED=false
```

The repository, corpus, model, processor, CUDA gradient path, resume contract,
and host resources satisfy the pre-training gates. This report stops before
formal training: `/home/user/cooper/posttrain_runs` contains no formal run
directory, the gradient smoke performed zero optimizer steps, and its output
guard directory is absent.

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

## Identity gates

```text
base_weight_identity=7923a3e4111586dd012e38daea52ffc7315bf2836080784750a18a8cb7b44dca
base_identity_record_sha256=6cfc39307a74b0b6e666cc1df0c486e5828120155d4b71734adba423d0933ec6
train_sha256=4ed0ea5f432737c8ae7bf6b6b4b8e5bf52516a337d33b9fd6541aaa2bc008918
validation_sha256=89c66e64588180afe1832961e43243b0d595e8d9b0d61bca94d74e9e7131bf7d
manifest_sha256=59b4979ee658c000fa606135df4d586c8a5313a41c66b43ac08867e2c644a2cb
prompt_sha256=5ca66d02fafc722b9946734dd09a8fcdb26a15b15b451d7a104ed088eedad54f
processor_chat_template_sha256=12de9034d5269e8b1fe6012b3538374fcb3941bf93c95fc81b0cf5acf3f08ce4
processor_special_tokens_sha256=995cbff8f8573df19478af406e7c898a4d08af4ce468ecaea1593ff6967543aa
```

Base identity record:
`/home/user/cooper/posttrain_formal_prep/readiness/qwen_base_identity.json`.

## Complete token audit

Artifact:
`/home/user/cooper/posttrain_formal_prep/readiness/qwen_token_audit.json`  
SHA-256: `d1685471d1d44fa986e9511a74ab0a289126bcb4fe991f11007c5699d232a533`

```text
status=PASS
untruncated=true
checked=108956
train=102758
validation=6198
min=737
p50=754
p95=771
p99=800
max=854
over_4096=0
over_8192=0
empty_supervision=0
encoding_errors=0
unique_image_files=15414
image_shapes=256x256+256x256 for all 108956 episodes
visual_token_delta=126
recommended_max_seq_length=4096
```

The audit loads the real Qwen processor and both real ordered images for every
episode without loading model weights or truncating input. The first audit
exposed an all-zero native assistant mask from a chat template without a
`generation` block. The encoder now fails over to the verified per-turn mask;
the complete rerun has zero empty-supervision records.

## Real CUDA gradient smoke

Artifact:
`/home/user/cooper/posttrain_formal_prep/readiness/qwen_gradient_smoke.json`  
SHA-256: `0c9318f7a0dfbc206004176fc43576c61d4389a44237c7b20e62465acc20f593`

```text
passed=true
steps=0
manifest=null
loss=2.835801839828491
trainable_parameter_tensors=502
parameter_tensors_with_grad=502
lora_parameter_tensors_with_grad=496
connector_parameter_tensors_with_grad=6
frozen_parameter_tensors_with_grad=0
nonfinite_gradient_tensors=0
gradient_l2_norm=29.783926625774978
gradient_max_abs=0.4986368715763092
maximum_resident_set_size_kbytes=19364208
elapsed=1:36.43
smoke_output_guard_absent=true
```

The smoke uses the approved `cuda:0`, BF16, sequence length 4096, micro batch
1, and accumulation setting 8. It exits after forward/backward, before an
optimizer step or checkpoint write. An initial device-mismatch attempt was
retained at
`/home/user/cooper/posttrain_formal_prep/readiness/qwen_gradient_smoke_attempt1_device_mismatch.log`;
the batch-to-model-device fix was then tested and the full smoke rerun passed.

## Resume and regression gates

```text
resume_cli_propagation=PASS
checkpoint_completion_marker=PASS
base_identity_validation=PASS
processor_semantic_and_content_identity=PASS
data_and_training_plan_identity=PASS
optimizer_scheduler_trainer_rng_restore=PASS
spark_full_pytest=2482 passed, 6 warnings in 37.76s
spark_full_pytest_log_sha256=c3ce2c770dc5e288124f498a1af184b0b909e64bc3ca9b6e6adf6d8d08f92caf
```

Full test log:
`/home/user/cooper/posttrain_formal_prep/readiness/pytest-full-final.log`.
The warnings are pre-existing deprecation/serializer/scheduler-test warnings;
there are no test failures.

Training-readiness changes were committed separately for auditability:

```text
4914129 posttrain: add final Qwen readiness gates
a312b67 fix: fall back from empty assistant masks
a45ac8c fix: move multimodal batches to model device
10f94da posttrain: report resolved parameter plan
```

## Final decision

Both ChangeHead and Qwen SFT may be started with their fixed commands and
identities. No training or deployment is part of this gate. Keep
`configs/spark.yaml` at `learned_change.enabled: false` until training,
independent test evaluation, release gates, export, and joint validation are
complete. The missing public ChangeHead checkpoint-to-test-probability CLI is a
release-closure item, not a formal-training blocker.
