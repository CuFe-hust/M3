# ChangeAgent Qwen SFT V2 re-gate

Recorded: 2026-08-25  
Qualified training-code commit: `a917c935369ab01b85aeb0cb2944377d80704834`

## Gate result

```text
TOKEN_AUDIT_GATE=PASS
ASSISTANT_LABEL_GATE=PASS
QWEN_PREFLIGHT_GATE=PASS
GRADIENT_SMOKE_GATE=PASS
FULL_SUITE_GATE=PASS
READY_FOR_FORMAL_POSTTRAINING=true
FORMAL_QWEN_SFT_STARTED=false
```

Token and assistant-label audit:

```text
checked=108956
untruncated=true
min/p50/p95/p99/max=733/750/767/796/850
over_4096=0
over_8192=0
empty_supervision=0
encoding_errors=0
recommended_max_seq_length=4096
artifact_sha256=bf6c3ac71a39cf2c01f994351d8a3d719594a4c12ff8e9bc862e597f90c8e9a1
```

Qwen preflight-only loaded the real local 9B model on `cuda:0`, accepted the V2
manifest and rows, resolved 248 LoRA targets plus `model.visual.merger`, and
returned `steps=0`, `manifest=null`. Artifact SHA-256:
`c463b1a9594ec7e21031a4de8329181b9cbf6d37fd1dc35a221c63d2d681fe12`.

The real BF16 forward/backward smoke used sequence length 4096, micro batch 1,
and GAS 8. Loss was `2.894878625869751`; all 502 trainable tensors received
finite gradients, frozen-gradient and non-finite-gradient counts were both 0.
It performed zero optimizer steps and created no checkpoint or output guard.
Artifact SHA-256:
`26137a77cd6c91a03369f0b674134372d8b3cda7741a664c192d967ac7a10cb5`.

Spark full pytest passed: 2525 passed, 6 warnings in 27.64 seconds. Log SHA-256:
`03bb909b52c50a9f7d671de705ade80035e1bdcc08658a423e89e596f16990ad`.

Approved formal settings remain: LoRA rank/alpha/dropout 64/128/0.05, LoRA LR
`1e-4`, connector LR `1e-5`, weight decay `0.01`, warmup `0.03`, max grad norm
1.0, BF16, `cuda:0`, sequence 4096, batch 1, GAS 8, one epoch, save every 1000,
log every 10, keep 4, seed 1234.
