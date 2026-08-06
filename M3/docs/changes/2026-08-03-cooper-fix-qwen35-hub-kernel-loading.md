# Modification Note: Fix Qwen3.5 Hub Kernel Loading - 2026-08-03 15:02 CST

## Modification Time

2026-08-03 15:02:00 +08:00

## Modifier

Cooper

## Modification Goal

Load the pinned Qwen3.5 Gated DeltaNet Hub kernel through the Hub repository path instead of incorrectly interpreting `Atlas-Inference/gdn` as a local filesystem directory.

## Modified Files

- `spacers_agent/clients/qwen_transformers.py`
- `tests/test_multiagent_vqa_pipeline.py`
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-08-03-cooper-fix-qwen35-hub-kernel-loading.md`

## Core Changes

- Set the Qwen3.5 `KernelConfig` to use `LayerRepository` semantics.
- Preserve the single `Qwen3_5GatedDeltaNet` replacement and exact revision `ef12347fc77d6ddf1cb72c0bd0af1c7d6cc69172`.
- Document the one-time pinned kernel cache command required before offline execution.
- Extend the configuration regression test to assert the repository/class and revision.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No public model interface changed. The optional Qwen3.5 kernel loader now follows the installed Transformers/kernels Hub contract.

## Whether the Configuration Was Changed

No configuration key changed. Existing `models.qwen.use_kernels: true` now resolves the pinned Hub kernel correctly.

## Whether Evaluation Was Affected

No metric, split, reference answer, or evaluation output format changed.

## Whether Deployment Was Affected

Yes. An online deployment must cache the pinned kernel revision once before an offline run. No service is started.

## Whether pytest Was Updated

Yes.

## Whether .gitignore Was Updated

No. The external kernel cache is outside the repository and no new tracked artifact type was introduced.

## Validation Method

- Targeted Qwen kernel configuration pytest.
- Full Spark offline pytest suite.
- One pinned kernel cache operation followed by an offline Qwen3.5 sample run.

## Risks and Follow-up TODOs

- The pinned third-party kernel uses trusted remote code at a fixed commit and must be reviewed under deployment security policy.
- Torch 2.12.1/GB10 runtime compatibility remains to be established by the live non-resident benchmark.
