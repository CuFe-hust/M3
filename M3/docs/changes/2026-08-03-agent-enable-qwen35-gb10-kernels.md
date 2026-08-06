# Modification Note: Enable Qwen3.5 GB10 Kernels - 2026-08-03 11:54:48 +08:00

## Modification Time

2026-08-03 11:54:48 +08:00

## Modifier

Cooper

## Modification Goal

Allow a machine-local Qwen3.5 Transformers deployment to opt into the Hugging Face Hub kernel path and an explicit CUDA device map on NVIDIA GB10 without changing other Qwen defaults.

## Modified Files

- `spacers_agent/settings.py`
- `spacers_agent/clients/qwen_transformers.py`
- `configs/default.yaml`
- `tests/test_multiagent_vqa_pipeline.py`
- `DETAILS.md`
- `README.md`
- `pyproject.toml`

## Core Changes

Added the optional `models.qwen.use_kernels` setting. The Transformers client forwards `use_kernels=True` only for a Qwen3.5 checkpoint when explicitly configured and restricts kernelization to the official fixed-revision Gated DeltaNet mapping. The optional dependency is constrained to the `kernels` range accepted by the installed Transformers integration. Deployment documentation records the GB10-specific `use_kernels: true` and `device_map: cuda:0` settings.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

Yes. `QwenSettings` gained one backward-compatible optional field, and the Qwen3.5 loader can forward one optional kernel flag.

## Whether the Configuration Was Changed

Yes. `use_kernels` defaults to `false`.

## Whether Evaluation Was Affected

No metric, split, reference answer, or output format changed.

## Whether Deployment Was Affected

Yes. NVIDIA GB10 deployments can opt into the architecture-specific kernel and explicitly avoid CPU offload.

## Whether pytest Was Updated

Yes. The Qwen3.5 settings and model-load keyword behavior are covered.

## Whether .gitignore Was Updated

No. No new generated repository artifact or file type was introduced.

## Validation Method

Run the focused pytest files and a live GB10 same-image benchmark after installing the optional deployment dependency.

## Risks and Follow-up TODOs

The optional Hub kernel may require network access on first use and must support the installed Torch, CUDA, and SM121 environment. Keep `use_kernels` disabled on unsupported machines.
