# Modification Note: Isolate Qwen3.5 Kernel Mapping - 2026-08-03 17:30 CST

## Modification Time

2026-08-03 17:30:00 +08:00

## Modifier

Cooper

## Modification Goal

Keep the pinned Qwen3.5 Gated DeltaNet kernel while preventing Transformers from inheriting and resolving unrelated default kernels during an offline Spark run.

## Modified Files

- `spacers_agent/clients/qwen_transformers.py`
- `tests/test_multiagent_vqa_pipeline.py`
- `pyproject.toml`
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-08-03-cooper-isolate-qwen35-kernel-mapping.md`

## Core Changes

- Resolve `Atlas-Inference/gdn` at revision `ef12347fc77d6ddf1cb72c0bd0af1c7d6cc69172` with `snapshot_download`.
- Pass the resolved snapshot path through a local `KernelConfig` mapping.
- Set `use_local_kernel=True` so Transformers does not merge its global default kernel mapping.
- Normalize the local tuple produced by Transformers 5.14 registration back to the path string expected by its final compatibility conversion.
- Restore the fixed GDN forward ABI metadata hidden by Transformers 5.14.1's `force_accelerate_hooks` wrapper so kernels 0.15 can validate the replacement.
- Accept both `BatchFeature` and the plain tensor mapping returned by the Qwen3.5 processor when moving inputs to the model device.
- Adapt the pinned Atlas kernel's state-zero cache reads to the dictionary-backed cache ABI in Transformers 5.14.1 while delegating all cache updates unchanged.
- Add `huggingface-hub` as an explicit dependency of the GB10 optional runtime.
- Extend regression tests to verify repository type, exact revision, offline propagation, local path mapping, inheritance isolation, local metadata normalization, the exposed forward ABI, plain processor mappings, and cache state-zero views.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No public model interface changed. The internal optional kernel resolution path changed.

## Whether the Configuration Was Changed

No configuration key changed. `models.qwen.use_kernels: true` now resolves only the pinned GDN snapshot.

## Whether Evaluation Was Affected

No metric, split, reference answer, or output format changed.

## Whether Deployment Was Affected

Yes. The pinned snapshot must exist in the Hugging Face cache for an offline run. No resident service is introduced.

## Whether pytest Was Updated

Yes.

## Whether .gitignore Was Updated

No.

## Validation Method

- Targeted kernel configuration regression test.
- Full Spark pytest suite in `Cooper_for_qwen9b`.
- Non-resident offline VRSBench sample 8 run with the pinned kernel.

## Risks and Follow-up TODOs

- The pinned third-party kernel is AGPL-3.0-only and requires deployment-policy review.
- The live GB10 sample benchmark passed, but broader 10/200-sample quality and stability remain unverified.
- Candidate review reached the 512-token ceiling and remains the largest warm-inference latency target.
