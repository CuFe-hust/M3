# Modification Note: Fix Remote Benchmark Test Expectations - 2026-08-04 14:18:16 CST

## Modification Time

2026-08-04 14:18:16 CST

## Modifier

tRoy (791056216@qq.com)

## Modification Goal

Make the newly added remote benchmark tests match the actual loader/model-wrapper
behavior so the full pytest suite passes in the local `m3` conda environment.

## Modified Files

- `tests/test_benchmark_vlm.py`
- `tests/test_remote_benchmark_adapter.py`
- `docs/changes/2026-08-04-troy-fix-benchmark-test-expectations.md`

## Core Changes

1. `tests/test_benchmark_vlm.py`: the Qwen3-VL and InternVL loading-call tests asserted
   that `from_pretrained` received a `str`, but the wrapper passes a resolved `Path`.
   `transformers` accepts path-like inputs, so the assertions now compare against the
   resolved `Path`.
2. `tests/test_remote_benchmark_adapter.py`: the XLRS adapter test expected 3 samples,
   while the loader intentionally emits both `xlrs_grounding_condition` and
   `xlrs_grounding_fine` (train and test splits), for 4 samples total. The test now
   asserts 4 samples and validates the fine-split sample.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No. Test expectations only; no metric, split, reference-answer, or inference behavior
changed.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. Two existing tests were corrected to match the implemented contract; one new
assertion was added for the XLRS fine-grounding sample.

## Whether .gitignore Was Updated

No.

## Validation Method

`/opt/miniconda3/envs/m3/bin/python -m pytest -q -p no:cacheprovider` passed:
171 passed, 0 failed.

## Risks and Follow-up TODOs

- No product-code behavior changed; these are test-contract corrections only.
- The `m3` environment now has `requirements.txt` dependencies installed (including
  torch/transformers via accelerate); no real model inference was run.
