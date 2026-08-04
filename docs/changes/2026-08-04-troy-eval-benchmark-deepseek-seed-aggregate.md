# Modification Note: Remote Benchmark DeepSeek Switch, Seed Shuffle, and Combined Results - 2026-08-04 13:59:26 CST

## Modification Time

2026-08-04 13:59:26 CST

## Modifier

tRoy (791056216@qq.com)

## Modification Goal

Add an optional DeepSeek API evaluation switch to `scripts/eval_remote_benchmark.py`,
make `--seed` actually control the `--limit` subsample, and print/persist both per-dataset
and all-datasets-combined test results.

## Modified Files

- `scripts/eval_remote_benchmark.py`
- `tests/test_eval_remote_benchmark_cli.py`
- `DETAILS.md`
- `docs/changes/2026-08-04-troy-eval-benchmark-deepseek-seed-aggregate.md`

## Core Changes

1. Added `--deepseek-proxy`, `--deepseek-model`, and `--deepseek-base-url` CLI options.
   When enabled, each VQA result group is additionally evaluated through the existing
   `eval.metrics.evaluate_records(use_deepseek=True, ...)` path, with the API key read only
   from `DEEPSEEK_API_KEY`. The per-group DeepSeek audit list is persisted to
   `<result-stem>.report/deepseek_audit.jsonl` and passed into the audit report builder.
2. Replaced the `samples[:limit]` slicing with `_limit_samples()`, which performs a
   deterministic `random.Random(seed).shuffle()` per dataset before taking the first N
   samples. `summary.json` now records `seed_applied`.
3. Added per-dataset and overall combined results. Records from the same metric family
   (VQA / caption including change_caption / grounding) are aggregated with the unchanged
   `evaluate_records()` evaluator; sample ids are dataset-prefixed only inside the aggregate
   copies to prevent cross-dataset caption key collisions. Results are printed to stdout and
   stored in `summary.json` under `combined_results`.

## Whether the Canonical Sample Format Was Changed

No. Persisted result files and the canonical sample/prediction schemas are unchanged.

## Whether the Model Interface Was Changed

No. Qwen3-VL/InternVL loading and prediction interfaces are unchanged.

## Whether the Configuration Was Changed

Yes, additively. New optional CLI arguments `--deepseek-proxy`, `--deepseek-model`, and
`--deepseek-base-url`; existing arguments and defaults are preserved.

## Whether Evaluation Was Affected

Yes, additively. Existing per-label deterministic metrics are unchanged. With
`--deepseek-proxy`, the existing non-official text-only DeepSeek VQA semantic proxy metric
is added per VQA label; combined dataset/overall results are new aggregate views and do not
alter per-label metrics, references, or splits.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. `tests/test_eval_remote_benchmark_cli.py` now covers the new argument defaults, the
DeepSeek switch, seeded `--limit` sampling, and combined-result aggregation.

## Whether .gitignore Was Updated

No. No new generated file type was introduced; outputs remain under the ignored
`outputs/` directory.

## Validation Method

- `python3 -m compileall scripts/eval_remote_benchmark.py tests/test_eval_remote_benchmark_cli.py` passed.
- Inline assertion script covering argument parsing, seeded limit sampling, and combined
  aggregation passed with a fake evaluator.
- `pytest` could not be executed because no interpreter in the current environment has
  pytest installed.

## Risks and Follow-up TODOs

- The DeepSeek proxy requires a valid `DEEPSEEK_API_KEY`; a missing key fails visibly.
- No live DeepSeek API call was made during this modification.
- Combined caption metrics require `pycocoevalcap`, the same dependency as existing
  per-label caption metrics.
- Resume-skip semantics are unchanged: previously completed labels are not re-evaluated,
  so changing `--seed` does not affect labels that were resumed instead of recomputed.
