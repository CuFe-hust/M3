# Modification Note: Allow CLI-Specified DeepSeek API Key - 2026-08-04 14:37:33 CST

## Modification Time

2026-08-04 14:37:33 CST

## Modifier

tRoy (791056216@qq.com)

## Modification Goal

Let the remote benchmark script accept the DeepSeek API key through the CLI while
keeping the existing `DEEPSEEK_API_KEY` environment variable as a fallback.

## Modified Files

- `scripts/eval_remote_benchmark.py`
- `eval/metrics.py`
- `tests/test_eval_remote_benchmark_cli.py`
- `tests/test_baseline_audit_report.py`
- `DETAILS.md`
- `docs/changes/2026-08-04-troy-eval-benchmark-cli-deepseek-api-key.md`

## Core Changes

1. Added `--deepseek-api-key` to the benchmark CLI. When provided, it is passed inside
   `deepseek_config["api_key"]` and takes precedence over `DEEPSEEK_API_KEY`.
2. `eval.metrics._deepseek_semantic_metrics` now resolves the key as
   `config.get("api_key") or DEEPSEEK_API_KEY`, preserving the existing environment-only
   behavior for all other callers such as `main.py`.
3. `summary.json` records only `deepseek_api_key_source` (`cli` or `environment`); the key
   value itself is never persisted in summary, metrics, or audit files.
4. Added tests for the new CLI argument and for config-key precedence over the environment
   variable.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

Yes, additively. New optional CLI argument `--deepseek-api-key`; existing arguments,
defaults, and the environment fallback are preserved.

## Whether Evaluation Was Affected

No metric, split, reference-answer, or inference behavior changed. Only the source of the
DeepSeek API key becomes configurable.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. CLI argument parsing and key-precedence tests were added/updated.

## Whether .gitignore Was Updated

No.

## Validation Method

`/opt/miniconda3/envs/m3/bin/python -m pytest -q -p no:cacheprovider` passed:
172 passed, 0 failed.

## Risks and Follow-up TODOs

- This explicitly deviates from the project convention that API keys come only from
  environment variables: a CLI key can appear in shell history and process listings.
  Users should prefer `DEEPSEEK_API_KEY` unless a CLI-specified key is required.
- No live DeepSeek API call was made during this modification.
