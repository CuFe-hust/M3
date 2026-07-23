# Local Runbook

## Safety boundary

Commands are local unless they explicitly invoke Qwen or DeepSeek. `judge-vqa-run` is a live text-only DeepSeek operation and requires explicit authorization plus a key placed only in ignored `.env` or the process environment.

## Local environment

Use the required Conda interpreter:

```powershell
C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m pytest -q
```

Copy the environment template only when endpoint metadata or a later authorized key is needed:

```powershell
Copy-Item .env.example .env
```

Place `DEEPSEEK_API_KEY` only in `.env`; never add it to source, YAML, test fixtures, artifacts, or Git.

## Offline checks

```powershell
C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m compileall -q spacers_agent tests
C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m pytest -q
```

Live markers are registered as `live_qwen` and `live_deepseek`. Do not run a live-marked test without explicit authorization, endpoint confirmation, and an environment-only key.

## Dataset audit

This command reads a supplied local directory and writes a separate report. It never modifies source dataset files.

```powershell
C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m spacers_agent.cli inspect-data `
  --root .\dataset `
  --output .\outputs\dataset_audit.json
```

## Render point evidence

After a counting result is available, render its persisted points and owner-core grid without calling a model:

```powershell
C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m spacers_agent.cli render-count `
  --image .\path\image.png `
  --result .\outputs\runs\example\counting_result.json `
  --output .\outputs\runs\example\counting_overlay.png
```

Green circles are accepted points; red circles with `!` are rejected points. Blue rectangles are owner cores. Inspect overlays before treating a result as scientifically credible.

## Summarize evaluations

`EvaluationRecord` JSONL can be summarized without a model call. Deterministic benchmark metrics and LLM-judged quality remain separate in the output.

```powershell
C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m spacers_agent.cli summarize-evaluations `
  --input .\outputs\evaluation_records.jsonl `
  --output .\outputs\evaluation_summary.json
```

## Current acceptance boundary

The local suite validates geometry, schema invariants, sequential Mock calls, repair/retry/cache paths, resume behavior, recursive parent replacement, routing budgets, DeepSeek non-visual scope, overlays, and evaluation summaries. It does not yet validate real datasets, live endpoints, model accuracy, or benchmark/ablation results.

## Dataset and Spark operations

`run-dataset` requires an explicit `spacers_adapter.json` in the selected `--root`. The adapter rejects unproven layouts and prints the observed field names; source datasets are never modified. Successful sample status files are reused by `--resume`, while partial and failed samples remain visible in `dataset_summary.json`.

For Spark, copy `scripts/server/env.example` to a protected server-only environment file and run `scripts/server/bootstrap.sh` before selecting vLLM parameters. `start_qwen_vllm.sh` binds the configured host (the example is loopback), `healthcheck.sh` calls `/v1/models`, and `run_dataset.sh`/`resume_dataset.sh` invoke the new CLI. No real server, tunnel, model download, dataset run, start/stop, or API smoke test was performed during local development.

`count-image` accepts `--target-spec`, `--resume`, `--force`, `--no-seam-verify`, and call-budget limits. Its final stdout line is a JSON summary and its exit code distinguishes data, Qwen, partial, Judge, and invariant failures. `run-dataset` supports comma-separated tasks, a stable SHA-256 sample-ID shard, `--max-samples 0` for no limit, explicit sample-ID filtering, start index, fail-fast, and append-only `predictions.jsonl`. The default sample concurrency is one; each image still processes tiles sequentially.

For VQA, `run-dataset --evaluate` defaults to judging all persisted answers. Export `.env` into the
process environment first; a missing `DEEPSEEK_API_KEY` is a visible error. To judge an existing run
without any Qwen call, use:

```bash
set -a
source .env
set +a
python -m spacers_agent.cli --config configs/local.spark.yaml judge-vqa-run --run-id <run-id>
```

The command skips successful existing Judge records, retries missing/failed ones, updates each
`vqa_evaluation.json` and `agent_trace.json`, and rebuilds the HTML report. `--force` explicitly
rejudges successful records.
