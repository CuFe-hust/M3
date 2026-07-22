# Local Runbook

## Safety boundary

All default commands below are local and do not call Qwen, DeepSeek, an SSH tunnel, or a cloud API. A live model smoke test requires the user's explicit authorization after code completion and a key placed only in ignored `.env` or the process environment.

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
