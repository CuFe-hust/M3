# Experiment Record: VRSBench DeepSeek VQA Backfill

## Time

2026-07-22 23:15:03 +08:00

## Dataset

The persisted first 20 official VRSBench validation questions from run `vrsbench-qwen3vl-hybrid-count-v1-b24024d-20-20260722-r2`.

## Model

- Prediction model: previously persisted local Qwen3-VL-4B-Instruct outputs; Qwen was not loaded or called.
- Judge model: the configured DeepSeek text-only VQA Judge.

## Configuration File

Ignored Spark-local configuration `configs/local.spark-router.yaml`; the API key was exported from the ignored `.env` into the process environment and was not persisted.

## Run Command

```bash
set -a
source .env
set +a
python -m spacers_agent.cli --config configs/local.spark-router.yaml \
  judge-vqa-run --run-id vrsbench-qwen3vl-hybrid-count-v1-b24024d-20-20260722-r2
```

## Metric Results

- Judge execution: 20 succeeded, 0 failed, 0 skipped.
- Qwen calls: 0.
- Deterministic exact match: 15/20 (75%).
- DeepSeek semantic match proxy: 15/20 (75%).
- Strict-mismatch but semantically correct: none.
- Remaining semantic errors: IDs 2, 5, 12, 16, and 19.
- Strict-match/DeepSeek conflicts: none.

## Resource Consumption

- Aggregate DeepSeek request duration: 32.917 seconds.
- Prompt tokens: 6,242.
- Completion tokens: 3,346.
- Total tokens: 9,588.
- No Qwen model load or GPU inference occurred.

## Conclusion

The report now contains a successful DeepSeek result for every sample. The five strict mismatches are genuine semantic contradictions rather than harmless wording differences, so the semantic score remains 75%; no reference answer or Qwen prediction was altered.

## Reproducibility Statement

The backfill is tied to code commit `1b3316e`, the persisted run ID above, the versioned `deepseek_vqa_judge_v1` prompt, and the configured DeepSeek model. Successful records are cached and skipped by default on subsequent `judge-vqa-run` invocations unless `--force` is explicitly supplied.
