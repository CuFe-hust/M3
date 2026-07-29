# Experiment Record: VRSBench Singular Position Grounding V5

## Time

2026-07-22 23:39:14 +08:00

## Dataset

VRSBench official evaluation adapter, validation split. Targeted smoke used question IDs 5 and 16; the final regression used the same first 20 ordered samples as the preceding reports.

## Model

Local Qwen3-VL-4B-Instruct through Transformers on Spark. DeepSeek receives only question text, reference answers, candidate answers, and deterministic exact-match metadata.

## Configuration File

`configs/local.spark-router.yaml` on Spark; this ignored local configuration was not committed.

## Run Command

```bash
python -m spacers_agent.cli --config configs/local.spark-router.yaml run-dataset \
  --dataset VRSBench --root /home/user/下载/datasets/vrsbench --split validation \
  --task general_vqa --run-id vrsbench-qwen3vl-position-v5-f6ccb5c-2-20260722 \
  --sample-ids /tmp/cooper-position-v5-f6ccb5c-ids.txt --sample-concurrency 1 \
  --evaluate --judge-policy all

python -m spacers_agent.cli --config configs/local.spark-router.yaml run-dataset \
  --dataset VRSBench --root /home/user/下载/datasets/vrsbench --split validation \
  --task general_vqa --run-id vrsbench-qwen3vl-grid-scoped-b8f25c6-20-20260722 \
  --max-samples 20 --sample-concurrency 1 --evaluate --judge-policy all
```

## Metric Results

- Targeted run: 2 completed, 0 partial, 0 failed.
- Exact match: 2/2.
- DeepSeek score 1: 2/2.
- ID 5: `middle-left`, box `[0,450,100,600]`.
- ID 16: `top-right`, box `[840,270,920,310]`.
- Final 20-sample regression: 17/20 exact match and 17/20 DeepSeek score 1.
- Run status: 14 completed, 6 partial, 0 failed.
- Remaining errors: IDs 2, 12, and 19.
- Previous comparable report: 15/20 with errors at IDs 2, 5, 12, 16, and 19.
- Prompt audit: IDs 5 and 16 used `spatial-v5`; non-grid spatial questions used `spatial-v4`.

## Resource Consumption

- Qwen weight loading: approximately 49 seconds in the successful targeted rerun.
- Final 20-sample wall time: approximately 277 seconds, including Qwen loading, sequential inference, and DeepSeek judging.
- Calls were sequential (`sample_concurrency=1`).
- No vLLM service, detector, segmenter, or additional vision model was used.

## Conclusion

Both requested regressions were corrected by restoring physical-object grounding before deterministic grid classification. ID 5 required preserving the review's existing top-level box and attaching the vehicle class explicitly named by the question; no coordinates were synthesized. ID 16 was independently localized directly at the isolated upper-right small vehicle. Scoping the new prompts only to grid-position questions removed an intermediate ID 7 regression and improved the comparable 20-sample result from 15/20 to 17/20 without changing evaluation logic.

## Reproducibility Statement

The run used committed prompt and workflow code, the ignored Spark-local configuration, unchanged official reference answers, and persisted raw/parsed/timing artifacts. DeepSeek did not receive images or claim visual verification.
