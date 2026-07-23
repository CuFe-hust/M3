# Experiment Record: VRSBench Qwen Structured Tolerance 20

## Time

2026-07-22 19:18:14 +08:00

## Dataset

VRSBench official validation VQA release, first 20 samples selected by the unchanged adapter and CLI ordering.

## Model

Local `Qwen3-VL-4B-Instruct` checkpoint through the Transformers backend. DeepSeek was used only as the existing text-only answer Judge.

## Configuration File

Ignored Spark-local `configs/local.spark-router.yaml`, with local-only Qwen loading, `count-point-v3`, one minimum VRSBench scan depth, empty-tile review, and maximum transmitted crop side 768.

## Run Command

```bash
python -m spacers_agent.cli --config configs/local.spark-router.yaml run-dataset \
  --dataset VRSBench --root /home/user/下载/datasets/vrsbench \
  --split validation --task general_vqa \
  --run-id vrsbench-qwen3vl-structured-tolerance-bbb62c4-20-20260722 \
  --max-samples 20 --sample-concurrency 1 \
  --evaluate --judge-policy all
```

## Metric Results

- Dataset execution: 20 total, 12 succeeded, 8 partial, 0 failed, 0 skipped.
- Structured Qwen results: 20/20.
- DeepSeek Judge records: 20/20, with no Judge API failures.
- Exact-match accuracy: 11/20 (`0.55`).
- DeepSeek text-only semantic proxy: 11/20 (`0.55`); this is not the official VRSBench GPT metric.
- Previous run at commit `95dc091`: 16 structured results, 4 failed samples, exact match 8/16, and DeepSeek proxy 9/16.

## Resource Consumption

Samples ran sequentially with `sample_concurrency=1`. The run summary did not capture peak GPU memory or total wall-clock duration, so neither value is claimed here.

## Conclusion

The structured-output tolerance removed all four prior parsing/schema failures. The new report contains all 20 samples and raises the number of exact matches from 8 to 11 while preserving the dataset split, references, and metric implementation. Eight samples remain partial because their evidence-completeness safeguards did not declare a fully complete result; they are retained in the report and metric denominator.

## Reproducibility Statement

The code was run from commit `bbb62c49e5eec27b1ad4678a9410f5a72a9d5d41` on branch `Cooper_Technologies_Inc` with a fresh run ID and without `--resume`. The source dataset and model checkpoint were read only. Raw model responses, parsed results, validation metadata, Agent traces, Judge records, and the HTML report remain in the Spark run directory; datasets, weights, caches, logs, and report artifacts were not committed to Git.
