# Experiment Record: Qwen3-VL LEVIR-CC Two-Stage Change Evidence, 20 Samples

## Time

2026-07-29 to 2026-07-30, Asia/Shanghai

## Dataset

The fixed first 20 samples of the official LEVIR-CC test split: 15 `changeflag=0` samples and
5 `changeflag=1` samples. This is a smoke comparison, not a full-dataset or leaderboard result.

## Model

`Qwen/Qwen3-VL-4B-Instruct`, unchanged local weights. No fine-tuning, LoRA, or weight update was used.

## Configuration File

An ignored Spark-local copy of `configs/default.yaml` selected the local Transformers backend,
offline checkpoint loading, `max_pixels=1003520`, sequential sample execution, and two Qwen calls
for each change-caption sample.

## Run Command

```bash
python -m spacers_agent.cli --config <ignored-local-config> run-dataset \
  --dataset LEVIR-CC --root <derived-read-only-20-sample-root> \
  --split test --task change_caption \
  --run-id levir-change-v4-contrastive-20 \
  --max-samples 20 --sample-concurrency 1 --fail-fast
```

## Metric Results

All 20 samples succeeded. The auxiliary `changeflag` diagnostic improved from 8/20 (40%) for
the direct baseline to 15/20 (75%) for the final Agent result. This diagnostic is not an official
LEVIR-CC caption metric.

| Metric | Baseline | Agent | Delta |
|---|---:|---:|---:|
| BLEU-1 | 0.080405 | 0.199262 | +0.118857 |
| BLEU-2 | 0.042347 | 0.074546 | +0.032199 |
| BLEU-3 | 0.019711 | 0.036371 | +0.016660 |
| BLEU-4 | 0.011729 | 0.000004 | -0.011725 |
| METEOR | 0.106136 | 0.172424 | +0.066288 |
| ROUGE-L | 0.109153 | 0.162520 | +0.053368 |
| CIDEr | 0.000002 | 0.038129 | +0.038128 |

## Resource Consumption

Inference used one NVIDIA GB10 with sequential sample execution and two Qwen calls per sample.
The final 20-sample run did not persist a directly comparable peak-memory measurement, so no new
peak-memory claim is made. Per-sample inference duration remains in each `agent_trace.json` and is
aggregated by the comparison report.

## Conclusion

The two-stage workflow improved six of seven persisted caption metrics and the auxiliary binary
diagnostic on this small fixed set. BLEU-4 decreased. Same-model verification can still reinforce
visual hallucinations, so this workflow is not yet justified for full-dataset replacement solely
from these 20 samples.

The baseline binary diagnostic was corrected from an earlier 11/20 estimate after per-sample
review showed that a broad phrase heuristic had treated local clauses such as “the rest remains
unchanged” as global no-change answers. Under the final explicit rule, only global no-change
statements count as no-change predictions.

## Conditional-Verification Follow-up

The fixed test-20 artifacts were replayed with a deterministic risk gate: verification runs only
for incomplete or uncertain analyses, or for positive change claims without geometry or a
localized T1/T2 comparison. The replay triggered verification for 10/20 samples and retained the
analysis result for 10/20. Compared with always-on two-stage execution, the measured saved-stage
estimate reduced model calls from 40 to 30, model latency from 188.41 to 120.22 seconds, and tokens
from 43,426 to 30,700. Its auxiliary diagnostic was 17/20 (85%). Because this was a replay on the
rule-development set, it is not used as the primary generalization claim.

A fresh real-model run then used 20 deterministic, evenly spaced LEVIR-CC validation samples
selected independently of the fixed test set: 10 changed and 10 no-change. All 20 samples
succeeded. Verification ran for 10 samples, producing 30 total model calls and an average of 1.5
calls per sample. The analysis-only auxiliary diagnostic was 9/20 (45%), while the conditional
final result was 13/20 (65%).

| Validation metric | Analysis only | Conditional final | Delta |
|---|---:|---:|---:|
| BLEU-1 | 0.155080 | 0.201031 | +0.045951 |
| BLEU-2 | 0.086192 | 0.101971 | +0.015780 |
| BLEU-3 | 0.000000370 | 0.000000407 | +0.000000037 |
| BLEU-4 | 0.0000000008 | 0.0000000008 | +0.00000000005 |
| METEOR | 0.101807 | 0.129774 | +0.027967 |
| ROUGE-L | 0.143912 | 0.159267 | +0.015355 |
| CIDEr | 0.030921 | 0.045877 | +0.014957 |

The fresh conditional run used 117.25 measured model seconds, 117.31 workflow seconds, and 30,653
tokens, excluding one-time model loading. This independent validation supports conditional review
as a better cost/effect tradeoff than always-on review, but 13/20 is still a small-sample result
and does not establish production or leaderboard accuracy.

## Reproducibility Statement

The source dataset was not modified. The derived 20-sample directory contains copied images and a
versioned adapter manifest. Raw analysis, verification, final results, request artifacts, traces,
canonical comparison JSONL, metric comparison JSON, and the report inputs were retained outside Git.
