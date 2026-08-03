# Qwen3.5-9B compact spatial review — VRSBench first 20

## Environment

- Host: Spark, NVIDIA GB10
- Environment: `/home/user/miniconda3/envs/Cooper_for_qwen9b`
- Model: `/home/user/models/Qwen3.5-9B`
- Runtime: Torch 2.12.1+cu130, Transformers 5.14.1, kernels 0.15.2
- Run: `qwen35-torch212-compact-recovery-first20-v3-20260803`
- Scope: VRSBench validation `general_vqa`, start index 0, maximum 20, concurrency 1, no DeepSeek judge

## Result

- Attempted: 20
- Status: 18 succeeded, 1 partial, 1 failed
- Evaluated predictions: 19
- Exact match: 17/19 = 89.47%
- Model load: 81.85 s
- Recorded inference: 536.85 s
- Process wall time: 11:48.04
- Peak RSS: 19,243,940 KiB

The failed sample was ID 11. Its first answer was the correct `in rows`, but the model expanded twelve row boxes and prose until the 512-token global response limit truncated `evidence_items`; its model repair then returned the JSON Schema rather than a result. This failure is retained rather than hidden by changing the global response budget.

## Candidate-review performance

Three candidate reviews were issued. All three validated, none used a model repair, and their total completion output was 138 tokens.

- Mean: 4.553 s
- Median: 3.714 s
- Maximum: 6.703 s
- Observed complex review: 69 tokens / 6.703 s with `compact_candidate_sequence_recovered_locally`
- Observed simple review: 32 tokens / 3.241 s

The earlier sample-8 review used 512 completion tokens and 47.707 s. The compact contract therefore reduced the observed review latency by roughly 7–15x while preserving independently localized boxes.

## GPU utilization

The one-second GPU log contains 670 samples. Across active samples (`utilization.gpu >= 10%`):

- Mean utilization: 93.83%
- Median / p95 utilization: 94% / 94%
- Maximum utilization: 96%
- Mean power: 37.64 W
- p95 / maximum power: 37.86 W / 60.01 W
- Mean SM clock: 2486 MHz

Memory usage is reported as `N/A` by `nvidia-smi` on this GB10, so peak process RSS is retained as the available memory indicator. The high active utilization confirms that the remaining 30–40 s primary-call latency is dominated by model prefill/decode throughput, not Qwen thinking or an idle GPU.
