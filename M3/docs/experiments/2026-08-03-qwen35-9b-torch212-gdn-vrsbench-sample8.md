# Experiment Record: Qwen3.5-9B Torch 2.12 GDN VRSBench Sample 8

## Time

2026-08-03 20:21 +08:00

## Environment

- Server: Spark, NVIDIA GB10.
- Conda environment: `/home/user/miniconda3/envs/Cooper_for_qwen9b`.
- Python: 3.12.13.
- PyTorch: 2.12.1+cu130.
- Torchvision: 0.27.1+cu130.
- Transformers: 5.14.1.
- kernels: 0.15.2.
- CUDA reported by PyTorch: available on `cuda:0`.
- No resident model service was started.

## Model and Kernel

- Model: official Qwen3.5-9B checkpoint at `/home/user/models/Qwen3.5-9B`.
- Model weights were not modified.
- Kernel repository: `Atlas-Inference/gdn`.
- Fixed revision: `ef12347fc77d6ddf1cb72c0bd0af1c7d6cc69172`.
- Selected build: `torch212-cxx11-cu130-aarch64-linux`.
- Third-party kernel license metadata: `AGPL-3.0-only`.
- The repository snapshot remained unchanged. A project-side compatibility view adapts
  Transformers 5.14.1 state dictionaries to the kernel's state-zero cache reads.

## Dataset and Sample

- Dataset: VRSBench official validation general VQA.
- Dataset root: `/home/user/wwl/M3-main/datasets/vrsbench`.
- Sample index and question ID: 8.
- Image: `P0003_0004.png`.
- Question: `What object class is the bottom-most vehicle?`
- Reference answer: `small-vehicle`.

## Run Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/time -v \
  /home/user/miniconda3/envs/Cooper_for_qwen9b/bin/python \
  -m spacers_agent.cli --config configs/local.spark-router.yaml run-dataset \
  --dataset VRSBench \
  --root /home/user/wwl/M3-main/datasets/vrsbench \
  --split validation --task general_vqa \
  --run-id qwen35-torch212-dedup-sample8-v9-20260803 \
  --start-index 8 --max-samples 1 --sample-concurrency 1 \
  --evaluate --judge-policy none --fail-fast
```

## Results

- Dataset result: 1 succeeded, 0 partial, 0 failed.
- Candidate answer: `small-vehicle`.
- Exact match: true.
- Judge: not requested.
- Thinking template: disabled (`enable_thinking: false`).
- Spatial initial call: 32.725138 seconds, 1,323 prompt tokens, 340 completion tokens.
- Candidate review call: 47.706533 seconds, 1,082 prompt tokens, 512 completion tokens.
- Agent inference total: 80.436579 seconds.
- End-to-end wall time including model load and report generation: 161.85 seconds.
- Maximum resident set size: 19,243,764 KiB.
- Swap activity: none.

The final result contains four distinct evidence items: two road small vehicles, the
partially visible bottom-edge small vehicle, and the large vehicle in the left parking
lot. The overlapping initial/review boxes for the same two road vehicles were merged;
the final geometry reports `candidate_review_added: 2`, `candidate_count: 4`, and selects
the bottom-edge small vehicle by center y. The deterministic answer is correct.

## Performance Interpretation

The observed delay is not Qwen thinking output: thinking is explicitly disabled and no
thinking prefix was generated. The dominant warm-run cost is two long structured model
calls. The second call reached the configured 512 completion-token limit and took 47.7
seconds. The one-shot process also spent approximately 74 seconds loading 760 weight
entries. This setup is therefore not directly comparable to a warm persistent vLLM
Qwen-VL 8B endpoint returning a short answer in 1-2 seconds.

## Compatibility Findings

Transformers 5.14.1 required three scoped compatibility measures for this pinned kernel:

1. Resolve the exact Hub snapshot to a local path and set `use_local_kernel=True` so
   unrelated default mappings such as `kernels-community/activation` are not inherited.
2. Normalize the local mapping tuple produced during registration and restore the
   GatedDeltaNet forward signature hidden by `force_accelerate_hooks`.
3. Present state zero from dictionary-backed convolution/recurrent caches to the pinned
   kernel while delegating cache update methods unchanged.

The Qwen3.5 processor also returned a plain tensor dictionary rather than `BatchFeature`;
the client now supports both forms.

## Validation

- Targeted Qwen compatibility tests: 5 passed before the live run.
- Full Spark suite after the live run: 383 passed in 3.83 seconds.
- Runtime report: `outputs/runs/qwen35-torch212-dedup-sample8-v9-20260803/vrsbench_vqa.report/report.html`.

## Limitations

- This is a single-sample compatibility and correctness run, not a 200-sample score.
- The one-shot loading time is intentionally included because no resident service was authorized.
- The candidate-review prompt can still consume the entire 512-token budget; reducing
  its requested output is the clearest remaining latency optimization.
