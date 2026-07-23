# Experiment Record: Qwen3-VL-4B Four-Dataset Smoke Test

## Time

2026-07-22 +08

## Environment

- GPU: NVIDIA GB10
- NVIDIA driver: 580.142
- PyTorch: 2.11.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- Conda environment: `snow-qwen`

## Model

Qwen3-VL-4B-Instruct, original local checkpoint, deterministic zero-shot inference
(`do_sample=False`). No training or weight modification was performed.

## Datasets and Tasks

- VRSBench: captioning, VQA, and visual grounding.
- MME RealWorld Remote Sensing: multiple-choice VQA.
- XLRS-Bench: English captioning, English visual grounding, and Lite VQA.
- LEVIR-CC: bi-temporal change captioning.

The smoke test used two samples per evaluation target: eight targets and 16 predictions in total.

## Run Command

```bash
python main.py \
  --config config/local.baseline.json \
  infer \
  --dataset all \
  --limit 2 \
  --overwrite
```

`config/local.baseline.json` and runtime outputs are intentionally ignored by Git.

## Smoke Test Results

| Dataset / target | Samples | Success | Failed | max_new_tokens | Status |
|---|---:|---:|---:|---:|---|
| VRSBench caption | 2 | 2 | 0 | 512 | Passed with truncation risk |
| VRSBench VQA | 2 | 2 | 0 | 64 | Passed |
| VRSBench grounding | 2 | 2 | 0 | 128 | Passed |
| MME Real RS | 2 | 2 | 0 | 64 | Passed |
| XLRS-Bench caption (English) | 2 | 2 | 0 | 768 | Passed |
| XLRS-Bench grounding (English) | 2 | 2 | 0 | 128 | Passed |
| XLRS-Bench VQA Lite | 2 | 2 | 0 | 64 | Passed |
| LEVIR-CC change caption | 2 | 2 | 0 | 512 | Passed |
| **Total** | **16** | **16** | **0** | - | **Passed** |

The process exited with status 0 after 8 minutes 28.82 seconds. All prediction texts were
non-empty. Eight JSONL prediction files, eight metadata files, and the MME official-format JSON
file were generated.

## Output Normalization Checks

- XLRS caption and grounding use the prompts provided by the released dataset rows.
- Raw Qwen3-VL grounding boxes are retained in the model-native normalized 0-1000 protocol.
- Canonical grounding boxes are stored in normalized 0-100 coordinates.
- XLRS grounding also records pixel-coordinate boxes derived from image width and height.
- No model weights, dataset contents, or evaluation metrics were changed.

## Known Issues

- One VRSBench caption reached the 512-token limit and ended mid-sentence.
- Caption predictions may contain unsupported or hallucinated details.
- The two sampled LEVIR-CC records were labeled as no-change, but the model described changes;
  the LEVIR-CC prompt and no-change handling require a separate calibration experiment.
- XLRS grounding was structurally valid but qualitatively weak on the two sampled records.
- This run validates the pipeline only. It does not report full-dataset accuracy, grounding IoU,
  BLEU, ROUGE-L, METEOR, or CIDEr.

## Artifacts

Runtime artifacts are kept under `outputs/baseline/final_regression_smoke/` and remain excluded
from Git. This document is the repository-safe summary of that run.
