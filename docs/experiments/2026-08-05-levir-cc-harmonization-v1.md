# Experiment Record: LEVIR-CC Harmonization v1

## Time

2026-08-05 12:00–12:20 +08:00

## Dataset

Local read-only `LEVIR-CC`, `train`, sorted intersection of `images/train/A` and `images/train/B`, offset 0, first 100 pairs. The inspected root contained no `labels/` directory or sample-level change manifest, so changed/unchanged grouping and pixel-mask metrics were not available.

## Model

No model was used. This was deterministic OpenCV/NumPy preprocessing only.

## Configuration File

Repository defaults in `configs/default.yaml`; algorithm `pif_lab_midpoint_v1`; no pre-existing calibration file.

## Run Command

```powershell
C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m scripts.evaluate_levir_harmonization --dataset LEVIR-CC --root C:\Users\TZDEZACR\Desktop\spacers-agent\dataset\LEVIR-CC --split train --offset 0 --max-pairs 100 --output-dir outputs/levir_cc_harmonization_v1 --write-calibration
```

## Metric Results

All 100 pairs were processed with zero execution failures. The first quality-gated run identified that unsafe optional blur rollback was incorrectly rejecting the entire color transform; this was corrected so the blur alone is rolled back. The final run accepted 99 pairs and rejected one unstable transform with raw fallback. Mean PIF coverage was 0.4454. Mean full-image MAD was 52.57 before and 38.72 for the evaluated harmonization candidate; mean PIF MAD was 41.51 before and 26.44 after. Mean proportion of pixels with grayscale difference greater than 20 was 0.7847 before and 0.6217 after. Full-image correlation was 0.4095 before and 0.4117 after.

The earlier scratch implementation produced full MAD 36.46 and PIF MAD 24.43 on the same 100 names. The formal v1 is intentionally more conservative because unsafe sharpness candidates are rolled back and unstable transforms fall back; the metric difference is recorded rather than presented as an improvement claim.

## Resource Consumption

CPU-only offline run; approximately five seconds wall-clock in the local `m3` environment. GPU memory and model calls: zero.

## Conclusion

The formal implementation reproduces the expected PIF coverage and substantial domain-difference suppression while making one unstable sample visible and recoverable. These image statistics do not establish ChangeAgent accuracy. Background suppression, change retention, small-change performance, and semantic false-positive/recall claims remain unmeasured because ground-truth masks/labels and a live model ablation were unavailable.

## Reproducibility Statement

The output directory contains `summary.json`, `metrics.csv`, `grouped_summary.json`, `failed_pairs.json`, and `calibration.json`; it is ignored by Git. The summary records all 100 sample IDs, split, algorithm version, command parameters, and code revision. A/B/C ChangeAgent ablation was not executed because it requires an authorized local Qwen checkpoint/service and applicable semantic ground truth; mock results were not substituted.
