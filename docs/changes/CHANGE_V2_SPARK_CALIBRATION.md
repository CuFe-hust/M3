# Change V2 NVIDIA GB10 calibration

Date: 2026-08-10

Execution host: NVIDIA GB10 (Spark)

Runtime: Python 3.12.13, PyTorch 2.12.1+cu130, Transformers 5.14.1
Code baseline recorded by the live report: `e71b1b4109b1c71e7894d888e69e8f6409c458b5`

Transformers 5.14.1 is the calibration-qualified runtime. Other Transformers
versions are not calibration-qualified; this is not a claim that they are
unsupported. Hidden-state token geometry continues to fail closed when a
two-dimensional grid cannot be proven.

The full machine-local report is stored under
`outputs/change_v2_validation/`. It is intentionally excluded from Git because
it contains derived image artifacts. No secret, absolute checkpoint path,
feature tensor, or probability tensor is persisted.

## Asset gate

- iSAID weight SHA-256:
  `f8e60686ec41160b5cbc494e8a3c1d28a92f7afdd41708c7b77e3d5793908b9a`
- OEM weight SHA-256:
  `d2141c79b2fc27ea5505db378b48e90e75e5ee06751df1c5b4028ef662fb2fab`
- Both files were materialized safetensors, not Git LFS pointers.
- The first load found that both migrated checkpoint directories lacked a
  processor asset. The canonical NVIDIA MiT-B2 `preprocessor_config.json` from
  upstream revision `3a609931044a9a83814802af9d861a47a1397636` was added to
  both directories. Offline loading then succeeded. The runtime continues to
  enforce `do_resize=False` for every tile.

## Measured selection

| Parameter | Selected | Evidence |
|---|---:|---|
| checkpoint | iSAID | OEM produced much larger semantic divergence, including on no-change pairs; OEM class names remain empty |
| feature stage | 1 | 128 channels, stride 8; localized-change ratio 18.91 with brightness median residual 0.00298 |
| tile size / overlap | 768 / 64 | lowest measured seam jump (0.000178), 0.193 s mosaic runtime, 1.19 GB peak allocated memory, no OOM |
| local match radius | 1 | no regression at 1/2 px; at 4 px median residual fell by 0.03365 and p95 by 0.07257 |
| PIF threshold k | 4.5 | selected from the joint matrix under the no-change FP ≤20% constraint |
| fusion weights | 0.25 / 0.50 / 0.25 | among candidates with no-change FP ≤20%, this retained the best visible-change recall proxy (66.7%) |

The 1024 tile had the smallest probability MAE versus a single 1280 tile, but
used about 2.00 GB and was slower. The 768/64 point was selected for seam,
throughput, and memory balance rather than maximum tile size.

## Acceptance observations

- Same-image: feature median 0, semantic map exactly 0, fused proposal count 0.
- Brightness, contrast, color cast, mild blur, and JPEG Q70: legacy proposal
  count was 6 for every shift; fused proposal count was 0 for every shift.
- Real pairs: 20 deterministic LEVIR-CC validation pairs covering five
  no-change, five building-add, five building-remove, two road, one vegetation,
  one water, and one other visible-change caption category.
- Joint selected configuration: no-change false-positive rate 20%, visible
  change recall proxy 66.7%, mean proposal count 2.0. These are proposal-level
  proxies, not official segmentation or caption metrics.
- PIF threshold fallback rate 0%; semantic fallback rate 0%; no NaN; no OOM.
- Each real sample includes raw/harmonized/PIF/low/feature/semantic/fused/mask/
  overlay artifacts, trace JSON, and raw crop evidence manifest suitable for
  Qwen input. Qwen itself was not invoked by this SegFormer calibration run.

## Limits

- Radius 1 only changed the result at 4 px for the selected stride-8 feature
  stage; 1/2 px shifts were unchanged rather than improved.
- The 20-pair caption-stratified sample is a calibration slice, not a claim of
  dataset-wide quality.
- Live artifacts were generated in the existing `std_test_env` because the
  supplied `Cooper_tryagents` environment had CUDA PyTorch/Transformers but no
  OpenCV. The initial real-model smoke in `Cooper_tryagents` passed after the
  processor asset was added.
