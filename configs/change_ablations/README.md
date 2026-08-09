# Change V2 offline ablations

These partial YAML files are merged over the built-in `AppSettings` defaults.
They do not add runtime switches or evaluation metrics.

| Variant | File | Comparison |
|---|---|---|
| A | `legacy.yaml` | deterministic `D_low` / V1 path |
| B | `low_semantic.yaml` | low-level + semantic; feature fusion weight is zero |
| C | `low_feature.yaml` | low-level + feature; semantic fusion weight is zero |
| D | `three_source.yaml` | low-level + feature + semantic |
| E | `pif_robust.yaml` | production robust PIF normalization and threshold settings |
| F0/F1 | `local_match_r0.yaml`, `local_match_r1.yaml` | exact versus one-cell local match |

Variant E's non-robust comparator intentionally remains an analysis/test
baseline. Production always uses robust alignment; no public runtime bypass is
introduced merely for an experiment.

For each run, analysis should collect:

- proposal count per sample, mean proposal area, and mean fused score;
- semantic fallback rate and PIF-threshold fallback rate;
- runtime per sample and Qwen image/crop counts per sample;
- SegFormer peak VRAM only for explicitly live GPU runs.

When a dataset supplies pixel masks, analysis may additionally collect proposal
recall at a declared IoU threshold, false-proposal area ratio, and pixel
precision/recall/F1. These are analysis outputs, not new public evaluation
contracts; change caption/QA remains the downstream business objective.
