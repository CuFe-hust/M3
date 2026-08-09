# Counting C8 Offline Integration Gate

Date: 2026-08-10

Scope: `new_structure`, base HEAD `1bc6a8f` plus the uncommitted C7/C8 changes.
Mode: deterministic synthetic fixtures with fake model clients; no network, API, dataset
download, Qwen checkpoint load, YOLO inference, or GPU benchmark.

This report validates routing and evidence contracts. It is not a live model-quality claim.
Backend priority remains fixed and is not tuned from these results:

```text
Detection > Semantic Segmentation > QuantityProposal > QwenPoint
```

## Final behavior matrix

| Case | Planned/observed result | Contract result |
|---|---|---|
| YOLO supports and succeeds | YOLO primary | pass |
| YOLO unsupported, Seg supports | Segmentation primary | pass |
| YOLO/Seg unsupported, Quantity supports | QuantityProposal primary | pass |
| no specialist | QwenPoint primary | pass |
| YOLO unavailable/runtime error | next ordered expert | pass |
| Seg unavailable/runtime error | Quantity/Qwen | pass |
| Quantity unavailable/runtime error | QwenPoint | pass |
| specialist valid zero | retained unless explicit review | pass |
| all specialists fail | QwenPoint final | pass |
| QwenPoint fails | stable terminal error, no fake result | pass |

The matrix is covered by `test_backend_selector.py` and `test_executor.py`, including fixed
kind rank, same-kind stable ordering, multi-hop history, zero retain/override/review failure,
invalid contract termination, and path-free error traces.

Composition-level plans additionally prove that enabled `small-vehicle` produces
`detector_obb_csl_001 -> segmenter_mitb2_001 -> quantity_proposal -> qwen_point`, the default
YOLO-disabled configuration sends `swimming-pool` to SegFormer, and unknown `crane` goes to
QwenPoint. When the provided YOLO example is enabled, `swimming-pool` is a real detector class
and therefore correctly remains Detection-first; the generic “YOLO unsupported, Seg supports”
condition is not falsified by pretending that audited label is absent.

## Synthetic orchestration counters

The ten C8 contract scenarios (A-I, with both zero-retain and zero-override variants) produce
the following controlled final-backend distribution:

| Final expert kind | Usage count |
|---|---:|
| `yolo_obb` | 2 |
| `semantic_segmentation` | 4 |
| `quantity_proposal` | 2 |
| `qwen_point` | 2 |

- Final backend differs from primary, including explicit zero override: 4/10 (40%).
- Runtime-failure fallback only: 3/10 (30%).
- Zero-review override count: 1.
- Controlled Qwen calls: 5 (two Qwen final paths, one zero review, and two single-pass
  QuantityProposal paths). These are fake-client budget calls, not token or latency measures.
- Latency: not reported. Fake-client pytest timing is dominated by test/process overhead and
  would not represent model, GPU, provider, or deployment latency.

## SegFormer connected-component fixtures

| Fixture | Gold | Predicted | Exact | Absolute error | Relative error |
|---|---:|---:|---:|---:|---:|
| separated instances | 2 | 2 | 1 | 0 | 0 |
| touching instances | 2 | 1 | 0 | 1 | 0.333333 |
| tiny noise rejected | 0 | 0 | 1 | 0 | 0 |
| large connected region rejected | 0 | 0 | 1 | 0 | 0 |
| border object owned by core | 1 | 1 | 1 | 0 | 0 |
| tile-overlap duplicate merged | 1 | 1 | 1 | 0 | 0 |
| low-confidence region rejected | 0 | 0 | 1 | 0 | 0 |

Aggregate for the semantic synthetic set:

- exact accuracy: 6/7 = 85.7143%;
- count MAE: 1/7 = 0.142857;
- mean relative error using the project definition `abs_error / (gold + 1)`: 1/21 = 0.047619.

Per-backend live exact accuracy for YOLO, QuantityProposal, and QwenPoint is `N/A`: their C8
tests intentionally use deterministic fakes, and no real labelled dataset/model run was
authorized or available. Reporting 100% from a fake that returns its fixture answer would be
misleading.

Touching semantic regions may undercount instances because two objects can form one connected
component. SegFormer is semantic segmentation, not instance segmentation. C8 does not add
watershed or hidden instance splitting.

## Seam micro-fixtures

The eight deterministic/visual-review seam cases cover strong merge, clear separate,
ambiguous same/different/uncertain, reviewer exception, budget exhaustion, and no callback.
Four cases remain unresolved by safe policy (uncertain, exception, exhausted budget, and no
callback). Failure never defaults to merge. Strong and clearly separate pairs make zero visual
calls; YOLO OBB pairs are not processed twice.

## Capability and asset audit

- `ExpertCatalog` drives canonical labels, aliases, hints, kinds, priorities, assets, verified
  mappings, model labels, and counting modes.
- Runtime backend names are dataset-neutral: `detector_obb_csl_001` and
  `segmenter_mitb2_001`.
- iSAID `classes.json` is authoritative; placeholder `LABEL_N` values cannot override it.
- OEM remains disabled because it has no verified class map.
- SegFormer clients are lazy and reused by logical model id; assembly does not predict.
- SegFormer weights are Git LFS tracked. HEAD stores 134-byte pointer objects while a working
  tree may contain hydrated binaries. The YOLO asset is an external/local deployment asset.
- Default runtime does not download model assets.

## Anti-bloat and security audit

The required searches found no backend-kind guessing, VLM model selection, counting-layer
Transformers import, or new manager/router file. The only `vrsbench` hits in generic counting
code are a one-time settings-key migration and explicit rejection/forbidden-name markers; none
select an expert from `sample.dataset`.

Security keyword hits are declarations, API transport code, documentation examples, and
negative sanitization tests. Public trace/error tests verify that absolute paths, secrets,
Bearer values, prompts, masks, tensors, and base64 images are not persisted.

## Remaining live gate

Before making a model-quality or deployment-performance claim, run a separately versioned
benchmark with labelled samples, hydrated/verified weights, the target CUDA/ONNX providers,
Qwen call accounting, and wall-clock latency. Record the run manifest, config snapshot, prompt
hashes, logical model ids/revisions/hashes, per-backend metrics, fallback history, zero-review
overrides, unresolved seams, and touching-component examples. Such a benchmark may evaluate
the fixed policy but must not automatically reorder it.

## C8.1 hardening addendum

C8.1 restores the catalog-declared `vehicle` chain:

```text
detector_obb_csl_001
  -> segmenter_mitb2_001
  -> quantity_proposal
  -> qwen_point
```

It also declares `aircraft` as a two-label semantic capability. Different semantic labels are
componentized independently, including when their regions touch. QuantityProposal now supports
`vehicle`; after grounded localization, a parseable localizer answer is compared with accepted
points, while a disagreeing proposal remains a warning instead of forcing a partial result.

Generic fallback and zero-review switches now live in `CountingSettings`. `ExpertCatalog`
provides stable immutable public enumeration, and composition no longer reads private catalog
storage. YOLO composition validates logical model id, SHA256, priority, and resolved labels.
SegFormer runtime profiles may override physical deployment paths but not catalog identity or
verified class-map semantics.

The local package-data gate built `spacers_agent-0.1.0-py3-none-any.whl` with offline
`pip wheel --no-deps --no-build-isolation`. The wheel contained the expert catalog, verified
SegFormer `classes.json`/`config.json`/`preprocessor_config.json`, and prompt Markdown files;
it contained no `.safetensors`, `.onnx`, or `.pt` checkpoint. Installation into a temporary
`--system-site-packages` venv and model-free runtime assembly passed. The preferred
`python -m build` command was not available because the local `build` package is absent; no
network install was attempted.
