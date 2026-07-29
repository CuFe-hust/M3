# Modification Note: Improve VRSBench Grounded Inference - 2026-07-22 20:18:00 +08:00

## Modification Time

2026-07-22 20:18:00 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Improve general VRSBench counting and spatial reasoning after the structured-tolerance run exposed systematic missed vehicles, anchored candidate review, untrusted degenerate boxes, coarse question-type normalization, and duplicate point evidence. DeepSeek behavior is intentionally unchanged.

## Modified Files

- `spacers_agent/schemas.py`, `vqa_geometry.py`, `workflow.py`, `counting.py`, `settings.py`, `cli.py`, and `commands/common.py`
- `configs/default.yaml`
- `prompts/count_tile_v4.md`, `missing_point_review_v3.md`, `spatial_v4.md`, and `spatial_candidate_review_v2.md`
- VRSBench geometry, workflow, and point-counting tests
- `README.md` and `DETAILS.md`

## Core Changes

- Scan fitting VRSBench images as one overview by default; split only for reported detail limits, density, or an independently unconfirmed zero.
- Reduce recursive halo by depth so child crops provide materially finer visual detail.
- Classify semantic question subtypes independently from coarse official types and send only reference-independent answer vocabularies to Qwen.
- Run candidate review independently without first-pass evidence, merge duplicate boxes and points, and prefer a real box over a duplicate point.
- Retain degenerate labeled evidence as a point instead of expanding it into a one-pixel box; expose evidence quality and repair severity so deterministic box geometry cannot trust fabricated extents.
- Keep incomplete evidence visible as a partial workflow result.

## Whether the Canonical Sample Format Was Changed

No. `UnifiedSample`, baseline canonical samples, and persisted canonical predictions are unchanged.

## Whether the Model Interface Was Changed

No model, processor, tokenizer, checkpoint, or Transformers loading interface changed. The structured `ExpertResult.geometry` audit gained additive fields.

## Whether the Configuration Was Changed

Yes. The default counting prompt version is `count-point-v4`, and `vrsbench_min_scan_depth` changes from `1` to `0`. Existing explicit local configurations remain valid.

## Whether Evaluation Was Affected

Inference behavior changes, so new predictions are not directly comparable as the same inference configuration. Reference answers, dataset split, exact-match metric, DeepSeek payload, and Judge role are unchanged.

## Whether Deployment Was Affected

No. The local Transformers inference path remains intact and no deployment dependency was added.

## Whether pytest Was Updated

Yes. Tests cover semantic subtype precedence, closed vocabulary, arrangement normalization, degenerate evidence handling, independent candidate review, point/box deduplication, unconfirmed-zero splitting, and depth-reduced halo.

## Whether .gitignore Was Updated

No. No new generated artifact, cache, dataset, weight, or secret-file type was introduced.

## Validation Method

- `python -m compileall -q spacers_agent tests`
- `python -m pytest -q`
- `git diff --check`

## Risks and Follow-up TODOs

- Local Mock tests validate contracts and geometry but cannot measure Qwen visual recall.
- The next Spark run must use a new run ID because prompt hashes and inference behavior changed.
- Cardinal orientation still depends on the benchmark raster convention; no georeferencing metadata is invented.
- The closed arrangement vocabulary is intentionally small and reference-independent; expand it only from audited training/development data, never from test answers.
