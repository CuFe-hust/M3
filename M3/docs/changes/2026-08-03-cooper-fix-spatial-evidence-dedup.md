# Modification Note: Fix Spatial Evidence Deduplication - 2026-08-03 14:11 CST

## Modification Time

2026-08-03 14:11:17 +08:00

## Modifier

Cooper

## Modification Goal

Prevent the first spatial pass and the independent candidate-review pass from retaining duplicate boxes when one pass labels a vehicle by positional role and the other uses the explicit VRSBench vehicle class.

## Modified Files

- `spacers_agent/agents/spatial/evidence_merge.py`
- `tests/agents/spatial/test_evidence_merge.py`
- `DETAILS.md`
- `docs/changes/2026-08-03-cooper-fix-spatial-evidence-dedup.md`

## Core Changes

- Treat top/bottom/left/right-most vehicle labels as generic vehicle roles during duplicate matching.
- Keep explicit `small-vehicle` and `large-vehicle` labels incompatible with one another.
- Add a smaller-box coverage and normalized centre-distance guard for shifted small-object boxes that narrowly miss the existing IoU threshold.
- Prefer an explicit vehicle-class label over a positional role label when replacing duplicate evidence.
- Add a regression case using the observed duplicate VRSBench coordinates, including the distinct partially visible bottom-edge vehicle.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

The evaluation metric and reference-answer logic are unchanged. Spatial evidence and report overlays can contain fewer duplicate boxes, so historical visual audit artifacts are not byte-for-byte identical.

## Whether Deployment Was Affected

Runtime spatial post-processing behavior is affected; model loading and export behavior are unchanged.

## Whether pytest Was Updated

Yes. Spatial evidence merge regression coverage was added.

## Whether .gitignore Was Updated

No. No new generated artifact type or local path was introduced.

## Validation Method

- `python -m pytest -q tests/agents/spatial/test_evidence_merge.py tests/test_vrsbench_vqa_geometry.py`
- Full offline test suite before deployment.
- Spark-side targeted test and non-resident VRSBench smoke after deployment.

## Risks and Follow-up TODOs

- The generic-role compatibility rule is intentionally limited to explicit positional vehicle labels.
- The coverage fallback requires both strong smaller-box coverage and close normalized centres to avoid merging adjacent vehicles.
