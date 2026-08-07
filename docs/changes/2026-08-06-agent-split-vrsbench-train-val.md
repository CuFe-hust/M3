# Modification Note: Split VRSBench Train Annotations into Train/Val and Rename Val to Test - 2026-08-06 14:59:07 CST

## Modification Time

2026-08-06 14:59:07 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Repartition the prepared VRSBench train annotations into a smaller train set and
a new validation set, and rename the current val annotations to test, so the
derived annotation files provide train/val/test splits for training workflows.

## Modified Files

- `scripts/split_vrsbench_train_val.py` (new)
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-08-06-agent-split-vrsbench-train-val.md` (this note)
- Local generated files under `datasets/vrsbench/` (git-ignored):
  `VRSBench_train_{caption,vqa}.jsonl` (rewritten),
  `VRSBench_val_{caption,vqa}.jsonl` (rewritten), and
  `VRSBench_test_{caption,vqa}.jsonl` (new, from the old val rows)

## Core Changes

- Added `scripts/split_vrsbench_train_val.py`, which reads
  `VRSBench_{train,val}_{caption,vqa}.jsonl`, collects the union of train
  `image_id` values from both the caption and VQA files, shuffles the sorted
  image IDs with a fixed seed (default 42), and moves the first
  `ceil(ratio * image_count)` images (default ratio 0.1) into the new val split.
- The split is image-level, so caption and VQA rows stay consistent per image
  and no image appears in more than one split; an overlap check runs before any
  file is written, so a conflicting input never leaves partially rewritten
  annotation files.
- Rewrote `VRSBench_train_{caption,vqa}.jsonl` with the remaining train images
  in source order; all fields are unchanged.
- Wrote the new `VRSBench_val_{caption,vqa}.jsonl` from the selected train
  images, updating only `split` to `val` and the split prefix of `id` from
  `vrsbench/train/` to `vrsbench/val/`; `image` paths still point to
  `Images_train/...` because the images physically remain there.
- Wrote `VRSBench_test_{caption,vqa}.jsonl` from the old val rows, updating only
  `split` to `test` and the split prefix of `id` from `vrsbench/val/` to
  `vrsbench/test/`; `image` paths still point to `Images_val/...`.
- The script refuses to run when a test output already exists unless `--force`
  is supplied, and it validates that input rows carry the expected split values
  before rewriting anything.
- Resulting row counts: caption 18,237 train / 2,027 val / 9,350 test; VQA
  31,054 train / 3,416 val / 16,715 test. Train images 20,264 total, of which
  2,027 (10.0%) moved to val and 18,237 (90.0%) remain in train.

## Whether the Canonical Sample Format Was Changed

No. These files are derived dataset annotations, not canonical samples.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No evaluation metric, reference-answer reader, or official evaluation file was
changed. The official VRSBench evaluation adapter reads `VRSBench_EVAL_vqa.json`
and is unaffected. The derived annotation splits were repartitioned as
requested: the old derived val rows are now named `test`, and the new derived
val rows are a deterministic subset of the old derived train rows.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No. The change adds a standalone data-preparation script and generated data
files; no repository interface, adapter, router, metric, or configuration
parsing logic was touched. Existing tests are unaffected.

## Whether .gitignore Was Updated

No. Generated files already live under `datasets/`, which is ignored.

## Validation Method

- `python3 -m compileall -q scripts/split_vrsbench_train_val.py` passed.
- `python3 scripts/split_vrsbench_train_val.py --help` passed.
- Ran the script on `datasets/vrsbench`; output reported 20,264 train images,
  2,027 new val images, 18,237 remaining train images, and zero overlap.
- Verified row counts (18,237/2,027/9,350 caption and 31,054/3,416/16,715 VQA),
  unique record IDs, `split` field values, and `id` prefixes for all six files.
- Verified zero image overlap between train/val/test and caption/VQA consistency
  within each split; the total row count (80,799) is unchanged before/after.
- Mini-fixture smoke tests passed: a disjoint train/val fixture produced the
  expected 18/2/4 counts, and an overlapping train/val fixture was refused
  before writing with all input hashes unchanged.
- Reconstructed the pre-split inputs from the final files in a temporary
  directory, reran the script with seed 42 and ratio 0.1, and confirmed the
  resulting row counts and image sets match the files in
  `datasets/vrsbench/` exactly.
- Recorded SHA-256 of the resulting files:
  - `VRSBench_train_caption.jsonl` `08ddb5bc55b0798fb46961bca368e9e1284055aa2a03566651f2763bf9a6cd2c`
  - `VRSBench_val_caption.jsonl` `650f3ce2df1abea4f8b95356f9c5fc825b392c4404f6a25781ea10235a8b0416`
  - `VRSBench_test_caption.jsonl` `26ed130e34df179c573f7c95b0910d6507e7508506948fa2af70f123b55a2fbd`
  - `VRSBench_train_vqa.jsonl` `e7c0922efbd1951effeba86a67f1433897bb79b17c3534194adf023382e2ca64`
  - `VRSBench_val_vqa.jsonl` `3dc4c069711783cac7cc3c73c6f1790f026d68b63e75028efdc5d154079e499e`
  - `VRSBench_test_vqa.jsonl` `884ec2ba69a2bd538260b935a2545565cc1f2fd5e1ac92d371c2bdf78a4bd84d`
- Full pytest could not be run because pytest is not installed in the local
  environment (`No module named pytest`).

## Risks and Follow-up TODOs

- The new val rows keep `Images_train/...` image paths because the images have
  not been moved; if the physical images are later relocated into an
  `Images_val/` tree, those paths must be updated.
- `VRSBench_{train,val}_vqa_sampled.jsonl` (generated previously, not present
  in this local folder) was not repartitioned; its `val` file is still derived
  from the old val rows that are now named `test`.
- The new train/val split is deterministic but derived data, not an official
  VRSBench split; downstream work should treat it as a training-time partition.
- Re-running the script on an already-split folder requires `--force`; the
  existing test files are then rebuilt from the current val files and the
  current train files are split again.
