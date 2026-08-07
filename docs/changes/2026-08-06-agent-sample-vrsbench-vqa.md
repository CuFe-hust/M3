# Modification Note: Sample VRSBench VQA Annotations by Original QA Type - 2026-08-06 14:12:58 CST

## Modification Time

2026-08-06 14:12:58 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Sample the prepared VRSBench train/val VQA annotations down to 10,000/5,000
rows, keeping the four original QA types (object existence, object category,
scene type, rural or urban) in their source proportions, and produce two new
JSONL files both locally and on the remote dataset root.

## Modified Files

- `scripts/sample_vrsbench_vqa.py` (new)
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-08-06-agent-sample-vrsbench-vqa.md` (this note)
- Local and remote generated files: `datasets/vrsbench/VRSBench_{train,val}_vqa_sampled.jsonl`

## Core Changes

- Added `scripts/sample_vrsbench_vqa.py`, which reads
  `VRSBench_{split}_vqa.jsonl`, groups rows by `source.original_type`, allocates
  the target total with the largest remainder method, samples each group with a
  fixed seed (default 42), and writes rows back in source order.
- Generated `VRSBench_train_vqa_sampled.jsonl` (10,000 rows) and
  `VRSBench_val_vqa_sampled.jsonl` (5,000 rows) locally and on the remote
  VRSBench dataset root; SHA-256 values match on both ends.
- Sampled train counts: object existence 4,856, object category 2,718, scene
  type 1,926, rural or urban 500. Sampled val counts: object existence 2,330,
  object category 1,626, scene type 956, rural or urban 88.

## Whether the Canonical Sample Format Was Changed

No. The generated files are derived dataset annotations, not the canonical
sample format.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No. The sampled files are new derived artifacts; no evaluation metric, split,
or reference-answer reader was changed.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No. The change adds a standalone data-preparation script; no repository
interface or evaluation logic was touched. Existing tests are unaffected.

## Whether .gitignore Was Updated

No. Generated files already live under `datasets/`, which is ignored.

## Validation Method

- `python3 -m compileall -q scripts/sample_vrsbench_vqa.py` passed.
- `python3 scripts/sample_vrsbench_vqa.py --help` passed.
- Ran the script and verified row counts (10,000/5,000), unique record IDs,
  and per-type counts against the allocation output.
- Uploaded both files to the remote VRSBench dataset root and verified
  `sha256sum` matches local `shasum -a 256` for both files.

## Risks and Follow-up TODOs

- Sampling is deterministic for a fixed seed but not a published evaluation
  split; downstream work should treat these files as derived data, not as a
  replacement for official VRSBench evaluation files.
- If a different target size, seed, or file layout is later required, the
  script arguments cover size and seed; a different layout would need a small
  script extension.
