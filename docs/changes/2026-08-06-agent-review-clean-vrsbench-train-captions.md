# Modification Note: Review and Harden VRSBench Train Caption Cleaning - 2026-08-06 15:13:16 CST

## Modification Time

2026-08-06 15:13:16 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Clean the repartitioned VRSBench train captions by removing Google Earth source
mentions while keeping sentences fluent, leaving the val and test captions
untouched. Before running, review and harden the existing cleaning script logic
so it covers the actual phrase inventory and cannot silently emit broken or
incomplete sentences.

## Modified Files

- `scripts/clean_vrsbench_train_captions.py` (logic update)
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-08-06-agent-review-clean-vrsbench-train-captions.md` (this note)
- Local generated file: `datasets/vrsbench/VRSBench_train_caption_cleaned.jsonl`

## Core Changes

- Audited the existing regex cleaner against the repartitioned
  `VRSBench_train_caption.jsonl` (18,237 rows; 13,773 contain Google Earth
  mentions) and fixed the constructions that previously produced broken output,
  such as `The image appears to be.`, `The source of the image.`, and `as,`.
- Added rules for `as seen/viewed/shown/presented/provided ... on Google Earth`,
  hedged source phrases (`presumably/apparently/possibly/likely ... from Google
  Earth`), `originates/comes from Google Earth`, `visible/viewed on/from Google
  Earth`, and the `presented by Google Earth` verb form.
- Moved the `as ...` phrase rule before the plain verb-phrase rule so
  `as presented by Google Earth` is removed as a whole instead of leaving a
  dangling `as,`.
- Captions without any Google Earth mention now return byte-identical text, so
  cleaning never rewrites whitespace or punctuation in unrelated captions.
- After all rules run, source-only fragment sentences (for example `The image.`
  or `The image appears to be.`) are dropped, and the script performs a final
  self-check that fails when any Google Earth mention or source-only fragment
  sentence remains in the output.
- Generated `VRSBench_train_caption_cleaned.jsonl`: 18,237 rows, 13,773 changed
  and 4,464 unchanged, 0 empty captions, 0 leftover Google Earth mentions, and
  0 source-only fragment sentences.

## Whether the Canonical Sample Format Was Changed

No. The generated file is a derived annotation artifact, not the canonical
sample format.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No. Only the derived train caption file is cleaned; the val and test caption
files were not modified (SHA-256 unchanged), and no evaluation metric, split,
or reference-answer reader was touched.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No. The change touches a standalone data-preparation script and generated data
files; no repository interface, adapter, router, metric, or configuration
parsing logic was involved. Existing tests are unaffected.

## Whether .gitignore Was Updated

No. Generated files already live under `datasets/`, which is ignored.

## Validation Method

- `python3 -m compileall -q scripts/clean_vrsbench_train_captions.py` passed.
- Ran the script on a temporary copy first, then on `datasets/vrsbench`; both
  runs reported 18,237 rows, 13,773 changed / 4,464 unchanged, 0 empty, 0
  leftover Google Earth, and 0 fragment sentences.
- Verified every non-caption field is identical between
  `VRSBench_train_caption.jsonl` and the cleaned file.
- Verified SHA-256 of the val and test caption/VQA files is unchanged before and
  after cleaning (`650f3ce2...`, `26ed130e...`, `3dc4c069...`, `884ec2ba...`).
- Regression-checked targeted cases: `as seen on Google Earth`, `presumably
  from GoogleEarth`, `originates from Google Earth and`, `comes from
  GoogleEarth and`, `as presented by GoogleEarth`, `The image appears to be from
  GoogleEarth`, and `The source of the image is GoogleEarth`; all outputs are
  fluent with the source phrase removed.
- Random-sample review of 20 changed captions confirmed fluent output and no
  content loss beyond the Google Earth source mention.
- Full pytest could not be run because pytest is not installed in the local
  environment (`No module named pytest`).

## Risks and Follow-up TODOs

- Cleaning is regex-based and derived from the observed phrase inventory;
  captions with new Google Earth phrasings should be re-audited before reuse.
- Some source captions are already ungrammatical (for example participial
  fragments such as `This high-resolution image showing ...` or a caption
  ending in `roads and.`); cleaning removes the Google Earth mention but cannot
  fully repair the source sentence structure.
- The cleaned file is a derived training artifact, not an official VRSBench
  split; if a remote copy is needed, upload the file and verify its SHA-256
  (`1a5568f68cbcb3aee4f7beb3e062313ca9fac7c0942567a1af8f66261d2993ed`).
