# Modification Note: Clean Google Earth Mentions from VRSBench Train Captions - 2026-08-06 14:39:17 CST

## Modification Time

2026-08-06 14:39:17 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Remove Google Earth source mentions from the VRSBench train captions so the
text stays remote-sensing relevant and fluent, using regex-based cleaning only.
The val split is intentionally not cleaned.

## Modified Files

- `scripts/clean_vrsbench_train_captions.py` (new)
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-08-06-agent-clean-vrsbench-train-captions.md` (this note)
- Local and remote generated file: `datasets/vrsbench/VRSBench_train_caption_cleaned.jsonl`

## Core Changes

- Added `scripts/clean_vrsbench_train_captions.py`, which reads
  `VRSBench_train_caption.jsonl` and writes `VRSBench_train_caption_cleaned.jsonl`.
- Ordered regex rules remove comma-wrapped and plain source phrases
  (`sourced/captured/provided/... from/by/via/through Google Earth`),
  copular constructions (`is from Google Earth`, `The source of the image is
  Google Earth`), simple prepositional phrases (`from/by/on/via/through Google
  Earth`), and attributive uses (`Google Earth satellite image` becomes
  `satellite image`); a fallback removes any remaining `Google Earth` token.
- Whitespace and punctuation are normalized after deletion so sentences stay
  fluent; only the `caption` field changes, and other record fields are
  byte-identical to the source.
- Generated file: 20,264 rows, 15,437 changed and 4,827 unchanged captions,
  zero leftover Google Earth mentions, zero empty captions.

## Whether the Canonical Sample Format Was Changed

No. The generated file is a derived annotation artifact, not the canonical
sample format.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No. The cleaned file is a new derived artifact; no evaluation metric, split,
or reference-answer reader was changed.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No. The change adds a standalone data-cleaning script; no repository interface
or evaluation logic was touched. Existing tests are unaffected.

## Whether .gitignore Was Updated

No. Generated files already live under `datasets/`, which is ignored.

## Validation Method

- `python3 -m compileall -q scripts/clean_vrsbench_train_captions.py` passed.
- `python3 scripts/clean_vrsbench_train_captions.py --help` passed.
- Ran the script and verified 20,264 rows, 15,437 changed / 4,827 unchanged,
  0 empty captions, and 0 leftover Google Earth mentions.
- Verified non-caption fields are identical between source and cleaned records.
- Regression-checked edge cases: `The image, sourced from GoogleEarth, displays`
  becomes `The image displays`; `provided by GF` (a non-Google-Earth source)
  is preserved; `bare earth` (a normal remote-sensing term) is preserved.
- Random-sample review of 30 changed captions confirmed fluent output.
- Uploaded the cleaned file to the remote VRSBench dataset root and verified
  `sha256sum` matches local `shasum -a 256`
  (`0c443111272ebf2fc0c6d9555e2f8d2d0b6539f2df26ec27ac0f7b3c3d217b0a`).

## Risks and Follow-up TODOs

- Cleaning is regex-based and derived from the observed phrase inventory;
  future captions with new phrasings should be re-audited before reuse.
- The cleaned file is not an official split; it should be treated as derived
  training data, not as a replacement for official VRSBench evaluation files.
- If the user later wants the cleaned captions to replace
  `VRSBench_train_caption.jsonl` directly, a small follow-up can rename or
  overwrite the file; the original generated file is currently preserved.
