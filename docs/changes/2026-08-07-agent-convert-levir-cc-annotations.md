# Modification Note: Convert LEVIR-CC Official Annotations to Readable JSONL - 2026-08-07 14:56:00

## Modification Time

2026-08-07 14:56:00 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Provide a small standalone script that flattens the official LEVIR-CC
`LevirCCcaptions.json` (COCO-style nested JSON) into one readable JSONL line
per image pair, so the A/B image paths, change flag, five captions, and
global sentence IDs can be inspected without nested parsing.

## Modified Files

- `scripts/convert_levir_cc_annotations.py` (new)
- `tests/test_convert_levir_cc_annotations.py` (new)
- `README.md`
- `DETAILS.md`

## Core Changes

- Added `scripts/convert_levir_cc_annotations.py` with `--root`,
  `--annotation`, `--output`, and `--include-tokens` options. Default input is
  `<root>/LevirCCcaptions.json` and default output is
  `<root>/LevirCCcaptions_readable.jsonl`.
- Each output row contains `imgid`, `split`, `filepath`, `filename`,
  derived `image_a` / `image_b` paths, `changeflag`, `captions` (raw text),
  and `sentids`; `tokens` is included only when `--include-tokens` is set.
- Added unit tests covering flattening, derived image paths, optional tokens,
  missing-input failure, and invalid-record failure.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes, added `tests/test_convert_levir_cc_annotations.py`.

## Whether .gitignore Was Updated

No; the generated JSONL lives under `datasets/`, which is already ignored.

## Validation Method

- `python -m compileall scripts/convert_levir_cc_annotations.py`
- `pytest -q tests/test_convert_levir_cc_annotations.py`
- Ran the script against the local official annotation and verified row
  counts, derived A/B paths, and sample captions.

## Risks and Follow-up TODOs

The derived A/B paths assume the standard LEVIR-CC layout
`images/<split>/{A,B}/<filename>`; if a different local layout is used, the
paths need remapping. The output omits `tokens` by default to keep the file
readable; use `--include-tokens` when lossless tokenization is required.
