# Modification Note: Filter ChangeChat-105k Localization Rows - 2026-08-07 15:28:00

## Modification Time

2026-08-07 15:28:00 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Provide a repeatable script that removes the 3x3 grid localization rows from
`changechat_105k_train.json` while keeping open QA rows that merely mention
"location", and writes the result to a new file without touching the source.

## Modified Files

- `scripts/filter_changechat_loc_rows.py` (new)
- `tests/test_filter_changechat_loc_rows.py` (new)
- `README.md`
- `DETAILS.md`

## Core Changes

- Added `scripts/filter_changechat_loc_rows.py` with `--input` and `--output`.
- Rows are removed only when the first human turn contains the canonical
  ChangeChat-105k localization template
  `Please indicate the locations where changes have occurred in the buildings
  and roads, using a 3x3 grid`.
- The script validates that image-pair coverage is unchanged and that no
  localization row remains in the output.
- Added unit tests covering exact filtering, preservation of location-themed
  open QA, missing-input failure, and malformed-row failure.

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

Yes, added `tests/test_filter_changechat_loc_rows.py`.

## Whether .gitignore Was Updated

No; the generated JSON stays under `datasets/`, which is already ignored.

## Validation Method

- `python -m compileall scripts/filter_changechat_loc_rows.py tests/test_filter_changechat_loc_rows.py`
- Ran the script against the real `changechat_105k_train.json` and verified
  87,935 input rows, 6,815 removed rows, 81,120 output rows, and unchanged
  image-pair coverage (6,815).
- pytest was not available in this environment; the three test functions were
  executed manually and passed.

## Risks and Follow-up TODOs

The filter relies on the canonical localization prompt; if the upstream
dataset rewords that prompt, the marker must be updated. `train_loc.json`
contains the same localization rows in rewritten-answer form and is not
touched by the default invocation; run the same script with `--input
changechat_105k_train_loc.json` if those rows should also be removed.
