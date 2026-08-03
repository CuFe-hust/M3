# Modification Note: Add .gitignore Rules for Downloaded Model Weights - 2026-08-03 16:33:54 CST

## Modification Time

2026-08-03 16:33:54 CST

## Modifier

tRoy (791056216@qq.com)

## Modification Goal

Prevent the downloaded HuggingFace model weights (models/InternVL3_5-8B/, ~16GB) from being accidentally committed to Git.

## Modified Files

- .gitignore

## Core Changes

- Added `models/InternVL3_5-8B/` to ignore the downloaded model weight directory.
- Added `*.safetensors` to ignore safetensors weight files globally.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

Yes, only `.gitignore` was updated; no runtime configuration fields were changed.

## Whether Evaluation Was Affected

No.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No (no code behavior changed).

## Whether .gitignore Was Updated

Yes.

## Validation Method

- `git check-ignore models/InternVL3_5-8B/model-00001-of-00004.safetensors` returns the path, confirming the weights are ignored.
- `git status --short` no longer lists the downloaded weights as untracked.

## Risks and Follow-up TODOs

- The `models/InternVL3_5-8B/` directory is download-only and must not be committed; `.gitignore` now protects against accidental commits.
- The full model download was completed and verified; a functional smoke test loading the model has not been performed.
