# Modification Note: LEVIR-CC Harmonization and ChangeAgent Dual Path - 2026-08-05 12:16:15 +08:00

## Modification Time

2026-08-05 12:16:15 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Formalize the validated LEVIR-CC PIF/LAB harmonization as a typed, auditable module and make ChangeAgent use harmonized comparison evidence together with raw semantic evidence.

## Modified Files

Added change preprocessing schemas, validator, harmonizer, proposals, artifact orchestration, reviewer, prompt, tests, and offline evaluation script. Updated ChangeAgent, settings/default YAML, Composition Root wiring, prompt catalog, OpenCV/NumPy runtime dependencies, README, DETAILS, and Agent Runtime architecture documentation.

## Core Changes

- Validates exactly ordered T1/T2 files, EXIF orientation, RGB decoding, size, and alignment evidence without implicit stretching.
- Estimates one raw-derived PIF mask, maps both LAB distributions to their midpoint, limits blur to the sharper image, and gates unstable/degraded candidates.
- Generates deterministic difference maps, overlays, normalized boxes, and raw/harmonized crops.
- Supports `raw_only`, `harmonized_only`, and `dual_path`; all non-applied transforms fall back to raw evidence.
- Keeps one registered ChangeAgent and one budgeted structured Qwen request, followed by a warning-only rule reviewer.

## Whether the Canonical Sample Format Was Changed

No. `UnifiedSample` and temporal role ordering are unchanged.

## Whether the Model Interface Was Changed

The shared `VisionLanguageClient.complete_json` and `ExpertResult` interfaces are unchanged. The active ChangeAgent prompt/request evidence contract changed to `change-dual-path-v1`.

## Whether the Configuration Was Changed

Yes. Strict typed defaults were added under `agents.change`; older YAML files may omit the section.

## Whether Evaluation Was Affected

Existing benchmark metrics and splits were not changed. A separate offline LEVIR-CC harmonization evaluator was added.

## Whether Deployment Was Affected

No deployment or weight-loading path was changed. OpenCV/NumPy already exist in the documented project environment and no model is downloaded.

## Whether pytest Was Updated

Yes. Pair validation, harmonization/gating, difference proposals, ChangeAgent fallback/budget, prompt binding, and intentional parity-boundary coverage were added or updated.

## Whether .gitignore Was Updated

No additional update was required by this change; `outputs/` and `tmp/` are already ignored. An unrelated pre-existing `.gitignore` modification was preserved.

## Validation Method

`python -m compileall -q spacers_agent tests scripts` passed. The complete `python -m pytest -q` run passed all 439 tests. `python -m spacers_agent.cli --help` exited successfully. The offline 100-pair evaluation processed all pairs with 99 applied, one rejected/raw-fallback, and zero execution failures. See `docs/experiments/2026-08-05-levir-cc-harmonization-v1.md` for the exact command and observed metrics.

## Risks and Follow-up TODOs

The available local LEVIR-CC layout contains only paired images, so pixel-mask/group metrics could not be computed. Live Qwen A/B/C ablation was not run. Calibration thresholds require review on a versioned annotated training layout before being treated as permanent acceptance limits.
