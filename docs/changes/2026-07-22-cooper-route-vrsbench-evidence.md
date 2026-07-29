# Modification Note: Route VRSBench VQA with Retained Evidence - 2026-07-22 16:54:42 +08:00

## Modification Time

2026-07-22 16:54:42 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Route official VRSBench VQA subtypes through the appropriate existing Agent, retain labeled boxes or accepted points, apply reproducible geometry where supported, and expose the complete evidence path in the default HTML report.

## Modified Files

- `spacers_agent/dataset_adapters.py`, `routing.py`, `workflow.py`, `schemas.py`, and `vqa_geometry.py`
- `spacers_agent/cli.py`, `spacers_agent/commands/common.py`, and versioned prompt assets
- `spacers_agent/vqa_report.py` and `eval/audit_report.py`
- VRSBench routing, geometry, pipeline, and report tests
- `README.md` and `DETAILS.md`

## Core Changes

- Preserved the official VRSBench `type` and source-dataset fields in unified sample metadata.
- Added deterministic subtype routing: quantity to accepted-point counting; existence, position, category, and direction to spatial evidence; color to general VQA.
- Added labeled normalized evidence boxes or points while retaining the legacy `boxes` list.
- Added deterministic top/bottom box-center selection and three-by-three position derivation. Cardinal directions remain Qwen visual answers when no north metadata exists.
- Added report-only box/point overlays, prompt version, execution task, and geometry audit to the existing HTML report.

## Whether the Canonical Sample Format Was Changed

No. Existing canonical fields and `task="general_vqa"` are unchanged; official subtype values are stored in existing metadata.

## Whether the Model Interface Was Changed

Yes, additively. The local structured `ExpertResult` accepts labeled `evidence_items` and a geometry-audit object. Existing `boxes`, `answer`, and status fields remain compatible. The Qwen checkpoint, processor, tokenizer, and loading path were not changed.

## Whether the Configuration Was Changed

No configuration key was added or renamed. The README example raises `max_tokens` to 512 for longer structured evidence responses.

## Whether Evaluation Was Affected

The metric, split, source answers, exact-match normalization, and DeepSeek payload are unchanged. Candidate answers may now be derived from accepted points or supported deterministic box geometry, so new results are a new inference configuration and should not be mixed with the prior prompt-v1 run.

## Whether Deployment Was Affected

No. The Transformers-only Qwen loading path is preserved and vLLM is not introduced.

## Whether pytest Was Updated

Yes. Tests cover official-type routing, accepted-point quantity answers, box retention, top/bottom y selection, grid positions, direction limitations, and report evidence.

## Whether .gitignore Was Updated

No. Existing rules already ignore run outputs, local configurations, environment files, datasets, and Python caches.

## Validation Method

- `python -m pytest -q` — 97 passed.
- `python -m compileall -q spacers_agent eval` — passed.
- `git diff --check` — passed before documentation finalization.

## Risks and Follow-up TODOs

- Qwen remains responsible for producing correct evidence labels and geometry; deterministic logic cannot repair a missing or incorrect candidate box.
- Ordinary PNG files do not expose geographic north. Cardinal-direction answers retain the benchmark north-up assumption as a model-visible judgment and are marked as such in the geometry audit.
- A fresh Spark live run is required because prompt versions and execution routes changed; prior cached sample results must not be reused.
