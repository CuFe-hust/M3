# Experiment Record: Pre-Cutover Real-Dataset Runtime Baseline

## Time

2026-07-26 12:09:59 +08:00

## Dataset

- VRSBench official validation VQA release: 10 quantity, 5 grid-position, and 5 general-VQA samples selected by fixed source IDs.
- LEVIR-CC official test images/annotations: two source image pairs mapped to both `change_caption` and `change_qa` through ignored derived manifests.
- MME-RealWorld Remote Sensing: two source records mapped to `general_vqa` and `multiple_choice_vqa` through an ignored derived manifest.
- XLRS-Bench-lite Arrow release: two official overall-counting rows and two location-bearing rows, plus two derived caption runtime rows using official embedded images. The Lite release is VQA-only, so the caption rows use a fixed generic caption prompt and are not an official XLRS caption benchmark subset.

Original dataset files were read only. The LEVIR/MME derived manifests and copied small image selections, and the XLRS Arrow-derived manifest/images, are under ignored `outputs/pre-cutover-derived-inputs-v2/`.

## Model

`RecordingFakeQwen` and `RecordingFakeDeepSeek` only. Responses are keyed by `sample_id + response_model + request_id`; no live model, API, server, YOLO package, or weight was loaded.

This is a runtime, routing, resume, and artifact baseline. It is not a Qwen/DeepSeek quality experiment and its answers must not be reported as benchmark predictions.

## Configuration File

Programmatic `AppSettings` preserving repository defaults except for a run root below ignored `outputs/`, the audited dataset/derived-manifest root, and the deterministic fake model identifier.

## Run Command

The legacy harness invoked `spacers_agent.workflow.DatasetRunner` directly with injected recording clients and `sample_concurrency=1`. Run IDs:

- `pre-cutover-vrsbench-offline-20260726`
- `pre-cutover-levir-offline-20260726-v2`
- `pre-cutover-mme-offline-20260726-v2`
- `pre-cutover-xlrs-offline-20260726`

## Metric Results

No benchmark metric was calculated. Runtime outcomes:

| Dataset/task | Total | Succeeded | Partial | Failed | Fake Qwen calls |
|---|---:|---:|---:|---:|---:|
| VRSBench mixed `general_vqa` execution routes | 20 | 20 | 0 | 0 | 33 |
| LEVIR-CC `change_caption` | 2 | 2 | 0 | 0 | 2 |
| LEVIR-CC `change_qa` | 2 | 2 | 0 | 0 | 2 |
| MME-RealWorld `general_vqa` | 2 | 2 | 0 | 0 | 2 |
| MME-RealWorld `multiple_choice_vqa` | 2 | 2 | 0 | 0 | 2 |
| XLRS-Bench-lite `counting` | 2 | 2 | 0 | 0 | 52 |
| XLRS-Bench-lite `grounding` | 2 | 2 | 0 | 0 | 2 |
| XLRS-Bench-lite derived `caption` runtime rows | 2 | 0 | 0 | 2 | 0 |

Artifact-set legend:

- `C`: `agent_trace.json`, `counting_result.json`, `expert_result.json`, `routing_decision.json`, `sample.json`, `status.json`, `vqa_evaluation.json`
- `V`: `agent_trace.json`, `expert_result.json`, `routing_decision.json`, `sample.json`, `status.json`, `vqa_evaluation.json`
- `E`: `agent_trace.json`, `expert_result.json`, `routing_decision.json`, `sample.json`, `status.json`
- `N`: `counting_result.json`, `evaluation_record.json`, `routing_decision.json`, `sample.json`, `status.json`
- `F`: `routing_decision.json`, `sample.json`, `status.json`

### Fixed sample summary

| Sample ID | Task | Route | Agent | Status | Answer | Final/accepted | Warning codes | Requests | Artifacts |
|---|---|---|---|---|---|---|---|---:|---|
| 1 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 4 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 9 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 14 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 18 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 26 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 30 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 36 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 40 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 43 | general_vqa | counting | counting_agent | succeeded | 2 | 2/2 | COUNT_PROPOSAL_EVIDENCE_MISMATCH | 2 | C |
| 5 | general_vqa | spatial_relation | spatial_agent | succeeded | middle-middle | - | - | 2 | V |
| 16 | general_vqa | spatial_relation | spatial_agent | succeeded | top-left | - | - | 2 | V |
| 20 | general_vqa | spatial_relation | spatial_agent | succeeded | top-left | - | - | 2 | V |
| 31 | general_vqa | spatial_relation | spatial_agent | succeeded | top-left | - | - | 1 | V |
| 44 | general_vqa | spatial_relation | spatial_agent | succeeded | top-left | - | - | 1 | V |
| 10 | general_vqa | general_vqa | general_vqa_agent | succeeded | yes | - | - | 1 | V |
| 21 | general_vqa | general_vqa | general_vqa_agent | succeeded | yes | - | - | 1 | V |
| 22 | general_vqa | general_vqa | general_vqa_agent | succeeded | yes | - | - | 1 | V |
| 24 | general_vqa | general_vqa | general_vqa_agent | succeeded | yes | - | - | 1 | V |
| 28 | general_vqa | general_vqa | general_vqa_agent | succeeded | yes | - | - | 1 | V |
| levir-8148-caption | change_caption | change_caption | change_agent | succeeded | a building appeared | - | - | 1 | E |
| levir-8149-caption | change_caption | change_caption | change_agent | succeeded | a building appeared | - | - | 1 | E |
| levir-8148-qa | change_qa | change_qa | change_agent | succeeded | a building appeared | - | - | 1 | E |
| levir-8149-qa | change_qa | change_qa | change_agent | succeeded | a building appeared | - | - | 1 | E |
| mme-rs-0-vqa | general_vqa | general_vqa | general_vqa_agent | succeeded | yes | - | - | 1 | V |
| mme-rs-1-vqa | general_vqa | general_vqa | general_vqa_agent | succeeded | yes | - | - | 1 | V |
| mme-rs-0-mcq | multiple_choice_vqa | multiple_choice_vqa | general_vqa_agent | succeeded | yes | - | - | 1 | E |
| mme-rs-1-mcq | multiple_choice_vqa | multiple_choice_vqa | general_vqa_agent | succeeded | yes | - | - | 1 | E |
| xlrs-counting-00016-0 | counting | counting | counting_agent | succeeded | - | 25/25 | - | 26 | N |
| xlrs-counting-00016-1 | counting | counting | counting_agent | succeeded | - | 25/25 | - | 26 | N |
| xlrs-grounding-00000-18 | grounding | grounding | grounding_agent | succeeded | located | - | - | 1 | E |
| xlrs-grounding-00002-87 | grounding | grounding | grounding_agent | succeeded | located | - | - | 1 | E |
| xlrs-caption-00021-2 | caption | caption | caption_agent | failed (`KeyError`) | - | - | - | 0 | F |
| xlrs-caption-00021-3 | caption | caption | caption_agent | failed (`KeyError`) | - | - | - | 0 | F |

All short answers above are deterministic Fake outputs. Their full SHA256 values remain in each ignored run's `pre_cutover_summary.json`.

## Resource Consumption

- Network calls: `0`
- Live Qwen calls: `0`
- Live DeepSeek calls: `0`
- YOLO/ultralytics loads: `0`
- Deterministic Fake Qwen calls: `95`
- Sample concurrency: `1`

## Conclusion

The current legacy DatasetRunner can reproducibly process the fixed VRSBench, LEVIR-CC, MME-RealWorld, and XLRS-Bench-lite selections with deterministic clients while retaining route, status, answer, accepted count, warnings, request count, and artifact filenames. These records form the pre-cutover comparison target.

The XLRS counting invariant is visible in both samples: `final_count == accepted_count == 25`. The number is the deterministic Fake's one-point-per-tile runtime output, not a quality prediction and not the official ground-truth count. The two XLRS caption runtime rows consistently fail before a model request because the old workflow routes to `caption_agent` but has no `caption_expert` implementation. This visible legacy defect is frozen, not hidden or treated as a successful benchmark result.

## Reproducibility Statement

The committed `tests/parity/fixtures/` are static, contain no dataset images or Base64, and are regenerated only from the current legacy path with deterministic clients. The real-data run directories, copied sample images, and derived manifests remain ignored under `outputs/`. The XLRS Arrow extraction used the already cached CPython 3.10/`pyarrow` environment read-only; no dependency was installed or changed. Repeating the real-data baseline requires the same local dataset releases; no network fallback is permitted.
