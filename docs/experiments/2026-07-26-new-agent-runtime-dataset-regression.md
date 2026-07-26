# Experiment Record: New Agent Runtime Cutover — Full Regression Gate

## Time

2026-07-26 16:30 CST

## Dataset

- Frozen parity fixtures: 15 request/result/artifact scenarios (11 visual, 3 native counting, 1 VRSBench count)
- Architecture tests: 61
- Agent tests: 66
- Routing tests: 18
- Workflow tests: 12
- Parity tests: 18
- CLI tests: 5
- Compatibility tests: 2
- Full suite: 364

## Model

- Qwen: deterministic fake (RecordingFakeQwen) for parity tests
- DeepSeek: deterministic fake (RecordingFakeDeepSeek)
- No YOLO, no live network, no model loading

## Configuration File

`configs/default.yaml` (unchanged; YOLO disabled in bootstrap)

## Run Command

```bash
pytest -q
```

## Regression Results

| Category | Passed |
|---|---|
| architecture | 61 |
| agents | 66 |
| routing | 18 |
| workflows | 12 |
| parity | 18 |
| cli | 5 |
| compat | 2 |
| other | 182 |
| **total** | **364** |

**0 failed, 0 errors.**

Parity breakdown:
- non-counting visual Agents: 11 passed
- native point counting: 3 passed
- VRSBench proposal/localizer counting: 1 passed
- fixture and canonicalization coverage: 3 passed

## Resource Consumption

- Compile time: < 1s
- Full test suite: ~20s
- No GPU, no network

## Conclusion

The new Agent runtime (CountingAgent, non-counting agents, DatasetRunner, SampleRunner,
JudgeService, BackendSelector) matches the frozen request/result/artifact contracts for all
15 selected parity scenarios. The old modules (`counting.py`, `workflow.py`, `experts.py`)
are now thin import-compatibility layers. `run-dataset`, `resume-run`, and `count-image` use
`assemble_runtime()` with the new agent graph. YOLO remains disabled at the bootstrap gate.

## Requested Dataset Regression Fields

| Field | Offline fixture result | Real-dataset result |
|---|---|---|
| sample count | 15 frozen parity scenarios | Not executed |
| task breakdown | 11 visual; 3 native count; 1 VRSBench count | Not executed |
| old/new succeeded/partial/failed | Frozen result and status contracts compared where applicable | Not executed |
| answer exact match | Compared through frozen visual results | Not executed |
| count exact match / accepted-point match | Compared for all 4 counting scenarios | Not executed |
| warning code / route match | Compared through frozen artifact and routing fixtures | Not executed |
| request count / request fields match | Compared: count, response model, prompt version, request ID, image/message hash, target spec, Judge payload, fallback behavior | Not executed |
| artifact match | Compared after the approved volatile-field normalization | Not executed |

No real Qwen service, DeepSeek service, local dataset, baseline sample IDs, or Phase 1 run
configuration was supplied or authorized for this run. Therefore this record is an offline
request/result/artifact compatibility regression, not a real-dataset quality or performance result.
The real-dataset columns must be populated only after a separately authorized sequential run using
the exact Phase 1 IDs, configuration, Prompt versions, temperature, token limit, Judge policy, and
dataset version.

## Reproducibility Statement

All tests are offline-only with deterministic fake clients. No live Qwen/DeepSeek calls,
no model weights, no dataset downloads required. Run `pytest -q` from the repository root.
