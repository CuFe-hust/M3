# M3

## Qwen3-VL-4B Zero-Shot Baseline

This repository provides a Colab-ready, zero-shot evaluation baseline built on
`Qwen/Qwen3-VL-4B-Instruct`. It does not fine-tune the model or change its
weights. The baseline evaluates each release independently and writes canonical
JSONL predictions plus separate metadata.

Evaluation scope:

- VRSBench: captioning, VQA, and visual grounding on the official `validation` split.
- MME Real RS: only the Remote Sensing subdomain of MME-RealWorld.
- XLRS-Bench: full English captioning and visual grounding releases; the VQA result uses
  the official Lite release and must be reported separately.
- LEVIR-CC: bi-temporal change captioning on the official test split.

The source releases are [VRSBench](https://huggingface.co/datasets/xiang709/VRSBench),
[MME-RealWorld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld),
[XLRS-Bench](https://huggingface.co/collections/initiacms/xlrs-bench), and
[LEVIR-CC](https://huggingface.co/datasets/lcybuaa/LEVIR-CC).

### Run in Colab

Enable a GPU runtime, then clone or upload this repository. Run the following cells from
the repository root:

```bash
pip install -r requirements.txt
cp config/baseline.example.json config/local.baseline.json
```

Edit `config/local.baseline.json` only to choose storage paths or supported model runtime settings.
The default paths keep downloaded data in `datasets/` and outputs in `outputs/`, both ignored by Git.
Do not put API keys in this file.

The default `report` settings generate a visual audit report for up to 200 samples per result.
Increase `report.max_samples` only when the additional image and HTML size is acceptable, or set
`report.enabled` to `false` to disable report artifacts for a particular local run.

For a checkpoint that is already present on a local server, set `model.id` to that external
directory and set `model.local_files_only` to `true`. This prevents accidental Hugging Face
network fallback while preserving the original Qwen3-VL loading and prediction interfaces.
Keep the server-specific absolute path only in the ignored `config/local.baseline.json` file.

Download the official data releases:

```bash
python main.py --config config/local.baseline.json download
```

Inspect each release before a full run. This prints the canonical sample derived from its
released fields and fails visibly if a source release changes its format:

```bash
python main.py --config config/local.baseline.json inspect --dataset vrsbench_vqa
python main.py --config config/local.baseline.json inspect --dataset mme_real_rs
python main.py --config config/local.baseline.json inspect --dataset xlrs_vqa_lite
python main.py --config config/local.baseline.json inspect --dataset levir_cc
```

Run a smoke test before the full evaluation. The `--limit` flag is only for smoke tests and
must be omitted from final results.

```bash
python main.py --config config/local.baseline.json infer --dataset all --limit 2
python main.py --config config/local.baseline.json infer --dataset all --overwrite
```

Each inference command prints the absolute path of its default HTML report. For a result named
`outputs/baseline/vrsbench_vqa.jsonl`, the report is saved at
`outputs/baseline/vrsbench_vqa.report/report.html`. It includes the captured source images,
questions/prompts, Qwen raw and final answers, references, exact-match comparison, and per-sample
inference duration. Each sample also records the actual Agent class, call entrypoint, route name,
task type, and whether a Router was used. The direct baseline is reported truthfully as
`models.qwen3vl.Qwen3VLBaseline` with `route=direct_baseline` and `router_used=false`; a future
workflow may provide `prediction.meta.agent_trace` to replace that fallback with its actual trace.
Images are content-addressed so repeated source images are stored only once.

Compute deterministic metrics for one saved result file:

```bash
python main.py --config config/local.baseline.json evaluate \
  --result outputs/baseline/mme_real_rs.jsonl
```

For VRSBench open-ended VQA, the optional DeepSeek semantic proxy requires the user to set
the key in the Colab session, never in a repository file:

```bash
export DEEPSEEK_API_KEY='set-this-in-the-Colab-session'
python main.py --config config/local.baseline.json evaluate \
  --result outputs/baseline/vrsbench_vqa.jsonl --deepseek-proxy
```

The resulting `deepseek_semantic_match_proxy` is not the official GPT-based VRSBench score;
report it as a separate proxy metric. For official oriented-box grounding metrics, run the
upstream VRSBench or XLRS-Bench evaluator on the canonical prediction file after converting
its documented output fields.

When `--deepseek-proxy` is used, the same report is regenerated with per-sample DeepSeek scores,
raw API responses, parsed results, duration, attempts, and token usage. The key is read only from
`DEEPSEEK_API_KEY` and is never written to the report. DeepSeek receives text and reference
answers only; it does not inspect the source image.

### Output Format

Each `outputs/*.jsonl` line contains:

```json
{
  "sample": {"id": "...", "task_type": "vqa", "prompt": "...", "answers": ["..."]},
  "prediction": {"id": "...", "task_type": "vqa", "text": "...", "answer": "..."}
}
```

`*.metadata.json` records the model settings, timestamp, completed sample count, and any
dataset-scope qualification needed for a report. It also records model-load and inference timing.
The sibling `*.report/` directory contains `report.html`, `samples.csv`, a bounded `samples.jsonl`
visual subset, deduplicated images, and optional `deepseek_audit.jsonl`. These report artifacts do
not change the canonical prediction JSONL or the metric JSON format.

For MME Real RS, inference also writes `mme_real_rs.official.json`, preserving each official
record and replacing only its `Output` field. It can be passed directly to the upstream
MME-RealWorld evaluator.

## Local Multi-Agent Foundation (Phase 1)

The existing baseline remains unchanged. The additive local foundation creates reproducible
run artifacts without contacting Qwen or DeepSeek:

```bash
python -m spacers_agent.cli --help
python -m spacers_agent.cli health qwen
python -m spacers_agent.cli run-init --run-id local-foundation-smoke
```

Copy `.env.example` to the ignored `.env` file only when local endpoint metadata needs an
override. API keys are never included in run manifests, configuration snapshots, or events.
The `health` command in this phase only displays configured metadata; it does not make a
network request.

## Structured Client Development (Phase 2)

The project now includes an async OpenAI-compatible Qwen client and an offline Mock client.
The default test suite injects local fake completions; it does not contact an endpoint. A live
Qwen call remains a separately authorized action and requires `QWEN_API_KEY` only in the
process environment or ignored `.env` file.

## Read-Only Dataset Audit (Phase 3)

Inspect a local dataset layout before implementing an Adapter. The command never changes source
dataset files and writes its result to a separate report path:

```bash
C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m spacers_agent.cli inspect-data \
  --root ./dataset \
  --output outputs/dataset_audit.json
```

## Point Counting Orchestration (Phase 4)

`spacers_agent.counting.PointCountingOrchestrator` is an additive, async workflow for one
normalized image and a caller-supplied `CountTargetSpec`. It sends one crop at a time through
an injected structured client, uses non-overlapping owner cores with halo context, converts only
validated `0..999` local points to global pixels, and derives `final_count` solely from accepted
global points. It writes each tile's geometry, parsed response, conversion report, and checkpoint
below the selected run directory; a matching successful request hash is reused on resume.

The active counting prompt is versioned in `prompts/count_tile_v3.md`; the superseded v2 prompt
remains available for experiment reproducibility. The v3 contract expands known vehicle aliases,
requires a systematic owner-core scan before returning zero, and supports an optional versioned
empty-tile review. Point-counting tests use local Mock clients only. No Qwen, DeepSeek, SSH tunnel,
server, or cloud request is made by this module unless a caller explicitly constructs and invokes
a live client after authorization.

## Sparse Multi-Agent Routing (Phase 5)

`spacers_agent.routing.TaskRouter` uses fixed rule routes for declared tasks and does not make a
model call in that case. Only `route_unknown` uses an injected, text-only client; it requires and
consumes a `CallBudget` entry before the call. `CountingExpert` is a thin display wrapper around
the existing point pipeline: complete answers are derived from accepted global points, while partial
results explicitly report completed tiles and remain non-final.

Every prompt is an independent versioned file in `prompts/` and `run-init` snapshots all of them.
The included Phase 5 tests use Mock clients only; no live routing, visual critic, or DeepSeek judge
call is part of the default path.

## DeepSeek Structured Judge (Phase 6)

`spacers_agent.evaluation` calculates deterministic counting metrics first, then builds a compact
text-and-structured-evidence payload for `spacers_agent.clients.DeepSeekJudgeClient`. The judge
never receives imagery, Base64, file paths, or complete point lists. It explicitly declares that it
cannot verify visual truth. A Judge verdict of `correct` that conflicts with a known count mismatch
is preserved as raw output and flagged as `judge_inconsistency`; it never overrides the deterministic
metric.

When you are ready for an explicitly authorized live smoke test, create the ignored `.env` from the
template and replace only this placeholder with your key:

```dotenv
DEEPSEEK_API_KEY=replace-with-deepseek-key
```

Use [`.env.example`](C:\Users\TZDEZACR\Desktop\spacers-agent\code\.env.example) as the template; do not place the key in `configs/default.yaml`, source code, tests, run manifests, or documentation. No live DeepSeek call is performed by default.

## Offline Acceptance Tools (Phase 7)

Two local-only CLI commands help make point counting auditable after a result exists:

```powershell
python -m spacers_agent.cli render-count --image .\image.png --result .\counting_result.json --output .\counting_overlay.png
python -m spacers_agent.cli summarize-evaluations --input .\evaluation_records.jsonl --output .\evaluation_summary.json
```

The overlay renders owner cores, accepted points, and rejected points; the summary keeps deterministic benchmark metrics separate from optional DeepSeek quality metrics. See [the local runbook](C:\Users\TZDEZACR\Desktop\spacers-agent\code\docs\runbook.md) for required interpreter paths, safeguards, and commands.

## Runnable Qwen Agent and Dataset Commands

The baseline `main.py` remains unchanged. New operations use `python -m spacers_agent.cli` and make network calls only for commands explicitly marked `--live` or requiring inference:

```powershell
python -m spacers_agent.cli health qwen --live
python -m spacers_agent.cli list-datasets
python -m spacers_agent.cli smoke-qwen --image tests/fixtures/smoke.png --question "Describe this image"
python -m spacers_agent.cli count-image --image .\demo.png --question "How many buildings?" --run-id demo-count --evaluate --render
python -m spacers_agent.cli run-dataset --dataset XLRS-Bench-lite --root D:\data\XLRS-Bench-lite --split test --task counting --run-id xlrs-count-v1 --resume
python -m spacers_agent.cli resume-run --run-id xlrs-count-v1
python -m spacers_agent.cli evaluate-run --run-id xlrs-count-v1 --deepseek
```

The four dataset adapters are deliberately read-only. LEVIR-CC, MME-RealWorld, and XLRS-Bench-lite require a versioned `spacers_adapter.json`. VRSBench general VQA directly validates the official `VRSBench_EVAL_vqa.json` fields, including its `type`, and the image paths without modifying the dataset. The runner probes the selected layout before reading a sample and reports observed fields on mismatch.

To run the real VRSBench multi-Agent path with an already downloaded Qwen3-VL checkpoint, create an ignored `configs/local.spark.yaml` from `configs/default.yaml` and set only local runtime values such as:

```yaml
models:
  qwen:
    backend: transformers
    model: /path/to/Qwen3_vl_4b_instruct
    dtype: bfloat16
    device_map: auto
    local_files_only: true
    max_tokens: 512
```

Then run sequentially; this path does not require or contact vLLM:

```bash
python -m spacers_agent.cli --config configs/local.spark.yaml run-dataset \
  --dataset VRSBench --root /path/to/vrsbench --split validation \
  --task general_vqa --run-id vrsbench-qwen3vl-router-20 \
  --max-samples 20 --sample-concurrency 1 \
  --evaluate --judge-policy all
```

The official VRSBench type routes quantity questions through accepted-point counting,
spatial/category/direction/existence questions through the spatial expert, and color questions
through general VQA. The canonical evaluation task and reference answers remain unchanged. The
HTML report records the actual Agent route and prompt version, overlays labeled boxes or accepted
points on a report-only image copy, and includes the deterministic geometry audit.

VRSBench quantity routing uses a fixed vehicle ontology instead of an answer-aware target parse.
The default configuration forces one non-overlapping owner-core subdivision, enlarges transmitted
review crops to a maximum side of 768 pixels, and rechecks empty leaf tiles. A zero answer remains
point-derived; an empty second pass must explicitly return `confirmed_absent`, otherwise the result
is marked partial with `ZERO_COUNT_UNCONFIRMED`. For top-most/bottom-most category questions, the
spatial expert may perform one candidate-completeness review. Program geometry requires at least
two vehicle candidates before claiming an extreme comparison. Raw answers remain in the audit
metadata while declared yes/no, vehicle-category, grid-position, color, and cardinal-direction
vocabularies receive reference-independent normalization before evaluation.

Qwen runs locally through Transformers. DeepSeek is used only after Qwen returns and receives the question, official reference answers, candidate answer, and exact-match flag—not the image, boxes, or points. Its key is read only from `DEEPSEEK_API_KEY`. The default report is saved as `outputs/runs/<run-id>/vrsbench_vqa.report/report.html`; each card also includes Qwen raw/final answers, the standard answer, and DeepSeek validation.

The local Transformers client normalizes Qwen's common two-corner box representation before strict validation. A malformed JSON response receives at most one versioned text-only format-repair call; the repair call does not receive the source image, and both attempts remain in the sample artifacts.
