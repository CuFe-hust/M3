# M3

## Unified Multimodal Zero-Shot Baseline

This repository provides a zero-shot evaluation baseline with explicit wrappers for
Qwen3-VL-4B, Qwen3.5-4B, Qwen3.5-9B, InternVL3.5-8B, MiniCPM-V-4.6, and Ovis2.5-2B.
It does not fine-tune model weights. The baseline evaluates each release independently and writes canonical
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
pip install -r requirements-models.txt
cp config/baseline.example.json config/local.baseline.json
```

Edit `config/local.baseline.json` only to choose storage paths or supported model runtime settings.
The default paths keep downloaded data in `datasets/` and outputs in `outputs/`, both ignored by Git.
Do not put API keys in this file.

The default `report` settings generate a visual audit report for up to 200 samples per result.
Increase `report.max_samples` only when the additional image and HTML size is acceptable, or set
`report.enabled` to `false` to disable report artifacts for a particular local run.

Main-flow models are constructed only through the unified entry
`models.entry.create_model(name, ...)`: `qwen3_vl_baseline` (the sync baseline in
`models/qwen3_vl/baseline.py`) and `qwen_transformers` (the shared local Transformers client in
`models/qwen_transformers.py`). Each model folder keeps its wrapper and weights under `weights/`;
adding a model means adding one `@register` builder in `models/entry.py`. The remote vLLM client
has been removed, and `spacers_agent/clients/` keeps only the test/training
`DeepSeekJudgeClient` and `MockVisionClient`. CUDA-matched PyTorch remains the server operator's
responsibility, and mock tests download no weights.

For a checkpoint that is already present on a local server, set `model.id` to that external
directory and set `model.local_files_only` to `true`. This prevents accidental Hugging Face
network fallback while preserving the original Qwen3-VL loading and prediction interfaces.
Keep the server-specific absolute path only in the ignored `config/local.baseline.json` file.

Download the official data releases:

```bash
python main.py --config config/local.baseline.json download
```

Dataset reading is unified in `data/loader.py`, which streams canonical samples lazily and
accepts `limit` for smoke tests. The default local data root is `/data` (override with
`DATASET_ROOT` or `--root`). The new module entry points are:

```bash
python -m data.loader --dataset vrsbench_vqa --root /data --limit 3
python -m data.downloader --root /data --datasets vrsbench
python -m data.validator --root /data --datasets vrsbench
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
questions/prompts, model raw and final answers, references, exact-match comparison, and per-sample
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

Run the separately maintained team standard evaluator against any canonical result and merge its
primary metric into the existing HTML audit report:

```bash
python -m spacers_agent.cli standard-evaluate \
  --result outputs/runs/<run-id>/vrsbench_vqa.jsonl \
  --tool-dir ~/eval_standard
```

The command writes `<result-stem>.standard.json` beside the canonical JSONL and refreshes the
existing HTML report when its visual audit artifacts are present. `EVAL_LLM_API_KEY` and
`EVAL_LLM_BACKEND` remain owned by `eval_standard`; this integration neither reads nor persists
the key. Use `--output` or `--python` when the report location or evaluator environment differs.

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

The project now includes a unified local Transformers Qwen client and an offline Mock client.
Main-flow models are constructed only through the unified entry
`models.entry.create_model(name, ...)` (entries: `qwen_transformers`,
`qwen3_vl_baseline`, `qwen3_5_transformers`); the remote vLLM client has been removed.
The default test suite injects local fake completions; it does not contact an endpoint, and
Qwen inference uses no API key or remote server. `DeepSeekJudgeClient` and `MockVisionClient`
remain in `spacers_agent/clients/` for test/training flows.

## Read-Only Dataset Audit (Phase 3)

Inspect a local dataset layout before implementing an Adapter. The command never changes source
dataset files and writes its result to a separate report path:

```bash
C:\Users\TZDEZACR\miniconda3\envs\m3\python.exe -m spacers_agent.cli inspect-data \
  --root ./dataset \
  --output outputs/dataset_audit.json
```

## Point Counting Orchestration (Phase 4)

`spacers_agent.agents.counting.point_pipeline.PointCountingOrchestrator` is an additive, async workflow for one
normalized image and a caller-supplied `CountTargetSpec`. It sends one crop at a time through
an injected structured client, uses non-overlapping owner cores with halo context, converts only
validated `0..999` local points to global pixels, and derives `final_count` solely from accepted
global points. It writes each tile's geometry, parsed response, conversion report, and checkpoint
below the selected run directory; a matching successful request hash is reused on resume.

The active counting prompt is versioned in `prompts/count_tile_v4.md`; superseded prompts remain
available for experiment reproducibility. The v4 contract requires a systematic overview scan,
explicit uncertainty for small candidates, and an independent versioned empty-tile review. An
unconfirmed zero triggers finer crops with depth-reduced halo. Point-counting tests use local Mock
clients only. No Qwen, DeepSeek, SSH tunnel,
server, or cloud request is made by this module unless a caller explicitly constructs and invokes
a live client after authorization.

## Sparse Multi-Agent Routing (Phase 5)

`spacers_agent.routing.TaskRouter` uses fixed rule routes for declared tasks and does not make a
model call in that case. Only `route_unknown` uses an injected, text-only client; it requires and
consumes a `CallBudget` entry before the call. `CountingAgent` runs the existing point pipeline:
complete answers are derived from accepted global points, while partial results explicitly report
completed tiles and remain non-final.

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
python -m spacers_agent.cli judge-vqa-run --run-id vrsbench-qwen3vl-router-20
```

`run-dataset`, `resume-run`, and `count-image` all enter the same composed runtime:
`assemble_runtime` → `build_dataset_runner` (dataset commands) or `SampleRunner` (one image) →
`TaskRouter` → `RoutingDecision` → `AgentRegistry` → concrete `Agent.run(UnifiedSample, AgentContext)` →
`AgentExecution` → `ArtifactWriter` and optional `JudgeService`. The runtime has no deprecated
workflow, counting, or expert-module compatibility path.

## Optional YOLO OBB Counting

The repository default keeps YOLO disabled. It neither imports a detector runtime nor inspects
weights until a local configuration enables an audited detector. Install only the runtime selected
by the deployed detector profile:

```bash
python -m pip install -e '.[yolo]'
# or, for the YOLOv5-OBB CSL ONNX profile
python -m pip install -e '.[yolo-onnx]'
```

On Linux ARM64 with CUDA 13, install the official nightly GPU wheel first, then install the
project extra without replacing that wheel:

```bash
python -m pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu --no-deps
python -m pip install -e '.[yolo-onnx]'
```

Copy `configs/yolo.example.yaml` to an ignored `configs/local.*.yaml`, point `weights` to an
already-downloaded detector artifact, and keep its SHA256 equal to the configured value. The
`yolov5m_obb_csl_dotav20` profile uses GPU ONNX Runtime and DOTA-v2.0's 18-class map. The
`detectors` list remains a multi-weight interface, while a deployment can enable only one default
detector. Do not rely on automatic model downloads.

For `counting` and `fine_grained_counting`, auto mode selects the highest-priority supported YOLO
detector, preserves `final_count == accepted points`, and visibly falls back to Qwen for missing or
invalid weights, missing optional dependencies, task/class-map mismatch, or detector failure. A
zero YOLO result triggers an independent Qwen point review unless local configuration explicitly
trusts empty detections. In auto mode, supported VRSBench quantity targets use YOLO first and retain
the dedicated VRSBench Qwen backend for unavailable/error/empty-result fallback.

`run-dataset --task counting` now writes `outputs/runs/<run-id>/counting.report/report.html` and
`samples.csv`. The report shows the executed backend, YOLO attempt/fallback state, detector profile,
and trace-backed detector summary; it never renders an absolute weight path or credentials. Review
the Ultralytics AGPL license and your deployment obligations before use.

The four dataset adapters are deliberately read-only. LEVIR-CC, MME-RealWorld, and XLRS-Bench-lite require a versioned `spacers_adapter.json`. VRSBench general VQA directly validates the official `VRSBench_EVAL_vqa.json` fields, including its `type`, and the image paths without modifying the dataset. The runner probes the selected layout before reading a sample and reports observed fields on mismatch.

To run the real VRSBench multi-Agent path with an already downloaded Qwen3-VL checkpoint, create an ignored `configs/local.spark.yaml` from `configs/default.yaml` and set only local runtime values such as:

```yaml
models:
  qwen:
    model: /path/to/Qwen3_vl_4b_instruct
    dtype: bfloat16
    device_map: auto
    use_kernels: false
    local_files_only: true
    max_tokens: 512
    spatial_review_max_tokens: 128
```

The same local Transformers backend also supports the native multimodal
`Qwen/Qwen3.5-9B` checkpoint when the installed Transformers release provides
`Qwen3_5ForConditionalGeneration`. Point `models.qwen.model` at the downloaded
checkpoint directory, for example `/home/user/models/Qwen3.5-9B`, and retain
`local_files_only: true`. The loader selects the native class from the checkpoint's
`model_type`. For Qwen3.5 it disables thinking in the chat template so the existing
JSON-only Agent response contract is preserved; Qwen3-VL rendering remains unchanged.
Spatial candidate review uses its own 128-token ceiling and a compact box-only schema, so
it does not consume the full general-response budget reproducing prose evidence and geometry.
On NVIDIA GB10, install the optional runtime with
`python -m pip install -e '.[qwen35-gb10]'`, set `use_kernels: true`, and set
`device_map: cuda:0` in the ignored machine-local configuration. This resolves the
fixed Qwen3.5 Gated DeltaNet Hub snapshot to a local path, disables Transformers'
unrelated inherited kernel mappings, and avoids unintended CPU offload without
changing the default Qwen3-VL path. Before an offline run, cache the pinned kernel
revision once:

```bash
python -c "from kernels import install_kernel; print(install_kernel('Atlas-Inference/gdn', revision='ef12347fc77d6ddf1cb72c0bd0af1c7d6cc69172'))"
```

The loader retains that exact revision during both online download and offline cache
reuse, then passes the resolved snapshot path to `KernelConfig`. This local mapping is
intentional: it prevents Transformers from also resolving default kernels such as
`kernels-community/activation`.

Then run sequentially; this path does not require or contact vLLM:

If an existing ignored `configs/local.spark.yaml` copied older counting keys, set
`counting.prompt_version: count-point-v4` and `counting.vrsbench_min_scan_depth: 0` before creating
a new run. This prevents a v4 prompt from being mislabeled as an older inference configuration.

```bash
set -a
source .env
set +a

python -m spacers_agent.cli --config configs/local.spark.yaml run-dataset \
  --dataset VRSBench --root /path/to/vrsbench --split validation \
  --task general_vqa --run-id vrsbench-qwen3vl-router-20 \
  --max-samples 20 --sample-concurrency 1 \
  --judge-policy all
```

`run-dataset` enables evaluation by default, and VQA uses `--judge-policy all`. If
`DEEPSEEK_API_KEY` is not present in the process environment, the command fails visibly instead of
silently marking Judge as `not_requested`; use `--no-evaluate` or `--judge-policy none` only when
DeepSeek evaluation is intentionally disabled.
To add or retry DeepSeek results for an existing run without loading or calling Qwen, export the
same environment key and run `judge-vqa-run --run-id <run-id>`. Successful existing Judge records
are reused unless `--force` is supplied, and the HTML report is rebuilt from persisted Qwen results.

VRSBench routing is conservative and question-driven. The official `type` is retained for audit but
does not force an Agent or answer vocabulary. Explicit numerical questions use accepted-point
counting; direct single-target grid location, vehicle extreme-category, orientation, and arrangement
questions use the spatial expert; open categories, scenes, existence/relation questions, and unknown
types fall back to general VQA. A closed vocabulary is included only when the question text itself
entails it. The canonical evaluation task and reference answers remain unchanged. The HTML report
records the actual Agent route and prompt version, overlays labeled boxes or accepted points on a
report-only image copy, and includes the deterministic geometry audit.

VRSBench quantity routing uses the fixed vehicle ontology only when the question explicitly asks
for a vehicle, small vehicle, or large vehicle. Other quantity targets use the normal target parser,
so their requested noun is preserved rather than being widened to `vehicle`.
The default configuration scans a fitting image as one overview, enlarges small transmitted crops
to a maximum side of 768 pixels, and independently rechecks empty results. A review that reports
`zero_unconfirmed` triggers finer owner-core crops with a smaller halo; a zero answer remains solely
point-derived. Spatial extreme and arrangement questions receive an independent
candidate-enumeration pass that is not shown first-pass evidence. Grid-position questions instead
localize the singular physical target before the program derives its three-by-three label from the
box centre. Their visual localization call does not receive the grid-label vocabulary, and an
independent review is used only for missing, ambiguous, or corner-region placeholder evidence. A
review may attach the explicit small/large vehicle class to model-provided top-level boxes; otherwise
it uses a neutral target label while preserving the returned coordinates. It never fabricates
coordinates. Question semantics remain separate from the coarse official type. Program geometry
requires at least two candidates before claiming an extreme comparison, workflow status tokens are
never retained as semantic answers, and every local decision remains in the geometry audit.

Qwen runs locally through Transformers. DeepSeek is used only after Qwen returns and receives the question, official reference answers, candidate answer, and exact-match flag—not the image, boxes, or points. Its key is read only from `DEEPSEEK_API_KEY`. The default report is saved as `outputs/runs/<run-id>/vrsbench_vqa.report/report.html`; each card also includes Qwen raw/final answers, the standard answer, and DeepSeek validation.

The local Transformers client normalizes Qwen's common two-corner box representation before strict validation. It orders reversed corners but never expands a zero-area line or point into a fabricated box. Labeled degenerate observations are retained as points, unlabeled legacy boxes are dropped, and a valid box/point conflict retains the box. The geometry audit records normalization names, evidence quality, and repair severity. A malformed JSON response receives at most one versioned text-only format-repair call; the repair call does not receive the source image, and both attempts remain in the sample artifacts. A response truncated only at its final JSON member may be closed locally or have only that incomplete tail member removed; this recovery is explicitly recorded and never invents missing visual evidence.
