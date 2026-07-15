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

### Output Format

Each `outputs/*.jsonl` line contains:

```json
{
  "sample": {"id": "...", "task_type": "vqa", "prompt": "...", "answers": ["..."]},
  "prediction": {"id": "...", "task_type": "vqa", "text": "...", "answer": "..."}
}
```

`*.metadata.json` records the model settings, timestamp, completed sample count, and any
dataset-scope qualification needed for a report.

For MME Real RS, inference also writes `mme_real_rs.official.json`, preserving each official
record and replacing only its `Output` field. It can be passed directly to the upstream
MME-RealWorld evaluator.
