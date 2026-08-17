#!/usr/bin/env bash

# Run the public dataset input path over the official VRSBench validation
# release and leave JSON/JSONL plus the self-contained HTML report in the run.
# 通过公开数据集输入路径运行官方 VRSBench validation 发布，并在 run 中留下
# JSON/JSONL 与自包含 HTML 报告。

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${VRSBENCH_ROOT:-$REPO_ROOT/data/VRSBench-full}"
SPLIT="${VRSBENCH_SPLIT:-val}"
RUN_ID="${VRSBENCH_RUN_ID:-vrsbench-val-system-$(date +%Y%m%d-%H%M%S)}"
CONFIG="${M3_CONFIG:-}"
LIMIT="${VRSBENCH_LIMIT:-}"
SAMPLE_CONCURRENCY="${VRSBENCH_SAMPLE_CONCURRENCY:-1}"
SHARD_INDEX="${VRSBENCH_SHARD_INDEX:-0}"
SHARD_COUNT="${VRSBENCH_SHARD_COUNT:-1}"

command=(
  "$PYTHON" "$REPO_ROOT/main.py"
)
if [[ -n "$CONFIG" ]]; then
  command+=(--config "$CONFIG")
fi
command+=(
  run-dataset
  --dataset VRSBench
  --root "$DATA_ROOT"
  --split "$SPLIT"
  --task caption,grounding,general_vqa
  --run-id "$RUN_ID"
  --evaluate
  --judge-policy none
  --sample-concurrency "$SAMPLE_CONCURRENCY"
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
)
if [[ -n "$LIMIT" ]]; then
  command+=(--limit "$LIMIT")
fi

result="$("${command[@]}")"
printf '%s\n' "$result"

# Some optional pycocoevalcap scorers print diagnostic lines to stdout before
# the CLI JSON. Extract the final structured payload without weakening the
# public command's JSON contract or discarding the diagnostics.
# 某些可选 pycocoevalcap scorer 会在 CLI JSON 前向 stdout 输出诊断行；在不
# 弱化公共命令 JSON 契约且不丢弃诊断信息的前提下提取最后一个结构化载荷。
run_dir="$("$PYTHON" - "$result" <<'PY'
import json
import sys

text = sys.argv[1]
decoder = json.JSONDecoder()
for index in reversed([position for position, value in enumerate(text) if value == "{"]):
    try:
        payload, _ = decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(payload, dict) and isinstance(payload.get("run_dir"), str):
        print(payload["run_dir"])
        break
else:
    raise SystemExit("CLI output did not contain a run_dir JSON payload")
PY
)"
if [[ "$run_dir" != /* ]]; then
  run_dir="$REPO_ROOT/$run_dir"
fi

# Keep a machine-readable pointer file beside the standard report bundle.
# 在标准报告 bundle 旁保存机器可读的指针文件。
"$PYTHON" - "$run_dir/command_result.json" "$result" <<'PY'
import json
import sys
from pathlib import Path

destination = Path(sys.argv[1])
text = sys.argv[2]
decoder = json.JSONDecoder()
payload = None
for index in reversed([position for position, value in enumerate(text) if value == "{"]):
    try:
        candidate, _ = decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict) and isinstance(candidate.get("run_dir"), str):
        payload = candidate
        break
if payload is None:
    raise SystemExit("CLI output did not contain a run_dir JSON payload")
payload["artifacts"] = {
    "predictions_jsonl": "predictions.jsonl",
    "report_json": "report/report.json",
    "report_samples_jsonl": "report/samples.jsonl",
    "report_html": "report/report.html",
}
payload["metric_contract"] = {
    "caption": {
        "requested": ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr"],
        "not_computed": ["CHAIR2"],
        "note": "No approved CHAIR2 scorer is configured in the current runtime.",
    },
    "grounding": {
        "overlap": "Persisted deterministic IoU is fail-closed for unsupported official polygon frames.",
        "position": "report/report.html renders model and GT geometry on the source image.",
    },
    "general_vqa": {
        "raw_submodel_outputs": "report/report.html reads persisted request/raw/parsed artifacts when present.",
        "final_answer": "report/report.html shows the final AgentResult answer.",
    },
}
report_path = destination.parent / "report" / "report.json"
if report_path.is_file():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    caption_metrics = {}
    for task in report.get("tasks", []):
        if task.get("run_task") == "caption":
            caption_metrics = task.get("metrics", {}).get("caption", {})
            break
    if isinstance(caption_metrics, dict):
        payload["metric_contract"]["caption"]["report_metric_status"] = caption_metrics.get(
            "metric_status", "not_available"
        )
        payload["metric_contract"]["caption"]["available"] = sorted(
            key for key in caption_metrics
            if key not in {"metric_status", "not_computed", "dependency", "record_count", "total"}
        )
        payload["metric_contract"]["caption"]["not_computed"] = sorted(
            set(payload["metric_contract"]["caption"]["not_computed"])
            | set(caption_metrics.get("not_computed", []))
        )
destination.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "status": "ok",
    "run_dir": str(destination.parent),
    "jsonl": str(destination.parent / "report/samples.jsonl"),
    "html": str(destination.parent / "report/report.html"),
}, ensure_ascii=False, sort_keys=True))
PY
