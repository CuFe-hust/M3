#!/usr/bin/env bash
set -euo pipefail

# Run this script inside: tmux new-session -s qwen35_change_sft
cd /home/user/cooper/M3

PY=/home/user/miniconda3/envs/Cooper_for_qwen9b/bin/python
CORPUS=/home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence_mixed_v1
LEVIROOT=/home/user/cooper/datasets/levir_mci_changehead/source/LEVIR-MCI-dataset
RUN=/home/user/cooper/posttrain_runs/changeagent_qwen35_sft_mixed_v1_$(date +%Y%m%d_%H%M%S)

test "$(git branch --show-current)" = change_agent
git merge-base --is-ancestor b321241868be24983c2aac814d52fc4399b458dc HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/change_agent)"
test -z "$(git status --porcelain)"
echo '6a52aacf6aa384f736fb51eef732fbb704b737c430a6ced4a9b4753ef4dcc67d  /home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence_mixed_v1/train.jsonl' | sha256sum -c -
echo 'd709c8e01f4ced952aaf198efb0652bd72014749913bb1f0fff4f13925428297  /home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence_mixed_v1/validation.jsonl' | sha256sum -c -
echo '56fae448cdab494ea1f7647e2ebb3637a52305d35521e7c012eb30886b0ab61a  /home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence_mixed_v1/manifest.json' | sha256sum -c -
test ! -e "$RUN"
mkdir -p "$RUN"

CMD=(
  "$PY" scripts/finetune_multimodal_sft.py
  --model-id /home/user/models/Qwen3.5-9B
  --model-adapter qwen3_5
  --data-profile change_agent
  --train-file "$CORPUS/train.jsonl"
  --validation-manifest "$CORPUS/validation.jsonl"
  --data-manifest "$CORPUS/manifest.json"
  --image-root "levir=$LEVIROOT"
  --prompt-ref change_dual_path_v9
  --tuning-policy lora_plus_projector
  --lora-rank 64 --lora-alpha 128 --lora-dropout 0.05
  --lora-lr 1e-4 --connector-lr 1e-5
  --weight-decay 0.01 --warmup-ratio 0.03 --max-grad-norm 1.0
  --dtype bfloat16 --device cuda:0
  --batch-size 1 --gradient-accumulation 8 --max-seq-length 4096
  --epochs 1 --save-steps 1000 --logging-steps 10 --save-total-limit 4
  --seed 1234 --local-files-only --output-dir "$RUN"
)

{
  date --iso-8601=seconds
  git rev-parse HEAD
  git rev-parse 'HEAD^{tree}'
  git status --short --branch
  "$PY" --version
  "$PY" -m pip freeze
  sha256sum "$CORPUS/train.jsonl" "$CORPUS/validation.jsonl" "$CORPUS/manifest.json" prompts/change_dual_path_v9.md
  jq '.ordering' "$CORPUS/manifest.json"
  nvidia-smi
  free -h
} > "$RUN/run_identity.txt" 2>&1

printf '%q ' "${CMD[@]}" > "$RUN/launch_command.txt"
printf '\n' >> "$RUN/launch_command.txt"

"${CMD[@]}" 2>&1 | tee "$RUN/console.log"
