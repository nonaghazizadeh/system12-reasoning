#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${MODEL:?Set MODEL to a Hugging Face model ID or local LoRA adapter path}"
batch_size="${BATCH_SIZE:-8}"
prompt_mode="${PROMPT_MODE:-zero-shot}"

datasets=(
  multiarith gsm8k addsub aqua singleeq svamp agieval
  coin_flip last_letters
  commonsensqa strategyqa piqa siqa com2sense
)

for dataset in "${datasets[@]}"; do
  python src/evaluate.py \
    --model "$MODEL" \
    --dataset "$dataset" \
    --batch-size "$batch_size" \
    --prompt-mode "$prompt_mode"
done
