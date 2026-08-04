#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${SYSTEM1_ADAPTER:?Set SYSTEM1_ADAPTER to the System 1 LoRA adapter}"
: "${SYSTEM2_ADAPTER:?Set SYSTEM2_ADAPTER to the System 2 LoRA adapter}"

datasets=(
  multiarith gsm8k addsub aqua singleeq svamp agieval
  coin_flip last_letters
  commonsensqa strategyqa piqa siqa com2sense
)

for dataset in "${datasets[@]}"; do
  python src/evaluate_dynamic.py \
    --system1-adapter "$SYSTEM1_ADAPTER" \
    --system2-adapter "$SYSTEM2_ADAPTER" \
    --dataset "$dataset" \
    --prefix-tokens 32 \
    --mean-weight 0.4
done
