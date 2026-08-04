#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

base_model="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
algorithm="${ALGORITHM:-dpo}"
output_root="${OUTPUT_ROOT:-experiments/camera_ready}"
report_to="${REPORT_TO:-none}"

for style in system1 system2; do
  python src/train_alignment.py \
    --algorithm "$algorithm" \
    --base-model "$base_model" \
    --"$style" \
    --output-dir "$output_root/$algorithm/$style" \
    --report-to "$report_to"
done
