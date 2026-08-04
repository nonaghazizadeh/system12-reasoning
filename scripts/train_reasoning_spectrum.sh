#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

base_model="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
algorithm="${ALGORITHM:-dpo}"
output_root="${OUTPUT_ROOT:-experiments/camera_ready/spectrum}"
report_to="${REPORT_TO:-none}"

for fraction in 0.125 0.25 0.375 0.5 0.625 0.75 0.875; do
  python src/train_alignment.py \
    --algorithm "$algorithm" \
    --base-model "$base_model" \
    --system1-fraction "$fraction" \
    --output-dir "$output_root/$algorithm/system1-$fraction" \
    --report-to "$report_to"
done
