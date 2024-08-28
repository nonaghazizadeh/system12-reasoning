#!/bin/bash

cd .. 
source venv/bin/activate

lm="meta-llama/Meta-Llama-3.1-8B-Instruct"
data_dir="data/processed_instruction_tunning"
output_data_dir="data/system12_instruction_tunning"

dataset_files=(
    "flan_v2/flan_v2_data_exclude_long.jsonl"
    "cot/cot_data.jsonl"
    "dolly/dolly_data.jsonl"
    "oasst1/oasst1_data.jsonl"
)
gpus=(4 5 6 7)

dataset_files=(
    "flan_v2/flan_v2_data_exclude_long.jsonl"
)
gpus=(6)

sample_size=0

for i in "${!dataset_files[@]}"; do
    dataset="${dataset_files[$i]}"
    gpu=${gpus[$i]}
    dataset_name="${dataset%%/*}"
    output_folder="$output_data_dir/${dataset_name}_system12"

    echo "Processing $dataset_name on GPU $gpu"
    
    # Create a new screen session for each dataset
    screen -dmS "zero_shot_${dataset_name}" bash -c "
        CUDA_VISIBLE_DEVICES=$gpu python zero_shot.py \
            --model_name $lm \
            --system_12_folder $output_folder \
            --dataset_files \"$data_dir/$dataset\" \
            --sample_size $sample_size
    "
done