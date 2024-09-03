#!/bin/bash

cd .. 
source ./scripts/base_training_args.sh

source ./venv/bin/activate

gpu=$1
thinking_type=$2
sample_size=$3

model_path="meta-llama/Meta-Llama-3.1-8B"
data_seed=42

data_dir=./data/system12_instruction_tunning


output_dir=./experiments/instruction_tunning/${thinking_type}
if [[ $thinking_type == "system12" ]]; then
    output_dir=./experiments/instruction_tunning/${thinking_type}_${sample_size}
fi
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

train_files=("$data_dir/flan_v2/"
    "$data_dir/cot/"
    "$data_dir/dolly/"
    "$data_dir/oasst1/")

# use fsdp for large models
if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
elif [[ $model_path == "meta-llama/Meta-Llama-3.1-8B" ]]; then
    base_training_args="$base_training_args --torch_dtype bfloat16"
#     base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
fi

training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--sample_size $sample_size \
--data_seed $data_seed \
--train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

SESSION_NAME="${thinking_type}_${sample_size}_gpu_${gpu}"
echo "Starting session $SESSION_NAME"

screen -dmS $SESSION_NAME bash -c "CUDA_VISIBLE_DEVICES=$gpu $header $training_args"