#!/bin/bash

cd .. 
source venv/bin/activate

gpu=7
lm="meta-llama/Meta-Llama-3.1-8B-Instruct"

system_12_folder="data/system_12_from_instruction_tunning"

data_dir="data/processed_instruction_tunning"
dataset_files=("$data_dir/flan_v2/flan_v2_data.jsonl"
    "$data_dir/cot/cot_data.jsonl"
    "$data_dir/dolly/dolly_data.jsonl"
    "$data_dir/oasst1/oasst1_data.jsonl")

sample_size=0


CUDA_VISIBLE_DEVICES=$gpu python zero_shot.py \
                                --model_name $lm \
                                --system_12_folder $system_12_folder \
                                --dataset_files ${dataset_files[@]} \
                                --sample_size $sample_size ;