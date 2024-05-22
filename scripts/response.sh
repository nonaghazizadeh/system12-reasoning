#!/bin/bash

cd .. 
source venv/bin/activate
# meta-llama/Meta-Llama-3-8B-Instruct' 'google/gemma-1.1-7b-it' 'mistralai/Mistral-7B-Instruct-v0.2 mistralai/Mistral-7B-Instruct-v0.3 microsoft/Phi-3-small-128k-instruct
LM=('mistralai/Mistral-7B-Instruct-v0.3')
gpus=(2)

system_prompt_format='one-sentence'
dataset_name="system12_combined_questions"

for lm_index in "${!LM[@]}"; do
    lm=${LM[$lm_index]}
    gpu=${gpus[$lm_index]}
    SESSION_NAME="(${gpu})"
    echo "[$gpu] $lm" 
    screen -dmS "$SESSION_NAME" bash -c "
    CUDA_VISIBLE_DEVICES=$gpu python response.py --LM $lm  \
                                                --system_prompt_format $system_prompt_format \
                                                --dataset_name $dataset_name;
                                                sleep 15;
    exit"
done
