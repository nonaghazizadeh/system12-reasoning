#! /bin/bash
cd ..
#lm='/home/nona/experiments/simpo/lora-Meta-Llama-3-8B-Instruct-system2-6'
#lm='meta-llama/Meta-Llama-3-8B-Instruct'
SESSION_NAME="benchmark"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(5)
lm='/home/nona/experiments/simpo/lora-Meta-Llama-3-8B-Instruct-system1-6'
datasets=('disability_status')
method='zero_shot'
batch_size=128
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python benchmark2.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
exit'