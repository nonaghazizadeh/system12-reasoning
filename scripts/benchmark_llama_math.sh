#! /bin/bash
cd ..


SESSION_NAME="benchmark"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(7)
lm='./experiments/dpo/lora-Meta-Llama-3-8B-Instruct-system2'
# lm='meta-llama/Meta-Llama-3-8B-Instruct'
datasets=('math500')
method='zero_shot'
batch_size=32
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python src/benchmark_llama_math.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
exit'