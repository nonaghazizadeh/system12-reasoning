#! /bin/bash
cd ..


SESSION_NAME="benchmark"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(3)
lm='./experiments/dpo-ratio/lora-Meta-Llama-3-8B-Instruct-87.5-12.5'
# lm='meta-llama/Meta-Llama-3-8B-Instruct'
datasets=('agieval')
method='zero_shot'
batch_size=32
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python src/benchmark_llama_agieval.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
exit'