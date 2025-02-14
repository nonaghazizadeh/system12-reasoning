#! /bin/bash
cd ..
# datasets=('aqua' 'gsm8k' 'addsub' 'singleeq' 'commonsensqa' 'strategyqa' 'object_tracking' 'coin_flip' 'bigbench_date' 'svamp' 'last_letters' 'multiarith')
# lm='./experiments/dpo/lora-Mistral-7B-Instruct-v0.3-system1'

SESSION_NAME="benchmark-mistral"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(3)
lm='mistralai/Mistral-7B-Instruct-v0.1'
datasets=('aqua' 'gsm8k' 'addsub' 'singleeq' 'commonsensqa' 'strategyqa' 'object_tracking' 'coin_flip' 'bigbench_date' 'svamp' 'last_letters' 'multiarith')
method='zero_shot'
batch_size=32
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python src/benchmark_mistral.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
exit'