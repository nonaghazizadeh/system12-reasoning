#! /bin/bash
cd ..

# source ./venv/bin/activate
#datasets=('PIQA' 'socialIQa' 'com2sense')
SESSION_NAME="benchmark"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(0)
lm='./experiments/dpo/lora-Meta-Llama-3-8B-Instruct-system1'
datasets=('aqua' 'gsm8k' 'addsub' 'singleeq' 'commonsensqa' 'strategyqa' 'object_tracking' 'coin_flip' 'bigbench_date' 'svamp' 'last_letters' 'multiarith')
method='zero_shot'
batch_size=64
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python src/benchmark_dynamic_no_threshold.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
exit'