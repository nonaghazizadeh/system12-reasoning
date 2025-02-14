#! /bin/bash
cd ..

# source ./venv/bin/activate

SESSION_NAME="benchmark"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(5)
lm='./prev_experiments/models/dpo/lora-Meta-Llama-3-8B-Instruct-system2'
datasets=('gsm8k' 'aqua' 'addsub' 'multiarith' 'singleeq' 'commonsensqa' 'strategyqa' 'coin_flip' 'bigbench_date' 'object_tracking' 'last_letters' 'svamp')
method='zero_shot'
batch_size=2
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python src/benchmark_llama.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
exit'