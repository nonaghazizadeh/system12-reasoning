#! /bin/bash
cd ..

SESSION_NAME="benchmark"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(3)
lm='meta-llama/Meta-Llama-3-8B-Instruct'
datasets=('aqua' 'gsm8k' 'addsub' 'multiarth' 'svamp' 'singleeq' 'commonsensqa' 'strategyqa' 'coin_flip' 'bigbench_date' 'object_tracking')
method='zero_shot'
batch_size=128
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python benchmark_cot.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
exit'