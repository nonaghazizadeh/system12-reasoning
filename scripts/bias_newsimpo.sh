#! /bin/bash
cd ..

SESSION_NAME="benchmark"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(1)
lm='./experiments/simpo/lora-Meta-Llama-3-8B-Instruct-system2-llama-newhp'
datasets=('religion' 'ses' 'sexual_orientation')
method='zero_shot'
batch_size=128
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python benchmark.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
exit'