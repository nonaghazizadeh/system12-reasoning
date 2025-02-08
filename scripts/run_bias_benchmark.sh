#! /bin/bash
cd ..

SESSION_NAME="benchmark-orpo2"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(6)
lm='./experiments/models/orpo/lora-Meta-Llama-3-8B-Instruct-system2'
datasets=('disability_status' 'age' 'gender_identity' 'nationality' 'physical_appearance' 'religion' 'ses' 'sexual_orientation')
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