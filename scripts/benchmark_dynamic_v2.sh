#! /bin/bash
cd ..

# source ./venv/bin/activate

SESSION_NAME="benchmark"
# screen -dmS "$SESSION_NAME" bash -c '
gpu=(1)
datasets=('addsub')
model='llama'
algorithm='dpo'
batch_size=16
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python src/benchmark_dynamic.py \
                                        --limit_dataset_size=0 \
                                        --algorithm=${algorithm} \
                                        --model=${model} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
# exit'