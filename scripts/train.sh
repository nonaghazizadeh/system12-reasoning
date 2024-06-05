#!/bin/bash
cd .. 
source venv/bin/activate

label_col=('labels')
methods=('lora')
epochs=(20)
gpu=7
LM=('meta-llama/Meta-Llama-3-8B-Instruct')
dataset_name="system12_combined"

for label in "${label_col[@]}"; do
    for method in "${methods[@]}"; do
        for epoch in "${epochs[@]}"; do
            for lm in "${LM[@]}"; do
                echo "Training $method with $lm on $label for $epoch epochs"

                WANDB_PROJECT=system12 CUDA_VISIBLE_DEVICES=$gpu python train.py \
                                                                        --label_col "$label" \
                                                                        --LM "$lm" \
                                                                        --method "$method" \
                                                                        --EPOCHS "$epoch" \
                                                                        --dataset_name "$dataset_name" 
            done
        done
    done
done