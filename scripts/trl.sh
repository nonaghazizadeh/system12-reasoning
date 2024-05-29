#!/bin/bash
cd .. 
source venv/bin/activate

label='labels'
methods=('lora')
epochs=(20)
learning_rate=8e-06
gpu=0
LM=('meta-llama/Meta-Llama-3-8B-Instruct')
dataset_name="system12_combined"

for method in "${methods[@]}"; do
    for epoch in "${epochs[@]}"; do
        for lm in "${LM[@]}"; do
            echo "TRL $method with $lm on $label for $epoch epochs"

            WANDB_PROJECT=system12_orpo CUDA_VISIBLE_DEVICES=$gpu python train_trl.py \
                                                                    --label_col "$label" \
                                                                    --LM "$lm" \
                                                                    --method "$method" \
                                                                    --EPOCHS "$epoch" \
                                                                    --LEARNING_RATE "$learning_rate" \
                                                                    --dataset_name "$dataset_name"
                                                                    # --reject_system_1
        done
    done
done
