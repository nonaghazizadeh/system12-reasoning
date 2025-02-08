#!/bin/bash
cd .. 
# source myenv/bin/activate

label='labels'
method='lora'
epochs=(5)
learning_rate=1e-6
gpus=(3)
LM=('meta-llama/Meta-Llama-3-8B-Instruct')
# LM=('mistralai/Mistral-7B-Instruct-v0.3')
dataset_name="system12_combined"

for epoch in "${epochs[@]}"; do
    for lm_index in "${!LM[@]}"; do
        lm=${LM[$lm_index]}
        gpu=${gpus[$lm_index]}
        SESSION_NAME="${gpu}_TRL_simpo"
        echo "[$gpu] $lm" 
        screen -dmS "$SESSION_NAME" bash -c "
        WANDB_PROJECT=system12_simpo_ratio CUDA_VISIBLE_DEVICES=$gpu python train_simpo_mix.py \
                                                                --label_col "$label" \
                                                                --LM "$lm" \
                                                                --method "$method" \
                                                                --EPOCHS "$epoch" \
                                                                --LEARNING_RATE "$learning_rate" \
                                                                --dataset_name "$dataset_name" ;


        exit"
    done
done
