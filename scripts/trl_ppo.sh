#!/bin/bash
cd .. 
# source myenv/bin/activate

method='lora'
epochs=(5)
learning_rate=8e-06
gpus=(6)
LM=('meta-llama/Meta-Llama-3-8B-Instruct')
dataset_path="nona-ghazizadeh/CoT"
system_name="system1"

for epoch in "${epochs[@]}"; do
    for lm_index in "${!LM[@]}"; do
        lm=${LM[$lm_index]}
        gpu=${gpus[$lm_index]}
        SESSION_NAME="${gpu}_ppo"
        echo "[$gpu] $lm" 
        # screen -dmS "$SESSION_NAME" bash -c "
        WANDB_PROJECT=system12_orpo CUDA_VISIBLE_DEVICES=$gpu python train_ppo.py \
                                                                --LM "$lm" \
                                                                --EPOCHS "$epoch" \
                                                                --LEARNING_RATE "$learning_rate" \
                                                                --dataset_path "$dataset_path" \
                                                                --system_name "$system_name";
        # exit"
    done
done