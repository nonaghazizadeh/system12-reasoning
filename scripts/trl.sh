#!/bin/bash
cd .. 
source venv/bin/activate

label='labels'
method='lora'
epochs=(20)
learning_rate=8e-06
gpus=(7)
LM=('google/gemma-7b')
dataset_name="system12_combined"

for epoch in "${epochs[@]}"; do
    for lm_index in "${!LM[@]}"; do
        lm=${LM[$lm_index]}
        gpu=${gpus[$lm_index]}
        SESSION_NAME="${gpu}_TRL"
        echo "[$gpu] $lm" 
        screen -dmS "$SESSION_NAME" bash -c "
        WANDB_PROJECT=system12_orpo CUDA_VISIBLE_DEVICES=$gpu python train_trl.py \
                                                                --label_col "$label" \
                                                                --LM "$lm" \
                                                                --method "$method" \
                                                                --EPOCHS "$epoch" \
                                                                --LEARNING_RATE "$learning_rate" \
                                                                --dataset_name "$dataset_name" \
                                                                --reject_system_1;
        exit"

        # screen -dmS "$SESSION_NAME" bash -c "
        # WANDB_PROJECT=system12_orpo CUDA_VISIBLE_DEVICES=$gpu python train_trl.py \
        #                                                         --label_col "$label" \
        #                                                         --LM "$lm" \
        #                                                         --method "$method" \
        #                                                         --EPOCHS "$epoch" \
        #                                                         --LEARNING_RATE "$learning_rate" \
        #                                                         --dataset_name "$dataset_name";
        # exit"
    done
done
