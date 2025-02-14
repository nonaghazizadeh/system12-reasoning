#!/bin/bash
cd .. 
# source myenv/bin/activate

label='labels'
method='lora'
epochs=(5)
learning_rate=5e-14
data_balance = "1more"
gpus=(0)
eval_steps=100
#LM=('meta-llama/Meta-Llama-3-8B-Instruct')
LM=('mistralai/Mistral-7B-Instruct-v0.3')
dataset_name="system12_combined"

for epoch in "${epochs[@]}"; do
    for lm_index in "${!LM[@]}"; do
        lm=${LM[$lm_index]}
        gpu=${gpus[$lm_index]}
        SESSION_NAME="${gpu}_TRL_dpo"
        echo "[$gpu] $lm" 
        screen -dmS "$SESSION_NAME" bash -c "
        WANDB_PROJECT=system12_dpo CUDA_VISIBLE_DEVICES=$gpu python src/train_simpo_ratio.py \
                                                                --label_col "$label" \
                                                                --LM "$lm" \
                                                                --method "$method" \
                                                                --EPOCHS "$epoch" \
                                                                --LEARNING_RATE "$learning_rate" \
                                                                --eval_step "$eval_steps" \
                                                                --data_balance "$data_balance" \
                                                                --dataset_name "$dataset_name";
        exit"

    done
done
