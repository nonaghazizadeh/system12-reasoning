#!/bin/bash
cd .. 
source venv/bin/activate

method='lora'
# LM_responders=('Meta-Llama-3-8B-Instruct' 'gemma-1.1-7b-it' 'Mistral-7B-Instruct-v0.2' 'Mistral-7B-Instruct-v0.3')
# LM_responders=('Phi-3-small-128k-instruct' 'Meta-Llama-3-8B-Instruct' 'gemma-1.1-7b-it' 'Mistral-7B-Instruct-v0.2' 'Mistral-7B-Instruct-v0.3')
LM_responders=('Mistral-7B-Instruct-v0.3')
gpus=(7)
LM_classifier_id='meta-llama/Meta-Llama-3-8B-Instruct'
LM_classifier_path='2024-05-15_16-50-54-lora-labels-Meta-Llama-3-8B-Instruct/checkpoint-36'
system_prompt_format='one-sentence'
dataset_name="system12_combined_questions"

for lm_index in "${!LM_responders[@]}"; do
    lm_responder=${LM_responders[$lm_index]}
    gpu=${gpus[$lm_index]}
    SESSION_NAME="(${gpu})"
    echo "[$gpu] Evaluating $lm_responder with $LM_classifier_id + $method"
    screen -dmS "$SESSION_NAME" bash -c "
    CUDA_VISIBLE_DEVICES=$gpu python evaluate.py  --LM_responder $lm_responder \
                                           --method $method \
                                           --dataset_name $dataset_name \
                                           --LM_classifier_id $LM_classifier_id \
                                           --LM_classifier_path $LM_classifier_path \
                                           --system_prompt_format $system_prompt_format ;
                                           sleep 15;
    exit"
done
