#!/bin/bash
cd .. 
source venv/bin/activate

method='lora'
# LM_responders=('Meta-Llama-3-8B-Instruct' 'gemma-1.1-7b-it' 'Mistral-7B-Instruct-v0.2')
LM_responders=('gemma-1.1-7b-it')
LM_classifier_id='meta-llama/Meta-Llama-3-8B-Instruct'
LM_classifier_path='2024-05-15_16-50-54-lora-labels-Meta-Llama-3-8B-Instruct/checkpoint-36'
for lm_responder in "${LM_responders[@]}"; do
    echo "Evaluating $lm_responder with $LM_classifier_id + $method"
    CUDA_VISIBLE_DEVICES=6 python test.py  --LM_responder "$lm_responder" \
                                           --method "$method" \
                                           --LM_classifier_id "$LM_classifier_id" \
                                           --LM_classifier_path "$LM_classifier_path"
done
