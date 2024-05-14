#!/bin/bash

cd .. 
source venv/bin/activate

LM=('meta-llama/Meta-Llama-3-8B-Instruct')

for lm in "${LM[@]}"; do
    echo "$lm" 
    CUDA_VISIBLE_DEVICES=6 python response.py --LM "$lm" 
done
