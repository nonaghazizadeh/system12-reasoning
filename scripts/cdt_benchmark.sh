#!/bin/bash
cd .. 

SESSION_NAME="benchmark"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(4)
lm='./prev_experiments/models/dpo/lora-Meta-Llama-3-8B-Instruct-system1'
method='zero_shot'
batch_size=4
echo "Running on CRT"
CUDA_VISIBLE_DEVICES=$gpu python cdt_benchmark.py \
                                    --model=${lm} ;                      

exit'