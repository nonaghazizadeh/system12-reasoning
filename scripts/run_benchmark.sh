#! /bin/bash
cd ..

# source ./venv/bin/activate


# LMs=('./experiments/instruction_tunning/system1_10000' './experiments/instruction_tunning/system2_10000' './experiments/instruction_tunning/system12_10000' './experiments/instruction_tunning/system12_20000')



# SESSION_NAME="benchmark"
# screen -dmS "$SESSION_NAME" bash -c '
#     gpu=(7)
#     lm='./experiments/orpo/lora-Meta-Llama-3-8B-Instruct-system2'
#     datasets=('gsm8k' 'aqua' 'addsub' 'multiarth' 'singleeq' 'commonsensqa' 'strategyqa' 'coin_flip', 'bigbench_date', 'object_tracking')
#     method='zero_shot'
#     batch_size=32
#     for dataset in "${datasets[@]}"; do
#         echo "Running on ${dataset}"
#         CUDA_VISIBLE_DEVICES=$gpu python benchmark.py \
#                                             --limit_dataset_size=0 \
#                                             --method=${method} \
#                                             --model=${lm} \
#                                             --batch_size ${batch_size} \
#                                             --dataset=${dataset} ;                      
#     done 
# exit'

# SESSION_NAME="benchmark"
# screen -dmS "$SESSION_NAME" bash -c '
#     gpu=(7)
#     lm='meta-llama/Meta-Llama-3-8B-Instruct'
#     datasets=('gsm8k' 'aqua' 'addsub' 'multiarth' 'singleeq' 'commonsensqa' 'strategyqa' 'coin_flip', 'bigbench_date', 'object_tracking')
#     method='zero_shot'
#     batch_size=32
#     for dataset in "${datasets[@]}"; do
#         echo "Running on ${dataset}"
#         CUDA_VISIBLE_DEVICES=$gpu python benchmark2.py \
#                                             --limit_dataset_size=0 \
#                                             --method=${method} \
#                                             --model=${lm} \
#                                             --batch_size ${batch_size} \
#                                             --dataset=${dataset} ;                      
#     done 
# exit'

SESSION_NAME="benchmark-cpo-7525"
screen -dmS "$SESSION_NAME" bash -c '
gpu=(7)
lm='/home/nona/experiments/cpo/lora-Meta-Llama-3-8B-Instruct-75sys1-25sys2'
datasets=('aqua' 'gsm8k' 'addsub' 'multiarth' 'singleeq' 'commonsensqa' 'strategyqa' 'coin_flip', 'bigbench_date', 'object_tracking')
method='zero_shot'
batch_size=32
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python benchmark.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
exit'