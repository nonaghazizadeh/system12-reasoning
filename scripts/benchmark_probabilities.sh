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
# datasets=('aqua' 'gsm8k' 'addsub' 'multiarth' 'singleeq' 'commonsensqa' 'strategyqa' 'coin_flip', 'bigbench_date', 'object_tracking')
#lm='/home/nona/experiments/cpo/lora-Meta-Llama-3-8B-Instruct-system2-6'
# datasets=('aqua' 'gsm8k' 'addsub' 'multiarith' 'svamp' 'singleeq' 'commonsensqa' 'strategyqa' 'coin_flip' 'bigbench_date' 'object_tracking')

# SESSION_NAME="benchmark"
# screen -dmS "$SESSION_NAME" bash -c '
# gpu=(3)
# lm='./experiments/simpo-ratio/lora-Meta-Llama-3-8B-Instruct-50sys1-50sys2'
# datasets=('multiarith')
# method='zero_shot'
# batch_size=128
# for dataset in "${datasets[@]}"; do
#     echo "Running on ${dataset}"
#     CUDA_VISIBLE_DEVICES=$gpu python benchmark.py \
#                                         --limit_dataset_size=0 \
#                                         --method=${method} \
#                                         --model=${lm} \
#                                         --batch_size ${batch_size} \
#                                         --dataset=${dataset} ;                      
# done
# exit'

#datasets=('aqua' 'gsm8k' 'addsub' 'multiarith' 'singleeq' 'commonsensqa' 'strategyqa' 'coin_flip' 'bigbench_date' 'object_tracking' 'last_letters' 'svamp')
#datasets=('gsm8k' 'aqua' 'addsub' 'multiarith' 'singleeq' 'commonsensqa' 'strategyqa' 'coin_flip' 'bigbench_date' 'object_tracking' 'last_letters' 'svamp')

SESSION_NAME="benchmark"
# screen -dmS "$SESSION_NAME" bash -c '
gpu=(5)
lm='./prev_experiments/models/dpo/lora-Meta-Llama-3-8B-Instruct-system2'
datasets=('bigbench_date')
method='zero_shot'
batch_size=2
for dataset in "${datasets[@]}"; do
    echo "Running on ${dataset}"
    CUDA_VISIBLE_DEVICES=$gpu python benchmark_prob.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;                      
done
# exit'