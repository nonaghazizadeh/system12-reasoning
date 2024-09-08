#! /bin/bash
cd ..

source ./venv/bin/activate

gpu=$1
lm=$2
# LMs=('./experiments/instruction_tunning/system1_10000' './experiments/instruction_tunning/system2_10000' './experiments/instruction_tunning/system12_10000' './experiments/instruction_tunning/system12_20000')
# gpus=(0 1 2 3)

datasets=('gsm8k' 'aqua' 'commonsensqa' 'object_tracking' 'strategyqa')
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