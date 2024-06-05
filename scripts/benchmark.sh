#! /bin/bash
cd ..

source ./venv/bin/activate

# LMs=('meta-llama/Meta-Llama-3-8B-Instruct' './experiments/orpo/lora-Meta-Llama-3-8B-Instruct-system2' './experiments/orpo/lora-Meta-Llama-3-8B-Instruct-system1')

# LMs=('google/gemma-1.1-7b-it' './experiments/orpo/lora-gemma-1.1-7b-it-system2' './experiments/orpo/lora-gemma-1.1-7b-it-system1')

# LMs=('mistralai/Mistral-7B-Instruct-v0.3' './experiments/orpo/lora-Mistral-7B-Instruct-v0.3-system2' './experiments/orpo/lora-Mistral-7B-Instruct-v0.3-system1')

# LMs=('meta-llama/Meta-Llama-3-8B' './experiments/orpo/lora-Meta-Llama-3-8B-system2' './experiments/orpo/lora-Meta-Llama-3-8B-system1')

# LMs=('google/gemma-7b' './experiments/orpo/lora-gemma-7b-system2' './experiments/orpo/lora-gemma-7b-system1')

LMs=('./experiments/orpo/lora-Meta-Llama-3-8B-Instruct-system1')

gpus=(7)
dataset='addsub'
method='zero_shot'
batch_size=32
for lm_index in "${!LMs[@]}"; do
    lm=${LMs[$lm_index]}
    gpu=${gpus[$lm_index]}

    SESSION_NAME="${gpu}_${dataset}"
    echo "${SESSION_NAME}"
    screen -dmS "$SESSION_NAME" bash -c "
    CUDA_VISIBLE_DEVICES=$gpu python run_benchmark.py \
                                        --limit_dataset_size=0 \
                                        --method=${method} \
                                        --model=${lm} \
                                        --batch_size ${batch_size} \
                                        --dataset=${dataset} ;
                                        sleep 15;
    exit"
done 