# Fast and Slow thinking


## Overview

<p align="center">
  <img src="image.png" alt="pipeline" width="400">
</p>


## Prerequisites

- python 3.10.12
- Ubuntu GPU-enabled server with CUDA 12.1+
    - Check your GPUs with `nvidia-smi`
- python environment with packages installed as in `requirements.txt`
- [Weights and Biases](https://wandb.ai/) Account

## Setup Environment

```bash
cd ROOT_OF_THE_REPO
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
To run the experiments, you need to login to your Weights and Biases account. You can do that by running the following command and following the instructions:

```bash
wandb login
```

### Datasets

#### Alignment Data

1. Download the `cognitive_biases.csv` file from [X]()
2. Place the file in `data/system12`

#### Benchmark Data

1. Download the `dataset` folder from [Role Play Prompting](https://github.com/NKU-HLT/Role-Play-Prompting)
2. Place the folders in `data/benchmark`

## Running the Experiments

### Scripts for running the experiments

#### scripts/train_dpo.sh
The shell script will run the `train_dpo.py` for aligning our base models, Llama 3 and Mistral v0.1 with System 1 data and System 2 data with DPO algorithm. 

#### scripts/train_simpo.sh
The shell script will run the `train_simpo.py` for aligning our base models, Llama 3 and Mistral v0.1 with System 1 data and System 2 data with SIMPO algorithm. 

#### scripts/train_dpo_ratio.sh
The shell script will run the `train_dpo_ratio.py` for aligning our base models, Llama 3 and Mistral v0.1 with different ratios of System 1 data and System 2 data with DPO algorithm. 

#### scripts/train_simpo_ratio.sh
The shell script will run the `train_dpo_ratio.py` for aligning our base models, Llama 3 and Mistral v0.1 with different ratios of System 1 data and System 2 data with SIMPO algorithm. 
