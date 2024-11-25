import argparse
import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig
from utils import create_logger
import wandb


nltk.download("punkt")
def modify_dataset(dataset, seed):
    dataset = dataset.rename_column("question", "query")
    dataset = dataset.rename_column("answer", "response")
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["query", "response"]])
    train_validation_split = dataset.train_test_split(test_size=0.1, random_state=seed)

    dataset_dict = DatasetDict({
        'train': train_validation_split['train'],
        'validation': train_validation_split['test']
    })

    train_dataset = dataset_dict['train']
    validation_dataset = dataset_dict['validation']

    return train_dataset, validation_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to your dataset")
    parser.add_argument("--LM", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="LLaMA model path")
    parser.add_argument("--LEARNING_RATE", type=float, default=1.41e-5, help="Learning rate for PPO")
    parser.add_argument("--EPOCHS", type=int, default=10, help="Number of epochs")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=8, help="Training batch size")
    parser.add_argument("--VALID_BATCH_SIZE", type=int, default=8, help="Validation batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--system_name", type=str, default="", help="Dataset name")
    args = parser.parse_args()
    return args

def tokenize(sample, tokenizer):
    sample["input_ids"] = tokenizer.encode(sample["query"], return_tensors="pt", truncation=True).squeeze()
    return sample

def reward_function(response, system_name):
    sentences = sent_tokenize(response, language="english")
    num_sentences = len(sentences)
    if system_name == "system1":
        return 1.0 / num_sentences
    else:
        return num_sentences

def main():
    args = parse_args()

    run_name = f"ppo-{args.LM}-{args.system_name}-{args.seed}"
    wandb.init(project="system12_ppo", name=run_name, config=args)
    
    output_directory = os.path.join("experiments", "ppo",args.LM, args.system_name)
    
    if args.system_name == "system1":
        output_directory = os.path.join(args.output_dir, "system1")
    else:
        output_directory = os.path.join(args.output_dir, "system2")
    
    os.makedirs(output_directory, exist_ok=True)
    logger = create_logger(output_directory)
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")
    
    dataset = load_dataset(args.dataset_path, split="train")
    train_dataset, validation_dataset = modify_dataset(dataset, args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.LM)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    
    config = PPOConfig(
        output_dir=output_directory,
        learning_rate=args.LEARNING_RATE,
        num_train_epochs=args.EPOCHS,
        per_device_train_batch_size=args.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=args.VALID_BATCH_SIZE,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="wandb",
        run_name=run_name,
    )


    train_dataset = train_dataset.map(lambda sample: tokenize(sample, tokenizer))
    validation_dataset = validation_dataset.map(lambda sample: tokenize(sample, tokenizer))

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    generation_kwargs = {
        "min_length": 1,
        "top_k": 0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        
        rewards = [torch.tensor(reward_function(resp, args.system_name)) for resp in batch["response"]]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()


