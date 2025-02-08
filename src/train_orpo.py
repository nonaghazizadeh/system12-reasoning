import wandb
import argparse
import os
import torch
from sklearn.model_selection import train_test_split
from utils import create_logger, get_dataset_loader_func, add_pad_token_id, get_tokenizer
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    set_seed,
)
from trl import (
    ORPOConfig,
    ORPOTrainer,
)
from peft import (
    get_peft_model,
    LoraConfig,
)


def create_feedback_datasets(df, seed, label_col, train_size=0.8, reject_system_1=True):
    df = df.rename(columns={"Question": "prompt"})
    df_0 = df[df[label_col] == 0].reset_index(drop=True)
    df_1 = df[df[label_col] == 1].reset_index(drop=True)
    df_0, df_1 = df_0[['prompt', 'Answer']], df_1[['prompt', 'Answer']]
    if reject_system_1:
        df_0 = df_0.rename(columns={"Answer": "rejected"})
        df_1 = df_1.rename(columns={"Answer": "chosen"})
    else:
        df_0 = df_0.rename(columns={"Answer": "chosen"})
        df_1 = df_1.rename(columns={"Answer": "rejected"})

    df = df_0.merge(df_1, on="prompt", how="inner")
    print(df.iloc[-1])
    train_df, rest_df = train_test_split(
        df, train_size=train_size, random_state=seed)
    val_df, test_df = train_test_split(
        rest_df, train_size=0.5, random_state=seed)

    train_dataset = HFDataset.from_pandas(train_df)
    val_dataset = HFDataset.from_pandas(val_df)
    test_dataset = HFDataset.from_pandas(test_df)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=float,
                        default=0.8, help="Training data split size")
    parser.add_argument("--MAX_LEN", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int,
                        default=2, help="Training batch size")
    parser.add_argument("--VALID_BATCH_SIZE", type=int,
                        default=8, help="Validation batch size")
    parser.add_argument("--LEARNING_RATE", type=float,
                        default=8e-06, help="Learning rate")
    parser.add_argument("--label_col", type=str,
                        default="Strategy", help="Label column name")
    parser.add_argument("--EPOCHS", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--LM", type=str, default="roberta-large",
                        help="the pretrained language model to use")

    parser.add_argument("--method", type=str, default="finetune",
                        help="the method to use for training")
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=int)

    parser.add_argument("--orpo_beta", default=0.1, type=int)

    parser.add_argument("--dataset_name", type=str,
                        default="system12", help="the dataset for training")
    parser.add_argument("--reject_system_1",
                        action="store_true", help="Reject system 1 answers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    return args


def get_model(args):
    peft_config = None

    if args.method == "lora":
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout)
    else:
        raise ValueError(f"Method {args.method} not recognized")
    model = AutoModelForCausalLM.from_pretrained(args.LM)

    if peft_config:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model




if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if "/" in args.LM:
        LM_name = args.LM.split("/")[-1]
    run_name = f"{args.method}-{args.label_col}-{args.LM}-{args.seed}"

    wandb.init(project="system12-orpo", name=run_name, config=args)

    
    output_directory = os.path.join(
        "experiments", 'orpo',
        f"{args.method}-{LM_name}")
    if args.reject_system_1:
        output_directory += "-system2"
    else:
        output_directory += "-system1"
    os.makedirs(output_directory, exist_ok=True)
    logger = create_logger(output_directory)
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using {device} device")

    df = get_dataset_loader_func(args.dataset_name)

    train_dataset, val_dataset, test_dataset = create_feedback_datasets(df=df,
                                                                        seed=args.seed,
                                                                        label_col=args.label_col,
                                                                        train_size=args.train_size,
                                                                        reject_system_1=args.reject_system_1)
    
    tokenizer = get_tokenizer(args.LM)
    model = get_model(args)
    tokenizer, model = add_pad_token_id(tokenizer, model)

    training_args = ORPOConfig(
        output_dir=output_directory,
        learning_rate=args.LEARNING_RATE,
        per_device_train_batch_size=args.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=args.VALID_BATCH_SIZE,
        num_train_epochs=args.EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=run_name,
        save_total_limit=1,
        beta=args.orpo_beta,
        max_length=args.MAX_LEN,
        max_prompt_length=128,
    )
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    unload_model = trainer.model.merge_and_unload()
    
    unload_model.save_pretrained(output_directory)
    tokenizer.save_pretrained(output_directory)