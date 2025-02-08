import wandb
import argparse
import os
import torch
from sklearn.model_selection import train_test_split
from utils import create_logger, get_dataset_loader_func, add_pad_token_id, get_tokenizer
from datasets import Dataset as HFDataset
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    set_seed,
)
from trl import (
    DPOConfig,
    DPOTrainer,
)
from peft import (
    get_peft_model,
    LoraConfig,
)


def create_feedback_datasets(df, seed, label_col, train_size=0.8, data_balance="equal"):
    df = df.rename(columns={"Question": "prompt"})
    df_0 = df[df[label_col] == 0].reset_index(drop=True)
    df_1 = df[df[label_col] == 1].reset_index(drop=True)
    df_0, df_1 = df_0[['prompt', 'Answer']], df_1[['prompt', 'Answer']]

    if data_balance == "1more":
        quarter_index_0 = len(df_0) // 4
        quarter_index_1 = len(df_1) // 4

        df_first_quarter = pd.DataFrame({
            'prompt': df_1.iloc[:quarter_index_1]['prompt'],
            'rejected': df_0.iloc[:quarter_index_0]['Answer'],
            'chosen': df_1.iloc[:quarter_index_1]['Answer']
        })
        df_remaining_three_quarters = pd.DataFrame({
            'prompt': df_0.iloc[quarter_index_0:]['prompt'],
            'rejected': df_1.iloc[quarter_index_1:]['Answer'],
            'chosen': df_0.iloc[quarter_index_0:]['Answer']
        })
        df = pd.concat([df_first_quarter, df_remaining_three_quarters], ignore_index=True)
        train_df, rest_df = train_test_split(
            df, train_size=train_size, random_state=seed)
        val_df, test_df = train_test_split(
        rest_df, train_size=0.5, random_state=seed)

    elif data_balance == "2more":
        quarter_index_0 = len(df_0) // 4
        quarter_index_1 = len(df_1) // 4

        df_first_quarter = pd.DataFrame({
            'prompt': df_0.iloc[:quarter_index_0]['prompt'],
            'rejected': df_1.iloc[:quarter_index_1]['Answer'],
            'chosen': df_0.iloc[:quarter_index_0]['Answer']
        })
        df_remaining_three_quarters = pd.DataFrame({
            'prompt': df_1.iloc[quarter_index_1:]['prompt'],
            'rejected': df_0.iloc[quarter_index_0:]['Answer'],
            'chosen': df_1.iloc[quarter_index_1:]['Answer']
        })
        df = pd.concat([df_first_quarter, df_remaining_three_quarters], ignore_index=True)
        train_df, rest_df = train_test_split(
            df, train_size=train_size, random_state=seed)
        val_df, test_df = train_test_split(
            rest_df, train_size=0.5, random_state=seed)
    
    elif data_balance == "equal":
        half_index_0 = len(df_0) // 2
        half_index_1 = len(df_1) // 2

        df_first_half = pd.DataFrame({
            'prompt': df_0.iloc[:half_index_0]['prompt'],
            'rejected': df_0.iloc[:half_index_0]['Answer'],
            'chosen': df_1.iloc[:half_index_1]['Answer']
        })
        df_second_half = pd.DataFrame({
            'prompt': df_0.iloc[half_index_0:]['prompt'],
            'rejected': df_1.iloc[half_index_1:]['Answer'],
            'chosen': df_0.iloc[half_index_0:]['Answer']
        })
        df = pd.concat([df_first_half, df_second_half], ignore_index=True)
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

    parser.add_argument("--dpo_beta", default=0.1, type=int)

    parser.add_argument("--dataset_name", type=str,
                        default="system12", help="the dataset for training")
    parser.add_argument("--data_balance",
                        type=str, default="equal" ,help="ratio")
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

    wandb.init(project="system12-dpo-ratio", name=run_name, config=args)

    
    output_directory = os.path.join(
        "experiments", 'dpo-ratio',
        f"{args.method}-{LM_name}")

    if args.data_balance == "1more":
        output_directory += "-75sys1-25sys2"
    elif args.data_balance == "2more":
        output_directory += "-25sys1-75sys2"
    elif args.data_balance == "equal":
        output_directory += "-50sys1-50sys2"
        
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
    print(tokenizer)
    training_args = DPOConfig(
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
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # -------------- Test
    # test_dataset = test_dataset.map(trainer.tokenize_row)
    # res = trainer.predict(test_dataset)
    # test_metrics = res.metrics
    # test_metrics = {"test/"+k[5:]: v for k, v in test_metrics.items()}
    # wandb.log(test_metrics)

    # Unload model
    unload_model = trainer.model.merge_and_unload()
    
    # Save model and tokenizer
    unload_model.save_pretrained(output_directory)
    print("hello")
    tokenizer.save_pretrained(output_directory)