import wandb
import argparse
import os
import torch
from torch.optim import AdamW
import datetime
from sklearn.model_selection import train_test_split
from utils import compute_metrics, create_logger, set_seed, get_dataset_loader_func
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from peft import (
    get_peft_model,
    PromptEncoderConfig,
    LoraConfig,
)


def get_datasets(df, seed, tokenize_function, label_col, train_size=0.8):

    df['input'] = "[QUESTION] " + df['Question'] + " [ANSWER] " + df['Answer']
    # Split the DataFrame into a training and a test set while maintaining the label proportions.
    train_df, rest_df = train_test_split(
        df, train_size=train_size, stratify=df[label_col], random_state=seed)

    val_df, test_df = train_test_split(
        rest_df, train_size=0.5, stratify=rest_df[label_col], random_state=seed)

    train_dataset = HFDataset.from_pandas(train_df)
    val_dataset = HFDataset.from_pandas(val_df)
    test_dataset = HFDataset.from_pandas(test_df)

    train_tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
    )

    val_tokenized_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
    )

    test_tokenized_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
    )

    if label_col != 'labels':
        train_tokenized_dataset = train_tokenized_dataset.rename_column(
            label_col, "labels")
        val_tokenized_dataset = val_tokenized_dataset.rename_column(
            label_col, "labels")
        test_tokenized_dataset = test_tokenized_dataset.rename_column(
            label_col, "labels")

    # calculating the loss weights based on the labels
    pos_freq = sum(train_df[label_col])
    # setting lable weights to be inverse of the frequency of the label in the training set
    label_weights = [len(train_df)/(len(train_df)-pos_freq),
                     len(train_df)/(pos_freq)]

    return train_tokenized_dataset, val_tokenized_dataset, test_tokenized_dataset, label_weights


def setup_tokenizer(model_name_or_path):
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side=padding_side)

    def tokenize_function(example):
        # max_length=None => use the model's max length (it's actually the default)
        # outputs = tokenizer(examples["text"], truncation=True, max_length=400)
        # from IPython import embed; embed()
        # example['input'] = "[QUESTION] " + example['Question'] + " [ANSWER] " + example['Answer']
        outputs = tokenizer(example['input'], truncation=True)
        example["input_ids"] = outputs["input_ids"]
        example["attention_mask"] = outputs["attention_mask"]
        return example

    return tokenizer, tokenize_function


def sweep_train(config=None):

    wandb.init(name="", config=config, project="noise-studies")
    config = wandb.config
    config = dict(config)
    logger.info(config)

    if "paper" in config:
        config.pop("paper")

    t = Trainer(**config)
    t.train()

    t.eval_on = "test"
    t.eval(load_model=False, intermediate=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEV", action="store_true",
                        help="Whether it's a development run")
    parser.add_argument("--get_majority", action="store_true",
                        help="Whether it's a development run")
    parser.add_argument("--train_size", type=float,
                        default=0.8, help="Training data split size")
    parser.add_argument("--sampling_ratio", type=float,
                        default=0.3, help="Upsampling ratio for minority class")
    parser.add_argument("--MAX_LEN", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int,
                        default=32, help="Training batch size")
    parser.add_argument("--VALID_BATCH_SIZE", type=int,
                        default=64, help="Validation batch size")
    parser.add_argument("--LEARNING_RATE", type=float,
                        default=1e-04, help="Learning rate")
    parser.add_argument("--label_col", type=str,
                        default="Strategy", help="Label column name")
    parser.add_argument("--EPOCHS", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--LM", type=str, default="roberta-large",
                        help="the pretrained language model to use")
    parser.add_argument("--method", type=str, default="finetune",
                        help="the method to use for training")
    parser.add_argument("--dataset_name", type=str,
                        default="system12", help="the dataset for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    return args


def get_model(args):
    peft_config = None
    if args.method == "p_tuning":
        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
    elif args.method == "lora":
        peft_config = LoraConfig(
            task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.LM, return_dict=True)
    if peft_config:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


if __name__ == "__main__":
    args = parse_args()

    # ------------- Set Seed
    set_seed(args.seed)

    # ------------- Make Train/Val/Test Dataloaders
    if "/" in args.LM:
        LM_name = args.LM.split("/")[-1]
    run_name = f"{args.method}-{args.label_col}-{args.LM}-{args.seed}"
    wandb.init(project="system12", name=run_name, config=args)

    output_directory = os.path.join(
        "experiments", 'classifier', 
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{args.method}-{args.label_col}-{LM_name}")
    os.mkdir(output_directory)
    logger = create_logger(output_directory)
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using {device} device")

    df = get_dataset_loader_func(args.dataset_name)

    if args.DEV:
        df = df.sample(1000)

    # args_dict = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr)) and not attr.startswith("__")}
    # wandb.log(args_dict)

    tokenizer, tokenize_function = setup_tokenizer(args.LM)
    # Add special tokens
    special_tokens_dict = {
        'additional_special_tokens': ['[QUESTION]', '[ANSWER]']}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    train_dataset, val_dataset, test_dataset, label_weights = get_datasets(df=df,
                                                                           seed=args.seed,
                                                                           tokenize_function=tokenize_function,
                                                                           label_col=args.label_col,
                                                                           train_size=0.2)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest")

    # -------------- Set up Trainer
    model = get_model(args)

    model.resize_token_embeddings(len(tokenizer))

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_directory,
        learning_rate=1e-3 if args.method != "finetune" else 1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # metric_for_best_model="auc_roc",
        report_to="wandb",  # enable logging to W&B
        run_name=run_name,
        save_total_limit=1,
    )

    # Define the optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )

    # Add linear warmup scheduler
    num_warmup_steps = int(0.1 * args.EPOCHS * len(train_dataset) /
                           training_args.per_device_train_batch_size)
    num_training_steps = int(
        args.EPOCHS * len(train_dataset) / training_args.per_device_train_batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[ProgressCallback],
        # label_weights=torch.tensor(label_weights).to(device),
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    # -------------- Test

    res = trainer.predict(test_dataset)
    test_metrics = res.metrics
    test_metrics = {"test/"+k[5:]: v for k, v in test_metrics.items()}
    wandb.log(test_metrics)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description='Model trainer')
#     for k, v in get_base_parameters_trainer().items():
#         if type(v) == bool:
#             parser.add_argument(f"--{k.replace('_', '-')}", action="store_true")
#         else:
#             parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)

#     args = parser.parse_args()
#     print(args)

#     hyperparameter_defaults = vars(args)

#     sweep_hyperparameter_defaults = {k: {"value":hyperparameter_defaults[k]} for k in hyperparameter_defaults}


#     # ---------------- sweep -----------------
#     sweep_config = {
#     'method': 'random'
#     }

#     metric = {
#     'name': 'validation/f1score',
#     'goal': 'maximize'
#     }

#     sweep_config['metric'] = metric

#     parameters_dict = sweep_hyperparameter_defaults

#     parameters_dict.update({

#     'learning_rate': {
#         'values': [0.00001,0.0001, 0.001]
#         },
#         'classifier_dropout': {
#           'values': [0.3, 0.4, 0.5]
#         },
#         'num_train_epochs': {
#             'values': [5, 10, 15, 20]
#         },

#     })

#     print(sweep_config)

#     sweep_config['parameters'] = parameters_dict
#     sweep_id = wandb.sweep(sweep_config, project="noise-studies")

#     wandb.agent(sweep_id, sweep_train, count=5)


#     sweep_config['parameters'] = parameters_dict

#     wandb.agent(sweep_id, sweep_train, count=5)
