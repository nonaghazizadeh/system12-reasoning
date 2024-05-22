import argparse
import os
import torch
from tqdm import tqdm
from utils import set_seed, get_dataset_loader_func
from custom_datasets import CustomDataset
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from peft import PeftModel
from train import add_pad_token_id


def create_dataloader(df, tokenize_function):

    df['input'] = "[QUESTION] " + df['Question'] + " [ANSWER] " + df['response']

    df = df.apply(tokenize_function, axis=1)

    tokenized_dataset = CustomDataset(df)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest")

    dataloader = DataLoader(tokenized_dataset,
                            shuffle=False, batch_size=args.BATCH_SIZE,
                            collate_fn=data_collator)

    return dataloader


def load_model(model, model_path, method):
    if method == "finetune":  # TODO: add file name
        model.load_state_dict(torch.load(model_path))
    else:
        model = PeftModel.from_pretrained(model, model_path)
    return model


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
        # example['input'] = "[QUESTION] " + example['Question'] + " [ANSWER] " + example['Answer']

        outputs = tokenizer(
            example['input'], truncation=True, return_tensors="pt")
        example["input_ids"] = outputs["input_ids"]
        example["attention_mask"] = outputs["attention_mask"]
        return example

    return tokenizer, tokenize_function


@torch.no_grad()
def predict(model, dataloader, device):
    all_probs, all_preds = [], []
    model.to(device)
    model.eval()
    for batch in tqdm(dataloader):
        # Move input tensors to the appropriate device (e.g., GPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Perform forward pass through the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract logits from the output
        logits = outputs.logits

        # For evaluation or inference, you might want to calculate accuracy or make predictions
        logits_max = torch.max(logits, dim=1)
        predictions, probabilities = logits_max.indices, logits_max.values

        all_probs.extend(probabilities.cpu().tolist())
        all_preds.extend(predictions.cpu().tolist())

    return {"probabilities": all_probs, "predictions": all_preds}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BATCH_SIZE", type=int,
                        default=16, help="inference batch size")
    parser.add_argument("--LM_classifier_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="the language model id")
    parser.add_argument("--LM_classifier_path", type=str,
                        help="the language model path")
    parser.add_argument("--method", type=str, default="lora",
                        choices=['lora', 'finetune'],
                        help="the method to use for training")
    parser.add_argument("--LM_responder", type=str,
                        choices=['gemma-1.1-7b-it',
                                 'Meta-Llama-3-8B-Instruct',
                                 'Mistral-7B-Instruct-v0.3', 'Mistral-7B-Instruct-v0.2',
                                 'Phi-3-small-128k-instruct'],
                        default="gemma-1.1-7b-it", help="the dataset for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--system_prompt_format", type=str,
                        choices=["simple", "short", "straightforward",
                                 "annotator", "shortandsimple", "one-sentence"],
                        default="simple", help="system prompt that we asked the question with.")
    parser.add_argument("--dataset_name", type=str,
                        choices=["system12_questions",
                                 'system12_combined_questions',
                                 "system12_gpt_questions"],
                        default="system12_questions", help="Questions to ask the language model")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # ------------- Set Seed
    set_seed(args.seed)

    # ------------- Make Train/Val/Test Dataloaders

    model_directory = os.path.join(
        "experiments", 'classifier', args.LM_classifier_path)
    output_directory = os.path.join(
        "experiments", 'responder', args.LM_responder, args.dataset_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    tokenizer, tokenize_function = setup_tokenizer(model_directory)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.LM_classifier_id, return_dict=True)
    model.resize_token_embeddings(len(tokenizer))
    model = load_model(model, model_directory, args.method)

    tokenizer, model = add_pad_token_id(tokenizer, model)

    df = get_dataset_loader_func(os.path.join(output_directory,
                                              args.system_prompt_format + "_responses.csv"))

    dataloader = create_dataloader(df=df,
                                   tokenize_function=tokenize_function)

    results = predict(model, dataloader, device)

    df['probabilities'] = results['probabilities']
    df['predictions'] = results['predictions']

    df.to_csv(os.path.join(output_directory,
                           args.system_prompt_format + "_predictions.csv"), index=False)
