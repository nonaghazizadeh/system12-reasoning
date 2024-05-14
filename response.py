import argparse
import os
import torch
import datetime
from tqdm import tqdm
from utils import create_logger, set_seed, get_dataset_loader_func
from transformers import pipeline


def create_prompt(question, system_content, user_first):
    if user_first:
        user_message = {
            "role": "user",
            "content": f'{system_content} question: {question}'
        }
        message = [user_message]
    else:
        system_message = {
            "role": "assistant",
            "content": system_content
        }
        user_message = {
            "role": "user",
            "content": f'question: {question}'
        }
        message = [system_message, user_message]
    return message


def generate_response(text_generation_pipeline, df, user_first=False):
    responses = []
    system_content = "Please answer the question in the simplest form."
    for idx, row in tqdm(df.iterrows()):
        question = row["Question"]
        messages = create_prompt(question, system_content, user_first)
        response = text_generation_pipeline(messages, max_new_tokens=args.MAX_LEN)[
            0]['generated_text'][-1]['content']
        responses.append(response)
    df['response'] = responses
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--MAX_LEN", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--LM", type=str, default="roberta-large",
                        help="the pretrained language model to use")
    parser.add_argument("--dataset_name", type=str, choices=["system12_questions"],
                        default="system12_questions", help="Questions to ask the language model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # ------------- Set Seed
    set_seed(args.seed)

    # ------------- Make Train/Val/Test Dataloaders
    if "/" in args.LM:
        LM_name = args.LM.split("/")[-1]

    output_directory = os.path.join(
        "experiments", 'responder', f"{LM_name}")
    os.mkdir(output_directory)
    logger = create_logger(output_directory)
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using {device} device")

    df = get_dataset_loader_func(args.dataset_name)

    # Initialize the pipeline for text generation
    text_generation_pipeline = pipeline("text-generation", model=args.LM)
    user_first = False
    if 'Mistral' in args.LM or 'google' in args.LM:
        user_first = True
    # Generate responses
    df = generate_response(text_generation_pipeline, df, user_first=user_first)

    # Save responses
    df.to_csv(os.path.join(output_directory, "responses.csv"), index=False)
