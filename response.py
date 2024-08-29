import argparse
import os
import torch
from tqdm import tqdm
from utils import get_dataset_loader_func, get_pipeline
from transformers import set_seed


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


def generate_response(text_generation_pipeline, df, system_content, user_first=False):
    responses = []
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
    parser.add_argument("--LM", type=str, default="google/gemma-1.1-7b-it",
                        help="the pretrained language model to use")
    parser.add_argument("--dataset_name", type=str,
                        choices=["system12_questions",
                                 "system12_combined_questions",
                                 "system12_gpt_questions"],
                        default="system12_questions", help="Questions to ask the language model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--system_prompt_format", type=str,
                        choices=["simple", "short", "straightforward",
                                 "annotator", "shortandsimple", "one-sentence"],
                        default="simple", help="Questions to ask the language model")
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
        "experiments", 'responder', f"{LM_name}", args.dataset_name)
    os.makedirs(output_directory, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = get_dataset_loader_func(args.dataset_name)

    # Initialize the pipeline for text generation
    text_generation_pipeline = get_pipeline(args.LM, device)
    user_first = True
    if 'Llama' in args.LM:
        user_first = False

    system_content_mapping = {
        "simple": "Please answer the question in the simplest form.",
        "short": "Please provide a brief response to the question.",
        "straightforward": "Please respond to the question in the most straightforward manner possible",
        "annotator": "Imagine you are an annotator and you need to answer the question.",
        "shortandsimple": "Please provide a short and simple response to the question.",
        "one-sentence": "Please provide a one-sentence response to the question.",
    }
    # Generate responses
    df = generate_response(text_generation_pipeline,
                           df,
                           system_content_mapping[args.system_prompt_format],
                           user_first=user_first)

    # Save responses
    df.to_csv(os.path.join(output_directory,
                           args.system_prompt_format + "_responses.csv"),
              index=False)
