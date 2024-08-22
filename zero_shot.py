import os
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm.auto import tqdm
from utils import add_pad_token_id, get_tokenizer


def load_model_and_tokenizer(model_name):
    tokenizer = get_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer, model = add_pad_token_id(tokenizer, model)
    return model, tokenizer


def add_prompt(example):
    system_content = """
    You are a helpful assistant. Please classify the interaction between a user and an assistant as either System 1 or System 2 thinking. 
    
    System 1 thinking is fast, automatic, and intuitive.
    System 2 thinking is slow, deliberate, and analytical.

    Please respond with only System 1 or System 2 without any additional explanation.
    """

    user_message = example['messages'][0]['content']
    assistant_message = example['messages'][1]['content']

    user_content = f"""
    user: {user_message}
    
    Assistant: {assistant_message}
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    example['prompt'] = messages
    return example


def separate_datasets(dataset, pipe, batch_size=2):
    system1_data, system2_data = [], []
    for data in tqdm(dataset['train']):
        response = pipe(data['prompt'], max_new_tokens=10)
        clean_response = response[0]['generated_text'][-1]['content'].strip().lower()
        if clean_response == "system 1":
            system1_data.append(data)
        elif clean_response == "system 2":
            system2_data.append(data)
        else:
            print(f"Error: {clean_response}")

    return system1_data, system2_data


def create_and_save_datasets(system1_data, system2_data, system_12_folder):
    system1_dataset = Dataset.from_list(
        system1_data).remove_columns(['prompt'])
    system2_dataset = Dataset.from_list(
        system2_data).remove_columns(['prompt'])

    print(f"System 1 dataset size: {len(system1_dataset)}")
    print(f"System 2 dataset size: {len(system2_dataset)}")

    system1_dataset.to_json(os.path.join(
        system_12_folder, "system1_dataset.jsonl"), lines=True)
    system2_dataset.to_json(os.path.join(
        system_12_folder, "system2_dataset.jsonl"), lines=True)


def main(args):
    os.makedirs(args.system_12_folder, exist_ok=True)

    dataset = load_dataset("json", data_files=args.dataset_files)
    if args.sample_size > 0:
        dataset['train'] = dataset['train'].select(range(args.sample_size))
    dataset = dataset.map(add_prompt)
    print(f"Dataset size: {len(dataset['train'])}")

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    device="cuda" if torch.cuda.is_available() else "cpu")

    system1_data, system2_data = separate_datasets(dataset, pipe)
    create_and_save_datasets(system1_data, system2_data, args.system_12_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot classification of System 1 and System 2 thinking")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Name of the model to use")
    parser.add_argument("--system_12_folder", type=str, default="data/system_12_from_instruction_tunning",
                        help="Folder to save the System 1 and System 2 datasets")
    parser.add_argument("--dataset_files", nargs="+",
                        required=True, help="List of dataset files to process")
    parser.add_argument("--sample_size", type=int, default=10,
                        help="Number of samples to process (0 for all)")

    args = parser.parse_args()
    main(args)
