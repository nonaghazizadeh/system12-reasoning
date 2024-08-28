import os
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, pipeline
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


def model_api(prompt, pipe):
    response = pipe(prompt, max_new_tokens=10)
    clean_response = response[0]['generated_text'][-1]['content'].strip().lower()
    return clean_response


def separate_datasets(dataset, pipe, start_index=0, cache_size=100):
    system1_data, system2_data = [], []
    for index in tqdm(range(start_index, len(dataset['train']))):
        data = dataset['train'][index]
        response = model_api(prompt=data['prompt'], pipe=pipe)
        if response == "system 1":
            system1_data.append(data)
        elif response == "system 2":
            system2_data.append(data)
        else:
            print(f"Error: {response}")

        if (index + 1) % cache_size == 0:
            print(f"Saving datasets at length of {index + 1}")
            create_and_save_datasets(
                system1_data, system2_data, args.system_12_folder, index + 1)
            system1_data, system2_data = [], []

    create_and_save_datasets(system1_data, system2_data, args.system_12_folder)
    combine_and_remove_cache_datasets(args.system_12_folder)


def combine_and_remove_cache_datasets(system_12_folder):
    files = os.listdir(system_12_folder)
    system1_files = [os.path.join(system_12_folder, file)
                     for file in files if file.startswith("system1_")]
    system2_files = [os.path.join(system_12_folder, file)
                     for file in files if file.startswith("system2_")]

    system_1_dataset = load_dataset("json", data_files=system1_files)
    system_2_dataset = load_dataset("json", data_files=system2_files)

    system_1_dataset['train'].to_json(os.path.join(
        system_12_folder, "system1.jsonl"), lines=True)
    system_2_dataset['train'].to_json(os.path.join(
        system_12_folder, "system2.jsonl"), lines=True)

    for file in files:
        os.remove(os.path.join(system_12_folder, file))


def create_and_save_dataset(data, file):
    if len(data) == 0:
        return
    dataset = Dataset.from_list(data).remove_columns(['prompt'])
    dataset.to_json(file, lines=True)


def create_and_save_datasets(system1_data, system2_data, system_12_folder, start_index=-1):
    create_and_save_dataset(system1_data, os.path.join(
        system_12_folder, f"system1_{start_index}.jsonl"))
    create_and_save_dataset(system2_data, os.path.join(
        system_12_folder, f"system2_{start_index}.jsonl"))

    print(f"System 1 dataset size: {len(system1_data)}")
    print(f"System 2 dataset size: {len(system2_data)}")


def get_start_index(system_12_folder):
    files = os.listdir(system_12_folder)
    if len(files) == 0:
        return 0

    indcies = [int(file.split("_")[-1].split(".")[0]) for file in files]
    return max(indcies)


def check_done(system_12_folder):
    system1_file = os.path.join(system_12_folder, "system1.jsonl")
    system2_file = os.path.join(system_12_folder, "system2.jsonl")
    return os.path.exists(system1_file) and os.path.exists(system2_file)


def main(args):
    os.makedirs(args.system_12_folder, exist_ok=True)
    start_index = get_start_index(args.system_12_folder)
    if check_done(args.system_12_folder):
        print("Already done")
        return

    dataset = load_dataset("json", data_files=args.dataset_files)
    if args.sample_size > 0:
        dataset['train'] = dataset['train'].select(range(args.sample_size))
    dataset = dataset.map(add_prompt)
    print(f"Dataset size: {len(dataset['train'])}, start index: {start_index}")

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    device="cuda" if torch.cuda.is_available() else "cpu")

    separate_datasets(dataset, pipe, start_index)


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
