import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_entropy import create_logger, LocalDecoder, answer_cleansing
from custom_datasets import BenchmarkDataset
from transformers import set_seed
import re


# --------------------------------------------
# Composite score decision logic
# --------------------------------------------
def choose_best_system(ent1, var1, ent2, var2, w_v=0.6, w_e=0.4, eps_ratio=0.01):
    """
    Decide which system is better based on entropy and variance entropy.
    Uses normalized composite score and 1% margin tie rule.

    ent1, var1 = entropy and variance-entropy of system 1
    ent2, var2 = entropy and variance-entropy of system 2
    w_v, w_e   = weights for variance vs entropy
    eps_ratio  = relative margin for tie (default: 1% of mean score)
    """

    # Normalize relative to both systems
    v1 = var1 / (var1 + var2)
    v2 = var2 / (var1 + var2)
    e1 = ent1 / (ent1 + ent2)
    e2 = ent2 / (ent1 + ent2)

    # Composite score (lower is better)
    s1 = w_v * v1 + w_e * e1
    s2 = w_v * v2 + w_e * e2

    # Tie margin = 1% of mean score
    eps = eps_ratio * ((s1 + s2) / 2.0)

    if abs(s1 - s2) <= eps:
        # Random choice when nearly tied
        choice = np.random.choice(["sys1", "sys2"])
    elif s1 < s2:
        choice = "sys1"
    else:
        choice = "sys2"

    return choice, s1, s2


@torch.inference_mode
def main():
    args = parse_arguments()
    output_directory = os.path.join("experiments", 'dynamic', args.model, args.algorithm, args.dataset)
    os.makedirs(output_directory, exist_ok=True)
    csv_file = os.path.join(output_directory, "result.csv")
    logger = create_logger(output_directory)
    logger.info('*****************************')
    logger.info(args)
    logger.info('*****************************')

    set_seed(args.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "llama":
        if args.algorithm == "dpo":
            decoder_system1 = LocalDecoder(
                model_name_or_path="./experiments/dpo/lora-Meta-Llama-3-8B-Instruct-system1",
                batch_size=args.batch_size, device=device
            )
            decoder_system2 = LocalDecoder(
                model_name_or_path="./experiments/dpo/lora-Meta-Llama-3-8B-Instruct-system2",
                batch_size=args.batch_size, device=device
            )
        elif args.algorithm == "simpo":
            decoder_system1 = LocalDecoder(
                model_name_or_path="./experiments/simpo/lora-Meta-Llama-3-8B-Instruct-system1",
                batch_size=args.batch_size, device=device
            )
            decoder_system2 = LocalDecoder(
                model_name_or_path="./experiments/simpo/lora-Meta-Llama-3-8B-Instruct-system2",
                batch_size=args.batch_size, device=device
            )
        else:
            raise ValueError(f"Algorithm {args.algorithm} not supported")
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    logger.info("setup data loader ...")
    dataset = BenchmarkDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    total = 0
    correct_list = []
    csv_data = {
        "input": [],
        "pred_before": [],
        "pred_after": [],
        "GT": [],
        "best_model": [],
        "score_sys1": [],
        "score_sys2": []
    }

    # dynamic evaluation
    for data in tqdm(dataloader):
        x, y = data
        x = list(x)
        
        z1 = decoder_system1.decode(x)
        z2 = decoder_system2.decode(x)

        # For each sample in batch
        for i in range(len(z1)):
            entropy1 = z1[i]['sequence_entropy']
            entropy2 = z2[i]['sequence_entropy']
            variance_entropy1 = z1[i]['entropy_variance']
            variance_entropy2 = z2[i]['entropy_variance']

            best_model, s1, s2 = choose_best_system(
                entropy1, variance_entropy1,
                entropy2, variance_entropy2
            )

            if best_model == "sys1":
                z = z1[i]['generated_text']
            elif best_model == "sys2":
                z = z2[i]['generated_text']

            print(f"Sys1 -> Ent={entropy1:.3f}, Var={variance_entropy1:.3f}, Score={s1:.3f}")
            print(f"Sys2 -> Ent={entropy2:.3f}, Var={variance_entropy2:.3f}, Score={s2:.3f}")
            print(f"Chosen: {best_model}")
            print("--------------------------------")

            # Feed final answer back into model for consistency
            z_final = [temp + "\n" + temp_out + args.direct_answer_trigger for temp, temp_out in zip([x[i]], [z])]
            if best_model == "sys1":
                pred = decoder_system1.decode(z_final)
            elif best_model == "sys2":
                pred = decoder_system2.decode(z_final)

            csv_data["input"] += z_final
            csv_data["pred_before"] += pred
            pred = answer_cleansing(args, pred)
            csv_data["pred_after"] += pred
            csv_data["best_model"] += [best_model]
            csv_data["GT"] += [y[i]]
            csv_data["score_sys1"] += [s1]
            csv_data["score_sys2"] += [s2]

            pred = clean_pred(pred)
            gt_clean = clean_ans([y[i]])
            correct = (np.array(pred) == np.array(gt_clean)).sum().item()
            correct_list.append(correct)
            total += 1
            
            if (args.limit_dataset_size != 0) and (total >= args.limit_dataset_size):
                break
            
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    logger.info(f"accuracy : {accuracy}")
    csv_data = pd.DataFrame(csv_data)
    data = final_clean_ans(args, csv_data)
    csv_data.to_csv(csv_file, index=False)


# ------------------------------------------------
# Helper functions (unchanged from your code)
# ------------------------------------------------
def clean_ans(answers):
    new_answers = []
    for ans in answers:
        new_ans = ""
        for i in range(len(ans)):
            if ans[i] == ",":
                continue
            new_ans += ans[i]

        if '.' in new_ans:
            pos = new_ans.find('.')
            if len(new_ans) - pos - 1 > 7:
                new_ans = new_ans[:pos + 7]
        new_answers.append(new_ans)
    return new_answers


def clean_pred(preds):
    clean_preds = []
    for pred in preds:
        if '.' in pred:
            pred = pred.rstrip('0')
            if pred.endswith('.'):
                pred = pred[:-1]

        if '.' in pred:
            pos = pred.find('.')
            if len(pred) - pos - 1 > 7:
                pred = pred[:pos + 7]
        clean_preds.append(pred)
    return clean_preds


def extract_last_number(text):
    matches = re.findall(r'\d+\.?\d*', text)
    return matches[-1] if matches else None

def extract_last_letter_in_parenthesis_af(text):
    matches = re.findall(r'\(?(A|B|C|D|E|F)\)', text)
    return matches[-1] if matches else None

def extract_last_letter_in_parenthesis_ae(text):
    matches = re.findall(r'\(?(A|B|C|D|E)\)', text)
    return matches[-1] if matches else None

def extract_last_letter_in_parenthesis_ac(text):
    matches = re.findall(r'\(?(A|B|C)\)', text)
    return matches[-1] if matches else None

def extract_last_yes_no(text):
    matches = re.findall(r'\b(?:yes|no)\b', text, re.IGNORECASE)
    return matches[-1].lower() if matches else None

def process_text_last_letters(text):
    text = str(text)
    if ":" in text:
        text = text.split(":")[-1]
    elif "is" in text:
        text = text.split("is")[-1]
    return text.lower().replace("-", "").replace(" ","").replace(".","")

def final_clean_ans(args, csv):
    if args.dataset in ["gsm8k", "multiarith", "svamp", "addsub", "singleeq"]:
        csv['pred_after'] = csv['pred_before'].astype(str).apply(extract_last_number)
    elif args.dataset in ["coin_flip", "strategyqa"]:
        csv['pred_after'] = csv['pred_before'].astype(str).apply(extract_last_yes_no)
    elif args.dataset in ["commonsensqa", "aqua"]:
        csv['pred_after'] = csv['pred_before'].astype(str).apply(extract_last_letter_in_parenthesis_ae)
    elif args.dataset == "bigbench_date":
        csv['pred_after'] = csv['pred_before'].astype(str).apply(extract_last_letter_in_parenthesis_af)
    elif args.dataset == "object_tracking":
        csv['pred_after'] = csv['pred_before'].astype(str).apply(extract_last_letter_in_parenthesis_ac)
    elif args.dataset == "last_letters":
        csv['pred_after'] = csv['pred_before'].astype(str).apply(process_text_last_letters)
    return csv


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--dataset", type=str, default="aqua", 
                        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",
                                 "strategyqa", "svamp", "singleeq", "bigbench_date",
                                 "object_tracking", "coin_flip", "last_letters", 
                                 "age", "disability_status", "gender_identity", 
                                 "nationality", "physical_appearance", "race_ethnicity", 
                                 "race_x_gender", "race_x_ses", "religion", "ses", 
                                 "sexual_orientation", "socialIQa", "PIQA", "com2sense"],
                        help="dataset used for experiment")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="model used for decoding.")
    parser.add_argument("--algorithm", type=str, default="dpo", help="algorithm to use")
    parser.add_argument("--limit_dataset_size", type=int, default=10,
                        help="limit test dataset size. if 0, use full dataset")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
