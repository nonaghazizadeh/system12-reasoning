"""
Dynamic benchmark with Multi-LoRA serving for reduced inference cost.

Key optimizations:
1. Single base model loaded once (instead of twice)
2. Both LoRA adapters loaded and swappable at runtime
3. Batched inference across both systems using the same base model
4. Significantly reduced GPU memory usage and inference latency
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_entropy import create_logger, answer_cleansing
from custom_datasets import BenchmarkDataset
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch.nn.functional as F
import re


# --------------------------------------------
# BitsAndBytes quantization config
# --------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


def add_pad_token_id(tokenizer, model):
    """Add pad token if not present."""
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id") is None:
        model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model


class MultiLoRADecoder:
    """
    Multi-LoRA Decoder that loads a single base model with multiple LoRA adapters.
    
    This dramatically reduces inference cost compared to loading separate models:
    - Single base model in memory (vs 2x memory for dual models)
    - Fast adapter switching at runtime (no model reload)
    - Efficient batching with adapter-aware generation
    
    References:
    - PEFT multi-adapter support: https://huggingface.co/docs/peft/developer_guides/adapter_save_load
    - vLLM multi-LoRA: https://docs.vllm.ai/en/latest/configuration/engine_args.html
    """
    
    def __init__(
        self, 
        base_model_path: str,
        adapter_paths: dict,  # {"system1": path1, "system2": path2}
        device: torch.device,
        batch_size: int,
        max_len: int = 256
    ):
        """
        Initialize Multi-LoRA decoder with shared base model.
        
        Args:
            base_model_path: Path to base model (e.g., meta-llama/Meta-Llama-3-8B-Instruct)
            adapter_paths: Dictionary mapping adapter names to their paths
            device: Torch device for inference
            batch_size: Batch size for inference
            max_len: Maximum generation length
        """
        self.device = device
        self.max_len = max_len
        self.adapter_names = list(adapter_paths.keys())
        
        print(f"[MultiLoRA] Loading base model: {base_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, 
            padding_side="left"
        )
        
        # Load base model with quantization
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Add pad token
        self.tokenizer, self.base_model = add_pad_token_id(self.tokenizer, self.base_model)
        
        # Load first adapter and merge as PeftModel
        first_adapter_name = self.adapter_names[0]
        first_adapter_path = adapter_paths[first_adapter_name]
        print(f"[MultiLoRA] Loading primary adapter '{first_adapter_name}' from: {first_adapter_path}")
        
        self.model = PeftModel.from_pretrained(
            self.base_model,
            first_adapter_path,
            adapter_name=first_adapter_name
        )
        
        # Load additional adapters
        for adapter_name in self.adapter_names[1:]:
            adapter_path = adapter_paths[adapter_name]
            print(f"[MultiLoRA] Loading adapter '{adapter_name}' from: {adapter_path}")
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)
        
        print(f"[MultiLoRA] All adapters loaded: {self.adapter_names}")
        print(f"[MultiLoRA] Active adapter: {self.model.active_adapter}")
        
    def set_adapter(self, adapter_name: str):
        """Switch to a specific LoRA adapter (fast operation, no model reload)."""
        if adapter_name not in self.adapter_names:
            raise ValueError(f"Unknown adapter: {adapter_name}. Available: {self.adapter_names}")
        self.model.set_adapter(adapter_name)
        
    def decode(self, inputs: list, adapter_name: str = None) -> list:
        """
        Generate responses using specified adapter.
        
        Args:
            inputs: List of input strings
            adapter_name: Which adapter to use (optional, uses current if not specified)
            
        Returns:
            List of dicts with 'generated_text', 'sequence_entropy', 'entropy_variance'
        """
        if adapter_name is not None:
            self.set_adapter(adapter_name)
            
        # Prepare conversations
        conversations = []
        for input_text in inputs:
            conversation = [{"role": "user", "content": input_text}]
            conversations.append(conversation)
        
        # Tokenize
        tokenized = self.tokenizer.apply_chat_template(
            conversations,
            add_special_tokens=False,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_len,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        
        # Process results
        results = []
        input_length = tokenized["input_ids"].shape[1]
        
        for batch_idx in range(tokenized["input_ids"].shape[0]):
            generated_sequence = outputs.sequences[batch_idx, input_length:]
            generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            
            # Calculate token-level entropies
            token_entropies = []
            for i, token_id in enumerate(generated_sequence):
                if i < len(outputs.scores):
                    logits = outputs.scores[i][batch_idx]
                    probs = F.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1).item()
                    token_entropies.append(entropy)
            
            # Aggregate entropy metrics
            if token_entropies:
                sequence_entropy = np.mean(token_entropies)
                entropy_variance = np.var(token_entropies)
            else:
                sequence_entropy = 0.0
                entropy_variance = 0.0
            
            results.append({
                'generated_text': generated_text,
                'sequence_entropy': sequence_entropy,
                'entropy_variance': entropy_variance
            })
        
        return results
    
    def decode_both_systems(self, inputs: list) -> tuple:
        """
        Efficiently generate from both systems for the same inputs.
        
        This method switches adapters between generations, which is very fast
        (just pointer swap) compared to loading separate models.
        
        Args:
            inputs: List of input strings
            
        Returns:
            Tuple of (system1_results, system2_results)
        """
        z1 = self.decode(inputs, adapter_name="system1")
        z2 = self.decode(inputs, adapter_name="system2")
        return z1, z2


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

    # Handle edge case where both are zero
    if (var1 + var2) == 0:
        v1, v2 = 0.5, 0.5
    else:
        v1 = var1 / (var1 + var2)
        v2 = var2 / (var1 + var2)
    
    if (ent1 + ent2) == 0:
        e1, e2 = 0.5, 0.5
    else:
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


@torch.inference_mode()
def main():
    args = parse_arguments()
    output_directory = os.path.join("experiments", 'dynamic_multilora', args.model, args.algorithm, args.dataset)
    os.makedirs(output_directory, exist_ok=True)
    csv_file = os.path.join(output_directory, "result.csv")
    logger = create_logger(output_directory)
    logger.info('*****************************')
    logger.info(args)
    logger.info('*****************************')

    set_seed(args.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configure Multi-LoRA decoder (single model, multiple adapters)
    if args.model == "llama":
        base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        if args.algorithm == "dpo":
            adapter_paths = {
                "system1": "./experiments/dpo/lora-Meta-Llama-3-8B-Instruct-system1",
                "system2": "./experiments/dpo/lora-Meta-Llama-3-8B-Instruct-system2",
            }
        elif args.algorithm == "simpo":
            adapter_paths = {
                "system1": "./experiments/simpo/lora-Meta-Llama-3-8B-Instruct-system1",
                "system2": "./experiments/simpo/lora-Meta-Llama-3-8B-Instruct-system2",
            }
        else:
            raise ValueError(f"Algorithm {args.algorithm} not supported")
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    # Initialize Multi-LoRA decoder (loads base model + all adapters once)
    logger.info("[MultiLoRA] Initializing decoder with shared base model...")
    decoder = MultiLoRADecoder(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        device=device,
        batch_size=args.batch_size
    )
    logger.info("[MultiLoRA] Decoder initialized. Memory cost reduced by ~50% compared to dual model loading.")
    
    logger.info("Setting up dataloader ...")
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

    # Dynamic evaluation with Multi-LoRA
    for data in tqdm(dataloader):
        x, y = data
        x = list(x)
        
        # Generate from both systems efficiently (single model, adapter swap)
        z1, z2 = decoder.decode_both_systems(x)

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
                selected_adapter = "system1"
            else:
                z = z2[i]['generated_text']
                selected_adapter = "system2"

            print(f"Sys1 -> Ent={entropy1:.3f}, Var={variance_entropy1:.3f}, Score={s1:.3f}")
            print(f"Sys2 -> Ent={entropy2:.3f}, Var={variance_entropy2:.3f}, Score={s2:.3f}")
            print(f"Chosen: {best_model}")
            print("--------------------------------")

            # Feed final answer back into model for consistency (using selected adapter)
            z_final = [temp + "\n" + temp_out + args.direct_answer_trigger 
                      for temp, temp_out in zip([x[i]], [z])]
            pred = decoder.decode(z_final, adapter_name=selected_adapter)

            csv_data["input"] += z_final
            csv_data["pred_before"] += pred
            pred_text = [p['generated_text'] for p in pred]
            pred_text = answer_cleansing(args, pred_text)
            csv_data["pred_after"] += pred_text
            csv_data["best_model"] += [best_model]
            csv_data["GT"] += [y[i]]
            csv_data["score_sys1"] += [s1]
            csv_data["score_sys2"] += [s2]

            pred_clean = clean_pred(pred_text)
            gt_clean = clean_ans([y[i]])
            correct = (np.array(pred_clean) == np.array(gt_clean)).sum().item()
            correct_list.append(correct)
            total += 1
            
            if (args.limit_dataset_size != 0) and (total >= args.limit_dataset_size):
                break
            
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    logger.info(f"Accuracy: {accuracy}")
    csv_data = pd.DataFrame(csv_data)
    data = final_clean_ans(args, csv_data)
    csv_data.to_csv(csv_file, index=False)


# ------------------------------------------------
# Helper functions
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
    parser = argparse.ArgumentParser(description="Dynamic benchmark with Multi-LoRA serving")

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
    parser.add_argument("--model", type=str, default="llama", help="model used for decoding.")
    parser.add_argument("--algorithm", type=str, default="dpo", help="algorithm to use (dpo/simpo)")
    parser.add_argument("--limit_dataset_size", type=int, default=10,
                        help="limit test dataset size. if 0, use full dataset")
    parser.add_argument("--direct_answer_trigger", type=str, default="\nTherefore, the answer is",
                        help="trigger for direct answer")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
