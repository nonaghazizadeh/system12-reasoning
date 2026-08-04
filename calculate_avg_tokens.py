#!/usr/bin/env python3
"""
Calculate the average token numbers for prob_benchmark addsub system1 and system2.

The data files contain arrays of [token, probability] pairs.
Each response ends with <|eot_id|> (prob 1.0), followed by padding <|eot_id|> (prob 0.0).
We count tokens up to and including the first <|eot_id|> with prob 1.0 for each response.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple


def count_tokens_per_response(data: List[List]) -> List[int]:
    """
    Count tokens for each response in the data.
    
    A response ends with <|eot_id|> (prob 1.0).
    Padding tokens are <|eot_id|> with prob 0.0 (we skip these).
    
    Returns a list of token counts, one per response.
    """
    token_counts = []
    current_count = 0
    in_padding = False
    
    for token, prob in data:
        if token == "<|eot_id|>":
            if prob == 1.0 and not in_padding:
                # End of a response (include this token in count)
                current_count += 1
                token_counts.append(current_count)
                current_count = 0
                in_padding = True
            elif prob == 0.0:
                # Padding token, skip it
                continue
            else:
                # Edge case: eot_id with other probability
                current_count += 1
        else:
            # Regular token
            in_padding = False
            current_count += 1
    
    # Handle case where last response doesn't end with eot_id
    if current_count > 0:
        token_counts.append(current_count)
    
    return token_counts


def process_directory(dir_path: str) -> Tuple[int, int, float]:
    """
    Process all data.json files in a directory.
    
    Returns: (total_tokens, total_responses, average_tokens)
    """
    total_tokens = 0
    total_responses = 0
    
    # Get all *data.json files
    data_files = sorted(Path(dir_path).glob("*data.json"))
    
    print(f"\nProcessing directory: {dir_path}")
    print(f"Found {len(data_files)} data files")
    
    for file_path in data_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            token_counts = count_tokens_per_response(data)
            file_total = sum(token_counts)
            file_responses = len(token_counts)
            
            total_tokens += file_total
            total_responses += file_responses
            
            if file_responses > 0:
                file_avg = file_total / file_responses
                print(f"  {file_path.name}: {file_responses} responses, avg {file_avg:.2f} tokens/response")
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
    
    avg_tokens = total_tokens / total_responses if total_responses > 0 else 0
    return total_tokens, total_responses, avg_tokens


def main():
    base_path = "/home/nona/system12/experiments/prob_benchmark/dpo"
    
    # System 1
    system1_path = os.path.join(base_path, "lora-Meta-Llama-3-8B-Instruct-system1", "addsub")
    total1, responses1, avg1 = process_directory(system1_path)
    
    # System 2
    system2_path = os.path.join(base_path, "lora-Meta-Llama-3-8B-Instruct-system2", "addsub")
    total2, responses2, avg2 = process_directory(system2_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSystem 1 (addsub):")
    print(f"  Total responses: {responses1}")
    print(f"  Total tokens: {total1}")
    print(f"  Average tokens per response: {avg1:.2f}")
    
    print(f"\nSystem 2 (addsub):")
    print(f"  Total responses: {responses2}")
    print(f"  Total tokens: {total2}")
    print(f"  Average tokens per response: {avg2:.2f}")
    
    if avg1 > 0 and avg2 > 0:
        diff = avg2 - avg1
        pct_diff = (diff / avg1) * 100
        print(f"\nDifference (System2 - System1): {diff:+.2f} tokens ({pct_diff:+.2f}%)")


if __name__ == "__main__":
    main()
