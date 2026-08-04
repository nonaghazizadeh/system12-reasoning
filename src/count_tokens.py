"""
Count the number of tokens in benchmark results for System1 and System2.
Uses the tokenizer to get accurate token counts.
"""

import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict

# Paths
SYSTEM1_DIR = "/home/nona/system12/src/interpretability/results/DPO/System1"
SYSTEM2_DIR = "/home/nona/system12/src/interpretability/results/DPO/System2"

# Load tokenizer (using Llama tokenizer for accurate counts)
print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
except:
    # Fallback to a simpler estimation if tokenizer not available
    print("Could not load Llama tokenizer, using rough word-based estimation")
    tokenizer = None


def count_tokens(text, tokenizer=None):
    """Count tokens in text using tokenizer or rough estimation."""
    if pd.isna(text) or text == "":
        return 0
    text = str(text)
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        # Rough estimation: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)


def analyze_benchmark(csv_path, tokenizer=None):
    """Analyze token counts for a single benchmark CSV."""
    df = pd.read_csv(csv_path)
    
    results = {
        'num_samples': len(df),
        'input_tokens': [],
        'output_tokens': [],  # pred_before column
    }
    
    for idx, row in df.iterrows():
        # Count input tokens
        input_text = str(row.get('input', ''))
        input_tokens = count_tokens(input_text, tokenizer)
        results['input_tokens'].append(input_tokens)
        
        # Count output tokens (pred_before is the generated response)
        output_text = str(row.get('pred_before', ''))
        output_tokens = count_tokens(output_text, tokenizer)
        results['output_tokens'].append(output_tokens)
    
    return results


def process_directory(dir_path, system_name, tokenizer=None):
    """Process all benchmark CSVs in a directory."""
    all_results = {}
    
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    
    for csv_file in sorted(csv_files):
        # Extract benchmark name from filename
        benchmark_name = csv_file.replace('dpo - ', '').replace('.csv', '').upper()
        csv_path = os.path.join(dir_path, csv_file)
        
        print(f"  Processing {benchmark_name}...")
        results = analyze_benchmark(csv_path, tokenizer)
        all_results[benchmark_name] = results
    
    return all_results


def print_summary(system1_results, system2_results):
    """Print comprehensive summary of token counts."""
    
    print("\n" + "="*100)
    print("TOKEN COUNT SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'Benchmark':<15} | {'System':^8} | {'Samples':^8} | {'Avg Input':^12} | {'Avg Output':^12} | {'Min Out':^8} | {'Max Out':^8} | {'Std Out':^10}")
    print("-"*100)
    
    all_benchmarks = sorted(set(system1_results.keys()) | set(system2_results.keys()))
    
    total_s1_output = []
    total_s2_output = []
    
    for benchmark in all_benchmarks:
        for system_name, results in [("System1", system1_results), ("System2", system2_results)]:
            if benchmark in results:
                r = results[benchmark]
                n = r['num_samples']
                avg_input = np.mean(r['input_tokens'])
                avg_output = np.mean(r['output_tokens'])
                min_output = np.min(r['output_tokens'])
                max_output = np.max(r['output_tokens'])
                std_output = np.std(r['output_tokens'])
                
                if system_name == "System1":
                    total_s1_output.extend(r['output_tokens'])
                else:
                    total_s2_output.extend(r['output_tokens'])
                
                print(f"{benchmark:<15} | {system_name:^8} | {n:^8} | {avg_input:^12.1f} | {avg_output:^12.1f} | {min_output:^8} | {max_output:^8} | {std_output:^10.1f}")
        print("-"*100)
    
    # Overall summary
    print(f"\n{'OVERALL SUMMARY':^100}")
    print("="*100)
    
    if total_s1_output:
        print(f"\nSystem1 Output Tokens:")
        print(f"  - Total samples: {len(total_s1_output)}")
        print(f"  - Average: {np.mean(total_s1_output):.1f} tokens")
        print(f"  - Median: {np.median(total_s1_output):.1f} tokens")
        print(f"  - Std Dev: {np.std(total_s1_output):.1f}")
        print(f"  - Min: {np.min(total_s1_output)} tokens")
        print(f"  - Max: {np.max(total_s1_output)} tokens")
        print(f"  - 25th percentile: {np.percentile(total_s1_output, 25):.1f} tokens")
        print(f"  - 75th percentile: {np.percentile(total_s1_output, 75):.1f} tokens")
    
    if total_s2_output:
        print(f"\nSystem2 Output Tokens:")
        print(f"  - Total samples: {len(total_s2_output)}")
        print(f"  - Average: {np.mean(total_s2_output):.1f} tokens")
        print(f"  - Median: {np.median(total_s2_output):.1f} tokens")
        print(f"  - Std Dev: {np.std(total_s2_output):.1f}")
        print(f"  - Min: {np.min(total_s2_output)} tokens")
        print(f"  - Max: {np.max(total_s2_output)} tokens")
        print(f"  - 25th percentile: {np.percentile(total_s2_output, 25):.1f} tokens")
        print(f"  - 75th percentile: {np.percentile(total_s2_output, 75):.1f} tokens")
    
    # Combined stats
    all_output = total_s1_output + total_s2_output
    if all_output:
        print(f"\nCombined (Both Systems):")
        print(f"  - Total samples: {len(all_output)}")
        print(f"  - Average: {np.mean(all_output):.1f} tokens")
        print(f"  - Median: {np.median(all_output):.1f} tokens")
        print(f"  - Std Dev: {np.std(all_output):.1f}")
        
        # Recommendation for first-k tokens
        print(f"\n{'FIRST-K TOKEN RECOMMENDATION':^100}")
        print("="*100)
        avg = np.mean(all_output)
        median = np.median(all_output)
        p25 = np.percentile(all_output, 25)
        
        print(f"\nBased on your data:")
        print(f"  - Average output length: {avg:.0f} tokens")
        print(f"  - Median output length: {median:.0f} tokens")
        print(f"  - 25th percentile: {p25:.0f} tokens")
        print(f"\nRecommended k values for first-k token approach:")
        print(f"  - Aggressive (fastest): k = {min(32, int(p25 * 0.3))}-{min(32, int(p25 * 0.5))} tokens")
        print(f"  - Balanced: k = 32 tokens")
        print(f"  - Conservative: k = {min(64, int(median * 0.5))} tokens")
    
    return total_s1_output, total_s2_output


def main():
    print("="*100)
    print("TOKEN COUNT ANALYSIS FOR DPO BENCHMARKS")
    print("="*100)
    
    print(f"\nSystem1 directory: {SYSTEM1_DIR}")
    print(f"System2 directory: {SYSTEM2_DIR}")
    
    # Process System1
    print(f"\nProcessing System1...")
    system1_results = process_directory(SYSTEM1_DIR, "System1", tokenizer)
    
    # Process System2
    print(f"\nProcessing System2...")
    system2_results = process_directory(SYSTEM2_DIR, "System2", tokenizer)
    
    # Print summary
    s1_tokens, s2_tokens = print_summary(system1_results, system2_results)
    
    # Save detailed results to CSV
    output_data = []
    for system_name, results in [("System1", system1_results), ("System2", system2_results)]:
        for benchmark, r in results.items():
            output_data.append({
                'System': system_name,
                'Benchmark': benchmark,
                'Samples': r['num_samples'],
                'Avg_Input_Tokens': np.mean(r['input_tokens']),
                'Avg_Output_Tokens': np.mean(r['output_tokens']),
                'Median_Output_Tokens': np.median(r['output_tokens']),
                'Min_Output_Tokens': np.min(r['output_tokens']),
                'Max_Output_Tokens': np.max(r['output_tokens']),
                'Std_Output_Tokens': np.std(r['output_tokens']),
            })
    
    df_summary = pd.DataFrame(output_data)
    output_path = "/home/nona/system12/src/interpretability/results/token_count_summary.csv"
    df_summary.to_csv(output_path, index=False)
    print(f"\n\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
