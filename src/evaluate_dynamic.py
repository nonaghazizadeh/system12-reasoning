#!/usr/bin/env python3
"""Evaluate entropy-guided System 1/System 2 Multi-LoRA arbitration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from system12.benchmarks import (
    BENCHMARKS,
    exact_match,
    load_benchmark,
    normalize_answer,
    second_stage_prompt,
)
from system12.modeling import tokenize_chat
from system12.routing import select_reasoning_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system1-adapter", required=True)
    parser.add_argument("--system2-adapter", required=True)
    parser.add_argument("--dataset", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/dynamic"))
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--mean-weight", type=float, default=0.4)
    parser.add_argument("--max-reasoning-tokens", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_multi_adapter(system1_adapter: str, system2_adapter: str):
    config1 = PeftConfig.from_pretrained(system1_adapter)
    config2 = PeftConfig.from_pretrained(system2_adapter)
    if config1.base_model_name_or_path != config2.base_model_name_or_path:
        raise ValueError("System 1 and System 2 adapters must share one base model")
    base_model_id = config1.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(
        base_model, system1_adapter, adapter_name="system1"
    )
    model.load_adapter(system2_adapter, adapter_name="system2")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer, base_model_id


@torch.inference_mode()
def generate_prefix(model, tokenizer, prompt: str, adapter: str, token_count: int):
    model.set_adapter(adapter)
    encoded = tokenize_chat(tokenizer, [prompt], model_device(model))
    output = model.generate(
        **encoded,
        max_new_tokens=token_count,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_length = encoded["input_ids"].shape[1]
    prefix_ids = output.sequences[:, prompt_length:]
    entropies = []
    for logits in output.scores:
        probabilities = F.softmax(logits[0].float(), dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
        entropies.append(float(entropy.cpu()))
    return encoded, prefix_ids, entropies


@torch.inference_mode()
def continue_reasoning(
    model,
    tokenizer,
    encoded,
    prefix_ids,
    adapter: str,
    max_reasoning_tokens: int,
) -> str:
    model.set_adapter(adapter)
    prefix_length = prefix_ids.shape[1]
    complete_input = torch.cat([encoded["input_ids"], prefix_ids], dim=1)
    if prefix_length >= max_reasoning_tokens or (
        prefix_length and prefix_ids[0, -1].item() == tokenizer.eos_token_id
    ):
        complete_output = complete_input
    else:
        attention_mask = torch.ones_like(complete_input)
        complete_output = model.generate(
            input_ids=complete_input,
            attention_mask=attention_mask,
            max_new_tokens=max_reasoning_tokens - prefix_length,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = encoded["input_ids"].shape[1]
    return tokenizer.decode(complete_output[0, prompt_length:], skip_special_tokens=True)


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, adapter: str, max_tokens: int) -> str:
    model.set_adapter(adapter)
    encoded = tokenize_chat(tokenizer, [prompt], model_device(model))
    output = model.generate(
        **encoded,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )


def main() -> None:
    args = parse_args()
    if args.prefix_tokens <= 0:
        raise ValueError("--prefix-tokens must be positive")
    if args.prefix_tokens > args.max_reasoning_tokens:
        raise ValueError("--prefix-tokens cannot exceed --max-reasoning-tokens")
    set_seed(args.seed)
    examples = load_benchmark(args.dataset, data_root=args.data_root, seed=args.seed)
    if args.limit:
        examples = examples[: args.limit]
    model, tokenizer, base_model_id = load_multi_adapter(
        args.system1_adapter, args.system2_adapter
    )

    rows = []
    for example in tqdm(examples, desc=f"dynamic:{args.dataset}"):
        encoded1, prefix1, entropies1 = generate_prefix(
            model, tokenizer, example.prompt, "system1", args.prefix_tokens
        )
        encoded2, prefix2, entropies2 = generate_prefix(
            model, tokenizer, example.prompt, "system2", args.prefix_tokens
        )
        selected, scores = select_reasoning_system(
            entropies1, entropies2, mean_weight=args.mean_weight
        )
        if selected == "system1":
            encoded, prefix = encoded1, prefix1
        else:
            encoded, prefix = encoded2, prefix2
        reasoning = continue_reasoning(
            model,
            tokenizer,
            encoded,
            prefix,
            selected,
            args.max_reasoning_tokens,
        )
        final_prompt = second_stage_prompt(args.dataset, example.prompt, reasoning)
        final_output = generate_answer(
            model, tokenizer, final_prompt, selected, args.max_answer_tokens
        )
        rows.append(
            {
                "prompt": example.prompt,
                "selected_system": selected,
                "system1_mean_entropy": scores.system1_mean_entropy,
                "system2_mean_entropy": scores.system2_mean_entropy,
                "system1_entropy_variance": scores.system1_entropy_variance,
                "system2_entropy_variance": scores.system2_entropy_variance,
                "system1_score": scores.system1_score,
                "system2_score": scores.system2_score,
                "reasoning": reasoning,
                "final_output": final_output,
                "prediction": normalize_answer(args.dataset, final_output),
                "reference": normalize_answer(args.dataset, example.answer),
                "correct": exact_match(args.dataset, final_output, example.answer),
            }
        )

    accuracy = sum(row["correct"] for row in rows) / len(rows)
    algorithm = Path(args.system1_adapter).parent.name
    run_dir = args.output_dir / algorithm / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "base_model": base_model_id,
        "system1_adapter": args.system1_adapter,
        "system2_adapter": args.system2_adapter,
        "dataset": args.dataset,
        "seed": args.seed,
        "examples": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": accuracy,
        "prefix_tokens": args.prefix_tokens,
        "mean_weight": args.mean_weight,
        "system1_routes": sum(row["selected_system"] == "system1" for row in rows),
        "system2_routes": sum(row["selected_system"] == "system2" for row in rows),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
