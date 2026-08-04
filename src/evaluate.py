#!/usr/bin/env python3
"""Run the paper's two-stage exact-match evaluation on one model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm
from transformers import set_seed

from system12.benchmarks import (
    BENCHMARKS,
    benchmark_spec,
    exact_match,
    load_benchmark,
    normalize_answer,
    second_stage_prompt,
)
from system12.modeling import generate_text, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="base model or LoRA adapter path/ID")
    parser.add_argument("--dataset", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-reasoning-tokens", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--prompt-mode", choices=["zero-shot", "cot"], default="zero-shot")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the complete split")
    return parser.parse_args()


def batches(values: list, size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    examples = load_benchmark(args.dataset, data_root=args.data_root, seed=args.seed)
    if args.limit:
        examples = examples[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args.model)

    rows = []
    for batch in tqdm(list(batches(examples, args.batch_size)), desc=args.dataset):
        stage1_prompts = [example.prompt for example in batch]
        if args.prompt_mode == "cot":
            stage1_prompts = [f"{prompt}\nLet's think step by step." for prompt in stage1_prompts]
        reasonings = generate_text(
            model,
            tokenizer,
            stage1_prompts,
            max_new_tokens=args.max_reasoning_tokens,
        )
        final_prompts = [
            second_stage_prompt(args.dataset, example.prompt, reasoning)
            for example, reasoning in zip(batch, reasonings)
        ]
        final_outputs = generate_text(
            model,
            tokenizer,
            final_prompts,
            max_new_tokens=args.max_answer_tokens,
        )
        for example, reasoning, final_output in zip(batch, reasonings, final_outputs):
            rows.append(
                {
                    "prompt": example.prompt,
                    "reasoning": reasoning,
                    "final_output": final_output,
                    "prediction": normalize_answer(args.dataset, final_output),
                    "reference": normalize_answer(args.dataset, example.answer),
                    "correct": exact_match(args.dataset, final_output, example.answer),
                }
            )

    accuracy = sum(row["correct"] for row in rows) / len(rows)
    model_name = args.model.rstrip("/").split("/")[-1]
    run_dir = args.output_dir / model_name / args.prompt_mode / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "category": benchmark_spec(args.dataset).category,
        "prompt_mode": args.prompt_mode,
        "seed": args.seed,
        "examples": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": accuracy,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
