#!/usr/bin/env python3
"""Train System 1, System 2, or interpolated LoRA preference adapters."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed
from trl import CPOConfig, CPOTrainer, DPOConfig, DPOTrainer

from system12.preferences import prepare_preference_splits


@dataclass(frozen=True)
class PaperHyperparameters:
    learning_rate: float
    beta: float
    gamma_over_beta: float | None = None


def paper_hyperparameters(base_model: str, algorithm: str) -> PaperHyperparameters:
    is_mistral = "mistral" in base_model.lower()
    if algorithm == "dpo":
        return PaperHyperparameters(
            learning_rate=5e-7 if is_mistral else 7e-7,
            beta=0.001 if is_mistral else 0.01,
        )
    return PaperHyperparameters(
        learning_rate=5e-7 if is_mistral else 1e-6,
        beta=2.5,
        gamma_over_beta=0.1 if is_mistral else 0.55,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["dpo", "simpo"], required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alignment-data", type=Path, default=Path("data/system12/cogbias.csv")
    )
    style = parser.add_mutually_exclusive_group(required=True)
    style.add_argument("--system1", action="store_true")
    style.add_argument("--system2", action="store_true")
    style.add_argument(
        "--system1-fraction",
        type=float,
        help="fraction of prompts preferring System 1; use for spectrum models",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma-over-beta", type=float)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb-project", default="reasoning-on-a-spectrum")
    return parser.parse_args()


def resolve_system1_fraction(args: argparse.Namespace) -> float:
    if args.system1:
        return 1.0
    if args.system2:
        return 0.0
    if args.system1_fraction is None or not 0 <= args.system1_fraction <= 1:
        raise ValueError("--system1-fraction must be between 0 and 1")
    return args.system1_fraction


def main() -> None:
    args = parse_args()
    system1_fraction = resolve_system1_fraction(args)
    defaults = paper_hyperparameters(args.base_model, args.algorithm)
    learning_rate = args.learning_rate or defaults.learning_rate
    beta = args.beta if args.beta is not None else defaults.beta
    gamma_over_beta = (
        args.gamma_over_beta
        if args.gamma_over_beta is not None
        else defaults.gamma_over_beta
    )

    set_seed(args.seed)
    splits = prepare_preference_splits(
        args.alignment_data,
        system1_fraction=system1_fraction,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    train_dataset = Dataset.from_pandas(
        splits.train.loc[:, ["prompt", "chosen", "rejected"]],
        preserve_index=False,
    )
    validation_dataset = Dataset.from_pandas(
        splits.validation.loc[:, ["prompt", "chosen", "rejected"]],
        preserve_index=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{args.algorithm}-{Path(args.base_model).name}-"
        f"s1-{system1_fraction:g}-seed-{args.seed}"
    )
    common = dict(
        output_dir=str(args.output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/accuracies",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to=[] if args.report_to == "none" else ["wandb"],
        run_name=run_name,
        save_total_limit=1,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        seed=args.seed,
        data_seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    callback = EarlyStoppingCallback(early_stopping_patience=5)

    if args.report_to == "wandb":
        import os

        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    if args.algorithm == "dpo":
        training_args = DPOConfig(beta=beta, **common)
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[callback],
        )
    else:
        if gamma_over_beta is None:
            raise AssertionError("SimPO requires gamma_over_beta")
        training_args = CPOConfig(
            beta=beta,
            simpo_gamma=beta * gamma_over_beta,
            loss_type="simpo",
            cpo_alpha=0.0,
            **common,
        )
        trainer = CPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[callback],
        )

    run_metadata = {
        "algorithm": args.algorithm,
        "base_model": args.base_model,
        "system1_fraction": system1_fraction,
        "seed": args.seed,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "learning_rate": learning_rate,
        "beta": beta,
        "gamma_over_beta": gamma_over_beta,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "paper_defaults": asdict(defaults),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n"
    )
    trainer.train()

    # Save the small LoRA adapter, not a merged multi-gigabyte base checkpoint.
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
