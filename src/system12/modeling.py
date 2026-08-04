"""Shared model loading and deterministic generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _is_adapter(path_or_id: str) -> bool:
    path = Path(path_or_id)
    if path.exists() and (path / "adapter_config.json").exists():
        return True
    try:
        PeftConfig.from_pretrained(path_or_id)
    except (OSError, ValueError):
        return False
    return True


def load_model_and_tokenizer(model_or_adapter: str):
    """Load either a base/merged model or a PEFT adapter with its base model."""

    adapter_config = None
    model_id = model_or_adapter
    if _is_adapter(model_or_adapter):
        adapter_config = PeftConfig.from_pretrained(model_or_adapter)
        model_id = adapter_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_config is not None:
        model = PeftModel.from_pretrained(model, model_or_adapter)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def tokenize_chat(tokenizer, prompts: Iterable[str], device: torch.device):
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    encoded = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        truncation=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    encoded = tokenize_chat(tokenizer, prompts, _model_device(model))
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_length = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)
