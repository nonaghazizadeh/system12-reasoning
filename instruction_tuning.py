import os
import torch
from time import time
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer,setup_chat_format

model_name = ""
device = ""

compute_dtype = torch.bfloat16
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True)

model_config = AutoConfig.from_pretrained(
    model_name,
    trust_remote_code=True,
    max_new_tokens=1024
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

dataset_name = "HuggingFaceH4/ultrachat_200k"
dataset = load_dataset(dataset_name, split="train_sft")
dataset = dataset.shuffle(seed=42).select(range(10000))

def format_chat_template(row):
    chat = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    return {"text":chat}

processed_dataset = dataset.map(
    format_chat_template,
)

dataset = processed_dataset.train_test_split(test_size=0.01)

peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.05,
        r=4,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
)

training_arguments = TrainingArguments(
        output_dir="./results_llama3_sft/",
        evaluation_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        log_level="debug",
        save_steps=20,
        logging_steps=1,
        learning_rate=8e-6,
        eval_steps=1,
        max_steps=20,
        num_train_epochs=20,
        warmup_steps=3,
        lr_scheduler_type="linear",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
)

trainer.train()