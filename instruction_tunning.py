import logging
import os
import random
import sys
import torch
from functools import partial
from typing import List

import datasets
import torch.distributed as dist
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          HfArgumentParser,
                          Trainer,
                          set_seed)

from data_arguments import DataArguments, get_data_statistics
from model_arguments import ModelArguments, add_padding_to_tokenizer
from training_arguments import TrainingArguments

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer)
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def get_datasets(thinking_type: str,
                 train_files: List[str],
                 tokenizer,
                 max_seq_length,
                 sample_size=10000,
                 eval_sample_size=5000,
                 seed=0):
    """ get datasets with a specified seed """
    if thinking_type == "system12":
        dataset_list, eval_dataset_list = [], []
        sample_size, eval_sample_size = sample_size // 2, eval_sample_size // 2
        for tp in ['system1', 'system2']:
            tp_files = [os.path.join(
                file, f"{tp}.jsonl") for file in train_files]
            raw_dataset = datasets.load_dataset(
                "json", data_files=tp_files, split="train")
            raw_dataset = raw_dataset.shuffle(seed=seed)
            dataset = raw_dataset.select(range(sample_size))
            eval_dataset = raw_dataset.select(
                range(sample_size, sample_size + eval_sample_size))
            dataset_list.append(dataset)
            eval_dataset_list.append(eval_dataset)
            logger.info(
                f"{tp} training dataset: {len(dataset)} from {len(raw_dataset)}")
            logger.info(
                f"{tp} eval dataset: {len(eval_dataset)} from {len(raw_dataset)}")
        dataset = datasets.concatenate_datasets(dataset_list)
        eval_dataset = datasets.concatenate_datasets(eval_dataset_list)
    else:
        train_files = [os.path.join(
            file, f"{thinking_type}.jsonl") for file in train_files]
        raw_dataset = datasets.load_dataset(
            "json", data_files=train_files, split="train")
        raw_dataset = raw_dataset.shuffle(seed=seed)
        dataset = raw_dataset.select(range(sample_size))
        eval_dataset = raw_dataset.select(
            range(sample_size, sample_size + eval_sample_size))
        logger.info(
            f"{thinking_type} training dataset: {len(dataset)} from {len(raw_dataset)}")
        logger.info(
            f"{thinking_type} eval dataset: {len(eval_dataset)} from {len(raw_dataset)}")

    encode_function = partial(
        encode_with_messages_format,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    dataset = dataset.map(encode_function,
                          batched=False,
                          num_proc=10,
                          load_from_cache_file=True,
                          desc="Tokenizing and reformatting instruction training data",
                          )
    eval_dataset = eval_dataset.map(encode_function,
                                    batched=False,
                                    num_proc=10,
                                    load_from_cache_file=True,
                                    desc="Tokenizing and reformatting instruction evaluation data",
                                    )
    dataset.set_format(type="pt")
    eval_dataset.set_format(type="pt")
    return dataset, eval_dataset


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # Load training dataset
    train_dataset, eval_dataset = get_datasets(thinking_type=data_args.thinking_type,
                                               train_files=data_args.train_files,
                                               tokenizer=tokenizer,
                                               max_seq_length=data_args.max_seq_length,
                                               sample_size=data_args.sample_size,
                                               eval_sample_size=data_args.eval_sample_size,
                                               seed=training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype)
    add_padding_to_tokenizer(tokenizer)

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(
            f"Resizing embedding size from {embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        training_args.save_embedding_layers = True

    get_data_statistics(train_dataset)

    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(
            ["dataset", "id", "messages"])

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    # model_params = sum(p.numel()
    #                    for p in model.parameters() if p.requires_grad)
    # logger.info(f"trainable model_params: {model_params}")

    # analysis_dataset = None
    # if training_args.analysis_mode:
    #     from less.data_selection.get_validation_dataset import get_dataset
    #     analysis_dataset = get_dataset(training_args.analysis_dataset,
    #                                    data_dir=data_args.data_dir,
    #                                    tokenizer=tokenizer,
    #                                    max_length=data_args.max_seq_length)

    # if dist.is_initialized() and dist.get_rank() == 0:
    #     print(model)
    # elif not dist.is_initialized():
    #     print(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest")
    )

    # Training
    train_result = trainer.train()

    # Get the best model from the trainer if load_best_model_at_end is True
    model = trainer.model

    # Create the directory if it doesn't exist
    best_model_output_dir = os.path.join(
        training_args.output_dir, "best_model")
    os.makedirs(best_model_output_dir, exist_ok=True)

    # Save the model and tokenizer in output_dir/best_model
    model.save_pretrained(best_model_output_dir, save_embedding_layers=True)
    tokenizer.save_pretrained(best_model_output_dir)

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None


if __name__ == "__main__":
    main()
