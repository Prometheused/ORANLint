#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Continually pretrain a LoRA adapter for the local domain generator."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data/generated/pretraining/merged_4G_5G_ORAN.txt"
DEFAULT_OUTPUT = ROOT / "models/domain_generator"
DEFAULT_LOGGING = ROOT / "logs/domain_generator_training"
DEFAULT_DEEPSPEED = ROOT / "configs/deepspeed_generator.json"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-path",
        "--corpus_path",
        dest="corpus_path",
        type=Path,
        default=DEFAULT_CORPUS,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--logging-dir",
        "--logging_dir",
        dest="logging_dir",
        type=Path,
        default=DEFAULT_LOGGING,
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_MODEL,
        help="Public model identifier or compatible local checkpoint.",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_REVISION,
        help="Model revision. Use 'none' for a local checkpoint.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Trainer checkpoint from which to resume.",
    )
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-total-limit", "--save_total_limit", dest="save_total_limit", type=int, default=3)
    parser.add_argument(
        "--save-only-model",
        "--save_only_model",
        dest="save_only_model",
        action="store_true",
        help="Store model weights without optimizer state.",
    )
    parser.add_argument(
        "--checkpoint-strategy",
        "--checkpoint_strategy",
        dest="checkpoint_strategy",
        choices=("steps", "epoch"),
        default="steps",
    )
    parser.add_argument(
        "--deepspeed-config",
        "--deepspeed_config",
        dest="deepspeed_config",
        type=Path,
        default=DEFAULT_DEEPSPEED,
    )
    parser.add_argument("--no-deepspeed", "--no_deepspeed", dest="no_deepspeed", action="store_true")
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer")
    if args.save_total_limit <= 0:
        parser.error("--save-total-limit must be positive")
    if args.save_only_model and args.resume_from_checkpoint:
        parser.error("--save-only-model cannot be combined with checkpoint resume")
    if args.save_only_model and not args.no_deepspeed:
        parser.error("--save-only-model requires --no-deepspeed")
    return args


def local_rank_for(args: argparse.Namespace) -> int:
    rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    return max(rank, 0)


def group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, List[List[int]]]:
    concatenated = {key: sum(values, []) for key, values in examples.items()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    grouped = {
        key: [values[index : index + block_size] for index in range(0, total_length, block_size)]
        for key, values in concatenated.items()
    }
    grouped["labels"] = grouped["input_ids"].copy()
    return grouped


def tokenize_and_pack(dataset: Dataset, tokenizer: AutoTokenizer, block_size: int) -> Dataset:
    tokenized = dataset.map(
        lambda rows: tokenizer(rows["text"], return_special_tokens_mask=True),
        batched=True,
        remove_columns=["text"],
    )
    packed = tokenized.map(
        lambda rows: group_texts(rows, block_size),
        batched=True,
    )
    if len(packed) == 0:
        raise ValueError("The corpus does not contain enough tokens for one training block")
    return packed


def main() -> None:
    args = parse_args()
    corpus_path = args.corpus_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    logging_dir = args.logging_dir.expanduser().resolve()
    deepspeed_config = args.deepspeed_config.expanduser().resolve()
    resume_checkpoint = (
        args.resume_from_checkpoint.expanduser().resolve()
        if args.resume_from_checkpoint
        else None
    )

    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)
    if not args.no_deepspeed and not deepspeed_config.is_file():
        raise FileNotFoundError(deepspeed_config)
    if resume_checkpoint and not resume_checkpoint.is_dir():
        raise FileNotFoundError(resume_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA generator training")
    local_rank = local_rank_for(args)
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    revision = None if args.model_revision.lower() == "none" else args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=revision,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float16,
    )
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        revision=revision,
        quantization_config=quantization,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            inference_mode=False,
        ),
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.print_trainable_parameters()

    dataset = load_dataset("text", data_files={"train": str(corpus_path)})["train"]
    if len(dataset) < 2:
        raise ValueError("The pretraining corpus must contain at least two records")
    splits = dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = splits["train"]
    validation_dataset = splits["test"]

    length_probe = train_dataset.map(
        lambda rows: tokenizer(rows["text"], truncation=False),
        batched=True,
        remove_columns=["text"],
    )
    lengths = [len(ids) for ids in length_probe["input_ids"]]
    percentiles = {int(level): float(np.percentile(lengths, level)) for level in (50, 75, 90, 99)}
    print(f"Token-length percentiles: {percentiles}")

    block_size = 2048
    train_packed = tokenize_and_pack(train_dataset, tokenizer, block_size)
    validation_packed = tokenize_and_pack(validation_dataset, tokenizer, block_size)
    print(
        f"Records: train={len(train_dataset):,}, validation={len(validation_dataset):,}; "
        f"packed blocks: train={len(train_packed):,}, validation={len(validation_packed):,}"
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_accumulation_steps=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        logging_strategy="steps",
        logging_steps=25,
        save_strategy=args.checkpoint_strategy,
        save_steps=25,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        eval_strategy=args.checkpoint_strategy,
        eval_steps=25,
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        run_name="domain-generator-training",
        logging_dir=str(logging_dir),
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        deepspeed=None if args.no_deepspeed else str(deepspeed_config),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_packed,
        eval_dataset=validation_packed,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    print(f"Base model: {args.base_model}@{revision}")
    print(f"CUDA local rank: {local_rank}")
    print(f"Output directory: {output_dir}")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    trainer.save_model(str(output_dir))
    eval_loss = float(trainer.evaluate(eval_dataset=validation_packed)["eval_loss"])
    perplexity = math.exp(eval_loss) if eval_loss < 700 else float("inf")
    print(f"Evaluation loss: {eval_loss:.6f}")
    print(f"Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
