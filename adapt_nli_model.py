#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Continual masked-language pretraining for DeBERTa-v3-large.

This stage adapts the public DeBERTa checkpoint to the full O-RAN corpus using
a deterministic held-out split and epoch-level evaluation.

The script deliberately uses the full parsed O-RAN corpus.  The security
filtered corpus is reserved for candidate-pair mining and is not the MLM
pretraining corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import shutil
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


SCRIPT_PATH = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT_PATH.parent
DEFAULT_CORPUS = (
    PIPELINE_ROOT
    / "data/processed/ORAN/corpus_ORAN.jsonl"
)
DEFAULT_OUTPUT = (
    PIPELINE_ROOT / "models/nli_domain_adapted"
)
DEFAULT_DEEPSPEED = PIPELINE_ROOT / "configs/deepspeed_nli.json"
DEFAULT_BASE_REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"
RETAINED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
)


def resolve_path(value: Path) -> Path:
    """Resolve a user-supplied path relative to the current working directory."""

    value = value.expanduser()
    if value.is_absolute():
        return value
    return (Path.cwd() / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continually pretrain DeBERTa-v3-large with masked language modeling."
    )
    parser.add_argument("--corpus_path", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--base-model", "--model_name", dest="model_name",
        default="microsoft/deberta-v3-large",
        help="Hugging Face model name or local model directory.",
    )
    parser.add_argument(
        "--model_revision",
        default=DEFAULT_BASE_REVISION,
        help="Pinned Hugging Face revision; use 'none' for a local model directory.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Resolve the pinned model entirely from the local Hugging Face cache.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--deepspeed_config",
        type=Path,
        default=DEFAULT_DEEPSPEED,
        help="DeepSpeed JSON config. Use --no_deepspeed to disable it.",
    )
    parser.add_argument("--no_deepspeed", action="store_true")

    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.05,
        help="Fraction of paragraphs held out for MLM validation (default: 5%%).",
    )
    parser.add_argument(
        "--split_group_field",
        default=None,
        help=(
            "Optional JSON field used for a group-disjoint validation split "
            "(for example pdf_file). The default uses a record-level split."
        ),
    )
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument(
        "--disable_early_stopping",
        action="store_true",
        help="Run the complete requested schedule.",
    )
    parser.add_argument(
        "--retain_epochs",
        default="",
        help="Comma-separated epoch numbers copied to retained/ before rotation.",
    )
    parser.add_argument(
        "--stop_after_epoch",
        type=int,
        default=None,
        help="Stop after this epoch while preserving the full --epochs LR schedule.",
    )
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--save_only_model",
        action="store_true",
        help="Store model weights without optimizer state (checkpoint resume is disabled).",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FP16 mixed precision (default: enabled).",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable BF16 mixed precision instead of FP16.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Allow an existing non-empty output directory to be reused.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint directory from which to resume training.",
    )
    # DeepSpeed injects this launcher argument into every worker process.
    # Trainer/Accelerate obtains the actual rank from the distributed
    # environment, so the parsed value is intentionally not used directly.
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {args.corpus_path}")
    if args.block_size <= 0:
        raise ValueError("--block_size must be positive")
    if args.epochs <= 0 and args.max_steps <= 0:
        raise ValueError("Set --epochs > 0 or --max_steps > 0")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max_steps must be -1 or a positive integer")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if not 0.0 < args.eval_split < 1.0:
        raise ValueError("--eval_split must be between 0 and 1")
    if not 0.0 < args.mlm_probability < 1.0:
        raise ValueError("--mlm_probability must be between 0 and 1")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience cannot be negative")
    if any(epoch <= 0 or epoch > math.ceil(args.epochs) for epoch in args.retain_epochs):
        raise ValueError("--retain_epochs must fall within the requested epoch schedule")
    if args.stop_after_epoch is not None and not 0 < args.stop_after_epoch <= args.epochs:
        raise ValueError("--stop_after_epoch must fall within the requested epoch schedule")
    if args.fp16 and args.bf16:
        raise ValueError("Use only one of FP16 and BF16")
    if (args.fp16 or args.bf16) and not torch.cuda.is_available():
        raise RuntimeError(
            "Mixed precision was requested but CUDA is unavailable. "
            "Use --no-fp16 and --no-bf16 for CPU execution."
        )
    if not args.no_deepspeed and not args.deepspeed_config.exists():
        raise FileNotFoundError(
            f"DeepSpeed config not found: {args.deepspeed_config}. "
            "Pass --no_deepspeed or provide --deepspeed_config."
        )
    if args.save_only_model and not args.no_deepspeed:
        raise ValueError(
            "--save_only_model is incompatible with DeepSpeed plus "
            "load_best_model_at_end; pass --no_deepspeed."
        )
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {args.resume_from_checkpoint}"
        )


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    block_size: int,
) -> Tuple[Dataset, int]:
    """Tokenize paragraphs and deterministically pack fixed-length MLM blocks."""

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=False,
            return_special_tokens_mask=True,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing paragraphs",
    )

    concatenated = {
        key: list(chain.from_iterable(tokenized[key]))
        for key in tokenized.column_names
    }
    total_tokens = len(concatenated["input_ids"])
    usable_tokens = (total_tokens // block_size) * block_size
    packed_values = {
        key: [
            values[index : index + block_size]
            for index in range(0, usable_tokens, block_size)
        ]
        for key, values in concatenated.items()
    }
    packed_values["labels"] = [list(row) for row in packed_values["input_ids"]]
    return Dataset.from_dict(packed_values), total_tokens - usable_tokens


def group_disjoint_dataset_split(
    dataset: Dataset,
    group_field: str,
    test_fraction: float,
    seed: int,
) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    """Make a document-disjoint holdout and remove exact-text leakage.

    Connecting every document that shares boilerplate creates a 30k-row giant
    component in this corpus. Instead, select whole documents near the target
    size, keep them entirely out of training, and exclude evaluation paragraphs
    whose normalized text also occurs in training (plus duplicate eval copies).
    """

    if group_field not in dataset.column_names:
        raise ValueError(
            f"Split group field {group_field!r} is absent; found {dataset.column_names}"
        )
    groups: Dict[str, list[int]] = defaultdict(list)
    normalized_texts: list[str] = []
    for index, row in enumerate(dataset):
        group = str(row.get(group_field) or "").strip()
        if not group:
            raise ValueError(f"Row {index} has no {group_field}")
        groups[group].append(index)
        normalized_texts.append(
            " ".join(str(row.get("text") or "").split()).casefold()
        )
    if len(groups) < 2:
        raise ValueError("Group split requires at least two documents")

    items = list(groups.items())
    random.Random(seed).shuffle(items)
    target_eval = len(dataset) * test_fraction
    upper_bound = max(1, math.ceil(target_eval * 2))
    reachable: Dict[int, tuple[int, ...]] = {0: ()}
    for item_index, (_, indexes) in enumerate(items):
        size = len(indexes)
        for total, selected in list(reachable.items())[::-1]:
            new_total = total + size
            if new_total <= upper_bound and new_total not in reachable:
                reachable[new_total] = selected + (item_index,)
    nonempty_totals = [total for total in reachable if total]
    if not nonempty_totals:
        selected_indexes = (min(range(len(items)), key=lambda i: len(items[i][1])),)
    else:
        closest = min(nonempty_totals, key=lambda total: (abs(total - target_eval), total))
        selected_indexes = reachable[closest]
    eval_groups = {items[index][0] for index in selected_indexes}
    raw_eval_indexes = sorted(index for group in eval_groups for index in groups[group])
    train_indexes = sorted(set(range(len(dataset))) - set(raw_eval_indexes))
    train_texts = {normalized_texts[index] for index in train_indexes}
    eval_indexes: list[int] = []
    eval_seen: set[str] = set()
    for index in raw_eval_indexes:
        text = normalized_texts[index]
        if text in train_texts or text in eval_seen:
            continue
        eval_seen.add(text)
        eval_indexes.append(index)
    if not train_indexes or not eval_indexes:
        raise ValueError("Group split produced an empty train or eval split")
    train_groups = set(groups) - eval_groups
    return (
        dataset.select(train_indexes),
        dataset.select(eval_indexes),
        {
            "strategy": "document_disjoint_exact_text_filtered_eval",
            "group_field": group_field,
            "train_groups": len(train_groups),
            "eval_groups": len(eval_groups),
            "shared_groups": len(train_groups & eval_groups),
            "target_eval_rows": target_eval,
            "actual_eval_fraction": len(eval_indexes) / len(dataset),
            "raw_eval_rows": len(raw_eval_indexes),
            "eval_rows_removed_as_train_duplicates_or_eval_repeats": (
                len(raw_eval_indexes) - len(eval_indexes)
            ),
            "largest_document_rows": max(len(indexes) for indexes in groups.values()),
            "train_membership_sha256": hashlib.sha256(
                json.dumps(train_indexes, separators=(",", ":")).encode()
            ).hexdigest(),
            "eval_membership_sha256": hashlib.sha256(
                json.dumps(eval_indexes, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )


def parse_retain_epochs(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        epochs = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError("--retain_epochs must be comma-separated integers") from exc
    return epochs


class RetainedEpochTrainer(Trainer):
    """Copy only selected model-only checkpoints outside normal rotation."""

    retained_epochs: tuple[int, ...] = ()

    def _save_checkpoint(self, model, trial):  # type: ignore[override]
        super()._save_checkpoint(model, trial)
        if not self.args.should_save or self.state.epoch is None:
            return
        epoch = int(round(float(self.state.epoch)))
        if epoch not in self.retained_epochs or not math.isclose(
            float(self.state.epoch), epoch, abs_tol=1e-4
        ):
            return
        source = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        target = Path(self.args.output_dir) / "retained" / f"epoch-{epoch}"
        if target.exists():
            return
        target.mkdir(parents=True)
        copied = []
        for filename in RETAINED_MODEL_FILES:
            source_file = source / filename
            if source_file.is_file():
                shutil.copy2(source_file, target / filename)
                copied.append(filename)
        if "model.safetensors" not in copied:
            raise FileNotFoundError(f"Retained checkpoint has no model weights: {source}")
        (target / "retention_manifest.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "global_step": self.state.global_step,
                    "source_checkpoint": str(source.resolve()),
                    "files": copied,
                    "model_sha256": sha256_file(target / "model.safetensors"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


class StopAfterEpochCallback(TrainerCallback):
    def __init__(self, stop_after_epoch: int):
        self.stop_after_epoch = stop_after_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and float(state.epoch) >= self.stop_after_epoch:
            control.should_training_stop = True
        return control


class AbortOnNonFiniteCallback(TrainerCallback):
    """Fail immediately instead of silently accepting skipped updates."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        for name in ("loss", "grad_norm", "eval_loss"):
            value = (logs or {}).get(name)
            if value is not None and not math.isfinite(float(value)):
                raise FloatingPointError(
                    f"Non-finite {name} at step {state.global_step}: {value}"
                )
        return control


def build_training_args(
    args: argparse.Namespace,
    output_dir: Path,
) -> TrainingArguments:
    training_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "save_only_model": args.save_only_model,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "eval_accumulation_steps": 1,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "gradient_checkpointing": True,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": args.report_to,
        "run_name": args.run_name,
    }

    # Transformers 4.52 uses eval_strategy; the fallback keeps this entrypoint
    # usable with older compatible versions as well.
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    if not args.no_deepspeed:
        training_kwargs["deepspeed"] = str(args.deepspeed_config)

    return TrainingArguments(**training_kwargs)


def main() -> None:
    args = parse_args()
    args.retain_epochs = parse_retain_epochs(args.retain_epochs)
    args.corpus_path = resolve_path(args.corpus_path)
    args.output_dir = resolve_path(args.output_dir)
    args.deepspeed_config = resolve_path(args.deepspeed_config)
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = resolve_path(args.resume_from_checkpoint)
    validate_args(args)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite_output_dir and args.resume_from_checkpoint is None:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Use a new directory, --overwrite_output_dir, or "
                "--resume_from_checkpoint."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print(f"Corpus: {args.corpus_path}")
    print(f"Corpus SHA-256: {sha256_file(args.corpus_path)}")
    print(f"Base model: {args.model_name}")
    print(f"Base revision: {args.model_revision}")
    print(f"Output: {args.output_dir}")
    print(f"Block size: {args.block_size}")
    print(f"Seed: {args.seed}")
    print(f"FP16: {args.fp16}; BF16: {args.bf16}")
    print(f"DeepSpeed: {'disabled' if args.no_deepspeed else args.deepspeed_config}")

    raw_dataset = load_dataset(
        "json",
        data_files={"train": str(args.corpus_path)},
    )["train"]
    if "text" not in raw_dataset.column_names:
        raise ValueError(
            f"Corpus must contain a 'text' field; found {raw_dataset.column_names}"
        )

    before_count = len(raw_dataset)
    raw_dataset = raw_dataset.filter(
        lambda row: isinstance(row.get("text"), str) and bool(row["text"].strip()),
        desc="Removing empty paragraphs",
    )
    print(f"Paragraphs: {len(raw_dataset)} (removed {before_count - len(raw_dataset)})")

    # Split before tokenization/packing so no paragraph is represented in both
    # training and evaluation and each side is packed independently.
    if args.split_group_field:
        train_raw, eval_raw, split_details = group_disjoint_dataset_split(
            raw_dataset,
            args.split_group_field,
            args.eval_split,
            args.seed,
        )
    else:
        splits = raw_dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_raw = splits["train"]
        eval_raw = splits["test"]
        split_details = {
            "strategy": "record_random",
            "group_field": None,
            "shared_groups": None,
        }
    print(f"Raw split: train={len(train_raw)}, eval={len(eval_raw)}")

    frozen_eval_path = args.output_dir / "frozen_eval_paragraphs.jsonl"
    if int(os.environ.get("RANK", "0")) == 0:
        temporary_eval_path = frozen_eval_path.with_suffix(".jsonl.tmp")
        with temporary_eval_path.open("w", encoding="utf-8") as handle:
            for row in eval_raw:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary_eval_path, frozen_eval_path)
    revision = None if args.model_revision.lower() == "none" else args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=revision,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        raise ValueError("The DeBERTa tokenizer does not define a pad token")

    train_dataset, train_dropped_tokens = tokenize_dataset(
        train_raw, tokenizer, args.block_size
    )
    eval_dataset, eval_dropped_tokens = tokenize_dataset(
        eval_raw, tokenizer, args.block_size
    )
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(
            "The packed train/eval split contains no complete blocks. "
            "Use a smaller --block_size or a larger corpus."
        )
    print(
        f"Packed blocks: train={len(train_dataset)}, eval={len(eval_dataset)}; "
        f"final dropped tails: train={train_dropped_tokens}, "
        f"eval={eval_dropped_tokens} tokens"
    )

    model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
    # Standard Trainer AMP keeps FP32 master parameters. Loading FP16 weights
    # is only safe here when DeepSpeed owns mixed-precision state handling.
    if not args.no_deepspeed:
        if args.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        elif args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name,
        revision=revision,
        local_files_only=args.local_files_only,
        **model_kwargs,
    )
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    training_args = build_training_args(args, args.output_dir)
    callbacks: list[TrainerCallback] = [AbortOnNonFiniteCallback()]
    if not args.disable_early_stopping and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )
    if args.stop_after_epoch is not None:
        callbacks.append(StopAfterEpochCallback(args.stop_after_epoch))
    trainer = RetainedEpochTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer.retained_epochs = args.retain_epochs

    print(
        "Training configuration: "
        f"epochs={args.epochs}, max_steps={args.max_steps}, "
        f"batch_size={args.batch_size}, "
        f"gradient_accumulation={args.gradient_accumulation_steps}, "
        f"patience={'disabled' if args.disable_early_stopping else args.early_stopping_patience}, "
        f"retained_epochs={args.retain_epochs}, stop_after_epoch={args.stop_after_epoch}"
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        )
    )
    eval_metrics = trainer.evaluate()

    # load_best_model_at_end=True means this save contains the best validation
    # checkpoint, while checkpoint-* directories remain available for resuming.
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    manifest = {
        "script": str(SCRIPT_PATH),
        "corpus_path": str(args.corpus_path),
        "corpus_sha256": sha256_file(args.corpus_path),
        "model_name": args.model_name,
        "model_revision": revision,
        "local_files_only": args.local_files_only,
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "eval_split": args.eval_split,
        "split_details": split_details,
        "frozen_eval_path": str(frozen_eval_path),
        "frozen_eval_sha256": sha256_file(frozen_eval_path),
        "raw_train_paragraphs": len(train_raw),
        "raw_eval_paragraphs": len(eval_raw),
        "train_blocks": len(train_dataset),
        "eval_blocks": len(eval_dataset),
        "train_dropped_tail_tokens": train_dropped_tokens,
        "eval_dropped_tail_tokens": eval_dropped_tokens,
        "block_size": args.block_size,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "mlm_probability": args.mlm_probability,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "deepspeed_config": None
        if args.no_deepspeed
        else str(args.deepspeed_config),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_disabled": args.disable_early_stopping,
        "retained_epochs": list(args.retain_epochs),
        "stop_after_epoch": args.stop_after_epoch,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    with (args.output_dir / "training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True, default=str)

    eval_loss = eval_metrics.get("eval_loss")
    perplexity = math.exp(eval_loss) if eval_loss is not None and eval_loss < 20 else None
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Final best-model eval metrics: {eval_metrics}")
    if perplexity is not None:
        print(f"Perplexity: {perplexity:.2f}")
    print(f"Saved domain-pretrained DeBERTa: {args.output_dir}")


if __name__ == "__main__":
    main()
