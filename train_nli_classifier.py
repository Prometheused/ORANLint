#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Validation-aware supervised DeBERTa NLI fine-tuning for O-RAN text.

The training and validation JSONL files contain anchor records with the
following fields:

    text, paraphrased, inconsistent, randomized, id

Each record is expanded into three NLI examples.  Training and validation are
kept as separate anchor pools; no row-level split is performed here.

Candidate-shaped synthetic files may instead contain direct ``text1``/``text2``
pairs with a ``label``.  That schema is auto-detected and can augment
training with the reverse order without altering validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_TRAIN_JSONL = (
    PROJECT_ROOT
    / "data/generated/nli_supervision/train_pairs.jsonl"
)
DEFAULT_VALIDATION_JSONL = (
    PROJECT_ROOT
    / "data/generated/nli_supervision/development_pairs.jsonl"
)
DEFAULT_BASE_MODEL = (
    PROJECT_ROOT
    / "models/nli_domain_adapted/retained/epoch-3"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models/nli_classifier"
DEFAULT_DEEPSPEED_CONFIG = PROJECT_ROOT / "configs/deepspeed_nli.json"

LABEL_NAMES = ["entailment", "neutral", "contradiction"]
LABEL_TO_ID = {name: index for index, name in enumerate(LABEL_NAMES)}
REQUIRED_RECORD_FIELDS = (
    "text",
    "paraphrased",
    "inconsistent",
    "randomized",
)
VARIANT_LABELS = (
    ("paraphrased", "entailment"),
    ("inconsistent", "contradiction"),
    ("randomized", "neutral"),
)
PAIR_REQUIRED_FIELDS = ("id1", "id2", "text1", "text2", "label")
PAIR_VERDICT_TO_LABEL = {
    "consistent": "entailment",
    "neutral": "neutral",
    "inconsistent": "contradiction",
}


class AbortOnNonFiniteCallback(TrainerCallback):
    """Abort training instead of accepting a skipped or invalid update."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        for name in ("loss", "grad_norm", "eval_loss"):
            value = (logs or {}).get(name)
            if value is not None and not math.isfinite(float(value)):
                raise FloatingPointError(
                    f"Non-finite {name} at step {state.global_step}: {value}"
                )
        return control


class StopAfterStepsCallback(TrainerCallback):
    """Stop at a fixed step without shortening the configured LR horizon."""

    def __init__(self, stop_after_steps: int):
        self.stop_after_steps = int(stop_after_steps)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps < self.stop_after_steps:
            raise ValueError(
                f"--stop_after_steps={self.stop_after_steps} exceeds the configured "
                f"schedule horizon of {state.max_steps} steps"
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_steps:
            control.should_training_stop = True
        return control


def resolve_path(path: Path) -> Path:
    """Resolve relative paths relative to the project directory."""

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}; "
                    f"got {type(record).__name__}."
                )
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def normalized_text(value: Any) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    return " ".join(text.split()).casefold()


def record_id(record: Dict[str, Any]) -> Optional[str]:
    value = record.get("id")
    return None if value is None else str(value)


def validate_records(records: Sequence[Dict[str, Any]], path: Path) -> None:
    missing: List[str] = []
    ids: List[str] = []

    for index, record in enumerate(records, start=1):
        absent = [
            field
            for field in REQUIRED_RECORD_FIELDS
            if clean_text(record.get(field)) is None
        ]
        if absent:
            missing.append(f"line {index}: {', '.join(absent)}")

        identifier = record_id(record)
        if identifier is None:
            missing.append(f"line {index}: id")
        else:
            ids.append(identifier)

    if missing:
        preview = "; ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f"; ... ({len(missing)} total)"
        raise ValueError(f"Invalid records in {path}: {preview}{suffix}")

    duplicate_ids = [identifier for identifier, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(
            f"Duplicate anchor IDs in {path}: {duplicate_ids[:10]}"
        )


def detect_data_format(records: Sequence[Dict[str, Any]], path: Path) -> str:
    anchor = all(field in records[0] for field in REQUIRED_RECORD_FIELDS)
    pairs = all(field in records[0] for field in PAIR_REQUIRED_FIELDS)
    if anchor == pairs:
        raise ValueError(
            f"Could not uniquely detect anchor or pair schema in {path}"
        )
    expected = REQUIRED_RECORD_FIELDS if anchor else PAIR_REQUIRED_FIELDS
    if any(not all(field in row for field in expected) for row in records):
        raise ValueError(f"Mixed data schemas are not supported in {path}")
    return "anchor_variants" if anchor else "direct_pairs"


def pair_documents(record: Dict[str, Any]) -> set[str]:
    values = record.get("source_pdf_files")
    if isinstance(values, list):
        return {str(value) for value in values if str(value)}
    value = record.get("pdf_file")
    return {str(value)} if value else set()


def validate_pair_records(
    records: Sequence[Dict[str, Any]], path: Path
) -> None:
    errors: list[str] = []
    pair_ids: list[tuple[str, str]] = []
    for index, record in enumerate(records, 1):
        absent = [
            field
            for field in PAIR_REQUIRED_FIELDS
            if clean_text(record.get(field)) is None
        ]
        if absent:
            errors.append(f"line {index}: {', '.join(absent)}")
            continue
        verdict = str(record["label"]).casefold()
        if verdict not in PAIR_VERDICT_TO_LABEL:
            errors.append(f"line {index}: invalid label={verdict!r}")
        pair_ids.append(tuple(sorted((str(record["id1"]), str(record["id2"])))))
    if errors:
        raise ValueError(f"Invalid direct pairs in {path}: {'; '.join(errors[:5])}")
    duplicates = [key for key, count in Counter(pair_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate direct-pair IDs in {path}: {duplicates[:10]}")


def validate_pair_disjoint_splits(
    train_records: Sequence[Dict[str, Any]],
    validation_records: Sequence[Dict[str, Any]],
    require_document_disjoint: bool,
) -> Dict[str, Any]:
    train_ids = {
        str(record[field]) for record in train_records for field in ("id1", "id2")
    }
    validation_ids = {
        str(record[field])
        for record in validation_records
        for field in ("id1", "id2")
    }
    train_texts = {
        normalized_text(record[field])
        for record in train_records
        for field in ("text1", "text2")
    }
    validation_texts = {
        normalized_text(record[field])
        for record in validation_records
        for field in ("text1", "text2")
    }
    shared_ids = train_ids & validation_ids
    shared_texts = train_texts & validation_texts
    train_documents = set().union(*(pair_documents(row) for row in train_records))
    validation_documents = set().union(
        *(pair_documents(row) for row in validation_records)
    )
    shared_documents = train_documents & validation_documents
    if shared_ids or shared_texts:
        raise ValueError(
            "Training and validation pairs overlap: "
            f"{len(shared_ids)} shared IDs and {len(shared_texts)} shared texts."
        )
    if require_document_disjoint and shared_documents:
        raise ValueError(
            "Training and validation documents overlap: "
            f"{len(shared_documents)} shared source documents."
        )
    return {
        "train_pairs": len(train_records),
        "validation_pairs": len(validation_records),
        "shared_ids": 0,
        "shared_texts": 0,
        "shared_documents": len(shared_documents),
        "document_disjoint_required": require_document_disjoint,
    }


def validate_disjoint_splits(
    train_records: Sequence[Dict[str, Any]],
    validation_records: Sequence[Dict[str, Any]],
    require_document_disjoint: bool = False,
) -> Dict[str, Any]:
    train_ids = {record_id(record) for record in train_records}
    validation_ids = {record_id(record) for record in validation_records}
    shared_ids = train_ids & validation_ids

    train_texts = {
        normalized_text(record.get("text")) for record in train_records
    }
    validation_texts = {
        normalized_text(record.get("text")) for record in validation_records
    }
    shared_texts = train_texts & validation_texts

    if shared_ids or shared_texts:
        raise ValueError(
            "Training and validation anchors overlap: "
            f"{len(shared_ids)} shared IDs and {len(shared_texts)} shared texts."
        )

    train_documents = {
        str(record.get("pdf_file"))
        for record in train_records
        if record.get("pdf_file")
    }
    validation_documents = {
        str(record.get("pdf_file"))
        for record in validation_records
        if record.get("pdf_file")
    }
    shared_documents = train_documents & validation_documents
    if require_document_disjoint and shared_documents:
        raise ValueError(
            "Training and validation documents overlap: "
            f"{len(shared_documents)} shared pdf_file values."
        )

    duplicate_train_texts = len(train_records) - len(train_texts)
    duplicate_validation_texts = len(validation_records) - len(validation_texts)
    print(
        "Disjoint anchor check: "
        f"train={len(train_records)}, validation={len(validation_records)}, "
        "shared_ids=0, shared_texts=0"
    )
    if duplicate_train_texts or duplicate_validation_texts:
        print(
            "Duplicate anchor text warning: "
            f"train={duplicate_train_texts}, validation={duplicate_validation_texts}"
        )
    return {
        "train_anchors": len(train_records),
        "validation_anchors": len(validation_records),
        "shared_ids": 0,
        "shared_texts": 0,
        "shared_documents": len(shared_documents),
        "document_disjoint_required": require_document_disjoint,
    }


def expand_records(
    records: Sequence[Dict[str, Any]], split_name: str
) -> Tuple[Dataset, Counter]:
    rows: List[Dict[str, Any]] = []
    label_counts: Counter = Counter()

    for record in records:
        premise = clean_text(record["text"])
        anchor_identifier = record_id(record)
        assert premise is not None
        assert anchor_identifier is not None

        for field, label_name in VARIANT_LABELS:
            hypothesis = clean_text(record[field])
            assert hypothesis is not None
            rows.append(
                {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "labels": LABEL_TO_ID[label_name],
                }
            )
            label_counts[label_name] += 1

    if not rows:
        raise ValueError(f"No NLI examples were generated for {split_name}.")

    print(
        f"{split_name}: {len(records)} anchors -> {len(rows)} NLI examples; "
        f"labels={dict(label_counts)}"
    )
    return Dataset.from_list(rows), label_counts


def expand_pair_records(
    records: Sequence[Dict[str, Any]], split_name: str, augment_reverse: bool
) -> Tuple[Dataset, Counter]:
    rows: list[dict[str, Any]] = []
    label_counts: Counter = Counter()
    for record in records:
        label_name = PAIR_VERDICT_TO_LABEL[
            str(record["label"]).casefold()
        ]
        orders = [(record["text1"], record["text2"])]
        if augment_reverse:
            orders.append((record["text2"], record["text1"]))
        for premise, hypothesis in orders:
            rows.append(
                {
                    "premise": str(premise).strip(),
                    "hypothesis": str(hypothesis).strip(),
                    "labels": LABEL_TO_ID[label_name],
                }
            )
            label_counts[label_name] += 1
    if not rows:
        raise ValueError(f"No NLI examples were generated for {split_name}.")
    print(
        f"{split_name}: {len(records)} direct pairs -> {len(rows)} NLI examples; "
        f"labels={dict(label_counts)}; reverse_augmented={augment_reverse}"
    )
    return Dataset.from_list(rows), label_counts


def prepare_nli_datasets(
    train_records: Sequence[Dict[str, Any]],
    validation_records: Sequence[Dict[str, Any]],
    train_path: Path,
    validation_path: Path,
    require_document_disjoint: bool,
    augment_reverse: bool,
) -> tuple[str, Dataset, Dataset, Counter, Counter, Dict[str, Any]]:
    train_format = detect_data_format(train_records, train_path)
    validation_format = detect_data_format(validation_records, validation_path)
    if train_format != validation_format:
        raise ValueError(
            f"Training format {train_format} does not match validation format "
            f"{validation_format}"
        )
    if train_format == "anchor_variants":
        validate_records(train_records, train_path)
        validate_records(validation_records, validation_path)
        split_validation = validate_disjoint_splits(
            train_records,
            validation_records,
            require_document_disjoint=require_document_disjoint,
        )
        train_dataset, train_counts = expand_records(train_records, "train")
        validation_dataset, validation_counts = expand_records(
            validation_records, "validation"
        )
    else:
        validate_pair_records(train_records, train_path)
        validate_pair_records(validation_records, validation_path)
        split_validation = validate_pair_disjoint_splits(
            train_records,
            validation_records,
            require_document_disjoint=require_document_disjoint,
        )
        train_dataset, train_counts = expand_pair_records(
            train_records, "train", augment_reverse=augment_reverse
        )
        validation_dataset, validation_counts = expand_pair_records(
            validation_records, "validation", augment_reverse=False
        )
    return (
        train_format,
        train_dataset,
        validation_dataset,
        train_counts,
        validation_counts,
        split_validation,
    )


def count_truncated_pairs(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int = 64,
) -> int:
    truncated = 0
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        encoded = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=False,
            add_special_tokens=True,
        )
        truncated += sum(
            len(input_ids) > max_length for input_ids in encoded["input_ids"]
        )
    return truncated


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
    split_name: str,
) -> Dataset:
    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=64,
        remove_columns=["premise", "hypothesis"],
        desc=f"Tokenizing {split_name}",
    )
    return tokenized


def compute_metrics(eval_prediction: Any) -> Dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    label_ids = list(range(len(LABEL_NAMES)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }
    for index, label_name in enumerate(LABEL_NAMES):
        metrics[f"precision_{label_name}"] = float(precision[index])
        metrics[f"recall_{label_name}"] = float(recall[index])
        metrics[f"f1_{label_name}"] = float(f1[index])
    return metrics


def parse_report_to(value: str) -> Any:
    if value.strip().lower() in {"", "none", "null"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Counter):
        return dict(value)
    return value


def runtime_environment() -> Dict[str, Any]:
    """Capture the software and accelerator context for a training run."""

    packages = {}
    for name in ("datasets", "scikit-learn", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": packages,
        "cuda_devices": cuda_devices,
    }


def build_training_args(
    args: argparse.Namespace,
    output_dir: Path,
    evaluation_enabled: bool,
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
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "optim": args.optim,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "save_strategy": "epoch" if evaluation_enabled else "no",
        "save_total_limit": args.save_total_limit,
        "save_only_model": args.save_only_model,
        "load_best_model_at_end": (
            evaluation_enabled
        ),
        "metric_for_best_model": "macro_f1" if evaluation_enabled else None,
        "greater_is_better": True if evaluation_enabled else None,
        "remove_unused_columns": True,
        "label_names": ["labels"],
        "eval_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "ddp_find_unused_parameters": False,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": parse_report_to(args.report_to),
        "run_name": args.run_name,
        "save_safetensors": True,
    }

    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["eval_strategy"] = "epoch" if evaluation_enabled else "no"
    else:
        training_kwargs["evaluation_strategy"] = (
            "epoch" if evaluation_enabled else "no"
        )

    if not args.no_deepspeed:
        training_kwargs["deepspeed"] = str(args.deepspeed_config)

    return TrainingArguments(**training_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune domain-pretrained DeBERTa on O-RAN NLI pairs."
    )
    parser.add_argument("--train_jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument(
        "--validation_jsonl",
        "--eval_jsonl",
        dest="validation_jsonl",
        type=Path,
        default=DEFAULT_VALIDATION_JSONL,
    )
    parser.add_argument(
        "--domain-checkpoint", "--base_model", dest="base_model",
        type=Path, default=DEFAULT_BASE_MODEL,
        help="Domain-adapted DeBERTa checkpoint used to initialize NLI training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Trainer checkpoint from which to resume.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--deepspeed_config",
        type=Path,
        default=DEFAULT_DEEPSPEED_CONFIG,
    )
    parser.add_argument("--no_deepspeed", action="store_true")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument(
        "--stop_after_steps",
        type=int,
        help=(
            "Stop after this optimizer step while retaining the epoch-based learning-"
            "rate schedule horizon. This differs from --max_steps, which also changes "
            "the scheduler horizon."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=10,
        help="Keep all epoch checkpoints for the default 10-epoch run so the "
        "best DeepSpeed checkpoint cannot be pruned before reload.",
    )
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
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require_document_disjoint",
        action="store_true",
        help="Reject train/validation files sharing any pdf_file value.",
    )
    parser.add_argument(
        "--augment_reverse",
        action="store_true",
        help="For direct-pair training data, add text2/text1 order to training only.",
    )
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--overwrite_output_dir", action="store_true")
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
    if not args.train_jsonl.exists():
        raise FileNotFoundError(f"Training JSONL not found: {args.train_jsonl}")
    if not args.validation_jsonl.exists():
        raise FileNotFoundError(
            f"Validation JSONL not found: {args.validation_jsonl}"
        )
    if not args.base_model.exists():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")
    if not args.no_deepspeed and not args.deepspeed_config.exists():
        raise FileNotFoundError(
            f"DeepSpeed config not found: {args.deepspeed_config}. "
            "Use --no_deepspeed or pass --deepspeed_config."
        )
    if args.save_only_model and not args.no_deepspeed:
        raise ValueError(
            "--save_only_model is incompatible with DeepSpeed plus "
            "load_best_model_at_end; pass --no_deepspeed."
        )
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max_steps must be -1 or a positive integer")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise ValueError("--stop_after_steps must be positive")
    if args.stop_after_steps is not None and args.max_steps != -1:
        raise ValueError(
            "--stop_after_steps requires epoch-based scheduling; do not combine it "
            "with --max_steps"
        )
    if args.epochs <= 0 and args.max_steps == -1:
        raise ValueError("--epochs must be positive when --max_steps is not set")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup_ratio must be between 0 and 1")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("Adam beta values must be in [0, 1)")
    if args.adam_epsilon <= 0:
        raise ValueError("--adam_epsilon must be positive")
    if args.max_grad_norm < 0:
        raise ValueError("--max_grad_norm cannot be negative")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience cannot be negative")
    if args.save_total_limit <= 0:
        raise ValueError("--save_total_limit must be positive")
    if args.logging_steps <= 0:
        raise ValueError("--logging_steps must be positive")
    if args.fp16 and args.bf16:
        raise ValueError("Use only one of FP16 and BF16")
    if (args.fp16 or args.bf16) and not torch.cuda.is_available():
        raise RuntimeError(
            "Mixed precision was requested but CUDA is unavailable. "
            "Use --no-fp16 and --no-bf16 for CPU execution."
        )




def main() -> None:
    args = parse_args()
    args.train_jsonl = resolve_path(args.train_jsonl)
    args.validation_jsonl = resolve_path(args.validation_jsonl)
    args.base_model = resolve_path(args.base_model)
    args.output_dir = resolve_path(args.output_dir)
    args.deepspeed_config = resolve_path(args.deepspeed_config)
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = resolve_path(args.resume_from_checkpoint)
        if not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume_from_checkpoint}"
            )
    validate_args(args)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite_output_dir:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Use a new directory or --overwrite_output_dir."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Training data: {args.train_jsonl}")
    print(f"Training SHA-256: {sha256_file(args.train_jsonl)}")
    print(f"Validation data: {args.validation_jsonl}")
    print(f"Validation SHA-256: {sha256_file(args.validation_jsonl)}")
    print(f"Base model: {args.base_model}")
    print(f"Output: {args.output_dir}")
    print(f"Maximum sequence length: {args.max_length}")
    print(f"Seed: {args.seed}")
    print(f"FP16: {args.fp16}; BF16: {args.bf16}")
    print(
        f"Training configuration: epochs={args.epochs}, "
        f"max_steps={args.max_steps}, batch_size={args.batch_size}, "
        f"eval_batch_size={args.eval_batch_size}, "
        f"gradient_accumulation={args.gradient_accumulation_steps}, "
        f"early_stopping_patience={args.early_stopping_patience}"
    )
    if args.stop_after_steps is not None:
        print(
            f"Fixed stop: step {args.stop_after_steps} within the "
            f"{args.epochs:g}-epoch scheduler horizon"
        )
    print(
        f"DeepSpeed: {'disabled' if args.no_deepspeed else args.deepspeed_config}"
    )

    train_records = load_jsonl(args.train_jsonl)
    validation_records = load_jsonl(args.validation_jsonl)
    (
        data_format,
        train_dataset,
        validation_dataset,
        train_label_counts,
        validation_label_counts,
        split_validation,
    ) = prepare_nli_datasets(
        train_records,
        validation_records,
        args.train_jsonl,
        args.validation_jsonl,
        args.require_document_disjoint,
        args.augment_reverse,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("The DeBERTa tokenizer has no usable pad token")

    train_truncated = count_truncated_pairs(
        train_dataset, tokenizer, args.max_length
    )
    validation_truncated = count_truncated_pairs(
        validation_dataset, tokenizer, args.max_length
    )
    print(
        f"Pairs exceeding max length: train={train_truncated}, "
        f"validation={validation_truncated}"
    )

    tokenized_train = tokenize_dataset(
        train_dataset, tokenizer, args.max_length, "train"
    )
    tokenized_validation = tokenize_dataset(
        validation_dataset, tokenizer, args.max_length, "validation"
    )

    model_kwargs: Dict[str, Any] = {
        "num_labels": len(LABEL_NAMES),
        "id2label": {index: name for index, name in enumerate(LABEL_NAMES)},
        "label2id": LABEL_TO_ID,
    }
    # Standard Trainer AMP expects FP32 master parameters and performs the
    # forward pass under autocast. Loading FP16 parameters here makes
    # GradScaler fail while unscaling FP16 gradients. Preserve the configured
    # low-precision load only for the DeepSpeed path, which owns precision and
    # optimizer-state handling itself.
    if not args.no_deepspeed:
        if args.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        elif args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "single_label_classification"
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    evaluation_enabled = args.max_steps == -1
    callbacks: list[TrainerCallback] = [AbortOnNonFiniteCallback()]
    if args.stop_after_steps is not None:
        callbacks.append(StopAfterStepsCallback(args.stop_after_steps))
    if evaluation_enabled and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    training_args = build_training_args(
        args, args.output_dir, evaluation_enabled=evaluation_enabled
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation if evaluation_enabled else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if evaluation_enabled else None,
        callbacks=callbacks,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if (
        args.stop_after_steps is not None
        and trainer.state.global_step != args.stop_after_steps
    ):
        raise RuntimeError(
            f"Training stopped at step {trainer.state.global_step}; expected fixed "
            f"step {args.stop_after_steps}"
        )
    final_eval_metrics: Dict[str, Any] = {}
    if evaluation_enabled:
        final_eval_metrics = trainer.evaluate()
        print(f"Final evaluation metrics: {final_eval_metrics}")
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    # With validation enabled, Trainer has restored the selected best model.
    # This root directory is the stable path consumed by later inference.
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if trainer.is_world_process_zero():
        manifest = {
            "script": str(Path(__file__).resolve()),
            "train_jsonl": str(args.train_jsonl),
            "train_sha256": sha256_file(args.train_jsonl),
            "validation_jsonl": str(args.validation_jsonl),
            "validation_sha256": sha256_file(args.validation_jsonl),
            "base_model": str(args.base_model),
            "output_dir": str(args.output_dir),
            "deepspeed_config": (
                None if args.no_deepspeed else str(args.deepspeed_config)
            ),
            "label_names": LABEL_NAMES,
            "data_format": data_format,
            "augment_reverse": args.augment_reverse,
            "train_anchors": len(train_records),
            "validation_anchors": len(validation_records),
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "train_label_counts": dict(train_label_counts),
            "validation_label_counts": dict(validation_label_counts),
            "train_pairs_truncated": train_truncated,
            "validation_pairs_truncated": validation_truncated,
            "split_validation": split_validation,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "train_metrics": make_jsonable(train_result.metrics),
            "final_eval_metrics": make_jsonable(final_eval_metrics),
            "schedule_horizon_steps": trainer.state.max_steps,
            "actual_stop_step": trainer.state.global_step,
            "runtime_environment": runtime_environment(),
            "arguments": make_jsonable(vars(args)),
        }
        manifest_path = args.output_dir / "training_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote training manifest: {manifest_path}")

    if trainer.is_world_process_zero():
        print(f"Saved DeBERTa NLI model: {args.output_dir}")


if __name__ == "__main__":
    main()
