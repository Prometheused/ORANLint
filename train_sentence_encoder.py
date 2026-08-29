#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Validation-aware supervised fine-tuning of the refreshed BGE model.

Phase 1 trains MultipleNegativesRankingLoss on (text, paraphrased) pairs.
Phase 2 trains TripletLoss on (text, paraphrased, inconsistent/randomized)
triplets.  The validation file is generated from anchors that are separate
from the training file, so no row-level split is performed here.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from transformers import EarlyStoppingCallback


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_JSONL = PROJECT_ROOT / "data/generated/encoder_supervision_train.jsonl"
DEFAULT_EVAL_JSONL = PROJECT_ROOT / "data/generated/encoder_supervision_validation.jsonl"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models/sentence_encoder_adapted"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models/sentence_encoder"
DEFAULT_LOGGING_DIR = PROJECT_ROOT / "logs/sentence_encoder_training"


def resolve_path(path: Path) -> Path:
    """Resolve relative CLI paths relative to the project directory."""

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch data/training randomness."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict]:
    """Load a non-empty JSON object from every nonblank JSONL line."""

    records: List[Dict] = []
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
                    f"Expected a JSON object in {path} at line {line_number}, "
                    f"got {type(record).__name__}."
                )
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def nonempty_text(record: Dict, field: str) -> Optional[str]:
    value = record.get(field)
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def normalized_anchor(record: Dict) -> Optional[str]:
    text = nonempty_text(record, "text")
    return text.casefold() if text is not None else None


def record_identifier(record: Dict) -> Optional[str]:
    value = record.get("id")
    if value is None:
        return None
    return str(value)


def validate_disjoint_splits(
    train_records: Sequence[Dict],
    eval_records: Sequence[Dict],
    require_document_disjoint: bool = False,
) -> Dict[str, Any]:
    """Reject validation anchors that are also present in training."""

    train_ids = {
        identifier
        for record in train_records
        if (identifier := record_identifier(record)) is not None
    }
    eval_ids = {
        identifier
        for record in eval_records
        if (identifier := record_identifier(record)) is not None
    }
    train_texts = {
        anchor for record in train_records if (anchor := normalized_anchor(record))
    }
    eval_texts = {
        anchor for record in eval_records if (anchor := normalized_anchor(record))
    }

    shared_ids = train_ids & eval_ids
    shared_texts = train_texts & eval_texts
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
    eval_documents = {
        str(record.get("pdf_file"))
        for record in eval_records
        if record.get("pdf_file")
    }
    shared_documents = train_documents & eval_documents
    if require_document_disjoint and shared_documents:
        raise ValueError(
            "Training and validation documents overlap: "
            f"{len(shared_documents)} shared pdf_file values."
        )

    print(
        "Disjoint anchor check: "
        f"train={len(train_records)}, validation={len(eval_records)}, "
        "shared_ids=0, shared_texts=0"
    )
    return {
        "train_anchors": len(train_records),
        "validation_anchors": len(eval_records),
        "shared_ids": 0,
        "shared_texts": 0,
        "shared_documents": len(shared_documents),
        "document_disjoint_required": require_document_disjoint,
    }


def make_mnlr_dataset(records: Sequence[Dict], split_name: str) -> Dataset:
    """Create raw-text (anchor, positive) examples for MNLR."""

    anchors: List[str] = []
    positives: List[str] = []
    skipped = 0

    for record in records:
        anchor = nonempty_text(record, "text")
        positive = nonempty_text(record, "paraphrased")
        if anchor is None or positive is None:
            skipped += 1
            continue
        anchors.append(anchor)
        positives.append(positive)

    if not anchors:
        raise ValueError(f"No usable MNLR records found in {split_name} split.")

    print(
        f"{split_name} MNLR examples: {len(anchors)} "
        f"(skipped {skipped} incomplete records)"
    )
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def make_triplet_dataset(
    records: Sequence[Dict],
    split_name: str,
    seed: int,
    inconsistent_probability: float,
) -> Dataset:
    """Create raw-text triplets with deterministic negative selection."""

    anchors: List[str] = []
    positives: List[str] = []
    negatives: List[str] = []
    negative_types = {"inconsistent": 0, "randomized": 0}
    skipped = 0
    rng = random.Random(seed)

    for record in records:
        anchor = nonempty_text(record, "text")
        positive = nonempty_text(record, "paraphrased")
        inconsistent = nonempty_text(record, "inconsistent")
        randomized = nonempty_text(record, "randomized")

        if anchor is None or positive is None or (
            inconsistent is None and randomized is None
        ):
            skipped += 1
            continue

        if inconsistent is not None and randomized is not None:
            if rng.random() < inconsistent_probability:
                negative = inconsistent
                negative_type = "inconsistent"
            else:
                negative = randomized
                negative_type = "randomized"
        elif inconsistent is not None:
            negative = inconsistent
            negative_type = "inconsistent"
        else:
            negative = randomized
            negative_type = "randomized"

        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
        negative_types[negative_type] += 1

    if not anchors:
        raise ValueError(f"No usable triplet records found in {split_name} split.")

    print(
        f"{split_name} triplet examples: {len(anchors)} "
        f"(inconsistent={negative_types['inconsistent']}, "
        f"randomized={negative_types['randomized']}, skipped={skipped})"
    )
    return Dataset.from_dict(
        {"anchor": anchors, "positive": positives, "negative": negatives}
    )


def run_phase(
    *,
    model: SentenceTransformer,
    phase_name: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    loss,
    output_root: Path,
    logging_root: Path,
    epochs: int,
    batch_size: int,
    max_steps: Optional[int],
    learning_rate: float,
    warmup_ratio: float,
    seed: int,
    fp16: bool,
    early_stopping_patience: int,
    early_stopping_threshold: float,
    save_total_limit: int,
    save_only_model: bool,
    drop_last: bool,
) -> None:
    """Train one phase and save the best SentenceTransformer model."""

    phase_output = output_root / phase_name
    phase_logging = logging_root / phase_name
    phase_output.mkdir(parents=True, exist_ok=True)
    phase_logging.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(phase_output),
        logging_dir=str(phase_logging),
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=500,
        save_total_limit=save_total_limit,
        save_only_model=save_only_model,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=False,
        report_to=[],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_drop_last=drop_last,
    )

    print(
        f"\nStarting {phase_name}: train={len(train_dataset)}, "
        f"validation={len(eval_dataset)}, epochs={epochs}, "
        f"batch_size={batch_size}, max_steps={max_steps}, "
        "evaluation=enabled"
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        callbacks=callbacks,
    )
    trainer.train()

    metrics: Dict[str, Any] = trainer.evaluate()
    print(f"{phase_name} final evaluation metrics: {metrics}")
    print(
        f"{phase_name} best checkpoint: "
        f"{trainer.state.best_model_checkpoint}"
    )

    # Save the best model at the phase root for downstream use.
    model.save(str(phase_output))
    print(f"Saved {phase_name} model: {phase_output}")
    return {
        "phase": phase_name,
        "epochs_completed": trainer.state.epoch,
        "global_step": trainer.state.global_step,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "eval_metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune domain-pretrained BGE on synthetic O-RAN pairs."
    )
    parser.add_argument("--train_jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument(
        "--eval_jsonl",
        "--validation_jsonl",
        dest="eval_jsonl",
        type=Path,
        default=DEFAULT_EVAL_JSONL,
    )
    parser.add_argument(
        "--model-checkpoint", "--base_model", dest="base_model",
        type=Path, default=DEFAULT_BASE_MODEL,
        help="Adapted sentence-encoder checkpoint used to initialize training.",
    )
    parser.add_argument(
        "--output_dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--logging_dir", type=Path, default=DEFAULT_LOGGING_DIR)
    parser.add_argument("--epochs_mnlr", type=int, default=3)
    parser.add_argument("--epochs_triplet", type=int, default=3)
    parser.add_argument("--batch_mnlr", type=int, default=8)
    parser.add_argument("--batch_triplet", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument(
        "--neg_mix_ratio",
        type=float,
        default=0.65,
        help="Probability of choosing inconsistent over randomized negatives.",
    )
    parser.add_argument("--max_steps_mnlr", type=int, default=None)
    parser.add_argument("--max_steps_triplet", type=int, default=None)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="0 disables early stopping; validation and best-model loading stay enabled.",
    )
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help=(
            "Maximum retained checkpoints per phase. Use at least the epoch "
            "training budget."
        ),
    )
    parser.add_argument(
        "--save_only_model",
        action="store_true",
        help="Store model weights without optimizer state (checkpoint resume is disabled).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require_document_disjoint",
        action="store_true",
        help="Reject train/validation files sharing any pdf_file value.",
    )

    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--fp16", dest="fp16", action="store_true")
    fp16_group.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for name, value in (
        ("epochs_mnlr", args.epochs_mnlr),
        ("epochs_triplet", args.epochs_triplet),
        ("batch_mnlr", args.batch_mnlr),
        ("batch_triplet", args.batch_triplet),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    for name, value in (
        ("max_steps_mnlr", args.max_steps_mnlr),
        ("max_steps_triplet", args.max_steps_triplet),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when provided, got {value}.")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1.")
    if not 0.0 <= args.neg_mix_ratio <= 1.0:
        raise ValueError("neg_mix_ratio must be between 0 and 1.")
    if args.margin <= 0:
        raise ValueError("margin must be positive.")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative.")
    if args.save_total_limit <= 0:
        raise ValueError("save_total_limit must be positive.")

    train_jsonl = resolve_path(args.train_jsonl)
    eval_jsonl = resolve_path(args.eval_jsonl)
    base_model = resolve_path(args.base_model)
    output_dir = resolve_path(args.output_dir)
    logging_dir = resolve_path(args.logging_dir)

    for path, label in (
        (train_jsonl, "training JSONL"),
        (eval_jsonl, "validation JSONL"),
        (base_model, "base model"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    seed_everything(args.seed)
    fp16 = torch.cuda.is_available() if args.fp16 is None else args.fp16

    print(f"Training data: {train_jsonl}")
    print(f"Training SHA-256: {sha256_file(train_jsonl)}")
    print(f"Validation data: {eval_jsonl}")
    print(f"Validation SHA-256: {sha256_file(eval_jsonl)}")
    print(f"Base model: {base_model}")
    print(f"Output root: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"FP16: {fp16}")
    print("Metadata prefixes and variant tags: disabled; raw text is used.")

    train_records = load_jsonl(train_jsonl)
    eval_records = load_jsonl(eval_jsonl)
    split_validation = validate_disjoint_splits(
        train_records,
        eval_records,
        require_document_disjoint=args.require_document_disjoint,
    )

    train_mnlr = make_mnlr_dataset(train_records, "train")
    eval_mnlr = make_mnlr_dataset(eval_records, "validation")
    train_triplets = make_triplet_dataset(
        train_records, "train", args.seed, args.neg_mix_ratio
    )
    eval_triplets = make_triplet_dataset(
        eval_records, "validation", args.seed + 1, args.neg_mix_ratio
    )

    model = SentenceTransformer(str(base_model))
    print("Loaded domain-pretrained BGE model.")

    mnlr_loss = losses.MultipleNegativesRankingLoss(model)
    mnlr_summary = run_phase(
        model=model,
        phase_name="phase_mnlr",
        train_dataset=train_mnlr,
        eval_dataset=eval_mnlr,
        loss=mnlr_loss,
        output_root=output_dir,
        logging_root=logging_dir,
        epochs=args.epochs_mnlr,
        batch_size=args.batch_mnlr,
        max_steps=args.max_steps_mnlr,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        fp16=fp16,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        drop_last=True,
    )

    triplet_loss = losses.TripletLoss(
        model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        triplet_margin=args.margin,
    )
    triplet_summary = run_phase(
        model=model,
        phase_name="phase_triplet",
        train_dataset=train_triplets,
        eval_dataset=eval_triplets,
        loss=triplet_loss,
        output_root=output_dir,
        logging_root=logging_dir,
        epochs=args.epochs_triplet,
        batch_size=args.batch_triplet,
        max_steps=args.max_steps_triplet,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        fp16=fp16,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        drop_last=False,
    )

    manifest = {
        "script": str(Path(__file__).resolve()),
        "train_jsonl": str(train_jsonl),
        "train_sha256": sha256_file(train_jsonl),
        "validation_jsonl": str(eval_jsonl),
        "validation_sha256": sha256_file(eval_jsonl),
        "base_model": str(base_model),
        "output_dir": str(output_dir),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "split_validation": split_validation,
        "phase_mnlr": mnlr_summary,
        "phase_triplet": triplet_summary,
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"BGE fine-tuning complete: {output_dir}")


if __name__ == "__main__":
    main()
