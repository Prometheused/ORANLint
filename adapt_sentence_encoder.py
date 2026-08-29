# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Domain-adaptive pretraining for the BGE sentence encoder.

This stage uses identity pairs from the refreshed 4G/5G/O-RAN text corpus and
Multiple Negatives Ranking Loss. It does not apply security or variant
filtering and does not consume the synthetic pair file.
"""

import argparse
import hashlib
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_PATHS = {
    "4G": PROJECT_ROOT / "data/generated/pretraining/processed_4G/corpus_4G.txt",
    "5G": PROJECT_ROOT / "data/generated/pretraining/processed_5G/corpus_5G.txt",
    "ORAN": PROJECT_ROOT / "data/generated/pretraining/processed_ORAN/corpus_ORAN.txt",
}
DEFAULT_PACKED_OUTPUT_DIR = (
    PROJECT_ROOT / "data/generated/sentence_encoder_packed"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models/sentence_encoder_adapted"
DEFAULT_LOGGING_DIR = PROJECT_ROOT / "logs/sentence_encoder_adaptation"
BASE_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 3
DEFAULT_MAX_CONTENT_TOKENS = 510


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain BGE on the refreshed 4G/5G/O-RAN corpus."
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Public model identifier or compatible local checkpoint.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Trainer checkpoint from which to resume.",
    )
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=None,
        help=(
            "Optional single UTF-8 corpus override. By default, the 4G, 5G, "
            "and O-RAN files are packed separately."
        ),
    )
    parser.add_argument(
        "--corpus_4g_path",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["4G"],
        help="UTF-8 4G paragraph corpus.",
    )
    parser.add_argument(
        "--corpus_5g_path",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["5G"],
        help="UTF-8 5G paragraph corpus.",
    )
    parser.add_argument(
        "--corpus_oran_path",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["ORAN"],
        help="UTF-8 O-RAN paragraph corpus.",
    )
    parser.add_argument(
        "--packed_output_dir",
        type=Path,
        default=DEFAULT_PACKED_OUTPUT_DIR,
        help="Directory for source-specific and combined packed corpora.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the pretrained SentenceTransformer is saved.",
    )
    parser.add_argument(
        "--logging_dir",
        type=Path,
        default=DEFAULT_LOGGING_DIR,
        help="Directory for trainer logs.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max_content_tokens",
        type=int,
        default=DEFAULT_MAX_CONTENT_TOKENS,
        help="Maximum non-special-token length for each BGE training chunk.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum number of optimizer steps.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve relative CLI paths from the project directory."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy, and Torch random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_paragraphs(corpus_path: Path) -> Tuple[List[str], str]:
    """Load nonempty UTF-8 paragraphs and return them with a SHA-256 hash."""
    raw = corpus_path.read_bytes()
    paragraphs = [
        line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    if not paragraphs:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    return paragraphs, hashlib.sha256(raw).hexdigest()


def effective_content_token_limit(
    tokenizer, model: SentenceTransformer, requested_limit: int
) -> Tuple[int, int, int]:
    """Reserve BGE special tokens and return usable content capacity."""
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    model_limit = getattr(model, "max_seq_length", None)
    sequence_limits = [
        int(limit)
        for limit in (tokenizer_limit, model_limit)
        if isinstance(limit, int) and 0 < limit < 100_000
    ]
    if not sequence_limits:
        raise ValueError("Could not determine BGE's maximum sequence length.")

    sequence_limit = min(sequence_limits)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    usable_limit = sequence_limit - special_tokens
    if usable_limit <= 0:
        raise ValueError("BGE sequence length leaves no room for content tokens.")

    effective_limit = min(requested_limit, usable_limit)
    if effective_limit != requested_limit:
        print(
            "Requested content-token limit "
            f"{requested_limit} exceeds BGE capacity; using {effective_limit}."
        )
    return effective_limit, sequence_limit, special_tokens


def pack_paragraphs(
    paragraphs: List[str], tokenizer, max_content_tokens: int
) -> Tuple[List[str], int, int]:
    """Pack adjacent paragraphs into BGE-sized chunks without dropping tokens.

    Paragraphs that fit are kept whole and accumulated. Paragraphs longer than
    the limit are split by tokenizer IDs; their final partial piece may be
    combined with the following paragraph.
    """
    packed_token_chunks: List[List[int]] = []
    current_tokens: List[int] = []
    long_paragraphs = 0
    total_tokens = 0

    def flush_current() -> None:
        nonlocal current_tokens
        if current_tokens:
            packed_token_chunks.append(current_tokens)
            current_tokens = []

    for paragraph in paragraphs:
        token_ids = tokenizer.encode(paragraph, add_special_tokens=False)
        if not token_ids:
            continue

        total_tokens += len(token_ids)
        if len(token_ids) > max_content_tokens:
            long_paragraphs += 1
            flush_current()

            while len(token_ids) > max_content_tokens:
                packed_token_chunks.append(token_ids[:max_content_tokens])
                token_ids = token_ids[max_content_tokens:]
            current_tokens = token_ids
            continue

        if current_tokens and len(current_tokens) + len(token_ids) > max_content_tokens:
            flush_current()
        current_tokens.extend(token_ids)

        if len(current_tokens) == max_content_tokens:
            flush_current()

    flush_current()

    packed_chunks = [
        tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        for token_ids in packed_token_chunks
    ]
    packed_chunks = [chunk for chunk in packed_chunks if chunk]
    return packed_chunks, long_paragraphs, total_tokens


def write_lines(path: Path, lines: List[str]) -> str:
    """Write UTF-8 line-oriented text and return its SHA-256 hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_sources(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    """Resolve either the three source files or one explicit corpus override."""
    if args.corpus_path is not None:
        return [("custom", resolve_path(args.corpus_path))]
    return [
        ("4G", resolve_path(args.corpus_4g_path)),
        ("5G", resolve_path(args.corpus_5g_path)),
        ("ORAN", resolve_path(args.corpus_oran_path)),
    ]


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.max_content_tokens <= 0:
        raise ValueError("--max_content_tokens must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be positive when provided")

    output_dir = resolve_path(args.output_dir)
    logging_dir = resolve_path(args.logging_dir)
    packed_output_dir = resolve_path(args.packed_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    print(f"Base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    max_content_tokens, sequence_limit, special_tokens = effective_content_token_limit(
        tokenizer, model, args.max_content_tokens
    )

    source_paths = resolve_sources(args)
    packed_chunks: List[str] = []
    source_hashes: Dict[str, str] = {}
    source_stats = []

    for source_name, source_path in source_paths:
        paragraphs, source_hash = load_paragraphs(source_path)
        source_hashes[source_name] = source_hash
        source_chunks, long_paragraphs, source_token_count = pack_paragraphs(
            paragraphs, tokenizer, max_content_tokens
        )
        packed_chunks.extend(source_chunks)

        packed_path = packed_output_dir / f"corpus_{source_name}_bge_packed.txt"
        packed_hash = write_lines(packed_path, source_chunks)
        source_stats.append(
            {
                "name": source_name,
                "path": source_path,
                "paragraphs": len(paragraphs),
                "tokens": source_token_count,
                "long_paragraphs": long_paragraphs,
                "packed_chunks": len(source_chunks),
                "packed_path": packed_path,
                "packed_hash": packed_hash,
            }
        )

    combined_path = packed_output_dir / "corpus_4G_5G_ORAN_bge_packed.txt"
    combined_hash = write_lines(combined_path, packed_chunks)
    duplicate_count = len(packed_chunks) - len(set(packed_chunks))

    print(f"BGE sequence limit: {sequence_limit} tokens")
    print(f"Special tokens reserved: {special_tokens}")
    print(f"Content-token limit: {max_content_tokens}")
    for stats in source_stats:
        print(
            f"{stats['name']}: {stats['paragraphs']:,} paragraphs -> "
            f"{stats['packed_chunks']:,} packed chunks; "
            f"{stats['long_paragraphs']:,} long paragraphs"
        )
        print(f"  Packed corpus: {stats['packed_path']}")
        print(f"  Source SHA-256: {source_hashes[stats['name']]}")
        print(f"  Packed SHA-256: {stats['packed_hash']}")
    print(f"Combined packed corpus: {combined_path}")
    print(f"Combined packed records: {len(packed_chunks):,}")
    print(f"Combined packed SHA-256: {combined_hash}")
    print(f"Duplicate packed records retained: {duplicate_count:,}")
    print(f"Output: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    if args.max_steps is not None:
        print(f"Maximum steps: {args.max_steps}")

    # Each packed chunk is paired with itself for identity-pair MNRL.
    train_dataset = Dataset.from_dict(
        {"anchor": packed_chunks, "positive": packed_chunks}
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save the loadable SentenceTransformer directory at the requested root.
    model.save(str(output_dir))
    print(f"Saved pretrained BGE model: {output_dir}")


if __name__ == "__main__":
    main()
