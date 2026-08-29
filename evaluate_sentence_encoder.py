#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Evaluate a sentence encoder on held-out synthetic quadruples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
REQUIRED_FIELDS = ("text", "paraphrased", "inconsistent", "randomized")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentence-encoder-checkpoint",
        default=str(ROOT / "models/sentence_encoder/phase_triplet"),
    )
    parser.add_argument(
        "--validation-jsonl",
        type=Path,
        default=ROOT / "data/generated/encoder_supervision_validation.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or any(
                not isinstance(value.get(field), str) or not value[field].strip()
                for field in REQUIRED_FIELDS
            ):
                raise ValueError(f"Invalid quadruple at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"No records found: {path}")
    return records


def main() -> None:
    args = parse_args()
    records = load_records(args.validation_jsonl.expanduser().resolve())
    device = None if args.device == "auto" else args.device
    model = SentenceTransformer(args.sentence_encoder_checkpoint, device=device)
    texts = [
        str(record[field])
        for record in records
        for field in REQUIRED_FIELDS
    ]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).reshape(len(records), len(REQUIRED_FIELDS), -1)
    positive = np.sum(embeddings[:, 0] * embeddings[:, 1], axis=1)
    inconsistent = np.sum(embeddings[:, 0] * embeddings[:, 2], axis=1)
    randomized = np.sum(embeddings[:, 0] * embeddings[:, 3], axis=1)
    selected_negative = np.maximum(inconsistent, randomized)
    result = {
        "records": len(records),
        "mean_anchor_positive_cosine": float(positive.mean()),
        "mean_anchor_inconsistent_cosine": float(inconsistent.mean()),
        "mean_anchor_randomized_cosine": float(randomized.mean()),
        "positive_ranking_accuracy": float(np.mean(positive > selected_negative)),
        "triplet_margin_accuracy": float(
            np.mean((1.0 - positive) + args.margin <= (1.0 - selected_negative))
        ),
        "margin": args.margin,
        "checkpoint": args.sentence_encoder_checkpoint,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
