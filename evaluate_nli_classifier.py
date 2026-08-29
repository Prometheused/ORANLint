#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Evaluate an NLI classifier on a labeled development JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.nn import functional as functional
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
LABELS = ("entailment", "neutral", "contradiction")
VERDICT_TO_ID = {"consistent": 0, "neutral": 1, "inconsistent": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nli-checkpoint",
        default=str(ROOT / "models/nli_classifier/checkpoint-3000"),
    )
    parser.add_argument(
        "--development-jsonl",
        type=Path,
        default=ROOT / "data/generated/nli_supervision/development_pairs.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if not all(isinstance(value.get(field), str) for field in ("text1", "text2", "label")):
                raise ValueError(f"Invalid NLI record at {path}:{line_number}")
            if value["label"] not in VERDICT_TO_ID:
                raise ValueError(f"Invalid label at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"No records found: {path}")
    return records


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("Batch size and maximum length must be positive")
    records = load_records(args.development_jsonl.expanduser().resolve())
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.nli_checkpoint, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_checkpoint)
    model.to(device).eval()
    expected = []
    predicted = []
    losses = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        labels = torch.tensor(
            [VERDICT_TO_ID[row["label"]] for row in batch],
            dtype=torch.long,
            device=device,
        )
        encoded = tokenizer(
            [row["text1"] for row in batch],
            [row["text2"] for row in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        losses.extend(functional.cross_entropy(logits, labels, reduction="none").cpu().tolist())
        expected.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(dim=-1).cpu().tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=list(range(len(LABELS))), zero_division=0
    )
    result = {
        "records": len(records),
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=list(range(len(LABELS)))
        ).tolist(),
        "checkpoint": args.nli_checkpoint,
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
