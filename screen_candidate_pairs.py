#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Run the trained DeBERTa NLI classifier over candidate pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/generated/candidate_pairs.jsonl"
)
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models/nli_classifier/checkpoint-3000"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "runs/default/nli_predictions.jsonl"
)
LABEL_NAMES = ("entailment", "neutral", "contradiction")
REQUIRED_FIELDS = ("text1", "text2")
SELECTION_POLICIES = (
    "argmax_contradiction",
    "contradiction_probability",
    "top_k_argmax_contradiction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append DeBERTa NLI predictions to O-RAN candidate pairs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--nli-checkpoint", "--model", dest="model",
        type=Path, default=DEFAULT_MODEL,
        help="Trained NLI classifier checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--contradiction-threshold",
        "--contradiction_threshold",
        dest="contradiction_threshold",
        type=float,
        help=(
            "Inclusive contradiction-probability threshold. The supported range depends "
            "on --selection-policy."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default="top_k_argmax_contradiction",
        help=(
            "argmax_contradiction requires contradiction to be the winning class "
            "and limits thresholds to [0.5, 1]. contradiction_probability applies a "
            "probability cutoff directly and permits [0, 1]. "
            "top_k_argmax_contradiction deterministically retains at most --max-selected "
            "contradiction-argmax rows."
        ),
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        default=10000,
        help="Maximum rows selected by top_k_argmax_contradiction (default: 10000).",
    )
    parser.add_argument(
        "--selected-output",
        "--selected_output",
        dest="selected_output",
        type=Path,
        help="Threshold-selected JSONL; defaults to <output stem>_selected.jsonl.",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device, e.g. auto, cpu, cuda, or cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp16", "fp32", "bf16"),
        default="auto",
        help="Model/inference dtype. auto uses fp16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON manifest path; defaults to <output>.manifest.json.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_threshold(
    threshold: Optional[float], selection_policy: str = "argmax_contradiction"
) -> Optional[float]:
    if threshold is None:
        return None
    value = float(threshold)
    if selection_policy not in SELECTION_POLICIES:
        raise ValueError(f"Unsupported selection policy: {selection_policy}")
    if selection_policy == "top_k_argmax_contradiction":
        raise ValueError(
            "--contradiction-threshold is incompatible with "
            "top_k_argmax_contradiction"
        )
    minimum = 0.5 if selection_policy == "argmax_contradiction" else 0.0
    if not minimum <= value <= 1.0:
        raise ValueError(
            f"--contradiction-threshold must be between {minimum:g} and 1.0 "
            f"for {selection_policy}"
        )
    return value


def derive_selected_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(
            f"{output_path.stem}_selected{output_path.suffix}"
        )
    return output_path.with_name(f"{output_path.name}_selected.jsonl")


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"Missing required field(s) {missing} at {path}:{line_number}"
                )
            for field in REQUIRED_FIELDS:
                if not isinstance(record[field], str):
                    raise ValueError(
                        f"Field {field!r} must be a string at {path}:{line_number}"
                    )
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def select_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if requested == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 inference requires a CUDA device")
        return torch.float16
    if requested == "bf16":
        if device.type != "cuda":
            raise ValueError("bf16 inference requires a CUDA device")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 inference is not supported by the selected CUDA device")
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(
    model_path: Path, device: torch.device, dtype: torch.dtype
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("The DeBERTa tokenizer has no usable pad token")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.id2label = {
        index: label for index, label in enumerate(LABEL_NAMES)
    }
    model.config.label2id = {
        label: index for index, label in enumerate(LABEL_NAMES)
    }
    model.to(device)
    model.eval()
    return tokenizer, model


def run_inference(
    records: List[Dict[str, Any]],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    contradiction_threshold: Optional[float] = None,
    selection_policy: str = "argmax_contradiction",
    max_selected: int = 10000,
) -> tuple[Counter, int, int]:
    prediction_counts: Counter = Counter()
    sequences_at_max_length = 0
    selected_count = 0
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(records), batch_size),
            desc="Running DeBERTa inference",
        ):
            batch = records[start : start + batch_size]
            encoded = tokenizer(
                [record["text1"] for record in batch],
                [record["text2"] for record in batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            # DeBERTa does not consume token_type_ids; older tokenizers may
            # nevertheless return them for pair inputs.
            encoded.pop("token_type_ids", None)
            sequences_at_max_length += int(
                (encoded["attention_mask"].sum(dim=1) >= max_length).sum().item()
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float().cpu()
            prediction_ids = logits.argmax(dim=-1).tolist()
            probabilities = (
                torch.softmax(logits, dim=-1).tolist()
                if contradiction_threshold is not None
                or selection_policy == "top_k_argmax_contradiction"
                else None
            )
            for offset, (record, prediction_id) in enumerate(zip(batch, prediction_ids)):
                label = LABEL_NAMES[int(prediction_id)]
                record["deberta_preds"] = label
                prediction_counts[label] += 1
                if probabilities is not None:
                    row_probabilities = probabilities[offset]
                    contradiction_probability = float(row_probabilities[2])
                    record["deberta_probabilities"] = {
                        name: float(row_probabilities[index])
                        for index, name in enumerate(LABEL_NAMES)
                    }
                    record["deberta_contradiction_probability"] = contradiction_probability
                    if selection_policy != "top_k_argmax_contradiction":
                        selected = is_threshold_selected(
                            label,
                            contradiction_probability,
                            contradiction_threshold,
                            selection_policy,
                        )
                        record["deberta_contradiction_threshold"] = contradiction_threshold
                        record["deberta_selected"] = selected
                        selected_count += int(selected)
    if selection_policy == "top_k_argmax_contradiction":
        selected_count, _ = apply_top_k_selection(records, max_selected)
    return prediction_counts, sequences_at_max_length, selected_count


def stable_identifier(value: Any) -> tuple[int, Any]:
    """Sort numeric pair IDs numerically and retain deterministic text fallback."""

    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, str(value)


def stable_pair_key(
    record: Mapping[str, Any], input_index: int
) -> tuple[tuple[int, Any], tuple[int, Any], int]:
    identifiers = sorted(
        (stable_identifier(record.get("id1")), stable_identifier(record.get("id2")))
    )
    return identifiers[0], identifiers[1], input_index


def apply_top_k_selection(
    records: List[Dict[str, Any]], max_selected: int
) -> tuple[int, Optional[float]]:
    """Rank contradiction argmax rows deterministically and cap the GPT queue."""

    if max_selected <= 0:
        raise ValueError("max_selected must be positive")
    candidates = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("deberta_preds") == "contradiction"
    ]
    candidates.sort(
        key=lambda item: (
            -float(item[1]["deberta_contradiction_probability"]),
            *stable_pair_key(item[1], item[0]),
        )
    )
    for record in records:
        record["deberta_selected"] = False
        record.pop("deberta_selection_rank", None)
    selected = candidates[:max_selected]
    for rank, (_, record) in enumerate(selected, 1):
        record["deberta_selected"] = True
        record["deberta_selection_rank"] = rank
    cutoff = (
        float(selected[-1][1]["deberta_contradiction_probability"])
        if selected
        else None
    )
    return len(selected), cutoff


def is_threshold_selected(
    label: str,
    contradiction_probability: float,
    contradiction_threshold: float,
    selection_policy: str = "argmax_contradiction",
) -> bool:
    if selection_policy == "argmax_contradiction":
        return (
            label == "contradiction"
            and contradiction_probability >= contradiction_threshold
        )
    if selection_policy == "contradiction_probability":
        return contradiction_probability >= contradiction_threshold
    raise ValueError(f"Unsupported selection policy: {selection_policy}")


def selected_records(records: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [record for record in records if record.get("deberta_selected") is True]


def write_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    model_path = resolve_path(args.model)
    output_path = resolve_path(args.output)
    contradiction_threshold = validate_threshold(
        args.contradiction_threshold, args.selection_policy
    )
    top_k_policy = args.selection_policy == "top_k_argmax_contradiction"
    selected_output_path = None
    if args.selected_output is not None:
        selected_output_path = resolve_path(args.selected_output)
    elif contradiction_threshold is not None or top_k_policy:
        selected_output_path = derive_selected_output(output_path)
    manifest_path = resolve_path(args.manifest) if args.manifest else Path(
        str(output_path) + ".manifest.json"
    )

    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.max_selected <= 0:
        raise ValueError("--max-selected must be positive")
    if (
        args.selected_output is not None
        and contradiction_threshold is None
        and not top_k_policy
    ):
        raise ValueError(
            "--selected-output requires --contradiction-threshold or the top-K policy"
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    output_paths = [output_path, manifest_path]
    if selected_output_path is not None:
        output_paths.append(selected_output_path)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("Output, selected output, and manifest paths must be different")
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    records = load_records(input_path)
    tokenizer, model = load_model_and_tokenizer(model_path, device, dtype)

    print(f"Input: {input_path}")
    print(f"Records: {len(records):,}")
    print(f"Model: {model_path}")
    print(f"Device: {device}; dtype: {dtype}")
    print(f"Maximum sequence length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    if contradiction_threshold is not None:
        print(f"Contradiction threshold: {contradiction_threshold}")
    if top_k_policy:
        print(f"Maximum selected contradiction argmax rows: {args.max_selected:,}")

    prediction_counts, sequences_at_max_length, selected_count = run_inference(
        records,
        tokenizer,
        model,
        device,
        args.max_length,
        args.batch_size,
        contradiction_threshold,
        args.selection_policy,
        args.max_selected,
    )
    write_records(output_path, records)
    selected = selected_records(records) if selected_output_path is not None else []
    if selected_output_path is not None:
        write_records(selected_output_path, selected)
    selected_argmax_contradictions = sum(
        bool(record.get("deberta_selected"))
        and record.get("deberta_preds") == "contradiction"
        for record in records
    )
    selected_non_argmax_contradictions = sum(
        bool(record.get("deberta_selected"))
        and record.get("deberta_preds") != "contradiction"
        for record in records
    )
    effective_cutoff = (
        min(
            float(record["deberta_contradiction_probability"])
            for record in records
            if record.get("deberta_selected")
        )
        if selected_count
        else None
    )

    manifest = {
        "script": str(Path(__file__).resolve()),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "model": str(model_path),
        "output": str(output_path),
        "records": len(records),
        "prediction_field": "deberta_preds",
        "prediction_counts": dict(prediction_counts),
        "contradiction_threshold": contradiction_threshold,
        "selection_policy": args.selection_policy,
        "selection_field": (
            "deberta_selected"
            if contradiction_threshold is not None or top_k_policy
            else None
        ),
        "selected_output": str(selected_output_path) if selected_output_path else None,
        "selected_count": (
            selected_count if contradiction_threshold is not None or top_k_policy else None
        ),
        "max_selected": args.max_selected if top_k_policy else None,
        "effective_contradiction_cutoff": effective_cutoff,
        "argmax_contradictions_below_threshold": (
            prediction_counts["contradiction"] - selected_argmax_contradictions
            if contradiction_threshold is not None
            else None
        ),
        "selected_non_argmax_contradictions": (
            selected_non_argmax_contradictions
            if contradiction_threshold is not None
            else None
        ),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": str(dtype),
        "label_names": list(LABEL_NAMES),
        "sequences_at_max_length": sequences_at_max_length,
        "model_files_sha256": {
            name: sha256_file(model_path / name)
            for name in (
                "model.safetensors",
                "config.json",
                "tokenizer_config.json",
                "spm.model",
            )
            if (model_path / name).is_file()
        },
        "output_sha256": {
            output_path.name: sha256_file(output_path),
            **(
                {selected_output_path.name: sha256_file(selected_output_path)}
                if selected_output_path is not None
                else {}
            ),
        },
    }
    write_json(manifest_path, manifest)

    print(f"Prediction counts: {dict(prediction_counts)}")
    if contradiction_threshold is not None or top_k_policy:
        print(f"Selected contradictions: {selected_count:,}")
    print(f"Wrote predictions: {output_path}")
    if selected_output_path is not None:
        print(f"Wrote selected predictions: {selected_output_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
