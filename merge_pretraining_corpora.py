#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Create the flat O-RAN corpus and merged 4G/5G/O-RAN pretraining text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
PROCESSED_DATA = PIPELINE_ROOT / "data/processed"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "data/generated/pretraining"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oran-jsonl",
        type=Path,
        default=PROCESSED_DATA / "ORAN/corpus_ORAN.jsonl",
    )
    parser.add_argument(
        "--corpus-4g",
        type=Path,
        default=PROCESSED_DATA / "4G/corpus_4G.jsonl",
    )
    parser.add_argument(
        "--corpus-5g",
        type=Path,
        default=PROCESSED_DATA / "5G/corpus_5G.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def write_flat_corpus(source: Path, destination: Path) -> None:
    """Normalize either a text corpus or segment JSONL into one passage per line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() != ".jsonl":
        destination.write_bytes(source.read_bytes())
        return
    with source.open(encoding="utf-8") as src, destination.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            text = str(record.get("text", "")).replace("\r", " ").replace("\n", " ").strip()
            if text:
                dst.write(text + "\n")


def main() -> None:
    args = parse_args()
    source_by_name = {
        "4G": args.corpus_4g.resolve(),
        "5G": args.corpus_5g.resolve(),
        "ORAN": args.oran_jsonl.resolve(),
    }
    for source in source_by_name.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_by_name = {
        name: output_dir / f"processed_{name}" / f"corpus_{name}.txt"
        for name in source_by_name
    }
    merged = output_dir / "merged_4G_5G_ORAN.txt"
    for output in (*flat_by_name.values(), merged):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")

    for name, source in source_by_name.items():
        write_flat_corpus(source, flat_by_name[name])

    with merged.open("wb") as dst:
        for source in flat_by_name.values():
            data = source.read_bytes()
            dst.write(data)
            if data and not data.endswith(b"\n"):
                dst.write(b"\n")

    for path in flat_by_name.values():
        print(f"Created {path}")
    print(f"Created {merged}")


if __name__ == "__main__":
    main()
