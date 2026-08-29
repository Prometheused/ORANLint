#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Select unique security-relevant O-RAN paragraphs with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping


PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_KEYWORDS = PIPELINE_DIR / "configs/security_keywords.txt"
DEFAULT_CORPUS = (
    PIPELINE_DIR
    / "data/processed/ORAN/corpus_ORAN.jsonl"
)
DEFAULT_OUTPUT = (
    PIPELINE_DIR
    / "data/generated/security_segments.jsonl"
)


def normalize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()


def count_keyword_matches(text: str, keywords: Iterable[str]) -> int:
    normalized = normalize(text)
    return sum(1 for keyword in keywords if keyword in normalized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"Missing string text at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No corpus records found: {path}")
    return rows


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter the O-RAN corpus using the security keyword whitelist.")
    parser.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keyword_path = args.keywords.expanduser().resolve()
    corpus_path = args.corpus.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else Path(str(output_path) + ".manifest.json")
    )
    for path in (keyword_path, corpus_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.resolve() in {keyword_path.resolve(), corpus_path.resolve()}:
        raise ValueError("Output must differ from inputs")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {', '.join(map(str, existing))}")

    keywords = [line.strip().lower() for line in keyword_path.open(encoding="utf-8") if line.strip()]
    if not keywords:
        raise ValueError("Keyword whitelist is empty")
    corpus = load_jsonl(corpus_path)
    match_counts = [count_keyword_matches(row["text"], keywords) for row in corpus]
    median_count = float(median(match_counts))
    filtered: List[Dict[str, Any]] = []
    seen = set()
    duplicate_texts = 0
    for source, count in zip(corpus, match_counts):
        if count <= median_count:
            continue
        normalized = normalize(source["text"])
        if normalized in seen:
            duplicate_texts += 1
            continue
        seen.add(normalized)
        row = dict(source)
        row["kw_count"] = count
        filtered.append(row)

    atomic_jsonl(output_path, filtered)
    manifest = {
        "schema_version": 1,
        "script": str(Path(__file__).resolve()),
        "keywords": str(keyword_path),
        "keywords_sha256": sha256_file(keyword_path),
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "keyword_count": len(keywords),
        "input_records": len(corpus),
        "median_keyword_match_count": median_count,
        "selection_rule": "keyword_match_count > global_median",
        "duplicate_normalized_texts_removed": duplicate_texts,
        "output_records": len(filtered),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, manifest)
    print(f"Loaded {len(keywords)} keywords and {len(corpus):,} corpus records")
    print(f"Global median keyword match count: {median_count}")
    print(f"Wrote {len(filtered):,} unique records: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
