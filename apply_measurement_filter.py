#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Apply the measurement counter template filter before contextual verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "runs/default/nli_selected_for_gpt.jsonl"

FILTER_VERSION = "measurement-counter-template"
MEASUREMENT_PATTERN_TEXT = r"Measurement\s+(sub)?counter\s+is\s+incremented"
MEASUREMENT_PATTERN = re.compile(MEASUREMENT_PATTERN_TEXT, re.IGNORECASE)
REQUIRED_FIELDS = ("id1", "id2", "text1", "text2")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove pairs matching the measurement counter/subcounter template."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", "--output_prefix", dest="output_prefix", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--expected-input-rows", type=int)
    parser.add_argument("--expected-removed-rows", type=int)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    pair_indexes: Dict[Tuple[object, object], int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            missing = [field for field in REQUIRED_FIELDS if field not in value]
            if missing:
                raise ValueError(
                    f"Missing required field(s) {missing} at {path}:{line_number}"
                )
            for field in ("text1", "text2"):
                if not isinstance(value[field], str):
                    raise ValueError(
                        f"Field {field!r} must be a string at {path}:{line_number}"
                    )
            pair_key = (value["id1"], value["id2"])
            if pair_key in pair_indexes:
                raise ValueError(
                    f"Duplicate pair {pair_key} at {path}:{line_number}; "
                    f"first seen at record {pair_indexes[pair_key]}"
                )
            pair_indexes[pair_key] = len(records) + 1
            records.append(value)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return count


def write_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def output_paths(input_path: Path, output_prefix: Optional[Path]) -> Dict[str, Path]:
    if output_prefix is None:
        stem = input_path.stem
        if stem.endswith("_for_gpt"):
            stem = stem[: -len("_for_gpt")]
        prefix = input_path.parent / stem
    else:
        prefix = output_prefix.expanduser().resolve()
    return {
        "filtered_for_gpt": Path(str(prefix) + "_measurement_filtered_for_gpt.jsonl"),
        "excluded": Path(str(prefix) + "_measurement_excluded.jsonl"),
        "summary": Path(str(prefix) + "_measurement_summary.json"),
        "manifest": Path(str(prefix) + "_measurement_manifest.json"),
    }


def default_source_manifest(input_path: Path) -> Path:
    stem = input_path.stem
    if not stem.endswith("_for_gpt"):
        raise ValueError(
            "Cannot derive the variant manifest from this input name; "
            "supply --source-manifest"
        )
    prefix = stem[: -len("_for_gpt")]
    return input_path.with_name(prefix + "_variant_manifest.json")


def validate_source_manifest(
    input_path: Path,
    input_hash: str,
    input_count: int,
    manifest_path: Path,
) -> Dict[str, object]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source variant manifest does not exist: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("mode") != "enforce":
        raise ValueError("Source variant manifest must describe an enforce-mode output")

    outputs = manifest.get("outputs")
    output_hashes = manifest.get("output_sha256")
    counts = manifest.get("counts")
    if not isinstance(outputs, Mapping):
        raise ValueError("Source variant manifest is missing outputs")
    if not isinstance(output_hashes, Mapping):
        raise ValueError("Source variant manifest is missing output_sha256")
    if not isinstance(counts, Mapping):
        raise ValueError("Source variant manifest is missing counts")

    recorded_input = outputs.get("for_gpt")
    if not isinstance(recorded_input, str) or Path(recorded_input).resolve() != input_path:
        raise ValueError("Source variant manifest does not identify the measurement-filter input")
    if output_hashes.get("for_gpt") != input_hash:
        raise ValueError("Source variant manifest hash does not match the input")
    if counts.get("for_gpt") != input_count:
        raise ValueError("Source variant manifest count does not match the input")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "mode": "enforce",
        "input_path_matches": True,
        "input_hash_matches": True,
        "input_count_matches": True,
    }


def matched_sides(record: Mapping[str, object]) -> List[str]:
    return [
        field
        for field in ("text1", "text2")
        if MEASUREMENT_PATTERN.search(str(record[field]))
    ]


def partition_records(
    records: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    retained: List[Dict[str, object]] = []
    excluded: List[Dict[str, object]] = []
    for original in records:
        sides = matched_sides(original)
        if not sides:
            retained.append(dict(original))
            continue
        annotated = dict(original)
        annotated.update(
            {
                "measurement_filter_version": FILTER_VERSION,
                "measurement_filter_decision": "excluded",
                "measurement_filter_matched_sides": sides,
                "measurement_filter_reason": "measurement_counter_increment_pattern",
            }
        )
        excluded.append(annotated)
    return retained, excluded


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.expected_input_rows is not None and args.expected_input_rows < 0:
        raise ValueError("--expected-input-rows must be non-negative")
    if args.expected_removed_rows is not None and args.expected_removed_rows < 0:
        raise ValueError("--expected-removed-rows must be non-negative")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")
    paths = output_paths(input_path, args.output_prefix)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")

    records = load_jsonl(input_path)
    if args.expected_input_rows is not None and len(records) != args.expected_input_rows:
        raise ValueError(
            f"Expected {args.expected_input_rows:,} input rows; found {len(records):,}"
        )
    retained, excluded = partition_records(records)
    if args.expected_removed_rows is not None and len(excluded) != args.expected_removed_rows:
        raise ValueError(
            f"Expected {args.expected_removed_rows:,} removed rows; found {len(excluded):,}"
        )

    input_hash = sha256_file(input_path)
    source_manifest_path = (
        args.source_manifest.expanduser().resolve()
        if args.source_manifest is not None
        else default_source_manifest(input_path)
    )
    source_manifest = validate_source_manifest(
        input_path, input_hash, len(records), source_manifest_path
    )

    match_side_counts: Counter[str] = Counter()
    for record in excluded:
        match_side_counts.update(record["measurement_filter_matched_sides"])
    output_counts = {
        "filtered_for_gpt": write_jsonl(paths["filtered_for_gpt"], retained),
        "excluded": write_jsonl(paths["excluded"], excluded),
    }
    summary: Dict[str, object] = {
        "filter_version": FILTER_VERSION,
        "pattern": MEASUREMENT_PATTERN_TEXT,
        "case_insensitive": True,
        "total_input_rows": len(records),
        "excluded_rows": len(excluded),
        "retained_rows": len(retained),
        "exclusion_rate": len(excluded) / len(records),
        "matched_side_counts": dict(sorted(match_side_counts.items())),
        "output_counts": output_counts,
    }
    write_json(paths["summary"], summary)
    output_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "manifest" and path.is_file()
    }
    write_json(
        paths["manifest"],
        {
            "filter_version": FILTER_VERSION,
            "pattern": MEASUREMENT_PATTERN_TEXT,
            "regex_flags": ["IGNORECASE"],
            "input": str(input_path),
            "input_sha256": input_hash,
            "expected_input_rows": args.expected_input_rows,
            "expected_removed_rows": args.expected_removed_rows,
            "source_variant_manifest": source_manifest,
            "outputs": {name: str(path) for name, path in paths.items()},
            "output_sha256": output_hashes,
            "counts": output_counts,
        },
    )

    print(f"Input rows: {len(records):,}")
    print(f"Removed (measurement counter/subcounter): {len(excluded):,}")
    print(f"Retained for GPT: {len(retained):,}")
    print(f"Output: {paths['filtered_for_gpt']}")
    print(f"Summary: {paths['summary']}")


if __name__ == "__main__":
    main()
