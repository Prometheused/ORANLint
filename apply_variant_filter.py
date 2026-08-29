#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Apply generalized variant filtering to NLI contradiction candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from variant_rules import (
    CorpusContextIndex,
    FILTER_OUTPUT_SCHEMA_VERSION,
    FILTER_VERSION,
    FilterDecision,
    annotate_pair,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "runs/default/nli_predictions.jsonl"
DEFAULT_CORPUS = (
    SCRIPT_DIR
    / "data/processed/ORAN/corpus_ORAN.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter DeBERTa contradiction predictions that are intentional O-RAN variants."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output_prefix", type=Path)
    parser.add_argument("--prediction_field", default="deberta_preds")
    parser.add_argument("--prediction_value", default="contradiction")
    parser.add_argument(
        "--expected-selected",
        "--expected_selected",
        dest="expected_selected",
        type=int,
        help="Fail unless exactly this many input rows are selected.",
    )
    parser.add_argument("--select_all", action="store_true", help="Analyze every pair without requiring predictions")
    parser.add_argument("--context_lookback", type=int, default=12)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_json(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_threshold_selection(
    rows: Sequence[Mapping[str, object]],
    expected_selected: Optional[int] = None,
    selection_policy: str = "argmax_contradiction",
    max_selected: Optional[int] = None,
) -> Dict[str, object]:
    if selection_policy not in (
        "argmax_contradiction",
        "contradiction_probability",
        "top_k_argmax_contradiction",
    ):
        raise ValueError(f"Unsupported DeBERTa selection policy: {selection_policy}")
    if selection_policy == "top_k_argmax_contradiction":
        if not isinstance(max_selected, int) or max_selected <= 0:
            raise ValueError("top-k selection requires a positive max_selected")
        ranked = []
        selected_ranks: List[int] = []
        for index, row in enumerate(rows):
            flag = row.get("deberta_selected")
            if not isinstance(flag, bool):
                raise ValueError(f"deberta_selected must be boolean at input row {index}")
            try:
                probability = float(row["deberta_contradiction_probability"])
                probabilities = row["deberta_probabilities"]
                if not isinstance(probabilities, Mapping):
                    raise TypeError
                probability_sum = sum(
                    float(probabilities[label])
                    for label in ("entailment", "neutral", "contradiction")
                )
                contradiction_probability = float(probabilities["contradiction"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid DeBERTa probabilities at input row {index}"
                ) from exc
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid contradiction probability at input row {index}")
            if abs(probability_sum - 1.0) > 1e-5:
                raise ValueError(f"DeBERTa probabilities do not sum to one at input row {index}")
            if abs(contradiction_probability - probability) > 1e-7:
                raise ValueError(f"Contradiction probability fields disagree at input row {index}")
            if str(row.get("deberta_preds") or "").strip().lower() == "contradiction":
                def stable_identifier(value: object) -> tuple[int, object]:
                    try:
                        return 0, int(value)
                    except (TypeError, ValueError):
                        return 1, str(value)

                identifiers = sorted(
                    (stable_identifier(row.get("id1")), stable_identifier(row.get("id2")))
                )
                ranked.append((-probability, identifiers[0], identifiers[1], index, row))
            rank = row.get("deberta_selection_rank")
            if flag:
                if not isinstance(rank, int) or rank <= 0:
                    raise ValueError(
                        f"Selected top-k row lacks a positive selection rank at input row {index}"
                    )
                selected_ranks.append(rank)
            elif rank is not None:
                raise ValueError(
                    f"Unselected top-k row has a selection rank at input row {index}"
                )

        ranked.sort(key=lambda item: item[:4])
        expected_rows = ranked[:max_selected]
        expected_indexes = {item[3] for item in expected_rows}
        for index, row in enumerate(rows):
            if bool(row["deberta_selected"]) != (index in expected_indexes):
                raise ValueError(
                    f"deberta_selected disagrees with deterministic top-k ranking at input row {index}"
                )
        expected_ranks = list(range(1, len(expected_rows) + 1))
        if sorted(selected_ranks) != expected_ranks:
            raise ValueError("Top-k selection ranks are not contiguous and unique")
        for rank, item in enumerate(expected_rows, 1):
            if item[4].get("deberta_selection_rank") != rank:
                raise ValueError("Top-k selection rank disagrees with deterministic ordering")
        selected_count = len(expected_rows)
        if expected_selected is not None and selected_count != expected_selected:
            raise ValueError(
                f"Expected {expected_selected:,} selected rows; found {selected_count:,}"
            )
        return {
            "validated": True,
            "selection_policy": selection_policy,
            "contradiction_threshold": None,
            "selected_count": selected_count,
            "max_selected": max_selected,
            "effective_contradiction_cutoff": (
                -expected_rows[-1][0] if expected_rows else None
            ),
            "argmax_contradiction_count": len(ranked),
        }

    thresholds = set()
    selected_count = 0
    probabilities_at_selection: List[float] = []
    probabilities_below_selection: List[float] = []

    for index, row in enumerate(rows):
        flag = row.get("deberta_selected")
        if not isinstance(flag, bool):
            raise ValueError(f"deberta_selected must be boolean at input row {index}")
        try:
            probability = float(row["deberta_contradiction_probability"])
            threshold = float(row["deberta_contradiction_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid DeBERTa threshold fields at input row {index}"
            ) from exc
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid contradiction probability at input row {index}")
        minimum_threshold = (
            0.5 if selection_policy == "argmax_contradiction" else 0.0
        )
        if not minimum_threshold <= threshold <= 1.0:
            raise ValueError(f"Invalid contradiction threshold at input row {index}")
        probabilities = row.get("deberta_probabilities")
        if not isinstance(probabilities, Mapping):
            raise ValueError(f"Missing DeBERTa probabilities at input row {index}")
        try:
            probability_sum = sum(
                float(probabilities[label])
                for label in ("entailment", "neutral", "contradiction")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid DeBERTa probabilities at input row {index}") from exc
        if abs(probability_sum - 1.0) > 1e-5:
            raise ValueError(f"DeBERTa probabilities do not sum to one at input row {index}")
        if abs(float(probabilities["contradiction"]) - probability) > 1e-7:
            raise ValueError(f"Contradiction probability fields disagree at input row {index}")

        should_select = probability >= threshold
        if selection_policy == "argmax_contradiction":
            should_select = (
                str(row.get("deberta_preds") or "").strip().lower()
                == "contradiction"
                and should_select
            )
        if flag != should_select:
            raise ValueError(f"deberta_selected disagrees with threshold at input row {index}")
        thresholds.add(threshold)
        selected_count += int(flag)
        if flag:
            probabilities_at_selection.append(probability)
        else:
            probabilities_below_selection.append(probability)

    if len(thresholds) != 1:
        raise ValueError(f"Expected one contradiction threshold; found {sorted(thresholds)}")
    if expected_selected is not None and selected_count != expected_selected:
        raise ValueError(
            f"Expected {expected_selected:,} selected rows; found {selected_count:,}"
        )
    return {
        "validated": True,
        "selection_policy": selection_policy,
        "contradiction_threshold": next(iter(thresholds)),
        "selected_count": selected_count,
        "minimum_selected_probability": (
            min(probabilities_at_selection) if probabilities_at_selection else None
        ),
        "maximum_rejected_probability": (
            max(probabilities_below_selection) if probabilities_below_selection else None
        ),
    }


def validate_source_manifest(
    input_path: Path,
    input_hash: str,
    threshold_diagnostics: Mapping[str, object],
) -> Dict[str, object]:
    manifest_path = Path(str(input_path) + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Threshold-selected input requires its NLI inference manifest: {manifest_path}"
        )
    manifest = load_json(manifest_path)
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, Mapping) or output_hashes.get(input_path.name) != input_hash:
        raise ValueError("NLI inference manifest does not match the variant-filter input")
    if manifest.get("selected_count") != threshold_diagnostics.get("selected_count"):
        raise ValueError("NLI inference manifest selected count does not match the input")
    if manifest.get("contradiction_threshold") != threshold_diagnostics.get(
        "contradiction_threshold"
    ):
        raise ValueError("NLI inference manifest threshold does not match the input")
    manifest_policy = manifest.get("selection_policy", "argmax_contradiction")
    if manifest_policy != threshold_diagnostics.get("selection_policy"):
        raise ValueError("NLI inference manifest selection policy does not match the input")
    if manifest_policy == "top_k_argmax_contradiction":
        if manifest.get("max_selected") != threshold_diagnostics.get("max_selected"):
            raise ValueError("NLI inference manifest top-k cap does not match the input")
        if manifest.get("effective_contradiction_cutoff") != threshold_diagnostics.get(
            "effective_contradiction_cutoff"
        ):
            raise ValueError("NLI inference manifest top-k cutoff does not match the input")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "input_hash_matches": True,
        "selected_count_matches": True,
        "threshold_matches": True,
        "selection_policy_matches": True,
        "top_k_matches": manifest_policy == "top_k_argmax_contradiction",
    }


def corpus_id_index(records: Sequence[Mapping[str, object]]) -> Dict[object, Mapping[str, object]]:
    return {record.get("id"): record for record in records if record.get("id") is not None}


def enrich_metadata(row: Dict[str, object], index: Mapping[object, Mapping[str, object]]) -> None:
    for side in ("id1", "id2"):
        record = index.get(row.get(side), {})
        mappings = {
            f"{side}_pdf_file": "pdf_file",
            f"{side}_section_number": "section_number",
            f"{side}_section_title": "section_title",
            f"{side}_page_number": "page_number",
            f"{side}_section_level": "section_level",
            f"{side}_ancestors": "ancestors",
        }
        for destination, source in mappings.items():
            if not row.get(destination) and record.get(source) is not None:
                row[destination] = record[source]


def output_paths(input_path: Path, output_prefix: Optional[Path]) -> Dict[str, Path]:
    if output_prefix is None:
        stem = input_path.stem
        if stem.endswith("_deberta_preds"):
            stem = stem[: -len("_deberta_preds")] + "_deberta"
        prefix = input_path.parent / stem
    else:
        prefix = output_prefix.expanduser().resolve()
    return {
        "annotated": Path(str(prefix) + "_variant_annotated.jsonl"),
        "auto_neutral": Path(str(prefix) + "_variant_auto_neutral.jsonl"),
        "for_gpt": Path(str(prefix) + "_for_gpt.jsonl"),
        "summary": Path(str(prefix) + "_variant_summary.json"),
        "manifest": Path(str(prefix) + "_variant_manifest.json"),
    }




def summarize(
    selected: Sequence[Mapping[str, object]], total_input: int, output_counts: Mapping[str, int]
) -> Dict[str, object]:
    decisions = Counter(str(row.get("variant_filter_decision") or "") for row in selected)
    primary_rules = Counter(str(row.get("variant_filter_primary_rule") or "none") for row in selected)
    matched_rules: Counter = Counter()
    vetoes: Counter = Counter()
    scenario_evidence: Counter = Counter()
    for row in selected:
        matched_rules.update(row.get("variant_filter_matched_rules") or [])
        vetoes.update(row.get("variant_filter_hard_vetoes") or [])
        scenario_evidence.update(row.get("variant_filter_scenario_evidence") or [])
    automatic = decisions.get(FilterDecision.AUTO_NEUTRAL.value, 0)
    return {
        "filter_version": FILTER_VERSION,
        "total_input_rows": total_input,
        "selected_rows": len(selected),
        "decisions": dict(sorted(decisions.items())),
        "primary_rules": dict(sorted(primary_rules.items())),
        "matched_rules": dict(sorted(matched_rules.items())),
        "hard_vetoes": dict(sorted(vetoes.items())),
        "scenario_evidence": dict(sorted(scenario_evidence.items())),
        "potential_gpt_requests_avoided": automatic,
        "automatic_rows_removed_from_gpt_queue": automatic,
        "output_counts": dict(output_counts),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    corpus_path = args.corpus.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus JSONL does not exist: {corpus_path}")
    if args.context_lookback < 0:
        raise ValueError("Context lookback must be non-negative")
    if args.expected_selected is not None and args.expected_selected < 0:
        raise ValueError("--expected-selected must be non-negative")

    paths = output_paths(input_path, args.output_prefix)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")

    rows = load_jsonl(input_path)
    threshold_diagnostics: Dict[str, object] = {}
    source_manifest: Dict[str, object] = {}
    input_hash = sha256(input_path)
    if args.prediction_field == "deberta_selected":
        prediction_manifest_path = Path(str(input_path) + ".manifest.json")
        if not prediction_manifest_path.is_file():
            raise FileNotFoundError(
                "Threshold-selected input requires its NLI inference manifest: "
                f"{prediction_manifest_path}"
            )
        prediction_manifest = load_json(prediction_manifest_path)
        selection_policy = str(
            prediction_manifest.get("selection_policy", "argmax_contradiction")
        )
        threshold_diagnostics = validate_threshold_selection(
            rows,
            args.expected_selected,
            selection_policy,
            prediction_manifest.get("max_selected"),
        )
        source_manifest = validate_source_manifest(
            input_path, input_hash, threshold_diagnostics
        )
    corpus = load_jsonl(corpus_path)
    metadata_index = corpus_id_index(corpus)
    context_index = CorpusContextIndex(corpus, max_lookback=args.context_lookback)
    selected: List[Dict[str, object]] = []
    annotated_all: List[Dict[str, object]] = []
    prediction_field_seen = False
    target = args.prediction_value.strip().lower()

    for original in rows:
        row = dict(original)
        enrich_metadata(row, metadata_index)
        if args.prediction_field in row:
            prediction_field_seen = True
        applies = args.select_all or str(row.get(args.prediction_field) or "").strip().lower() == target
        if applies:
            annotated = annotate_pair(row, context_index)
            selected.append(annotated)
            annotated_all.append(annotated)
        else:
            row.update({
                "variant_filter_version": FILTER_VERSION,
                "variant_filter_output_schema_version": FILTER_OUTPUT_SCHEMA_VERSION,
                "variant_filter_decision": "not_applicable",
                "variant_filter_reason": f"Not selected: {args.prediction_field} != {args.prediction_value!r}",
            })
            annotated_all.append(row)

    if not args.select_all and not prediction_field_seen:
        raise ValueError(
            f"Prediction field {args.prediction_field!r} is absent. Run DeBERTa inference first or use --select_all."
        )
    if args.expected_selected is not None and len(selected) != args.expected_selected:
        raise ValueError(
            f"Expected {args.expected_selected:,} selected rows; found {len(selected):,}"
        )

    automatic = [row for row in selected if row["variant_filter_decision"] == FilterDecision.AUTO_NEUTRAL.value]
    retained = [row for row in selected if row["variant_filter_decision"] == FilterDecision.SEND_TO_GPT.value]
    gpt_queue = retained
    output_counts = {
        "annotated": write_jsonl(paths["annotated"], annotated_all),
        "auto_neutral": write_jsonl(paths["auto_neutral"], automatic),
        "for_gpt": write_jsonl(paths["for_gpt"], gpt_queue),
    }
    summary = summarize(selected, len(rows), output_counts)
    write_json(paths["summary"], summary)
    output_hashes = {
        name: sha256(path)
        for name, path in paths.items()
        if name != "manifest" and path.is_file()
    }
    write_json(paths["manifest"], {
        "filter_version": FILTER_VERSION,
        "filter_output_schema_version": FILTER_OUTPUT_SCHEMA_VERSION,
        "input": str(input_path),
        "input_sha256": input_hash,
        "corpus": str(corpus_path),
        "corpus_sha256": sha256(corpus_path),
        "prediction_field": args.prediction_field,
        "prediction_value": args.prediction_value,
        "select_all": args.select_all,
        "mode": "enforce",
        "context_lookback": args.context_lookback,
        "expected_selected": args.expected_selected,
        "threshold_selection_validation": threshold_diagnostics,
        "source_prediction_manifest": source_manifest,
        "outputs": {name: str(path) for name, path in paths.items()},
        "output_sha256": output_hashes,
        "counts": output_counts,
    })

    print(f"Input: {input_path}")
    print(f"Selected rows: {len(selected):,}/{len(rows):,}")
    reduction = (100.0 * len(automatic) / len(selected)) if selected else 0.0
    print(f"Generalized variants: {len(automatic):,} ({reduction:.1f}% of selected rows)")
    print(f"Retained for GPT: {len(retained):,}")
    print(f"GPT queue written: {len(gpt_queue):,}")
    print(f"Summary: {paths['summary']}")


if __name__ == "__main__":
    main()
