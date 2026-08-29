#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Prepare, submit, monitor, and collect contextual-verification Batch jobs.

A run is prepared as one or more size-balanced JSONL files. Model, reasoning,
sampling, sharding, and pricing behavior are explicit in manifests.

The first-pass collector never submits retries.  It writes a retry-candidate
JSONL containing only unresolved requests for a later, explicit Batch run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


PIPELINE_DIR = Path(__file__).resolve().parent
import verify_candidates as verifier

DEFAULT_RUN_ROOT = PIPELINE_DIR / "runs/default/gpt-batch"
DEFAULT_INITIAL_OUTPUT = (
    PIPELINE_DIR / "runs/default/gpt_batch_initial.jsonl"
)
MANIFEST_NAME = "manifest.json"
INDEX_NAME = "request_index.jsonl"
SUMMARY_NAME = "batch_summary.json"
RETRY_NAME = "retry_candidates.jsonl"
NUM_SHARDS = 2
MAX_SHARD_BYTES = 190_000_000
MAX_SHARD_REQUESTS = 49_000
TIER3_BATCH_QUEUE_TOKENS = 40_000_000
QUEUE_SAFETY_TOKENS = 36_000_000
LONG_CONTEXT_THRESHOLD_TOKENS = 272_000
DEFAULT_CAPPED_INPUT_TOKENS = 270_000
MODEL_CONTEXT_WINDOW_TOKENS = 1_050_000
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

# The effective Batch rates are recorded in every manifest and can be
# overridden when preparing a run if pricing changes.
DEFAULT_BATCH_INPUT_PER_M = 1.0
DEFAULT_BATCH_CACHED_INPUT_PER_M = 0.10
DEFAULT_BATCH_OUTPUT_PER_M = 6.0

_WORKER_RESOLVER: Optional[Any] = None
_WORKER_SETTINGS: Optional[Any] = None
_WORKER_CUSTOM_ID_PREFIX = "verifier"
_WORKER_PROMPT_MODE = "semantic"
_WORKER_INPUT_TOKEN_CAP = DEFAULT_CAPPED_INPUT_TOKENS
PROMPT_MODES = {"semantic": verifier.PROMPT_VERSION}
CAPPED_PROMPT_MODES = {"semantic"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    return DEFAULT_RUN_ROOT / stamp


def resolve_pipeline_path(path: Path) -> Path:
    return path if path.is_absolute() else PIPELINE_DIR / path


def manifest_path(run_dir: Path) -> Path:
    return run_dir / MANIFEST_NAME


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _sdk_value(value: object, name: str, default: object = None) -> object:
    return verifier._get_attr(value, name, default)


def _sdk_dump(value: object) -> Dict[str, Any]:
    def jsonable(item: object) -> Any:
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        if isinstance(item, Mapping):
            return {str(key): jsonable(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(child) for child in item]
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            return jsonable(model_dump(mode="json"))
        if hasattr(item, "__dict__"):
            return {
                key: jsonable(child)
                for key, child in vars(item).items()
                if not key.startswith("_")
            }
        return str(item)

    if isinstance(value, dict):
        return jsonable(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return jsonable(model_dump(mode="json"))
    return jsonable(value)


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Batch manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid Batch manifest: {path}")
    return value


def save_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    verifier.write_json_atomic(manifest_path(run_dir), manifest)


def model_slug(model: str) -> str:
    value = "".join(character if character.isalnum() else "-" for character in model.lower())
    value = "-".join(part for part in value.split("-") if part)
    if value.startswith("gpt-5-6-"):
        value = value[len("gpt-5-6-") :]
    return value[:24] or "model"


def build_custom_id(index: int, prompt_sha256: str, prefix: str = "verifier") -> str:
    return f"{prefix}-{index:08d}-{prompt_sha256[:12]}"


def estimate_serialized_request_input_tokens(body: Mapping[str, Any]) -> int:
    """Conservatively estimate tokens from the complete serialized request body."""
    return math.ceil(_serialized_request_body_bytes(body) / 3)


def _serialized_request_body_bytes(body: Mapping[str, Any]) -> int:
    serialized = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return len(serialized)






def responses_request_body(prompt: str, settings: Any) -> Dict[str, Any]:
    return verifier.responses_request_kwargs(
        prompt,
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
        max_output_tokens=settings.max_output_tokens,
    )


def build_capped_target_window_prompt(
    record: Mapping[str, Any],
    resolver: Any,
    settings: Any,
    input_token_cap: int,
) -> Tuple[str, Dict[str, Any]]:
    """Render the largest deterministic semantic prompt within the token cap."""
    if input_token_cap <= 0 or input_token_cap >= LONG_CONTEXT_THRESHOLD_TOKENS:
        raise ValueError(
            "The input-token cap must be positive and below the long-context threshold"
        )

    def render(document_context: str) -> str:
        return verifier.build_semantic_verification_prompt(
            record, resolver, document_context
        )

    full_documents = verifier.full_documents_to_string(record, resolver)
    full_prompt = render(full_documents)
    uncapped_tokens = estimate_serialized_request_input_tokens(
        responses_request_body(full_prompt, settings)
    )
    if uncapped_tokens <= input_token_cap:
        return full_prompt, {
            "document_context_capped": False,
            "document_context_strategy": "full_document",
            "input_token_cap": input_token_cap,
            "uncapped_estimated_input_tokens": uncapped_tokens,
            "capped_estimated_input_tokens": uncapped_tokens,
            "document_context_budget_bytes": len(full_documents.encode("utf-8")),
            "document_context_details": {"strategy": "full_document"},
        }

    empty_body = responses_request_body(render(""), settings)
    upper = len(full_documents.encode("utf-8"))
    budget = min(
        upper,
        max(0, input_token_cap * 3 - _serialized_request_body_bytes(empty_body) - 12_000),
    )
    best: Optional[Tuple[str, Dict[str, Any], int, int]] = None
    attempted: Set[int] = set()
    for _ in range(10):
        if budget in attempted:
            break
        attempted.add(budget)
        document_context, details = verifier.target_window_documents_to_string(
            record, resolver, budget
        )
        prompt = render(document_context)
        tokens = estimate_serialized_request_input_tokens(
            responses_request_body(prompt, settings)
        )
        if tokens <= input_token_cap:
            best = (prompt, details, tokens, budget)
            remaining = input_token_cap - tokens
            if remaining <= 64 or budget >= upper:
                break
            budget = min(upper, budget + max(1, remaining * 3 // 2))
        else:
            budget = max(0, budget - max(1, (tokens - input_token_cap) * 3 + 1_024))

    if best is None:
        document_context, details = verifier.target_window_documents_to_string(
            record, resolver, 0
        )
        prompt = render(document_context)
        tokens = estimate_serialized_request_input_tokens(
            responses_request_body(prompt, settings)
        )
        if tokens <= input_token_cap:
            best = (prompt, details, tokens, 0)
    if best is None:
        raise ValueError("Mandatory target entries do not fit within the input-token cap")

    prompt, details, tokens, budget = best
    return prompt, {
        "document_context_capped": True,
        "document_context_strategy": details.get("strategy"),
        "input_token_cap": input_token_cap,
        "uncapped_estimated_input_tokens": uncapped_tokens,
        "capped_estimated_input_tokens": tokens,
        "document_context_budget_bytes": budget,
        "document_context_details": details,
    }


def build_batch_request(
    index: int,
    record: Mapping[str, Any],
    resolver: Any,
    settings: Any,
    custom_id_prefix: str = "verifier",
    prompt_mode: str = "semantic",
    input_token_cap: int = DEFAULT_CAPPED_INPUT_TOKENS,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if prompt_mode != "semantic":
        raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}")
    prompt, cap_metadata = build_capped_target_window_prompt(
        record, resolver, settings, input_token_cap
    )
    prompt_bytes = len(prompt.encode("utf-8"))
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    custom_id = build_custom_id(index, prompt_sha256, custom_id_prefix)
    body = responses_request_body(prompt, settings)
    estimated_input_tokens = estimate_serialized_request_input_tokens(body)
    if estimated_input_tokens > input_token_cap:
        raise ValueError(
            f"Request estimate {estimated_input_tokens:,} exceeds {input_token_cap:,}"
        )
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }
    metadata = {
        "custom_id": custom_id,
        "input_index": index,
        "pair_key": verifier.pair_key(record),
        "prompt_sha256": prompt_sha256,
        "prompt_bytes": prompt_bytes,
        "estimated_input_tokens": estimated_input_tokens,
        "token_estimator": "ceil(serialized_request_body_utf8_bytes/3)",
        "estimated_long_context": False,
        "long_context_threshold_tokens": LONG_CONTEXT_THRESHOLD_TOKENS,
        "same_source_document": (
            str((resolver.document_for(record.get("id1")) or {}).get("pdf_file") or "")
            == str((resolver.document_for(record.get("id2")) or {}).get("pdf_file") or "")
            != ""
        ),
        **cap_metadata,
    }
    return request, metadata


def _render_batch_request_worker(
    item: Tuple[int, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if _WORKER_RESOLVER is None or _WORKER_SETTINGS is None:  # pragma: no cover - worker guard
        raise RuntimeError("Batch render worker was not initialized")
    index, record = item
    return build_batch_request(
        index,
        record,
        _WORKER_RESOLVER,
        _WORKER_SETTINGS,
        _WORKER_CUSTOM_ID_PREFIX,
        _WORKER_PROMPT_MODE,
        _WORKER_INPUT_TOKEN_CAP,
    )


def render_batch_requests(
    selected: Sequence[Tuple[int, Mapping[str, Any]]],
    resolver: Any,
    settings: Any,
    workers: int,
    custom_id_prefix: str = "verifier",
    prompt_mode: str = "semantic",
    input_token_cap: int = DEFAULT_CAPPED_INPUT_TOKENS,
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Render with deterministic ordering, using fork workers when requested."""
    if workers <= 1:
        for index, record in selected:
            yield build_batch_request(
                index,
                record,
                resolver,
                settings,
                custom_id_prefix,
                prompt_mode,
                input_token_cap,
            )
        return
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("Parallel prompt rendering requires the multiprocessing 'fork' start method")
    global _WORKER_RESOLVER, _WORKER_SETTINGS, _WORKER_CUSTOM_ID_PREFIX, _WORKER_PROMPT_MODE, _WORKER_INPUT_TOKEN_CAP
    _WORKER_RESOLVER = resolver
    _WORKER_SETTINGS = settings
    _WORKER_CUSTOM_ID_PREFIX = custom_id_prefix
    _WORKER_PROMPT_MODE = prompt_mode
    _WORKER_INPUT_TOKEN_CAP = input_token_cap
    context = multiprocessing.get_context("fork")
    try:
        with context.Pool(processes=workers) as pool:
            chunksize = 1 if prompt_mode in CAPPED_PROMPT_MODES else 16
            for rendered in pool.imap(
                _render_batch_request_worker, selected, chunksize=chunksize
            ):
                yield rendered
    finally:
        _WORKER_RESOLVER = None
        _WORKER_SETTINGS = None
        _WORKER_CUSTOM_ID_PREFIX = "verifier"
        _WORKER_PROMPT_MODE = "semantic"
        _WORKER_INPUT_TOKEN_CAP = DEFAULT_CAPPED_INPUT_TOKENS


def select_batch_records(
    records: Sequence[Dict[str, Any]],
    offset: int,
    limit: Optional[int],
    sample_size: Optional[int],
    sample_seed: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    if sample_size is None:
        return verifier.select_records(records, offset, limit)
    if offset != 0 or limit is not None:
        raise ValueError("--sample-size cannot be combined with --offset or --limit")
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if sample_size > len(records):
        raise ValueError(
            f"Cannot sample {sample_size:,} records from a {len(records):,}-record input"
        )
    indexes = sorted(random.Random(sample_seed).sample(range(len(records)), sample_size))
    return [(index, records[index]) for index in indexes]


def select_document_stratified_records(
    records: Sequence[Dict[str, Any]],
    sample_size: int,
    sample_seed: int,
    resolver: Any,
    settings: Any,
    prompt_mode: str,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[str, Any]]:
    """Sample proportionally across same/cross-document and context-size strata."""
    if sample_size <= 0 or sample_size > len(records):
        raise ValueError("Stratified sample size must be within the input population")
    strata: Dict[Tuple[bool, bool], List[int]] = {}
    population_prompt_bytes = 0
    population_estimated_tokens = 0
    population_effective_input_tokens = 0
    population_long_context_requests = 0
    for index, record in enumerate(records):
        if prompt_mode != "semantic":
            raise ValueError(
                "Document-stratified sampling is defined only for the semantic prompt"
            )
        document1 = resolver.document_for(record.get("id1"))
        document2 = resolver.document_for(record.get("id2"))
        pdf1 = str(document1.get("pdf_file") or "")
        pdf2 = str(document2.get("pdf_file") or "")
        same_document = bool(pdf1 and pdf1 == pdf2)
        document_bytes = len(str(document1.get("text") or "").encode("utf-8"))
        if not same_document:
            document_bytes += len(str(document2.get("text") or "").encode("utf-8"))
        pair_bytes = len(str(record.get("text1") or "").encode("utf-8")) + len(
            str(record.get("text2") or "").encode("utf-8")
        )
        # Fixed prompt prose, metadata labels, and the maximum nearby-context
        # allowance make this intentionally conservative without performing
        # expensive paragraph matching for all population rows.
        prompt_bytes = document_bytes + pair_bytes + 12_000 + 2 * settings.max_paragraph_chars
        prompt_tokens = math.ceil(prompt_bytes / 3)
        metadata = {
            "prompt_bytes": prompt_bytes,
            "estimated_input_tokens": prompt_tokens,
            "same_source_document": same_document,
            "estimated_long_context": prompt_tokens > LONG_CONTEXT_THRESHOLD_TOKENS,
        }
        key = (
            bool(metadata["same_source_document"]),
            bool(metadata["estimated_long_context"]),
        )
        strata.setdefault(key, []).append(index)
        prompt_tokens = int(metadata["estimated_input_tokens"])
        population_prompt_bytes += int(metadata["prompt_bytes"])
        population_estimated_tokens += prompt_tokens
        if key[1]:
            population_long_context_requests += 1
            population_effective_input_tokens += 2 * prompt_tokens
        else:
            population_effective_input_tokens += prompt_tokens

    allocations: Dict[Tuple[bool, bool], int] = {}
    fractions: List[Tuple[float, Tuple[bool, bool]]] = []
    assigned = 0
    for key in sorted(strata):
        exact = sample_size * len(strata[key]) / len(records)
        allocation = math.floor(exact)
        allocations[key] = allocation
        assigned += allocation
        fractions.append((exact - allocation, key))
    for _, key in sorted(fractions, key=lambda item: (-item[0], item[1]))[: sample_size - assigned]:
        allocations[key] += 1

    rng = random.Random(sample_seed)
    indexes: List[int] = []
    for key in sorted(strata):
        indexes.extend(rng.sample(strata[key], allocations[key]))
    indexes.sort()
    if len(indexes) != sample_size or len(set(indexes)) != sample_size:
        raise RuntimeError("Stratified sampling did not produce the requested unique sample")
    summary = {
        "method": "proportional_same_document_x_long_context",
        "long_context_threshold_tokens": LONG_CONTEXT_THRESHOLD_TOKENS,
        "population_strata": {
            f"same_document={key[0]},long_context={key[1]}": len(strata[key])
            for key in sorted(strata)
        },
        "sample_strata": {
            f"same_document={key[0]},long_context={key[1]}": allocations[key]
            for key in sorted(strata)
        },
        "population_estimate": {
            "records": len(records),
            "prompt_utf8_bytes": population_prompt_bytes,
            "estimated_input_tokens": population_estimated_tokens,
            "long_context_requests": population_long_context_requests,
            "long_context_threshold_tokens": LONG_CONTEXT_THRESHOLD_TOKENS,
            "long_context_input_multiplier": 2.0,
            "long_context_output_multiplier": 1.5,
            "effective_input_tokens_for_pricing": population_effective_input_tokens,
            "token_estimator": "sum(ceil(prompt_utf8_bytes/3))",
            "estimator_scope": "full-document bytes plus conservative prompt/context allowance",
        },
    }
    return [(index, records[index]) for index in indexes], summary


def estimate_population_prompts(
    records: Sequence[Dict[str, Any]],
    resolver: Any,
    settings: Any,
    workers: int,
    custom_id_prefix: str,
    prompt_mode: str,
    input_token_cap: int = DEFAULT_CAPPED_INPUT_TOKENS,
) -> Dict[str, Any]:
    population = list(enumerate(records))
    estimated_tokens = 0
    effective_input_tokens = 0
    long_context_requests = 0
    prompt_bytes = 0
    capped_requests = 0
    uncapped_estimated_tokens = 0
    maximum_estimated_tokens = 0
    strategies: Counter[str] = Counter()
    for _, metadata in render_batch_requests(
        population,
        resolver,
        settings,
        workers,
        custom_id_prefix,
        prompt_mode,
        input_token_cap,
    ):
        prompt_bytes += int(metadata["prompt_bytes"])
        tokens = int(metadata["estimated_input_tokens"])
        estimated_tokens += tokens
        maximum_estimated_tokens = max(maximum_estimated_tokens, tokens)
        uncapped_estimated_tokens += int(
            metadata.get("uncapped_estimated_input_tokens", tokens)
        )
        if metadata.get("document_context_capped"):
            capped_requests += 1
        strategy = metadata.get("document_context_strategy")
        if strategy:
            strategies[str(strategy)] += 1
        if metadata.get("estimated_long_context"):
            long_context_requests += 1
            effective_input_tokens += 2 * tokens
        else:
            effective_input_tokens += tokens
    return {
        "records": len(records),
        "prompt_utf8_bytes": prompt_bytes,
        "estimated_input_tokens": estimated_tokens,
        "uncapped_estimated_input_tokens": uncapped_estimated_tokens,
        "maximum_estimated_input_tokens": maximum_estimated_tokens,
        "capped_requests": capped_requests,
        "document_context_strategies": dict(sorted(strategies.items())),
        "long_context_requests": long_context_requests,
        "long_context_threshold_tokens": LONG_CONTEXT_THRESHOLD_TOKENS,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
        "effective_input_tokens_for_pricing": effective_input_tokens,
        "token_estimator": (
            "sum(ceil(serialized_request_body_utf8_bytes/3))"
            if prompt_mode in CAPPED_PROMPT_MODES
            else "sum(ceil(prompt_utf8_bytes/3))"
        ),
    }


def _validate_prepare_paths(
    input_path: Path,
    corpus_path: Path,
    hierarchical_path: Path,
    output_path: Path,
    run_dir: Path,
) -> None:
    for path in (input_path, corpus_path, hierarchical_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Input and output must be different files")
    if output_path.resolve() == verifier.DEFAULT_OUTPUT.resolve():
        raise ValueError("The Batch runner refuses to overwrite the synchronous output")
    if output_path.exists():
        raise FileExistsError(f"Batch initial output already exists: {output_path}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")


def prepare_run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = resolve_pipeline_path(args.input)
    corpus_path = resolve_pipeline_path(args.corpus)
    hierarchical_path = resolve_pipeline_path(args.hierarchical)
    output_path = resolve_pipeline_path(args.output)
    run_dir = resolve_pipeline_path(args.run_dir or default_run_dir())
    _validate_prepare_paths(input_path, corpus_path, hierarchical_path, output_path, run_dir)

    if args.max_output_tokens <= 0 or args.max_paragraph_chars <= 0:
        raise ValueError("Token and context limits must be positive")
    if args.input_token_cap <= 0:
        raise ValueError("--input-token-cap must be positive")
    if (
        args.prompt_mode in CAPPED_PROMPT_MODES
        and args.input_token_cap >= LONG_CONTEXT_THRESHOLD_TOKENS
    ):
        raise ValueError(
            "The capped prompt mode requires --input-token-cap below 272,000"
        )
    if args.context_window < 0:
        raise ValueError("Context window must be non-negative")
    if args.workers <= 0:
        raise ValueError("Prompt-rendering workers must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")

    records = verifier.load_jsonl(input_path, required_fields=verifier.REQUIRED_FIELDS)
    verifier.validate_records(records, args.prediction_field, args.prediction_value)
    settings = verifier.InferenceSettings(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        max_paragraph_chars=args.max_paragraph_chars,
        context_window=args.context_window,
        max_retries=1,
        retry_base_delay=0.0,
    )
    resolver = verifier.ContextResolver.from_paths(
        corpus_path,
        hierarchical_path,
        max_par_chars=args.max_paragraph_chars,
        window=args.context_window,
    )
    sampling_details: Dict[str, Any] = {}
    if args.sample_method == "document-stratified":
        if args.sample_size is None or args.offset != 0 or args.limit is not None:
            raise ValueError(
                "--sample-method document-stratified requires --sample-size and cannot use offset/limit"
            )
        selected, sampling_details = select_document_stratified_records(
            records,
            args.sample_size,
            args.sample_seed,
            resolver,
            settings,
            args.prompt_mode,
        )
    else:
        selected = select_batch_records(
            records,
            args.offset,
            args.limit,
            args.sample_size,
            args.sample_seed,
        )
    verifier.validate_unique_pairs(selected)
    if len(selected) < args.num_shards:
        raise ValueError(
            f"At least {args.num_shards} records are required for "
            f"{args.num_shards} non-empty batches"
        )
    if args.expected_records is not None and len(selected) != args.expected_records:
        raise ValueError(
            f"Selected {len(selected):,} records; expected exactly {args.expected_records:,}"
        )

    custom_id_prefix = model_slug(settings.model)
    sampling_population_estimate = sampling_details.get("population_estimate")
    if args.prompt_mode in CAPPED_PROMPT_MODES:
        # The stratifier intentionally classifies rows by their uncapped
        # document size.  Its fast population estimate is therefore not a
        # valid substitute for rendering the capped prompts.
        sampling_population_estimate = None
    population_estimate = (
        sampling_population_estimate
        or estimate_population_prompts(
                records,
                resolver,
                settings,
                args.workers,
                custom_id_prefix,
                args.prompt_mode,
                args.input_token_cap,
            )
        if args.estimate_population
        else None
    )
    sampling_details.pop("population_estimate", None)

    run_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [
        run_dir / f"batch_input_{number:02d}.jsonl"
        for number in range(1, args.num_shards + 1)
    ]
    shard_temps = [run_dir / f".{path.name}.tmp" for path in shard_paths]
    index_path = run_dir / INDEX_NAME
    index_temp = run_dir / f".{INDEX_NAME}.tmp"
    shard_bytes = [0 for _ in range(args.num_shards)]
    shard_counts = [0 for _ in range(args.num_shards)]
    shard_estimated_tokens = [0 for _ in range(args.num_shards)]
    shard_effective_estimated_tokens = [0 for _ in range(args.num_shards)]
    custom_ids: Set[str] = set()
    selected_capped_requests = 0
    maximum_selected_estimated_tokens = 0
    selected_context_strategies: Counter[str] = Counter()

    try:
        handles = [path.open("w", encoding="utf-8") for path in shard_temps]
        index_handle = index_temp.open("w", encoding="utf-8")
        try:
            rendered_requests = render_batch_requests(
                selected,
                resolver,
                settings,
                args.workers,
                custom_id_prefix,
                args.prompt_mode,
                args.input_token_cap,
            )
            for ordinal, (request, metadata) in enumerate(rendered_requests, 1):
                if (
                    int(metadata["estimated_input_tokens"]) + settings.max_output_tokens
                    > MODEL_CONTEXT_WINDOW_TOKENS
                ):
                    raise ValueError(
                        f"Request {metadata['custom_id']} exceeds the conservative model-context preflight"
                    )
                request_line = _json_line(request)
                request_line_bytes = len(request_line.encode("utf-8"))
                shard = min(
                    range(args.num_shards),
                    key=lambda item: (shard_bytes[item], shard_counts[item], item),
                )
                metadata["shard"] = shard + 1
                if metadata["custom_id"] in custom_ids:
                    raise ValueError(f"Duplicate generated custom_id: {metadata['custom_id']}")
                custom_ids.add(str(metadata["custom_id"]))
                maximum_selected_estimated_tokens = max(
                    maximum_selected_estimated_tokens,
                    int(metadata["estimated_input_tokens"]),
                )
                if metadata.get("document_context_capped"):
                    selected_capped_requests += 1
                strategy = metadata.get("document_context_strategy")
                if strategy:
                    selected_context_strategies[str(strategy)] += 1

                handles[shard].write(request_line)
                index_handle.write(_json_line(metadata))
                shard_bytes[shard] += request_line_bytes
                shard_counts[shard] += 1
                shard_estimated_tokens[shard] += int(metadata["estimated_input_tokens"])
                shard_effective_estimated_tokens[shard] += int(
                    metadata["estimated_input_tokens"]
                ) * (2 if metadata.get("estimated_long_context") else 1)
                if args.progress_every and (ordinal % args.progress_every == 0 or ordinal == len(selected)):
                    print(f"Prepared {ordinal:,}/{len(selected):,} requests", flush=True)
        finally:
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            index_handle.flush()
            os.fsync(index_handle.fileno())
            index_handle.close()

        if sum(shard_counts) != len(selected) or len(custom_ids) != len(selected):
            raise RuntimeError("Prepared shard coverage does not match the selected records")
        for number, (count, size) in enumerate(zip(shard_counts, shard_bytes), 1):
            if count <= 0 or count > MAX_SHARD_REQUESTS:
                raise ValueError(f"Shard {number} has invalid request count: {count:,}")
            if size >= MAX_SHARD_BYTES:
                raise ValueError(f"Shard {number} is too large: {size:,} bytes")

        total_estimated_tokens = sum(shard_estimated_tokens)
        if total_estimated_tokens >= QUEUE_SAFETY_TOKENS:
            raise ValueError(
                f"Conservative input-token estimate {total_estimated_tokens:,} exceeds "
                f"the {QUEUE_SAFETY_TOKENS:,} safety threshold"
            )

        for temporary, final in zip(shard_temps, shard_paths):
            os.replace(temporary, final)
        os.replace(index_temp, index_path)
    finally:
        for temporary in [*shard_temps, index_temp]:
            if temporary.exists():
                temporary.unlink()

    input_sha = verifier.sha256_file(input_path)
    corpus_sha = verifier.sha256_file(corpus_path)
    hierarchical_sha = verifier.sha256_file(hierarchical_path)
    run_id = run_dir.name
    shards: List[Dict[str, Any]] = []
    for shard_index, path in enumerate(shard_paths):
        shards.append(
            {
                "number": shard_index + 1,
                "path": str(path),
                "sha256": verifier.sha256_file(path),
                "bytes": path.stat().st_size,
                "requests": shard_counts[shard_index],
                "estimated_input_tokens": shard_estimated_tokens[shard_index],
                "effective_estimated_input_tokens_for_pricing": (
                    shard_effective_estimated_tokens[shard_index]
                ),
                "input_file_id": "",
                "batch_id": "",
                "status": "prepared",
                "output_file_id": "",
                "error_file_id": "",
                "request_counts": {},
            }
        )

    if args.pricing_mode == "tokens-only":
        pricing: Dict[str, Any] = {
            "mode": "tokens_only",
            "batch_input_per_million": None,
            "batch_cached_input_per_million": None,
            "batch_output_per_million": None,
            "source": "No verified model-specific Batch rates supplied",
        }
        estimated_input_cost: Optional[float] = None
    else:
        pricing = {
            "mode": "priced",
            "batch_input_per_million": args.batch_input_price,
            "batch_cached_input_per_million": args.batch_cached_input_price,
            "batch_output_per_million": args.batch_output_price,
            "source": "CLI rates recorded at preparation time",
        }
        estimated_input_cost = round(
            sum(shard_effective_estimated_tokens)
            * args.batch_input_price
            / 1_000_000,
            6,
        )
    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "state": "prepared",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "script": str(Path(__file__).resolve()),
        "prompt_mode": args.prompt_mode,
        "prompt_version": PROMPT_MODES[args.prompt_mode],
        "output_schema_version": verifier.OUTPUT_SCHEMA_VERSION,
        "response_schema": "oran_verdict",
        "model": settings.model,
        "reasoning_effort": settings.reasoning_effort,
        "max_output_tokens": settings.max_output_tokens,
        "input_token_cap": (
            args.input_token_cap
            if args.prompt_mode in CAPPED_PROMPT_MODES
            else None
        ),
        "max_paragraph_chars": settings.max_paragraph_chars,
        "context_window": settings.context_window,
        "render_workers": args.workers,
        "input": str(input_path),
        "input_sha256": input_sha,
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_sha,
        "hierarchical": str(hierarchical_path),
        "hierarchical_sha256": hierarchical_sha,
        "output": str(output_path),
        "request_index": str(index_path),
        "request_index_sha256": verifier.sha256_file(index_path),
        "retry_candidates": str(run_dir / RETRY_NAME),
        "summary": str(run_dir / SUMMARY_NAME),
        "offset": args.offset,
        "limit": args.limit,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed if args.sample_size is not None else None,
        "sampling_method": (
            sampling_details.get("method")
            or ("random_without_replacement" if args.sample_size else "slice")
        ),
        "sampling_details": sampling_details,
        "sample_indexes_sha256": verifier.sha256_json([index for index, _ in selected]),
        "records": len(selected),
        "population_estimate": population_estimate,
        "prediction_field": args.prediction_field,
        "prediction_value": args.prediction_value,
        "shards": shards,
        "total_bytes": sum(shard_bytes),
        "estimated_input_tokens": sum(shard_estimated_tokens),
        "effective_estimated_input_tokens_for_pricing": sum(
            shard_effective_estimated_tokens
        ),
        "long_context_threshold_tokens": LONG_CONTEXT_THRESHOLD_TOKENS,
        "token_estimator": (
            "ceil(serialized_request_body_utf8_bytes/3)"
            if args.prompt_mode in CAPPED_PROMPT_MODES
            else "ceil(prompt_utf8_bytes/3)"
        ),
        "tier3_batch_queue_limit": TIER3_BATCH_QUEUE_TOKENS,
        "queue_safety_threshold": QUEUE_SAFETY_TOKENS,
        "estimated_batch_input_cost_usd": estimated_input_cost,
        "pricing": pricing,
        "validation": {
            "passed": True,
            "shards": args.num_shards,
            "unique_custom_ids": len(custom_ids),
            "covered_records": sum(shard_counts),
            "max_shard_bytes": MAX_SHARD_BYTES,
            "max_shard_requests": MAX_SHARD_REQUESTS,
            "capped_requests": selected_capped_requests,
            "maximum_estimated_input_tokens": maximum_selected_estimated_tokens,
            "document_context_strategies": dict(
                sorted(selected_context_strategies.items())
            ),
        },
    }
    signature_payload = {
        key: manifest[key]
        for key in (
            "prompt_version",
            "output_schema_version",
            "model",
            "reasoning_effort",
            "max_output_tokens",
            "input_token_cap",
            "input_sha256",
            "corpus_sha256",
            "hierarchical_sha256",
            "records",
            "sample_indexes_sha256",
        )
    }
    manifest["signature_sha256"] = verifier.sha256_json(signature_payload)
    save_manifest(run_dir, manifest)
    print(f"Prepared Batch run: {run_dir}")
    for shard in shards:
        print(
            f"Shard {shard['number']}: {shard['requests']:,} requests, "
            f"{shard['bytes'] / 1_000_000:.2f} MB, "
            f"~{shard['estimated_input_tokens']:,} input tokens"
        )
    print(f"Manifest: {manifest_path(run_dir)}")
    return manifest


def validate_prepared_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("Manifest does not contain a successful preparation validation")
    expected_shards = int(validation.get("shards") or NUM_SHARDS)
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != expected_shards:
        raise ValueError(f"Manifest must contain exactly {expected_shards} shards")
    total_requests = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("Invalid shard entry in manifest")
        path = Path(str(shard.get("path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"Prepared shard does not exist: {path}")
        if verifier.sha256_file(path) != shard.get("sha256"):
            raise ValueError(f"Prepared shard hash changed: {path}")
        if path.stat().st_size != shard.get("bytes"):
            raise ValueError(f"Prepared shard size changed: {path}")
        total_requests += int(shard.get("requests") or 0)
    if total_requests != manifest.get("records"):
        raise ValueError("Prepared shard request counts do not match the manifest total")
    if manifest.get("prompt_mode") in CAPPED_PROMPT_MODES:
        cap = int(manifest.get("input_token_cap") or 0)
        if cap <= 0 or cap >= LONG_CONTEXT_THRESHOLD_TOKENS:
            raise ValueError("Capped manifest has an invalid input-token cap")
        for row in _load_jsonl_values(Path(str(manifest["request_index"]))):
            if int(row.get("estimated_input_tokens") or 0) > cap:
                raise ValueError("Prepared capped request exceeds the manifest input-token cap")
    for key in ("input", "corpus", "hierarchical"):
        path = Path(str(manifest[key]))
        if verifier.sha256_file(path) != manifest[f"{key}_sha256"]:
            raise ValueError(f"Source changed after preparation: {path}")


def _find_existing_batch(client: Any, input_file_id: str, run_id: str, shard_number: int) -> Optional[object]:
    try:
        batches = client.batches.list(limit=100)
        for batch in batches:
            metadata = _sdk_value(batch, "metadata", {}) or {}
            if (
                str(_sdk_value(batch, "input_file_id", "")) == input_file_id
                and str(_sdk_value(metadata, "run_id", "")) == run_id
                and str(_sdk_value(metadata, "shard", "")) == str(shard_number)
            ):
                return batch
    except Exception:
        # Reconciliation is a duplicate-safety aid.  The SDK's normal create
        # error remains visible if listing is unavailable.
        return None
    return None


def submit_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = resolve_pipeline_path(run_dir)
    manifest = load_manifest(run_dir)
    validate_prepared_manifest(run_dir, manifest)
    if manifest.get("state") == "collected":
        raise ValueError("This run has already been collected")
    client = verifier.create_client()
    run_id = str(manifest["run_id"])

    for shard in manifest["shards"]:
        path = Path(shard["path"])
        if not shard.get("input_file_id"):
            with path.open("rb") as handle:
                uploaded = client.files.create(file=handle, purpose="batch")
            shard["input_file_id"] = str(_sdk_value(uploaded, "id", ""))
            shard["uploaded_at"] = utc_now()
            manifest["updated_at"] = utc_now()
            save_manifest(run_dir, manifest)
            print(f"Uploaded shard {shard['number']}: {shard['input_file_id']}", flush=True)

        if not shard.get("batch_id"):
            existing = _find_existing_batch(
                client,
                str(shard["input_file_id"]),
                run_id,
                int(shard["number"]),
            )
            batch = existing or client.batches.create(
                input_file_id=str(shard["input_file_id"]),
                endpoint="/v1/responses",
                completion_window="24h",
                metadata={
                    "pipeline": f"oran-gpt56-{model_slug(str(manifest['model']))}",
                    "run_id": run_id,
                    "shard": str(shard["number"]),
                },
            )
            shard["batch_id"] = str(_sdk_value(batch, "id", ""))
            shard["status"] = str(_sdk_value(batch, "status", "validating"))
            shard["submitted_at"] = utc_now()
            shard["batch_snapshot"] = _sdk_dump(batch)
            manifest["updated_at"] = utc_now()
            save_manifest(run_dir, manifest)
            print(f"Submitted shard {shard['number']}: {shard['batch_id']}", flush=True)

    manifest["state"] = "submitted"
    manifest["submitted_at"] = manifest.get("submitted_at") or utc_now()
    manifest["updated_at"] = utc_now()
    save_manifest(run_dir, manifest)
    return manifest


def _request_counts(batch: object) -> Dict[str, int]:
    counts = _sdk_value(batch, "request_counts")
    result: Dict[str, int] = {}
    for key in ("total", "completed", "failed"):
        value = _sdk_value(counts, key) if counts is not None else None
        if isinstance(value, int):
            result[key] = value
    return result


def refresh_status(run_dir: Path, client: Optional[Any] = None) -> Dict[str, Any]:
    run_dir = resolve_pipeline_path(run_dir)
    manifest = load_manifest(run_dir)
    client = client or verifier.create_client()
    statuses: List[str] = []
    for shard in manifest.get("shards", []):
        batch_id = str(shard.get("batch_id") or "")
        if not batch_id:
            raise ValueError(f"Shard {shard.get('number')} has not been submitted")
        batch = client.batches.retrieve(batch_id)
        status = str(_sdk_value(batch, "status", ""))
        statuses.append(status)
        shard["status"] = status
        shard["output_file_id"] = str(_sdk_value(batch, "output_file_id", "") or "")
        shard["error_file_id"] = str(_sdk_value(batch, "error_file_id", "") or "")
        shard["request_counts"] = _request_counts(batch)
        shard["batch_snapshot"] = _sdk_dump(batch)
        shard["last_checked_at"] = utc_now()
    manifest["state"] = "terminal" if statuses and all(item in TERMINAL_STATUSES for item in statuses) else "submitted"
    manifest["updated_at"] = utc_now()
    save_manifest(run_dir, manifest)
    return manifest


def print_status(manifest: Mapping[str, Any]) -> None:
    for shard in manifest.get("shards", []):
        counts = shard.get("request_counts") or {}
        print(
            f"Shard {shard['number']} {shard.get('batch_id')}: {shard.get('status')} "
            f"(total={counts.get('total', 0):,}, completed={counts.get('completed', 0):,}, "
            f"failed={counts.get('failed', 0):,})"
        )


def wait_for_terminal(
    run_dir: Path,
    poll_seconds: float = 60.0,
    max_wait_seconds: float = 0.0,
) -> Dict[str, Any]:
    if poll_seconds <= 0 or max_wait_seconds < 0:
        raise ValueError("Polling must be positive and max wait must be non-negative")
    run_dir = resolve_pipeline_path(run_dir)
    client = verifier.create_client()
    started = time.monotonic()
    previous: Optional[Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...]] = None
    while True:
        manifest = refresh_status(run_dir, client=client)
        current = tuple(
            (
                str(shard.get("status")),
                tuple(sorted((str(key), int(value)) for key, value in (shard.get("request_counts") or {}).items())),
            )
            for shard in manifest["shards"]
        )
        if current != previous:
            print_status(manifest)
            previous = current
        if manifest.get("state") == "terminal":
            return manifest
        if max_wait_seconds and time.monotonic() - started >= max_wait_seconds:
            raise TimeoutError("Batches are still running after the configured wait limit")
        time.sleep(poll_seconds)


def _download_file(client: Any, file_id: str, output_path: Path) -> None:
    response = client.files.content(file_id)
    content = getattr(response, "content", None)
    if not isinstance(content, (bytes, bytearray)):
        read = getattr(response, "read", None)
        content = read() if callable(read) else bytes(response)
    _atomic_write_bytes(output_path, bytes(content))


def _load_jsonl_values(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def extract_response_text(body: Mapping[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    texts: List[str] = []
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if part.get("type") == "output_text" and isinstance(text, str):
                    texts.append(text)
    return "".join(texts)


def _error_label(value: Mapping[str, Any], fallback: str = "batch_request_failed") -> str:
    error = value.get("error")
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        if code:
            return str(code)
    response = value.get("response")
    if isinstance(response, dict):
        status_code = response.get("status_code")
        if status_code:
            return f"api_status_{status_code}"
    return fallback


def _read_request_index(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    ordered = list(_load_jsonl_values(path))
    by_custom: Dict[str, Dict[str, Any]] = {}
    for value in ordered:
        custom_id = str(value.get("custom_id") or "")
        if not custom_id or custom_id in by_custom:
            raise ValueError(f"Invalid or duplicate custom_id in request index: {custom_id!r}")
        by_custom[custom_id] = value
    ordered.sort(key=lambda item: int(item["input_index"]))
    return ordered, by_custom


def _load_batch_results(
    manifest: Mapping[str, Any],
    expected: Mapping[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    successes: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, Dict[str, Any]] = {}
    integrity: List[str] = []
    for shard in manifest["shards"]:
        batch_id = str(shard.get("batch_id") or "")
        output_path = Path(str(shard.get("local_output_path") or ""))
        error_path = Path(str(shard.get("local_error_path") or ""))
        for value in _load_jsonl_values(output_path):
            custom_id = str(value.get("custom_id") or "")
            if custom_id not in expected:
                integrity.append(f"unknown output custom_id {custom_id!r}")
                continue
            if custom_id in successes or custom_id in failures:
                integrity.append(f"duplicate result custom_id {custom_id!r}")
                continue
            response = value.get("response")
            status_code = response.get("status_code") if isinstance(response, dict) else None
            body = response.get("body") if isinstance(response, dict) else None
            if status_code == 200 and isinstance(body, dict):
                successes[custom_id] = {
                    "body": body,
                    "batch_id": batch_id,
                    "shard": int(shard["number"]),
                }
            else:
                failures[custom_id] = {
                    "error": _error_label(value),
                    "batch_id": batch_id,
                    "shard": int(shard["number"]),
                    "body": body if isinstance(body, dict) else None,
                }
        for value in _load_jsonl_values(error_path):
            custom_id = str(value.get("custom_id") or "")
            if custom_id not in expected:
                integrity.append(f"unknown error custom_id {custom_id!r}")
                continue
            if custom_id in successes or custom_id in failures:
                integrity.append(f"duplicate result custom_id {custom_id!r}")
                continue
            failures[custom_id] = {
                "error": _error_label(value),
                "batch_id": batch_id,
                "shard": int(shard["number"]),
            }
    return successes, failures, integrity


def _batch_fields(
    result: Dict[str, Any],
    custom_id: str,
    batch_id: str,
    shard: int,
    model: str = verifier.MODEL,
) -> Dict[str, Any]:
    result.update(
        {
            "gpt56_execution_mode": "batch",
            "gpt56_batch_custom_id": custom_id,
            "gpt56_batch_id": batch_id,
            "gpt56_batch_shard": shard,
        }
    )
    return result


def _cost_summary(usage: Mapping[str, int], pricing: Mapping[str, float]) -> Dict[str, float]:
    input_tokens = int(usage.get("input_tokens", 0))
    cached_tokens = min(input_tokens, int(usage.get("cached_input_tokens", 0)))
    uncached_tokens = input_tokens - cached_tokens
    output_tokens = int(usage.get("output_tokens", 0))
    input_cost = uncached_tokens * float(pricing["batch_input_per_million"]) / 1_000_000
    cached_cost = cached_tokens * float(pricing["batch_cached_input_per_million"]) / 1_000_000
    output_cost = output_tokens * float(pricing["batch_output_per_million"]) / 1_000_000
    return {
        "uncached_input_usd": round(input_cost, 6),
        "cached_input_usd": round(cached_cost, 6),
        "output_usd": round(output_cost, 6),
        "total_usd": round(input_cost + cached_cost + output_cost, 6),
    }


def _cost_summary_records(
    records: Sequence[Mapping[str, Any]], pricing: Mapping[str, float]
) -> Dict[str, Any]:
    """Calculate Batch cost per request, including long-context multipliers."""
    uncached_cost = 0.0
    cached_cost = 0.0
    output_cost = 0.0
    long_context_requests = 0
    records_with_usage = 0
    for record in records:
        usage = _record_usage(record)
        if not usage:
            continue
        records_with_usage += 1
        input_tokens = int(usage.get("input_tokens", 0))
        cached_tokens = min(input_tokens, int(usage.get("cached_input_tokens", 0)))
        output_tokens = int(usage.get("output_tokens", 0))
        is_long = input_tokens > LONG_CONTEXT_THRESHOLD_TOKENS
        if is_long:
            long_context_requests += 1
        input_multiplier = 2.0 if is_long else 1.0
        output_multiplier = 1.5 if is_long else 1.0
        uncached_cost += (
            (input_tokens - cached_tokens)
            * input_multiplier
            * float(pricing["batch_input_per_million"])
            / 1_000_000
        )
        cached_cost += (
            cached_tokens
            * input_multiplier
            * float(pricing["batch_cached_input_per_million"])
            / 1_000_000
        )
        output_cost += (
            output_tokens
            * output_multiplier
            * float(pricing["batch_output_per_million"])
            / 1_000_000
        )
    return {
        "records_with_usage": records_with_usage,
        "long_context_requests": long_context_requests,
        "long_context_threshold_tokens": LONG_CONTEXT_THRESHOLD_TOKENS,
        "uncached_input_usd": round(uncached_cost, 6),
        "cached_input_usd": round(cached_cost, 6),
        "output_usd": round(output_cost, 6),
        "total_usd": round(uncached_cost + cached_cost + output_cost, 6),
    }


def _projected_cost_summary(
    projection: Mapping[str, Any],
    population_estimate: Mapping[str, Any],
    pricing: Mapping[str, float],
) -> Optional[Dict[str, Any]]:
    usage = projection.get("projected_usage")
    if not isinstance(usage, Mapping):
        return None
    raw_estimated_input = int(population_estimate.get("estimated_input_tokens") or 0)
    effective_estimated_input = int(
        population_estimate.get("effective_input_tokens_for_pricing") or 0
    )
    population_records = int(population_estimate.get("records") or 0)
    long_requests = int(population_estimate.get("long_context_requests") or 0)
    if raw_estimated_input <= 0 or population_records <= 0:
        return None
    input_multiplier = effective_estimated_input / raw_estimated_input
    output_multiplier = 1.0 + 0.5 * long_requests / population_records
    input_tokens = int(usage.get("input_tokens") or 0)
    cached_tokens = min(input_tokens, int(usage.get("cached_input_tokens") or 0))
    output_tokens = int(usage.get("output_tokens") or 0)
    uncached_cost = (
        (input_tokens - cached_tokens)
        * input_multiplier
        * float(pricing["batch_input_per_million"])
        / 1_000_000
    )
    cached_cost = (
        cached_tokens
        * input_multiplier
        * float(pricing["batch_cached_input_per_million"])
        / 1_000_000
    )
    output_cost = (
        output_tokens
        * output_multiplier
        * float(pricing["batch_output_per_million"])
        / 1_000_000
    )
    return {
        "population_records": population_records,
        "long_context_requests": long_requests,
        "input_pricing_multiplier": round(input_multiplier, 6),
        "output_pricing_multiplier": round(output_multiplier, 6),
        "uncached_input_usd": round(uncached_cost, 6),
        "cached_input_usd": round(cached_cost, 6),
        "output_usd": round(output_cost, 6),
        "total_usd": round(uncached_cost + cached_cost + output_cost, 6),
    }


def _record_usage(record: Mapping[str, Any]) -> Dict[str, int]:
    value = record.get("gpt56_usage")
    if not isinstance(value, Mapping):
        return {}
    return {
        key: int(value[key])
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        )
        if isinstance(value.get(key), int)
    }


def _percentile(values: Sequence[int], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def usage_distribution(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    usages = [_record_usage(record) for record in records]
    result: Dict[str, Any] = {"records_with_usage": sum(bool(item) for item in usages)}
    for key in (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        values = [item[key] for item in usages if key in item]
        if not values:
            continue
        result[key] = {
            "records": len(values),
            "mean": round(sum(values) / len(values), 3),
            "median": round(float(_percentile(values, 0.5)), 3),
            "p90": round(float(_percentile(values, 0.9)), 3),
            "p95": round(float(_percentile(values, 0.95)), 3),
            "max": max(values),
        }
    return result


def project_population_usage(
    records: Sequence[Mapping[str, Any]],
    request_index: Sequence[Mapping[str, Any]],
    population_estimate: Optional[Mapping[str, Any]],
    seed: int,
    bootstrap_iterations: int = 10_000,
) -> Optional[Dict[str, Any]]:
    if not population_estimate:
        return None
    population_records = int(population_estimate.get("records") or 0)
    population_input_estimate = int(
        population_estimate.get("estimated_input_tokens") or 0
    )
    if population_records <= 0 or population_input_estimate <= 0:
        return None
    if len(records) != len(request_index):
        raise ValueError("Collected records and request index have different lengths")

    points: List[Tuple[int, int, int, int, int]] = []
    for record, metadata in zip(records, request_index):
        usage = _record_usage(record)
        if "input_tokens" not in usage or "output_tokens" not in usage:
            continue
        points.append(
            (
                int(metadata["estimated_input_tokens"]),
                usage["input_tokens"],
                usage.get("cached_input_tokens", 0),
                usage["output_tokens"],
                usage.get("reasoning_tokens", 0),
            )
        )
    if not points:
        return {
            "status": "unavailable_no_usage",
            "population_records": population_records,
            "sample_records_with_usage": 0,
        }

    def estimate(sample: Sequence[Tuple[int, int, int, int, int]]) -> Dict[str, int]:
        estimated_input = sum(point[0] for point in sample)
        if estimated_input <= 0:
            raise ValueError("Sample prompt-token estimate must be positive")
        input_tokens = round(
            population_input_estimate
            * sum(point[1] for point in sample)
            / estimated_input
        )
        cached_tokens = round(
            population_input_estimate
            * sum(point[2] for point in sample)
            / estimated_input
        )
        output_tokens = round(
            population_records * sum(point[3] for point in sample) / len(sample)
        )
        reasoning_tokens = round(
            population_records * sum(point[4] for point in sample) / len(sample)
        )
        return {
            "input_tokens": input_tokens,
            "cached_input_tokens": min(input_tokens, cached_tokens),
            "output_tokens": output_tokens,
            "reasoning_tokens": min(output_tokens, reasoning_tokens),
            "total_tokens": input_tokens + output_tokens,
        }

    central = estimate(points)
    bootstrap: Dict[str, List[int]] = {key: [] for key in central}
    rng = random.Random(seed)
    for _ in range(bootstrap_iterations):
        resample = [points[rng.randrange(len(points))] for _ in points]
        iteration = estimate(resample)
        for key, value in iteration.items():
            bootstrap[key].append(value)
    confidence = {
        key: {
            "low": round(float(_percentile(values, 0.025))),
            "high": round(float(_percentile(values, 0.975))),
        }
        for key, values in bootstrap.items()
    }
    return {
        "status": "projected_from_seeded_sample",
        "population_records": population_records,
        "population_prompt_estimated_input_tokens": population_input_estimate,
        "sample_records": len(records),
        "sample_records_with_usage": len(points),
        "sample_usage_coverage": len(points) / len(records),
        "method": "input ratio estimator plus output sample mean",
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": seed,
        "projected_usage": central,
        "confidence_95": confidence,
        "reasoning_tokens_are_subset_of_output_tokens": True,
    }


def _retry_requests(shard_paths: Sequence[Path], unresolved: Set[str]) -> Iterable[Dict[str, Any]]:
    found: Set[str] = set()
    for path in shard_paths:
        for request in _load_jsonl_values(path):
            custom_id = str(request.get("custom_id") or "")
            if custom_id in unresolved:
                found.add(custom_id)
                yield request
    missing = unresolved - found
    if missing:
        raise RuntimeError(f"Could not recover {len(missing)} retry requests from prepared shards")


def collect_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = resolve_pipeline_path(run_dir)
    manifest = load_manifest(run_dir)
    output_path = Path(str(manifest["output"]))
    summary_path = Path(str(manifest["summary"]))
    retry_path = Path(str(manifest["retry_candidates"]))
    if manifest.get("state") == "collected":
        for path in (output_path, summary_path, retry_path):
            if not path.exists():
                raise FileNotFoundError(f"Collected manifest references a missing file: {path}")
        print(f"Run already collected: {run_dir}")
        return manifest
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Batch output: {output_path}")
    if output_path.resolve() == verifier.DEFAULT_OUTPUT.resolve():
        raise ValueError("The Batch runner refuses to overwrite the synchronous output")

    client = verifier.create_client()
    manifest = refresh_status(run_dir, client=client)
    if manifest.get("state") != "terminal":
        raise RuntimeError("All prepared batches must be terminal before collection")

    for shard in manifest["shards"]:
        number = int(shard["number"])
        output_file_id = str(shard.get("output_file_id") or "")
        error_file_id = str(shard.get("error_file_id") or "")
        local_output = run_dir / f"batch_output_{number:02d}.jsonl"
        local_error = run_dir / f"batch_error_{number:02d}.jsonl"
        if output_file_id and not local_output.exists():
            _download_file(client, output_file_id, local_output)
        if error_file_id and not local_error.exists():
            _download_file(client, error_file_id, local_error)
        shard["local_output_path"] = str(local_output) if local_output.exists() else ""
        shard["local_error_path"] = str(local_error) if local_error.exists() else ""
    manifest["updated_at"] = utc_now()
    save_manifest(run_dir, manifest)

    index_path = Path(str(manifest["request_index"]))
    ordered_index, expected = _read_request_index(index_path)
    successes, failures, integrity = _load_batch_results(manifest, expected)
    settings = verifier.InferenceSettings(
        model=str(manifest["model"]),
        reasoning_effort=str(manifest["reasoning_effort"]),
        max_output_tokens=int(manifest["max_output_tokens"]),
        max_paragraph_chars=int(manifest["max_paragraph_chars"]),
        context_window=int(manifest["context_window"]),
        max_retries=1,
        retry_base_delay=0.0,
    )
    all_records = verifier.load_jsonl(Path(str(manifest["input"])), required_fields=verifier.REQUIRED_FIELDS)
    final_records: List[Dict[str, Any]] = []
    unresolved: Set[str] = set()
    parse_failures = 0
    manifest_prompt_mode = str(manifest.get("prompt_mode") or "")
    if manifest_prompt_mode != "semantic":
        raise ValueError(f"Unsupported prompt mode in manifest: {manifest_prompt_mode!r}")
    for metadata in ordered_index:
        custom_id = str(metadata["custom_id"])
        source_index = int(metadata["input_index"])
        record = all_records[source_index]
        prompt_hash = str(metadata["prompt_sha256"])
        success = successes.get(custom_id)
        if success is not None:
            body = success["body"]
            try:
                response_text = extract_response_text(body)
                verdict = verifier.parse_verdict_response(response_text)
                result = verifier._decorate_result(
                    record, "", settings, "completed", 1, verdict=verdict, response=body
                )
                result["gpt56_prompt_sha256"] = prompt_hash
                final_records.append(
                    _batch_fields(
                        result,
                        custom_id,
                        success["batch_id"],
                        success["shard"],
                        settings.model,
                    )
                )
                continue
            except verifier.ResponseValidationError:
                parse_failures += 1
                failures[custom_id] = {
                    "error": "response_validation",
                    "batch_id": success["batch_id"],
                    "shard": success["shard"],
                    "body": body,
                }

        failure = failures.get(custom_id)
        if failure is None:
            shard_number = int(metadata["shard"])
            shard = manifest["shards"][shard_number - 1]
            failure = {
                "error": f"missing_batch_result:{shard.get('status', 'unknown')}",
                "batch_id": str(shard.get("batch_id") or ""),
                "shard": shard_number,
            }
        error = str(failure["error"])
        status = "expired" if error == "batch_expired" else "failed"
        failure_body = failure.get("body")
        result = verifier._decorate_result(
            record,
            "",
            settings,
            status,
            1,
            response=failure_body if isinstance(failure_body, Mapping) else None,
            error=error,
        )
        result["gpt56_prompt_sha256"] = prompt_hash
        final_records.append(
            _batch_fields(
                result,
                custom_id,
                str(failure["batch_id"]),
                int(failure["shard"]),
                settings.model,
            )
        )
        unresolved.add(custom_id)

    if len(final_records) != manifest["records"]:
        raise RuntimeError("Collected output does not cover every prepared record")
    verifier.write_jsonl_atomic(output_path, final_records)
    shard_paths = [Path(str(shard["path"])) for shard in manifest["shards"]]
    verifier.write_jsonl_atomic(retry_path, _retry_requests(shard_paths, unresolved))

    result_summary = verifier.summarize_results(final_records)
    usage = {key: int(value) for key, value in result_summary["usage_totals"].items()}
    pricing = manifest["pricing"]
    pricing_mode = str(pricing.get("mode") or "priced")
    batch_cost = (
        _cost_summary_records(final_records, pricing)
        if pricing_mode == "priced"
        else None
    )
    distribution = usage_distribution(final_records)
    projection = project_population_usage(
        final_records,
        ordered_index,
        manifest.get("population_estimate"),
        int(manifest.get("sample_seed") or 42),
    )
    projected_cost = (
        _projected_cost_summary(
            projection,
            manifest.get("population_estimate") or {},
            pricing,
        )
        if pricing_mode == "priced" and isinstance(projection, Mapping)
        else None
    )
    summary: Dict[str, Any] = {
        "run_id": manifest["run_id"],
        "created_at": utc_now(),
        "records": manifest["records"],
        "successful_records": len(final_records) - len(unresolved),
        "unresolved_records": len(unresolved),
        "parse_failures": parse_failures,
        "integrity_errors": integrity,
        "retry_candidates": len(unresolved),
        **result_summary,
        "usage_distribution": distribution,
        "projected_full_queue_usage": projection,
        "projected_full_queue_cost_usd": projected_cost,
        "batch_cost_usd": batch_cost,
        "pricing_status": (
            "priced"
            if pricing_mode == "priced"
            else f"unpriced_{model_slug(settings.model)}_tokens_only"
        ),
        "cost_formula": (
            "((input_tokens-cached_input_tokens)*input_rate + "
            "cached_input_tokens*cached_input_rate + output_tokens*output_rate) / 1e6"
        ),
        "reasoning_tokens_are_subset_of_output_tokens": True,
        "pricing": pricing,
    }
    verifier.write_json_atomic(summary_path, summary)
    manifest["state"] = "collected"
    manifest["collected_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    manifest["output_sha256"] = verifier.sha256_file(output_path)
    manifest["retry_candidates_sha256"] = verifier.sha256_file(retry_path)
    manifest["summary_sha256"] = verifier.sha256_file(summary_path)
    manifest["successful_records"] = summary["successful_records"]
    manifest["unresolved_records"] = summary["unresolved_records"]
    manifest["batch_cost_usd"] = summary["batch_cost_usd"]
    save_manifest(run_dir, manifest)
    print(f"Initial Batch results: {output_path}")
    print(f"Retry candidates (not submitted): {retry_path}")
    print(f"Successful: {summary['successful_records']:,}; unresolved: {len(unresolved):,}")
    if batch_cost is not None:
        print(f"Actual Batch cost: ${batch_cost['total_usd']:.6f}")
    else:
        print(
            "Actual token usage: "
            f"{usage.get('input_tokens', 0):,} input + "
            f"{usage.get('output_tokens', 0):,} output (USD unpriced)"
        )
    if integrity:
        print(f"Integrity warnings: {len(integrity):,}")
    return manifest


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=verifier.DEFAULT_INPUT)
    parser.add_argument("--corpus", type=Path, default=verifier.DEFAULT_CORPUS)
    parser.add_argument("--hierarchical", type=Path, default=verifier.DEFAULT_HIERARCHICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_INITIAL_OUTPUT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", default=verifier.MODEL)
    parser.set_defaults(prompt_mode="semantic")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default=verifier.REASONING_EFFORT,
    )
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--input-token-cap",
        type=int,
        default=DEFAULT_CAPPED_INPUT_TOKENS,
        help=(
            "Offline serialized-request estimate cap for the target-window prompt; "
            "must remain below 272000."
        ),
    )
    parser.add_argument("--max-paragraph-chars", type=int, default=2000)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-method",
        choices=("random", "document-stratified"),
        default="random",
        help="Use deterministic proportional same/cross-document and long-context strata when requested.",
    )
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    parser.add_argument(
        "--estimate-population",
        action="store_true",
        help="Render every input prompt locally for a full-queue token projection.",
    )
    parser.add_argument("--prediction-field", default="deberta_preds")
    parser.add_argument("--prediction-value", default="contradiction")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--pricing-mode",
        choices=("priced", "tokens-only"),
        default="priced",
    )
    parser.add_argument("--batch-input-price", type=float, default=DEFAULT_BATCH_INPUT_PER_M)
    parser.add_argument(
        "--batch-cached-input-price", type=float, default=DEFAULT_BATCH_CACHED_INPUT_PER_M
    )
    parser.add_argument("--batch-output-price", type=float, default=DEFAULT_BATCH_OUTPUT_PER_M)


def _add_remote_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, submit, monitor, and collect GPT-5.6 Batch jobs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Create and validate local Batch files")
    _add_prepare_arguments(prepare)
    submit = subparsers.add_parser("submit", help="Upload and submit prepared shards")
    _add_remote_arguments(submit)
    status = subparsers.add_parser("status", help="Refresh Batch job statuses")
    _add_remote_arguments(status)
    wait = subparsers.add_parser("wait", help="Wait until all jobs are terminal")
    _add_remote_arguments(wait)
    wait.add_argument("--poll-seconds", type=float, default=60.0)
    wait.add_argument("--max-wait-seconds", type=float, default=0.0)
    collect = subparsers.add_parser("collect", help="Collect jobs without submitting retries")
    _add_remote_arguments(collect)
    run = subparsers.add_parser("run", help="Prepare, submit, wait, and collect")
    _add_prepare_arguments(run)
    run.add_argument("--poll-seconds", type=float, default=60.0)
    run.add_argument("--max-wait-seconds", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare_run(args)
    elif args.command == "submit":
        manifest = submit_run(args.run_dir)
        print_status(manifest)
    elif args.command == "status":
        print_status(refresh_status(args.run_dir))
    elif args.command == "wait":
        try:
            wait_for_terminal(args.run_dir, args.poll_seconds, args.max_wait_seconds)
        except KeyboardInterrupt:
            print("Stopped local Batch monitoring; remote Batch jobs were not cancelled.")
    elif args.command == "collect":
        collect_run(args.run_dir)
    elif args.command == "run":
        prepared = prepare_run(args)
        # Derive the run directory from the stored absolute shard path instead
        # of relying on the original CLI's relative-path spelling.
        run_dir = Path(str(prepared["shards"][0]["path"])).parent
        submit_run(run_dir)
        try:
            wait_for_terminal(run_dir, args.poll_seconds, args.max_wait_seconds)
        except KeyboardInterrupt:
            print("Stopped local Batch monitoring; remote Batch jobs were not cancelled.")
            return
        collect_run(run_dir)
    else:  # pragma: no cover - argparse enforces the command choices
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
