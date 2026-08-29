#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Run the contextual semantic verifier as resumable Batch waves.

The coordinator optionally reuses a completed compatible run, renders every
remaining record with the selected verdict-only prompt, packs
contiguous waves below a configurable token ceiling, and delegates each Batch
lifecycle to ``manage_verification_batch.py``.

Retries are deliberately out of scope.  First-pass unresolved requests are
collected into a single output for later, explicit approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_QUEUE = (
    PIPELINE_DIR
    / "runs/default/verification_queue_measurement_filtered_for_gpt.jsonl"
)
DEFAULT_CORPUS = (
    PIPELINE_DIR / "data/processed/ORAN/corpus_ORAN.jsonl"
)
DEFAULT_HIERARCHICAL = (
    PIPELINE_DIR / "data/processed/ORAN/corpus_ORAN_hierarchical.json"
)
DEFAULT_RUN_ROOT = PIPELINE_DIR / "runs/default/contextual_verification"
PROMPT_MODE = "semantic"
PROMPT_VERSION = "oranlint-contextual-semantic"
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "max"
INPUT_TOKEN_CAP = 270_000
WAVE_TOKEN_CAP = 34_000_000
MAX_OUTPUT_TOKENS = 128_000
MIN_FREE_RESERVE_BYTES = 6 * 1024**3


import manage_verification_batch as batch
verifier = batch.verifier


def canonical_pair_key(record: Mapping[str, Any]) -> str:
    values = sorted((str(record.get("id1")), str(record.get("id2"))))
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    verifier.write_jsonl_atomic(path, records)
    return verifier.sha256_file(path)


def build_remainder(
    queue: Sequence[Mapping[str, Any]],
    reused: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    queue_keys = [canonical_pair_key(record) for record in queue]
    if len(queue_keys) != len(set(queue_keys)):
        raise ValueError("Canonical queue contains duplicate unordered pair IDs")
    reused_keys = [canonical_pair_key(record) for record in reused]
    if len(reused_keys) != len(set(reused_keys)):
        raise ValueError("Reused results contain duplicate unordered pair IDs")
    unknown = sorted(set(reused_keys) - set(queue_keys))
    if unknown:
        raise ValueError(f"Reused results contain {len(unknown)} pair(s) outside the queue")
    reused_set = set(reused_keys)
    return [dict(record) for record, key in zip(queue, queue_keys) if key not in reused_set]


def plan_waves(
    metadata: Sequence[Mapping[str, Any]],
    token_cap: int = WAVE_TOKEN_CAP,
) -> List[Dict[str, int]]:
    if token_cap <= 0:
        raise ValueError("Wave token cap must be positive")
    waves: List[Dict[str, int]] = []
    current: Optional[Dict[str, int]] = None
    for expected_index, record in enumerate(metadata):
        index = int(record["input_index"])
        tokens = int(record["estimated_input_tokens"])
        if index != expected_index:
            raise ValueError("Preflight metadata must cover contiguous input indexes")
        if tokens <= 0 or tokens > INPUT_TOKEN_CAP:
            raise ValueError(f"Invalid capped token estimate at input index {index}: {tokens}")
        if current is None or current["estimated_input_tokens"] + tokens > token_cap:
            if current is not None:
                waves.append(current)
            current = {
                "number": len(waves) + 1,
                "offset": index,
                "limit": 0,
                "estimated_input_tokens": 0,
                "maximum_request_tokens": 0,
            }
        current["limit"] += 1
        current["estimated_input_tokens"] += tokens
        current["maximum_request_tokens"] = max(
            current["maximum_request_tokens"], tokens
        )
    if current is not None:
        waves.append(current)
    if sum(wave["limit"] for wave in waves) != len(metadata):
        raise RuntimeError("Wave plan does not cover every preflight record")
    return waves


def merge_canonical_results(
    queue: Sequence[Mapping[str, Any]],
    sources: Sequence[Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for records in sources:
        for record in records:
            key = canonical_pair_key(record)
            if key in by_key:
                raise ValueError(f"Duplicate collected pair: {key}")
            by_key[key] = dict(record)
    ordered: List[Dict[str, Any]] = []
    for record in queue:
        key = canonical_pair_key(record)
        result = by_key.pop(key, None)
        if result is None:
            raise ValueError(f"Missing collected pair: {key}")
        ordered.append(result)
    if by_key:
        raise ValueError(f"Collected results contain {len(by_key)} pair(s) outside the queue")
    return ordered


def _validate_reuse(
    queue_path: Path,
    reuse_root: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if reuse_root is None:
        return [], None
    reuse_results = verifier.load_jsonl(
        reuse_root / "results.jsonl", required_fields=verifier.REQUIRED_FIELDS
    )
    reuse_manifest = batch.load_manifest(reuse_root / "batch")
    required = {
        "state": "collected",
        "input_sha256": verifier.sha256_file(queue_path),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_version": PROMPT_VERSION,
        "response_schema": "oran_verdict",
        "input_token_cap": INPUT_TOKEN_CAP,
        "unresolved_records": 0,
    }
    for field, expected in required.items():
        if reuse_manifest.get(field) != expected:
            raise ValueError(
                f"Reuse manifest {field}={reuse_manifest.get(field)!r}; expected {expected!r}"
            )
    expected_records = int(reuse_manifest.get("records", -1))
    if expected_records < 0 or len(reuse_results) != expected_records:
        raise ValueError("Reuse results count does not match its Batch manifest")
    if int(reuse_manifest.get("successful_records", -1)) != expected_records:
        raise ValueError("Reuse run did not successfully complete every record")
    if any(
        record.get("gpt56_status") != "completed"
        or record.get("gpt56_verdict") not in verifier.VERDICTS
        for record in reuse_results
    ):
        raise ValueError("Reuse results contain a non-completed or invalid verdict")
    return reuse_results, reuse_manifest


def _render_preflight(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    workers: int,
    corpus: Path,
    hierarchical: Path,
) -> List[Dict[str, Any]]:
    resolver = verifier.ContextResolver.from_paths(
        corpus,
        hierarchical,
        max_par_chars=2000,
        window=1,
    )
    settings = verifier.InferenceSettings(
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_paragraph_chars=2000,
        context_window=1,
        max_retries=1,
        retry_base_delay=0.0,
    )
    selected = [(index, record) for index, record in enumerate(records)]
    compact: List[Dict[str, Any]] = []
    for ordinal, (_, metadata) in enumerate(
        batch.render_batch_requests(
            selected,
            resolver,
            settings,
            workers,
            batch.model_slug(MODEL),
            PROMPT_MODE,
            INPUT_TOKEN_CAP,
        ),
        1,
    ):
        compact.append(
            {
                "input_index": int(metadata["input_index"]),
                "pair_key": str(metadata["pair_key"]),
                "prompt_sha256": str(metadata["prompt_sha256"]),
                "estimated_input_tokens": int(metadata["estimated_input_tokens"]),
                "document_context_capped": bool(metadata["document_context_capped"]),
                "document_context_strategy": str(metadata["document_context_strategy"]),
            }
        )
        if ordinal % 250 == 0 or ordinal == len(records):
            print(f"Preflight rendered {ordinal:,}/{len(records):,}", flush=True)
    _write_jsonl(output_path, compact)
    return compact


def _prepare_args(
    remainder_path: Path,
    run_root: Path,
    wave: Mapping[str, int],
    workers: int,
    corpus: Path,
    hierarchical: Path,
) -> SimpleNamespace:
    wave_root = run_root / "waves" / f"wave-{int(wave['number']):02d}"
    return SimpleNamespace(
        input=remainder_path,
        corpus=corpus,
        hierarchical=hierarchical,
        output=wave_root / "results.jsonl",
        run_dir=wave_root / "batch",
        model=MODEL,
        prompt_mode=PROMPT_MODE,
        reasoning_effort=REASONING_EFFORT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input_token_cap=INPUT_TOKEN_CAP,
        max_paragraph_chars=2000,
        context_window=1,
        offset=int(wave["offset"]),
        limit=int(wave["limit"]),
        sample_size=None,
        sample_seed=42,
        sample_method="random",
        expected_records=int(wave["limit"]),
        num_shards=1,
        estimate_population=False,
        prediction_field="deberta_preds",
        prediction_value="contradiction",
        progress_every=100,
        workers=workers,
        pricing_mode="tokens-only",
        batch_input_price=batch.DEFAULT_BATCH_INPUT_PER_M,
        batch_cached_input_price=batch.DEFAULT_BATCH_CACHED_INPUT_PER_M,
        batch_output_price=batch.DEFAULT_BATCH_OUTPUT_PER_M,
    )


def _series_path(run_root: Path) -> Path:
    return run_root / "series_manifest.json"


def _save_series(run_root: Path, value: Mapping[str, Any]) -> None:
    verifier.write_json_atomic(_series_path(run_root), value)


def _load_series(run_root: Path) -> Dict[str, Any]:
    path = _series_path(run_root)
    if not path.is_file():
        raise FileNotFoundError(f"Series manifest does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid series manifest: {path}")
    return value


def _validate_series_inputs(series: Mapping[str, Any]) -> None:
    """Refuse submission or merging if any frozen local input has drifted."""
    checks = (
        ("queue", "queue_sha256"),
        ("corpus", "corpus_sha256"),
        ("hierarchical", "hierarchical_sha256"),
        ("remainder", "remainder_sha256"),
        ("preflight_index", "preflight_index_sha256"),
    )
    for path_field, hash_field in checks:
        path = Path(str(series[path_field]))
        actual = verifier.sha256_file(path)
        if actual != series.get(hash_field):
            raise ValueError(f"Frozen input changed after preparation: {path}")
    reuse_root = series.get("reuse_root")
    if reuse_root:
        reuse_results = Path(str(reuse_root)) / "results.jsonl"
        if verifier.sha256_file(reuse_results) != series.get("reuse_results_sha256"):
            raise ValueError(f"Reused results changed after preparation: {reuse_results}")


def prepare_full_run(args: argparse.Namespace) -> Dict[str, Any]:
    queue_path = args.queue.resolve()
    corpus = args.corpus.resolve()
    hierarchical = args.hierarchical.resolve()
    reuse_root = args.reuse_run.resolve() if args.reuse_run is not None else None
    run_root = args.run_root.resolve()
    queue_sha = verifier.sha256_file(queue_path)
    if args.expected_queue_sha256 and queue_sha != args.expected_queue_sha256:
        raise ValueError("Queue SHA-256 does not match --expected-queue-sha256")
    queue = verifier.load_jsonl(queue_path, required_fields=verifier.REQUIRED_FIELDS)
    verifier.validate_records(queue, "deberta_preds", "contradiction")
    if args.expected_records is not None and len(queue) != args.expected_records:
        raise ValueError(
            f"Expected {args.expected_records:,} queue records; found {len(queue):,}"
        )
    reused_results, reuse_manifest = _validate_reuse(queue_path, reuse_root)
    remainder = build_remainder(queue, reused_results)
    corpus_sha = verifier.sha256_file(corpus)
    hierarchical_sha = verifier.sha256_file(hierarchical)
    reuse_results_sha = (
        verifier.sha256_file(reuse_root / "results.jsonl")
        if reuse_root is not None else None
    )

    run_root.mkdir(parents=True, exist_ok=True)
    existing_series = _series_path(run_root)
    if existing_series.is_file():
        old = _load_series(run_root)
        if old.get("queue_sha256") != queue_sha:
            raise ValueError("Run root already belongs to a different queue")
        if old.get("corpus_sha256") != corpus_sha:
            raise ValueError("Run root already belongs to a different corpus")
        if old.get("hierarchical_sha256") != hierarchical_sha:
            raise ValueError("Run root already belongs to a different hierarchy")
        if old.get("reuse_results_sha256") != reuse_results_sha:
            raise ValueError("Run root already uses a different reuse set")
        if int(old.get("wave_token_cap", -1)) != int(args.wave_token_cap):
            raise ValueError("Run root was prepared with a different wave token cap")
        if old.get("state") != "preparing":
            _validate_series_inputs(old)
            print(f"Run already {old.get('state')}: {run_root}", flush=True)
            return old
    remainder_path = run_root / "pending_requests.jsonl"
    remainder_sha = _write_jsonl(remainder_path, remainder)
    preflight_path = run_root / "preflight_index.jsonl"
    if preflight_path.is_file():
        preflight = verifier.load_jsonl(preflight_path)
    else:
        preflight = _render_preflight(
            remainder, preflight_path, args.workers, corpus, hierarchical
        )
    if len(preflight) != len(remainder):
        raise ValueError("Preflight index does not cover the full remainder")
    for metadata, record in zip(preflight, remainder):
        if str(metadata["pair_key"]) != verifier.pair_key(record):
            raise ValueError("Preflight pair order does not match the remainder")
    waves = plan_waves(preflight, args.wave_token_cap)
    estimated_total = sum(int(item["estimated_input_tokens"]) for item in preflight)

    existing_bytes = sum(
        path.stat().st_size
        for path in run_root.rglob("batch_input_*.jsonl")
        if path.is_file()
    )
    projected_bytes = math.ceil(estimated_total * 3.2)
    available = shutil.disk_usage(run_root).free
    remaining_projected = max(0, projected_bytes - existing_bytes)
    if available < remaining_projected + MIN_FREE_RESERVE_BYTES:
        raise OSError(
            f"Insufficient disk: {available:,} free bytes; need approximately "
            f"{remaining_projected + MIN_FREE_RESERVE_BYTES:,}"
        )

    series: Dict[str, Any] = {
        "schema_version": "contextual-verification-series",
        "state": "preparing",
        "created_at": batch.utc_now(),
        "updated_at": batch.utc_now(),
        "queue": str(queue_path),
        "queue_sha256": queue_sha,
        "queue_records": len(queue),
        "corpus": str(corpus),
        "corpus_sha256": corpus_sha,
        "hierarchical": str(hierarchical),
        "hierarchical_sha256": hierarchical_sha,
        "reuse_root": str(reuse_root) if reuse_root is not None else None,
        "reused_records": len(reused_results),
        "reuse_results_sha256": reuse_results_sha,
        "reuse_batch_ids": (
            [str(shard.get("batch_id") or "") for shard in reuse_manifest["shards"]]
            if reuse_manifest is not None else []
        ),
        "remainder": str(remainder_path),
        "remainder_sha256": remainder_sha,
        "new_request_records": len(remainder),
        "preflight_index": str(preflight_path),
        "preflight_index_sha256": verifier.sha256_file(preflight_path),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_mode": PROMPT_MODE,
        "prompt_version": PROMPT_VERSION,
        "response_schema": "oran_verdict",
        "input_token_cap": INPUT_TOKEN_CAP,
        "wave_token_cap": int(args.wave_token_cap),
        "estimated_input_tokens": estimated_total,
        "maximum_request_tokens": max(
            (int(item["estimated_input_tokens"]) for item in preflight), default=0
        ),
        "capped_requests": sum(
            bool(item["document_context_capped"]) for item in preflight
        ),
        "document_context_strategies": dict(
            sorted(Counter(str(item["document_context_strategy"]) for item in preflight).items())
        ),
        "projected_batch_input_bytes": projected_bytes,
        "free_bytes_before_wave_preparation": available,
        "waves": [],
    }
    _save_series(run_root, series)

    prepared_waves: List[Dict[str, Any]] = []
    for wave in waves:
        wave_args = _prepare_args(
            remainder_path, run_root, wave, args.workers, corpus, hierarchical
        )
        wave_root = Path(wave_args.run_dir).parent
        batch_manifest_path = Path(wave_args.run_dir) / batch.MANIFEST_NAME
        if batch_manifest_path.is_file():
            manifest = batch.load_manifest(Path(wave_args.run_dir))
        else:
            print(
                f"Preparing wave {wave['number']}/{len(waves)}: "
                f"offset={wave['offset']:,}, records={wave['limit']:,}",
                flush=True,
            )
            manifest = batch.prepare_run(wave_args)
        if manifest.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"Wave {wave['number']} prompt version mismatch")
        if manifest.get("input_sha256") != remainder_sha:
            raise ValueError(f"Wave {wave['number']} remainder hash mismatch")
        if int(manifest.get("offset", -1)) != int(wave["offset"]):
            raise ValueError(f"Wave {wave['number']} offset mismatch")
        if int(manifest.get("limit", -1)) != int(wave["limit"]):
            raise ValueError(f"Wave {wave['number']} record-limit mismatch")
        if manifest.get("corpus_sha256") != corpus_sha:
            raise ValueError(f"Wave {wave['number']} corpus hash mismatch")
        if manifest.get("hierarchical_sha256") != hierarchical_sha:
            raise ValueError(f"Wave {wave['number']} hierarchy hash mismatch")
        if int(manifest["estimated_input_tokens"]) != int(wave["estimated_input_tokens"]):
            raise ValueError(f"Wave {wave['number']} preflight token total changed")
        if int(manifest["estimated_input_tokens"]) > int(args.wave_token_cap):
            raise ValueError(f"Wave {wave['number']} exceeds the wave token cap")
        prepared_waves.append(
            {
                **wave,
                "root": str(wave_root),
                "run_dir": str(wave_args.run_dir),
                "output": str(wave_args.output),
                "manifest_sha256": verifier.sha256_file(batch_manifest_path),
                "state": str(manifest["state"]),
                "batch_ids": [
                    str(shard.get("batch_id") or "") for shard in manifest["shards"]
                ],
            }
        )
        series["waves"] = prepared_waves
        series["updated_at"] = batch.utc_now()
        _save_series(run_root, series)

    series["state"] = "prepared"
    series["prepared_at"] = batch.utc_now()
    series["updated_at"] = batch.utc_now()
    series["waves"] = prepared_waves
    _save_series(run_root, series)
    print(
        f"Prepared {len(waves)} wave(s) covering {len(remainder):,} new requests; "
        f"estimated input {estimated_total:,} tokens",
        flush=True,
    )
    return series


def execute_full_run(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = args.run_root.resolve()
    series = _load_series(run_root)
    _validate_series_inputs(series)
    if series.get("state") not in {"prepared", "running", "first_pass_collected", "needs_retry_approval", "complete"}:
        raise ValueError(f"Series is not ready for execution: {series.get('state')!r}")
    if series.get("state") in {"first_pass_collected", "needs_retry_approval", "complete"}:
        return finalize_full_run(SimpleNamespace(run_root=run_root))
    series["state"] = "running"
    series["started_at"] = series.get("started_at") or batch.utc_now()
    series["updated_at"] = batch.utc_now()
    _save_series(run_root, series)

    for wave in series["waves"]:
        number = int(wave["number"])
        run_dir = Path(str(wave["run_dir"]))
        manifest = batch.load_manifest(run_dir)
        if manifest.get("state") == "collected":
            print(f"Wave {number} already collected; skipping", flush=True)
        else:
            if manifest.get("state") == "prepared":
                print(f"Submitting wave {number}/{len(series['waves'])}", flush=True)
                manifest = batch.submit_run(run_dir)
            if manifest.get("state") != "terminal":
                print(
                    f"Waiting for wave {number}; status checks every {args.poll_seconds:g}s",
                    flush=True,
                )
                manifest = batch.wait_for_terminal(run_dir, args.poll_seconds, 0.0)
            manifest = batch.collect_run(run_dir)
        summary = json.loads(Path(str(manifest["summary"])).read_text(encoding="utf-8"))
        if summary.get("integrity_errors"):
            raise RuntimeError(f"Wave {number} has Batch integrity errors")
        wave["state"] = "collected"
        wave["batch_ids"] = [str(shard.get("batch_id") or "") for shard in manifest["shards"]]
        wave["successful_records"] = int(summary["successful_records"])
        wave["unresolved_records"] = int(summary["unresolved_records"])
        wave["verdict_counts"] = summary["verdict_counts"]
        wave["usage_totals"] = summary["usage_totals"]
        wave["output_sha256"] = verifier.sha256_file(Path(str(manifest["output"])))
        series["updated_at"] = batch.utc_now()
        _save_series(run_root, series)
        print(
            f"Collected wave {number}/{len(series['waves'])}: "
            f"{summary['successful_records']:,} successful, "
            f"{summary['unresolved_records']:,} unresolved",
            flush=True,
        )

    series["state"] = "first_pass_collected"
    series["first_pass_collected_at"] = batch.utc_now()
    series["updated_at"] = batch.utc_now()
    _save_series(run_root, series)
    return finalize_full_run(SimpleNamespace(run_root=run_root))


def _aggregate_retry_requests(series: Mapping[str, Any]) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for wave in series["waves"]:
        manifest = batch.load_manifest(Path(str(wave["run_dir"])))
        retry_path = Path(str(manifest["retry_candidates"]))
        if not retry_path.is_file():
            continue
        for request in batch._load_jsonl_values(retry_path):
            custom_id = str(request.get("custom_id") or "")
            if not custom_id or custom_id in seen:
                raise ValueError(f"Invalid duplicate retry custom_id: {custom_id!r}")
            seen.add(custom_id)
            requests.append(request)
    return requests


def finalize_full_run(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = args.run_root.resolve()
    series = _load_series(run_root)
    _validate_series_inputs(series)
    queue = verifier.load_jsonl(Path(str(series["queue"])), required_fields=verifier.REQUIRED_FIELDS)
    reused: List[Dict[str, Any]] = []
    if series.get("reuse_root"):
        reused = verifier.load_jsonl(
            Path(str(series["reuse_root"])) / "results.jsonl",
            required_fields=verifier.REQUIRED_FIELDS,
        )
    wave_records: List[List[Dict[str, Any]]] = []
    for wave in series["waves"]:
        manifest = batch.load_manifest(Path(str(wave["run_dir"])))
        if manifest.get("state") != "collected":
            raise ValueError(f"Wave {wave['number']} is not collected")
        wave_records.append(
            verifier.load_jsonl(Path(str(manifest["output"])), required_fields=verifier.REQUIRED_FIELDS)
        )
    final_records = merge_canonical_results(queue, [reused, *wave_records])
    if len(final_records) != int(series["queue_records"]):
        raise RuntimeError("Final merge did not reproduce the complete queue")
    inconsistent = [
        record for record in final_records if record.get("gpt56_verdict") == "inconsistent"
    ]
    unresolved = [
        record
        for record in final_records
        if record.get("gpt56_status") != "completed"
        or record.get("gpt56_verdict") not in verifier.VERDICTS
    ]
    retry_requests = _aggregate_retry_requests(series)
    if len(retry_requests) != len(unresolved):
        raise RuntimeError("Aggregated retry requests do not match unresolved results")

    final_path = run_root / "final_verdicts.jsonl"
    inconsistent_path = run_root / "inconsistent_verdicts.jsonl"
    unresolved_path = run_root / "unresolved_results.jsonl"
    retry_path = run_root / "retry_candidates.jsonl"
    _write_jsonl(final_path, final_records)
    _write_jsonl(inconsistent_path, inconsistent)
    _write_jsonl(unresolved_path, unresolved)
    _write_jsonl(retry_path, retry_requests)

    result_summary = verifier.summarize_results(final_records)
    usage = result_summary["usage_totals"]
    calibration = 1.31 / (16_289_292 * 0.1 / 1_000_000 + 31_911 * 0.6 / 1_000_000)
    local_rate_cost = (
        int(usage.get("input_tokens", 0)) * 0.1 / 1_000_000
        + int(usage.get("output_tokens", 0)) * 0.6 / 1_000_000
    )
    batch_ids = [
        batch_id
        for wave in series["waves"]
        for batch_id in wave.get("batch_ids", [])
        if batch_id
    ]
    summary: Dict[str, Any] = {
        "schema_version": "contextual-verification-summary",
        "created_at": batch.utc_now(),
        "state": "complete" if not unresolved else "needs_retry_approval",
        "queue_records": len(queue),
        "reused_records": len(reused),
        "new_request_records": sum(len(records) for records in wave_records),
        "wave_count": len(series["waves"]),
        "batch_ids": batch_ids,
        "successful_records": len(final_records) - len(unresolved),
        "unresolved_records": len(unresolved),
        "inconsistent_records": len(inconsistent),
        **result_summary,
        "dashboard_cost_status": "not_available",
        "dashboard_calibrated_cost_estimate_usd": round(local_rate_cost * calibration, 2),
        "cost_estimate_note": (
            "Estimate only; applies the original 100-request dashboard-to-local-rate "
            "calibration to aggregate API usage."
        ),
        "queue_sha256": verifier.sha256_file(Path(str(series["queue"]))),
        "reuse_results_sha256": series.get("reuse_results_sha256"),
        "final_verdicts": str(final_path),
        "final_verdicts_sha256": verifier.sha256_file(final_path),
        "inconsistent_verdicts": str(inconsistent_path),
        "inconsistent_verdicts_sha256": verifier.sha256_file(inconsistent_path),
        "unresolved_results": str(unresolved_path),
        "unresolved_results_sha256": verifier.sha256_file(unresolved_path),
        "retry_candidates": str(retry_path),
        "retry_candidates_sha256": verifier.sha256_file(retry_path),
    }
    summary_path = run_root / "final_summary.json"
    verifier.write_json_atomic(summary_path, summary)
    series["state"] = summary["state"]
    series["finalized_at"] = batch.utc_now()
    series["updated_at"] = batch.utc_now()
    series["final_summary"] = str(summary_path)
    series["final_summary_sha256"] = verifier.sha256_file(summary_path)
    _save_series(run_root, series)
    print(
        f"Finalized {len(final_records):,} rows: {len(inconsistent):,} inconsistent, "
        f"{len(unresolved):,} unresolved",
        flush=True,
    )
    return summary


def show_status(args: argparse.Namespace) -> Dict[str, Any]:
    series = _load_series(args.run_root.resolve())
    counts = Counter(str(wave.get("state") or "unknown") for wave in series["waves"])
    print(
        json.dumps(
            {
                "state": series.get("state"),
                "waves": len(series["waves"]),
                "wave_states": dict(sorted(counts.items())),
                "reused_records": series.get("reused_records"),
                "new_request_records": series.get("new_request_records"),
            },
            indent=2,
        )
    )
    return series


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    prepare.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    prepare.add_argument("--hierarchical", type=Path, default=DEFAULT_HIERARCHICAL)
    prepare.add_argument(
        "--reuse-run",
        type=Path,
        help="Optional compatible collected run whose completed results should be reused",
    )
    prepare.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    prepare.add_argument("--expected-records", type=int)
    prepare.add_argument("--expected-queue-sha256")
    prepare.add_argument("--wave-token-cap", type=int, default=WAVE_TOKEN_CAP)
    prepare.add_argument("--workers", type=int, default=8)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    execute.add_argument("--poll-seconds", type=float, default=900.0)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    status = subparsers.add_parser("status")
    status.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare_full_run(args)
    elif args.command == "execute":
        execute_full_run(args)
    elif args.command == "finalize":
        finalize_full_run(args)
    elif args.command == "status":
        show_status(args)
    else:  # pragma: no cover - argparse constrains commands
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
