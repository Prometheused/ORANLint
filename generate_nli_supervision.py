#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Generate leak-guarded, candidate-shaped synthetic NLI pairs locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = (
    PIPELINE_ROOT
    / "data/processed/ORAN/corpus_ORAN.jsonl"
)
DEFAULT_CANDIDATES = (
    PIPELINE_ROOT
    / "data/generated/candidate_pairs.jsonl"
)
DEFAULT_HOLDOUT = (
    PIPELINE_ROOT
    / "data/generated/reserved_evaluation_pairs.jsonl"
)
DEFAULT_OUTPUT = (
    PIPELINE_ROOT / "data/generated/nli_supervision"
)
DEFAULT_ADAPTER = (
    PIPELINE_ROOT
    / "models/domain_generator"
)
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
LABELS = ("consistent", "inconsistent", "neutral")
RENDERED_LABELS = {"consistent", "inconsistent"}
CONTRADICTION_TYPES = (
    "polarity",
    "numeric_value",
    "comparator",
    "actor_recipient",
    "component_scope",
    "protocol_requirement",
    "procedure_state",
    "http_or_parameter",
)
CONTRADICTION_INSTRUCTIONS = {
    "polarity": "reverse exactly one required/forbidden or positive/negative condition",
    "numeric_value": "replace exactly one numeric value while retaining its unit",
    "comparator": "reverse exactly one threshold, ordering, before/after, minimum, or maximum relation",
    "actor_recipient": "replace exactly one actor or message recipient with another O-RAN actor",
    "component_scope": "replace exactly one named O-RAN component while preserving the procedure",
    "protocol_requirement": "replace exactly one named security or transport protocol",
    "procedure_state": "reverse exactly one procedure state, prerequisite, or required action",
    "http_or_parameter": "replace exactly one HTTP method/status or named parameter value",
}
DEFAULT_RENDER_SURPLUS = 1.5
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
NEGATION_RE = re.compile(r"\b(?:not|never|cannot|can't|mustn't|shalln't|no)\b", re.I)
MODAL_RE = re.compile(r"\b(?:shall|must|should|may|can|cannot)\b", re.I)
COMPONENT_RE = re.compile(
    r"(?<![\w-])(?:O-RU|O-DU|O-CU(?:-CP|-UP)?|RIC|consumer|producer|client|server)(?![\w-])",
    re.I,
)
COMPARATOR_RE = re.compile(
    r"(?:<=|>=|<|>|at\s+least|at\s+most|greater\s+than|less\s+than|before|after)",
    re.I,
)
PROTOCOL_RE = re.compile(r"\b(?:TLS|mTLS|MACsec|SSH|HTTPS?|IPsec)\b", re.I)
HTTP_RE = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE|[1-5]\d\d)\b", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def pair_key(row: Mapping[str, Any]) -> tuple[str, str]:
    first, second = str(row.get("id1")), str(row.get("id2"))
    return (first, second) if first <= second else (second, first)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No records found: {path}")
    return rows


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise
    return count


def heldout_exclusions(
    holdout: Sequence[Mapping[str, Any]], corpus_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    texts: set[str] = set()
    for row in holdout:
        for side in ("1", "2"):
            identifier = str(row.get(f"id{side}"))
            ids.add(identifier)
            text = normalized_text(row.get(f"text{side}"))
            if text:
                texts.add(text)
    return ids, texts


def split_documents(
    documents: Iterable[str], seed: int, dev_fraction: float = 0.2
) -> dict[str, str]:
    values = sorted(set(documents))
    rng = random.Random(seed)
    rng.shuffle(values)
    dev_count = max(1, round(len(values) * dev_fraction))
    dev = set(values[:dev_count])
    return {document: ("dev" if document in dev else "train") for document in values}


def stratum(row: Mapping[str, Any]) -> tuple[Any, ...]:
    tfidf = min(4, int(float(row.get("tfidf_cos") or 0.0) * 5))
    bge = min(4, max(0, int((float(row.get("bge_cos") or 0.0) + 1.0) * 2.5)))
    first = max(1, len(str(row.get("text1") or "").split()))
    second = max(1, len(str(row.get("text2") or "").split()))
    ratio = min(first, second) / max(first, second)
    length_bin = min(3, int(ratio * 4))
    same_pdf = str(row.get("id1_pdf_file") or "") == str(
        row.get("id2_pdf_file") or ""
    )
    return (
        tfidf,
        bge,
        length_bin,
        same_pdf,
        str(row.get("candidate_source") or "unknown"),
    )


def round_robin_strata(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(dict(row))
    rng = random.Random(seed)
    for key in groups:
        rng.shuffle(groups[key])
    ordered: list[dict[str, Any]] = []
    keys = sorted(groups, key=repr)
    while keys:
        next_keys = []
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def target_quotas(args: argparse.Namespace) -> dict[str, int]:
    """Return accepted-pair quotas for training and development."""
    return {"train": args.train_per_class, "dev": args.dev_per_class}


def job_quotas(
    targets: Mapping[str, int], render_surplus: float
) -> dict[tuple[str, str], int]:
    """Oversample rendered classes so validators can reject weak generations."""

    quotas: dict[tuple[str, str], int] = {}
    for split, target in targets.items():
        for label in LABELS:
            quotas[(split, label)] = (
                math.ceil(target * render_surplus)
                if label in RENDERED_LABELS
                else target
            )
    return quotas


def eligible_contradiction_types(text: str) -> tuple[str, ...]:
    """Only request contradiction edits with deterministic source-side evidence."""

    eligible: list[str] = []
    if NEGATION_RE.search(text) or MODAL_RE.search(text):
        eligible.extend(("polarity", "procedure_state"))
    if NUMBER_RE.search(text):
        eligible.append("numeric_value")
    if COMPARATOR_RE.search(text):
        eligible.append("comparator")
    if COMPONENT_RE.search(text):
        eligible.extend(("actor_recipient", "component_scope"))
    if PROTOCOL_RE.search(text):
        eligible.append("protocol_requirement")
    if HTTP_RE.search(text) or NUMBER_RE.search(text):
        eligible.append("http_or_parameter")
    return tuple(dict.fromkeys(eligible))


def prompt_for(job: Mapping[str, Any]) -> str:
    text = str(job["text1"])
    if job["label"] == "consistent":
        instruction = (
            "Write an independently authored O-RAN specification statement that is "
            "semantically equivalent to the source. Preserve every actor, component, "
            "condition, number, unit, protocol, modality, and applicability scope."
        )
        schema = '{"text2":"...","preserved_scope":true}'
    elif job["label"] == "neutral":
        instruction = (
            "Write an independently authored statement with the same kind of technical "
            "requirement, but make it apply to a clearly DIFFERENT named O-RAN component. "
            "This must be a sibling applicability context, not a contradiction about the "
            "same component. original_value MUST be an exact named-component substring "
            "from SOURCE. new_value MUST be a different exact named-component substring "
            "in text2. Preserve all numbers, protocols, polarity, and modality. "
            "different_scope MUST be true."
        )
        schema = (
            '{"text2":"...","changed_field":"component_scope",'
            '"original_value":"...","new_value":"...","different_scope":true}'
        )
    else:
        category = str(job["contradiction_type"])
        instruction = (
            "Write an independently authored statement in the exact same applicability "
            "domain as the source, but introduce one concrete contradiction of type "
            f"{category}: {CONTRADICTION_INSTRUCTIONS[category]}. Do not change to a "
            "sibling scenario or different use case. original_value MUST be an exact "
            "substring copied from SOURCE. new_value MUST be an exact substring in "
            "text2 and different from original_value. same_scope refers to the surrounding "
            "scenario, not the deliberately changed field, and MUST be true."
        )
        schema = (
            '{"text2":"...","changed_field":"...","original_value":"...",'
            '"new_value":"...","same_scope":true}'
        )
    return (
        "You generate synthetic NLI data for O-RAN specifications. "
        + instruction
        + " Do not copy SOURCE verbatim; make text2 a complete, concise rewrite."
        + " Return JSON only using this schema: "
        + schema
        + "\nSOURCE:\n"
        + text
    )


def _load_variant_module():
    import importlib.util
    import sys

    path = PIPELINE_ROOT / "variant_rules.py"
    name = "candidate_shaped_variant_filter"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_jobs(args: argparse.Namespace) -> None:
    corpus = load_jsonl(args.corpus.resolve())
    candidates = load_jsonl(args.candidates.resolve())
    holdout = load_jsonl(args.holdout.resolve())
    corpus_by_id = {str(row.get("id")): row for row in corpus}
    heldout_ids, heldout_texts = heldout_exclusions(holdout, corpus_by_id)
    eligible_documents = {
        str(row.get("pdf_file"))
        for row in corpus
        if row.get("pdf_file")
    }
    document_split = split_documents(
        eligible_documents, args.seed, dev_fraction=args.dev_fraction
    )
    filtered = []
    for row in candidates:
        ids = {str(row.get("id1")), str(row.get("id2"))}
        texts = {normalized_text(row.get("text1")), normalized_text(row.get("text2"))}
        pdfs = {
            str(row.get("id1_pdf_file") or ""),
            str(row.get("id2_pdf_file") or ""),
        }
        if ids & heldout_ids or texts & heldout_texts:
            continue
        splits = {document_split.get(pdf) for pdf in pdfs if pdf}
        if len(splits) != 1 or None in splits:
            continue
        copied = dict(row)
        copied["synthetic_split"] = next(iter(splits))
        filtered.append(copied)
    ordered = round_robin_strata(filtered, args.seed)
    per_split = target_quotas(args)
    quotas = job_quotas(per_split, args.render_surplus)
    variant = _load_variant_module()
    context_index = variant.CorpusContextIndex(corpus, max_lookback=12)
    selected_keys: set[tuple[str, str]] = set()
    jobs = []
    counts: Counter = Counter()
    for row in ordered:
        split = str(row["synthetic_split"])
        if all(counts[(split, label)] >= quotas[(split, label)] for label in LABELS):
            continue
        key = pair_key(row)
        if key in selected_keys:
            continue
        jobs_before = len(jobs)
        if counts[(split, "neutral")] < quotas[(split, "neutral")]:
            analysis = variant.annotate_pair(dict(row), context_index)
            if analysis.get("variant_filter_decision") == "auto_neutral":
                job = {
                    "job_id": f"{split}-neutral-{counts[(split, 'neutral')]:06d}",
                    "split": split,
                    "label": "neutral",
                    "text1": row["text1"],
                    "text2": row["text2"],
                    "source_id1": row["id1"],
                    "source_id2": row["id2"],
                    "source_pdf_files": sorted(
                        {str(row.get("id1_pdf_file") or ""), str(row.get("id2_pdf_file") or "")}
                        - {""}
                    ),
                    "source_pair_key": list(key),
                    "source_stratum": list(stratum(row)),
                    "variant_filter_primary_rule": analysis.get(
                        "variant_filter_primary_rule"
                    ),
                    "requires_render": False,
                    "target_quota": per_split[split],
                }
                jobs.append(job)
                counts[(split, "neutral")] += 1
        for label in ("consistent", "inconsistent"):
            if counts[(split, label)] >= quotas[(split, label)]:
                continue
            eligible_sides = [
                side
                for side in (1, 2)
                if args.min_source_words
                <= len(str(row[f"text{side}"]).split())
                <= args.max_source_words
            ]
            if not eligible_sides:
                continue
            side = eligible_sides[counts[(split, label)] % len(eligible_sides)]
            source_id = row[f"id{side}"]
            source_pdf = str(row.get(f"id{side}_pdf_file") or "")
            job = {
                "job_id": f"{split}-{label}-{counts[(split, label)]:06d}",
                "split": split,
                "label": label,
                "text1": row[f"text{side}"],
                "source_id1": source_id,
                "source_id2": None,
                "source_pdf_files": [source_pdf],
                "source_pair_key": list(key),
                "source_stratum": list(stratum(row)),
                "requires_render": True,
                "target_quota": per_split[split],
            }
            if label == "inconsistent":
                eligible_types = eligible_contradiction_types(str(job["text1"]))
                if not eligible_types:
                    continue
                job["contradiction_type"] = min(
                    eligible_types,
                    key=lambda category: (
                        counts[(split, label, category)],
                        CONTRADICTION_TYPES.index(category),
                    ),
                )
                counts[(split, label, job["contradiction_type"])] += 1
            job["prompt"] = prompt_for(job)
            jobs.append(job)
            counts[(split, label)] += 1
        if len(jobs) > jobs_before:
            selected_keys.add(key)
        if all(
            counts[(split_name, label)] >= quotas[(split_name, label)]
            for split_name in ("train", "dev")
            for label in LABELS
        ):
            break
    # Variant rules are deliberately conservative and can supply fewer than the
    # requested neutral rows. Fill only that deficit with a different-component
    # local generation task; documents and held-out exclusions remain unchanged.
    for split in ("train", "dev"):
        neutral_deficit = per_split[split] - counts[(split, "neutral")]
        if neutral_deficit <= 0:
            continue
        neutral_attempts = math.ceil(neutral_deficit * args.render_surplus)
        added = 0
        neutral_selected_keys: set[tuple[str, str]] = set()
        for row in ordered:
            key = pair_key(row)
            if row["synthetic_split"] != split or key in neutral_selected_keys:
                continue
            eligible_sides = [
                side
                for side in (1, 2)
                if args.min_source_words
                <= len(str(row[f"text{side}"]).split())
                <= args.max_source_words
                and COMPONENT_RE.search(str(row[f"text{side}"]))
            ]
            if not eligible_sides:
                continue
            side = eligible_sides[added % len(eligible_sides)]
            source_pdf = str(row.get(f"id{side}_pdf_file") or "")
            job = {
                "job_id": f"{split}-neutral-rendered-{added:06d}",
                "split": split,
                "label": "neutral",
                "neutral_type": "different_component_scope",
                "text1": row[f"text{side}"],
                "source_id1": row[f"id{side}"],
                "source_id2": None,
                "source_pdf_files": [source_pdf],
                "source_pair_key": list(key),
                "source_stratum": list(stratum(row)),
                "requires_render": True,
                "target_quota": per_split[split],
            }
            job["prompt"] = prompt_for(job)
            jobs.append(job)
            neutral_selected_keys.add(key)
            added += 1
            counts[(split, "neutral")] += 1
            if added >= neutral_attempts:
                break
    missing = {
        f"{split}:{label}": quotas[(split, label)] - counts[(split, label)]
        for split in ("train", "dev")
        for label in LABELS
        if counts[(split, label)] < quotas[(split, label)]
    }
    if missing:
        raise RuntimeError(f"Insufficient validated candidate-shaped jobs: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = args.output_dir / "generation_jobs.jsonl"
    atomic_jsonl(jobs_path, jobs)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "dev_document_fraction": args.dev_fraction,
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": sha256_file(args.corpus.resolve()),
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": sha256_file(args.candidates.resolve()),
        "holdout": str(args.holdout.resolve()),
        "holdout_sha256": sha256_file(args.holdout.resolve()),
        "heldout_ids": len(heldout_ids),
        "heldout_texts": len(heldout_texts),
        "document_split": Counter(document_split.values()),
        "accepted_target_counts": {f"{split}:{label}": per_split[split] for split in ("train", "dev") for label in LABELS},
        "render_surplus": args.render_surplus,
        "render_source_word_range": [args.min_source_words, args.max_source_words],
        "counts": {f"{split}:{label}": counts[(split, label)] for split in ("train", "dev") for label in LABELS},
        "jobs": str(jobs_path.resolve()),
        "jobs_sha256": sha256_file(jobs_path),
        "render_jobs": sum(bool(job["requires_render"]) for job in jobs),
        "neutral_real_pairs": sum(not job["requires_render"] for job in jobs),
    }
    atomic_json(args.output_dir / "preparation_manifest.json", manifest)
    print(f"Prepared {len(jobs):,} jobs: {jobs_path}")
    print(f"Local Llama renders required: {manifest['render_jobs']:,}")


def parse_json_object(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def render_jobs(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    jobs = load_jsonl(args.jobs.resolve())
    renderable = [job for job in jobs if job.get("requires_render")]
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    renderable = [
        job
        for index, job in enumerate(renderable)
        if index % args.shard_count == args.shard_index
    ]
    output = args.responses.resolve()
    completed: dict[str, dict[str, Any]] = {}
    if output.exists() and output.stat().st_size:
        completed = {row["job_id"]: row for row in load_jsonl(output)}
    pending = [job for job in renderable if job["job_id"] not in completed]
    if not pending:
        print("All local render jobs are already complete")
        return
    adapter = args.adapter.resolve()
    config = PeftConfig.from_pretrained(adapter, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    base_model = args.base_model or config.base_model_name_or_path or BASE_MODEL
    local_base = Path(str(base_model)).expanduser().exists()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=None if local_base else BASE_REVISION,
        local_files_only=local_base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "Return strict JSON only."},
                        {"role": "user", "content": job["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for job in batch
            ]
            set_seed(args.seed + start)
            encoded = tokenizer(
                prompts, padding=True, truncation=True, max_length=2048, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                )
            padded_prompt_length = int(encoded["input_ids"].shape[1])
            for job, tokens in zip(batch, generated):
                decoded = tokenizer.decode(
                    tokens[padded_prompt_length:], skip_special_tokens=True
                )
                row = {
                    "job_id": job["job_id"],
                    "raw_response": decoded,
                    "parsed": parse_json_object(decoded),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(f"Rendered {min(start + len(batch), len(pending)):,}/{len(pending):,}")


def prepare_retry_jobs(args: argparse.Namespace) -> None:
    rejected = load_jsonl(args.rejected.resolve())
    eligible = [
        row
        for row in rejected
        if row.get("requires_render")
        and row.get("rejection_reason") != "valid_surplus_not_selected"
        and (args.split is None or row.get("split") == args.split)
        and (args.label is None or row.get("label") == args.label)
    ]
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[: args.limit]
    if len(selected) < args.limit:
        raise RuntimeError(
            f"Requested {args.limit} retry jobs, found only {len(selected)}"
        )
    for row in selected:
        row.pop("rejection_reason", None)
    atomic_jsonl(args.output.resolve(), selected)
    atomic_json(
        Path(str(args.output.resolve()) + ".manifest.json"),
        {
            "schema_version": 1,
            "source": str(args.rejected.resolve()),
            "source_sha256": sha256_file(args.rejected.resolve()),
            "split": args.split,
            "label": args.label,
            "seed": args.seed,
            "eligible": len(eligible),
            "selected": len(selected),
            "output_sha256": sha256_file(args.output.resolve()),
        },
    )
    print(f"Prepared {len(selected):,} local retry jobs: {args.output.resolve()}")


def prepare_neutral_augmentation(args: argparse.Namespace) -> None:
    source_jobs = load_jsonl(args.jobs.resolve())
    candidates = []
    seen: set[tuple[str, str]] = set()
    for row in source_jobs:
        key = tuple(str(value) for value in row.get("source_pair_key", []))
        if (
            row.get("split") != "train"
            or not row.get("requires_render")
            or row.get("label") not in {"consistent", "inconsistent"}
            or key in seen
            or not COMPONENT_RE.search(str(row.get("text1") or ""))
        ):
            continue
        seen.add(key)
        candidates.append(dict(row))
    random.Random(args.seed).shuffle(candidates)
    if len(candidates) < args.limit:
        raise RuntimeError(
            f"Requested {args.limit} neutral augmentation jobs, found {len(candidates)}"
        )
    output = []
    for index, source in enumerate(candidates[: args.limit]):
        job = {
            **source,
            "job_id": f"train-neutral-augmentation-{index:06d}",
            "label": "neutral",
            "neutral_type": "different_component_scope",
            "target_quota": args.accepted_target,
        }
        job.pop("contradiction_type", None)
        job["prompt"] = prompt_for(job)
        output.append(job)
    atomic_jsonl(args.output.resolve(), output)
    atomic_json(
        Path(str(args.output.resolve()) + ".manifest.json"),
        {
            "schema_version": 1,
            "source": str(args.jobs.resolve()),
            "source_sha256": sha256_file(args.jobs.resolve()),
            "seed": args.seed,
            "jobs": len(output),
            "accepted_target": args.accepted_target,
            "output_sha256": sha256_file(args.output.resolve()),
        },
    )
    print(f"Prepared {len(output):,} train neutral augmentation jobs")


def finalize_neutral_augmentation(args: argparse.Namespace) -> None:
    jobs = load_jsonl(args.jobs.resolve())
    responses: dict[str, dict[str, Any]] = {}
    response_paths = [path.resolve() for path in args.responses]
    for path in response_paths:
        for row in load_jsonl(path):
            if row["job_id"] in responses:
                raise ValueError(f"Duplicate augmentation response: {row['job_id']}")
            responses[row["job_id"]] = row
    holdout_texts = {
        normalized_text(row[field])
        for row in load_jsonl(args.holdout.resolve())
        for field in ("text1", "text2")
    }
    forbidden_texts = {
        normalized_text(row[field])
        for path in args.forbidden_split
        for row in load_jsonl(path.resolve())
        for field in ("text1", "text2")
    }
    accepted = []
    rejected = []
    seen_pairs: set[tuple[str, str]] = set()
    for job in jobs:
        response = responses.get(job["job_id"])
        parsed = (response or {}).get("parsed")
        if not isinstance(parsed, Mapping):
            rejected.append({**job, "rejection_reason": "missing_or_invalid_json"})
            continue
        valid, reason = validate_rendered(job, parsed)
        repair_method = None
        if not valid:
            repaired, repair_method = deterministic_repair(job, parsed)
            if repaired is not None:
                valid, repaired_reason = validate_rendered(job, repaired)
                if valid:
                    parsed = repaired
                    reason = f"{repaired_reason}:{repair_method}"
        text1 = str(job["text1"])
        text2 = str(parsed.get("text2") or "").strip()
        normalized = {normalized_text(text1), normalized_text(text2)}
        key = tuple(sorted(normalized))
        if normalized & holdout_texts:
            valid, reason = False, "heldout_text_overlap"
        elif normalized & forbidden_texts:
            valid, reason = False, "forbidden_split_text_overlap"
        elif key in seen_pairs:
            valid, reason = False, "duplicate_pair"
        if not valid:
            rejected.append({**job, "rejection_reason": reason})
            continue
        seen_pairs.add(key)
        accepted.append(
            {
                "id1": f"synthetic:{job['job_id']}:1",
                "id2": f"synthetic:{job['job_id']}:2",
                "text1": text1,
                "text2": text2,
                "label": "neutral",
                "pdf_file": "|".join(job["source_pdf_files"]),
                "source_pdf_files": job["source_pdf_files"],
                "synthetic_split": "train",
                "synthetic_generator": "candidate_shaped_local",
                "synthetic_job_id": job["job_id"],
                "synthetic_validation": reason,
                "synthetic_repair": repair_method,
                "synthetic_neutral_type": job["neutral_type"],
                "source_pair_key": job["source_pair_key"],
                "source_stratum": job["source_stratum"],
            }
        )
        if len(accepted) >= args.accepted_target:
            break
    atomic_jsonl(args.output.resolve(), accepted)
    rejected_path = Path(str(args.output.resolve()) + ".rejected.jsonl")
    atomic_jsonl(rejected_path, rejected)
    manifest = {
        "schema_version": 1,
        "jobs": str(args.jobs.resolve()),
        "responses": [str(path) for path in response_paths],
        "accepted_target": args.accepted_target,
        "accepted": len(accepted),
        "rejected_before_target": len(rejected),
        "heldout_overlap": 0,
        "forbidden_split_overlap": 0,
        "output_sha256": sha256_file(args.output.resolve()),
    }
    atomic_json(Path(str(args.output.resolve()) + ".manifest.json"), manifest)
    if len(accepted) < args.accepted_target:
        raise RuntimeError(
            f"Neutral augmentation deficit: {args.accepted_target - len(accepted)}"
        )
    print(f"Accepted {len(accepted):,} neutral augmentation pairs")


def rebuild_training_neutral_mix(args: argparse.Namespace) -> None:
    train = load_jsonl(args.train.resolve())
    dev = load_jsonl(args.dev.resolve())
    augmentation = load_jsonl(args.augmentation.resolve())
    non_neutral = [row for row in train if row["label"] != "neutral"]
    real_neutral = [row for row in train if row["label"] == "neutral"]
    rng = random.Random(args.seed)
    indexes = list(range(len(real_neutral)))
    rng.shuffle(indexes)
    kept_indexes = set(indexes[: args.real_neutral])
    kept_real = [row for index, row in enumerate(real_neutral) if index in kept_indexes]
    generated = augmentation[: args.generated_neutral]
    if len(kept_real) != args.real_neutral or len(generated) != args.generated_neutral:
        raise RuntimeError("Insufficient neutral rows for requested mix")
    output = [*non_neutral, *kept_real, *generated]
    counts = Counter(row["label"] for row in output)
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Rebuilt training labels are not balanced: {counts}")
    dev_texts = {
        normalized_text(row[field]) for row in dev for field in ("text1", "text2")
    }
    train_texts = {
        normalized_text(row[field]) for row in output for field in ("text1", "text2")
    }
    if train_texts & dev_texts:
        raise RuntimeError("Rebuilt training data overlaps dev text")
    atomic_jsonl(args.output.resolve(), output)
    atomic_json(
        Path(str(args.output.resolve()) + ".manifest.json"),
        {
            "schema_version": 1,
            "base_train": str(args.train.resolve()),
            "base_train_sha256": sha256_file(args.train.resolve()),
            "dev": str(args.dev.resolve()),
            "dev_sha256": sha256_file(args.dev.resolve()),
            "augmentation": str(args.augmentation.resolve()),
            "augmentation_sha256": sha256_file(args.augmentation.resolve()),
            "seed": args.seed,
            "counts": dict(counts),
            "real_neutral": len(kept_real),
            "generated_neutral": len(generated),
            "shared_dev_texts": 0,
            "output_sha256": sha256_file(args.output.resolve()),
        },
    )
    print(f"Rebuilt balanced training data: {args.output.resolve()}")


def _tokens(regex: re.Pattern, text: str) -> tuple[str, ...]:
    return tuple(sorted(match.group(0).casefold() for match in regex.finditer(text)))


def validate_rendered(job: Mapping[str, Any], parsed: Mapping[str, Any]) -> tuple[bool, str]:
    text1 = str(job.get("text1") or "")
    text2 = str(parsed.get("text2") or "").strip()
    if len(text2.split()) < 5:
        return False, "render_too_short"
    if normalized_text(text1) == normalized_text(text2):
        return False, "unchanged"
    if job["label"] == "consistent":
        if not parsed.get("preserved_scope"):
            return False, "scope_not_confirmed"
        for regex, name in (
            (NUMBER_RE, "number"),
            (NEGATION_RE, "negation"),
            (COMPONENT_RE, "component"),
            (PROTOCOL_RE, "protocol"),
        ):
            if _tokens(regex, text1) != _tokens(regex, text2):
                return False, f"consistent_{name}_changed"
        return True, "validated_consistent"
    if job["label"] == "neutral":
        if parsed.get("different_scope") is not True:
            return False, "different_scope_not_confirmed"
        for field in ("changed_field", "original_value", "new_value"):
            if not str(parsed.get(field) or "").strip():
                return False, f"missing_{field}"
        original = normalized_text(parsed.get("original_value"))
        replacement = normalized_text(parsed.get("new_value"))
        if (
            not original
            or not replacement
            or original == replacement
            or original not in normalized_text(text1)
            or replacement not in normalized_text(text2)
        ):
            return False, "ungrounded_changed_values"
        if _tokens(COMPONENT_RE, text1) == _tokens(COMPONENT_RE, text2):
            return False, "missing_neutral_component_evidence"
        for regex, name in (
            (NUMBER_RE, "number"),
            (NEGATION_RE, "negation"),
            (MODAL_RE, "modal"),
            (PROTOCOL_RE, "protocol"),
        ):
            if _tokens(regex, text1) != _tokens(regex, text2):
                return False, f"neutral_non_target_{name}_changed"
        return True, "validated_neutral_different_component"
    for field in ("changed_field", "original_value", "new_value"):
        if not str(parsed.get(field) or "").strip():
            return False, f"missing_{field}"
    original = normalized_text(parsed.get("original_value"))
    replacement = normalized_text(parsed.get("new_value"))
    if (
        not original
        or not replacement
        or original == replacement
        or original not in normalized_text(text1)
        or replacement not in normalized_text(text2)
    ):
        return False, "ungrounded_changed_values"
    category = str(job.get("contradiction_type"))
    evidence = {
        "polarity": _tokens(NEGATION_RE, text1) != _tokens(NEGATION_RE, text2),
        "numeric_value": _tokens(NUMBER_RE, text1) != _tokens(NUMBER_RE, text2),
        "comparator": _tokens(COMPARATOR_RE, text1) != _tokens(COMPARATOR_RE, text2),
        "actor_recipient": _tokens(COMPONENT_RE, text1) != _tokens(COMPONENT_RE, text2),
        "component_scope": _tokens(COMPONENT_RE, text1) != _tokens(COMPONENT_RE, text2),
        "protocol_requirement": _tokens(PROTOCOL_RE, text1) != _tokens(PROTOCOL_RE, text2),
        "procedure_state": (
            _tokens(NEGATION_RE, text1) != _tokens(NEGATION_RE, text2)
            or _tokens(MODAL_RE, text1) != _tokens(MODAL_RE, text2)
        ),
        "http_or_parameter": (
            _tokens(HTTP_RE, text1) != _tokens(HTTP_RE, text2)
            or _tokens(NUMBER_RE, text1) != _tokens(NUMBER_RE, text2)
        ),
    }
    if not evidence.get(category, False):
        return False, f"missing_{category}_evidence"
    target_slots = {
        "polarity": {"negation", "modal"},
        "numeric_value": {"number", "http"},
        "comparator": {"comparator"},
        "actor_recipient": {"component"},
        "component_scope": {"component"},
        "protocol_requirement": {"protocol", "http"},
        "procedure_state": {"negation", "modal"},
        "http_or_parameter": {"http", "number", "protocol"},
    }[category]
    slots = (
        (NUMBER_RE, "number"),
        (NEGATION_RE, "negation"),
        (MODAL_RE, "modal"),
        (COMPONENT_RE, "component"),
        (COMPARATOR_RE, "comparator"),
        (PROTOCOL_RE, "protocol"),
        (HTTP_RE, "http"),
    )
    for regex, name in slots:
        if name not in target_slots and _tokens(regex, text1) != _tokens(regex, text2):
            return False, f"non_target_{name}_changed"
    return True, f"validated_{category}"


def deterministic_repair(
    job: Mapping[str, Any], parsed: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Construct only mechanically provable fallbacks, then let validators recheck."""

    source = str(job.get("text1") or "")
    repaired = dict(parsed)
    if job["label"] == "consistent":
        substitutions = (
            (r"\bshall not\b", "is not permitted to"),
            (r"\bmust not\b", "is not permitted to"),
            (r"\bshall\b", "is required to"),
            (r"\bmust\b", "is required to"),
            (r"\bmay\b", "is permitted to"),
        )
        text2 = source
        for pattern, replacement in substitutions:
            text2, changes = re.subn(
                pattern, replacement, text2, count=1, flags=re.I
            )
            if changes:
                repaired = {"text2": text2, "preserved_scope": True}
                return repaired, "deterministic_modal_rewrite"
        return None, None

    original = str(parsed.get("original_value") or "").strip()
    replacement = str(parsed.get("new_value") or "").strip()
    if not original or not replacement or normalized_text(original) == normalized_text(replacement):
        return None, None
    text2, changes = re.subn(
        re.escape(original), replacement, source, count=1, flags=re.I
    )
    if not changes:
        return None, None
    repaired["text2"] = text2
    if job["label"] == "neutral":
        repaired["different_scope"] = True
    else:
        repaired["same_scope"] = True
    return repaired, "deterministic_grounded_span_replacement"


def finalize(args: argparse.Namespace) -> None:
    jobs = load_jsonl(args.jobs.resolve())
    response_paths = [path.resolve() for path in args.responses]
    responses: dict[str, dict[str, Any]] = {}
    response_retries = 0
    for response_path in response_paths:
        for row in load_jsonl(response_path):
            if row["job_id"] in responses:
                response_retries += 1
            responses[row["job_id"]] = row
    holdout = load_jsonl(args.holdout.resolve())
    heldout_texts = {
        normalized_text(row.get(field))
        for row in holdout
        for field in ("text1", "text2")
        if normalized_text(row.get(field))
    }
    valid_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for job in jobs:
        if job.get("requires_render"):
            response = responses.get(job["job_id"])
            parsed = (response or {}).get("parsed")
            if not isinstance(parsed, Mapping):
                rejected.append({**job, "rejection_reason": "missing_or_invalid_json"})
                continue
            valid, reason = validate_rendered(job, parsed)
            repair_method = None
            if not valid:
                repaired, repair_method = deterministic_repair(job, parsed)
                if repaired is not None:
                    repaired_valid, repaired_reason = validate_rendered(job, repaired)
                    if repaired_valid:
                        parsed = repaired
                        valid = True
                        reason = f"{repaired_reason}:{repair_method}"
            text2 = str(parsed.get("text2") or "").strip()
        else:
            valid, reason = True, "validated_variant_neutral"
            repair_method = None
            text2 = str(job["text2"])
        text1 = str(job["text1"])
        if normalized_text(text1) in heldout_texts or normalized_text(text2) in heldout_texts:
            valid, reason = False, "heldout_text_overlap"
        key = tuple(sorted((normalized_text(text1), normalized_text(text2))))
        if key in seen_pairs:
            valid, reason = False, "duplicate_pair"
        if not valid:
            rejected.append({**job, "rejection_reason": reason})
            continue
        seen_pairs.add(key)
        valid_rows.append(
            {
                "id1": f"synthetic:{job['job_id']}:1",
                "id2": f"synthetic:{job['job_id']}:2",
                "text1": text1,
                "text2": text2,
                "label": job["label"],
                "pdf_file": "|".join(job["source_pdf_files"]),
                "source_pdf_files": job["source_pdf_files"],
                "synthetic_split": job["split"],
                "synthetic_generator": "candidate_shaped_local",
                "synthetic_job_id": job["job_id"],
                "synthetic_validation": reason,
                "synthetic_repair": repair_method,
                "synthetic_contradiction_type": job.get("contradiction_type"),
                "source_pair_key": job["source_pair_key"],
                "source_stratum": job["source_stratum"],
            }
        )
    targets = {
        (str(job["split"]), str(job["label"])): int(job["target_quota"])
        for job in jobs
    }
    accepted: list[dict[str, Any]] = []
    accepted_counts: Counter = Counter()
    accepted_texts: dict[str, set[str]] = {"train": set(), "dev": set()}
    cross_split_text_rejections = 0
    for row in valid_rows:
        key = (row["synthetic_split"], row["label"])
        if accepted_counts[key] < targets[key]:
            other_split = "dev" if row["synthetic_split"] == "train" else "train"
            row_texts = {
                normalized_text(row["text1"]), normalized_text(row["text2"])
            }
            if row_texts & accepted_texts[other_split]:
                cross_split_text_rejections += 1
                rejected.append(
                    {
                        "synthetic_job_id": row["synthetic_job_id"],
                        "split": row["synthetic_split"],
                        "label": row["label"],
                        "rejection_reason": "cross_split_text_overlap",
                    }
                )
                continue
            accepted.append(row)
            accepted_counts[key] += 1
            accepted_texts[row["synthetic_split"]].update(row_texts)
        else:
            rejected.append(
                {
                    "synthetic_job_id": row["synthetic_job_id"],
                    "split": row["synthetic_split"],
                    "label": row["label"],
                    "rejection_reason": "valid_surplus_not_selected",
                }
            )
    by_split = {
        split: [row for row in accepted if row["synthetic_split"] == split]
        for split in ("train", "dev")
    }
    expected = Counter(targets)
    actual = Counter((row["synthetic_split"], row["label"]) for row in accepted)
    deficits = {
        f"{split}:{label}": expected[(split, label)] - actual[(split, label)]
        for split in ("train", "dev")
        for label in LABELS
        if actual[(split, label)] != expected[(split, label)]
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_pairs.jsonl"
    dev_path = output_dir / "development_pairs.jsonl"
    rejected_path = output_dir / "rejected_generations.jsonl"
    atomic_jsonl(train_path, by_split["train"])
    atomic_jsonl(dev_path, by_split["dev"])
    atomic_jsonl(rejected_path, rejected)
    manifest = {
        "schema_version": 1,
        "jobs": str(args.jobs.resolve()),
        "jobs_sha256": sha256_file(args.jobs.resolve()),
        "responses": [str(path) for path in response_paths],
        "responses_sha256": {path.name: sha256_file(path) for path in response_paths},
        "response_retries_overriding_initial": response_retries,
        "accepted": len(accepted),
        "valid_before_quota": len(valid_rows),
        "rejected": len(rejected),
        "expected_counts": {f"{split}:{label}": expected[(split, label)] for split in ("train", "dev") for label in LABELS},
        "accepted_counts": {f"{split}:{label}": actual[(split, label)] for split in ("train", "dev") for label in LABELS},
        "deficits": deficits,
        "leakage": {
            "heldout_text_overlap": 0,
            "duplicate_pairs": 0,
            "cross_split_text_overlap": 0,
            "cross_split_text_candidates_rejected": cross_split_text_rejections,
        },
        "outputs": {
            "train": str(train_path),
            "dev": str(dev_path),
            "rejected": str(rejected_path),
        },
        "output_sha256": {
            "train": sha256_file(train_path),
            "dev": sha256_file(dev_path),
            "rejected": sha256_file(rejected_path),
        },
        "quality_gate_passed": not deficits,
    }
    atomic_json(output_dir / "finalization_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if deficits:
        raise RuntimeError(f"Synthetic quality gate failed: {deficits}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    prepare.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    prepare.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--train-per-class", type=int, default=4000)
    prepare.add_argument("--dev-per-class", type=int, default=1000)
    prepare.add_argument("--render-surplus", type=float, default=DEFAULT_RENDER_SURPLUS)
    prepare.add_argument("--min-source-words", type=int, default=8)
    prepare.add_argument("--max-source-words", type=int, default=120)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dev-fraction", type=float, default=0.35)
    render = subparsers.add_parser("render")
    render.add_argument("--jobs", type=Path, required=True)
    render.add_argument("--responses", type=Path, required=True)
    render.add_argument(
        "--adapter-checkpoint", "--adapter", dest="adapter",
        type=Path, default=DEFAULT_ADAPTER,
    )
    render.add_argument(
        "--base-model",
        help="Optional public model identifier or compatible local base checkpoint.",
    )
    render.add_argument("--batch-size", type=int, default=4)
    render.add_argument("--max-new-tokens", type=int, default=256)
    render.add_argument("--seed", type=int, default=42)
    render.add_argument("--shard-count", type=int, default=1)
    render.add_argument("--shard-index", type=int, default=0)
    retry = subparsers.add_parser("prepare-retry")
    retry.add_argument("--rejected", type=Path, required=True)
    retry.add_argument("--output", type=Path, required=True)
    retry.add_argument("--split", choices=("train", "dev"))
    retry.add_argument("--label", choices=LABELS)
    retry.add_argument("--limit", type=int, required=True)
    retry.add_argument("--seed", type=int, default=3407)
    neutral = subparsers.add_parser("prepare-neutral-augmentation")
    neutral.add_argument("--jobs", type=Path, required=True)
    neutral.add_argument("--output", type=Path, required=True)
    neutral.add_argument("--limit", type=int, default=6000)
    neutral.add_argument("--accepted-target", type=int, default=1660)
    neutral.add_argument("--seed", type=int, default=42)
    neutral_finish = subparsers.add_parser("finalize-neutral-augmentation")
    neutral_finish.add_argument("--jobs", type=Path, required=True)
    neutral_finish.add_argument("--responses", type=Path, nargs="+", required=True)
    neutral_finish.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    neutral_finish.add_argument("--forbidden-split", type=Path, nargs="+", required=True)
    neutral_finish.add_argument("--output", type=Path, required=True)
    neutral_finish.add_argument("--accepted-target", type=int, default=1660)
    rebuild = subparsers.add_parser("rebuild-neutral-mix")
    rebuild.add_argument("--train", type=Path, required=True)
    rebuild.add_argument("--dev", type=Path, required=True)
    rebuild.add_argument("--augmentation", type=Path, required=True)
    rebuild.add_argument("--output", type=Path, required=True)
    rebuild.add_argument("--real-neutral", type=int, default=2340)
    rebuild.add_argument("--generated-neutral", type=int, default=1660)
    rebuild.add_argument("--seed", type=int, default=42)
    finish = subparsers.add_parser("finalize")
    finish.add_argument("--jobs", type=Path, required=True)
    finish.add_argument("--responses", type=Path, nargs="+", required=True)
    finish.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    finish.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_jobs(args)
    elif args.command == "render":
        render_jobs(args)
    elif args.command == "prepare-retry":
        prepare_retry_jobs(args)
    elif args.command == "prepare-neutral-augmentation":
        prepare_neutral_augmentation(args)
    elif args.command == "finalize-neutral-augmentation":
        finalize_neutral_augmentation(args)
    elif args.command == "rebuild-neutral-mix":
        rebuild_training_neutral_mix(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
