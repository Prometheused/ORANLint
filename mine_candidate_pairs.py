#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Mine TF-IDF/BGE candidate pairs from the security-filtered O-RAN corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_JSONL = (
    PIPELINE_DIR
    / "data/generated/security_segments.jsonl"
)
DEFAULT_CLUSTER_PATH = (
    PIPELINE_DIR
    / "data/generated/clusters/bge_kmeans_clusters.jsonl"
)
DEFAULT_BGE_MODEL = (
    PIPELINE_DIR / "models/sentence_encoder/phase_triplet"
)
DEFAULT_OUTPUT = (
    PIPELINE_DIR
    / "data/generated/candidate_pairs.jsonl"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path):
    root = Path(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def ids_sha256(ids):
    digest = hashlib.sha256()
    for identifier in ids:
        digest.update(str(identifier).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_cached_embeddings(cache_path, corpus_path, ids, model_path, dtype):
    """Load and validate normalized embeddings produced during clustering."""
    cache_path = Path(cache_path).expanduser().resolve()
    manifest_path = Path(str(cache_path) + ".manifest.json")
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Embedding cache and its .manifest.json sidecar are both required"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "corpus_sha256": sha256_file(corpus_path),
        "ids_sha256": ids_sha256(ids),
        "model_sha256": sha256_tree(model_path),
        "records": len(ids),
        "inference_dtype": dtype,
        "embedding_cache_sha256": sha256_file(cache_path),
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError("Embedding cache manifest mismatch: " + ", ".join(mismatches))
    embeddings = np.load(cache_path, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError("Embedding cache shape does not match the corpus")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embedding cache contains non-finite values")
    return torch.from_numpy(embeddings.astype(np.float32, copy=False)), manifest_path


def load_corpus(jsonl_path):
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_number}") from exc
            if not entry.get("text"):
                continue
            entries.append(entry)

    corpus = [entry["text"] for entry in entries]
    ids = [entry.get("id") for entry in entries]
    if any(identifier is None for identifier in ids):
        raise ValueError("Every corpus entry must contain an id.")
    if len({str(identifier) for identifier in ids}) != len(ids):
        raise ValueError("Corpus contains duplicate ids; refusing to misalign clusters and texts.")
    return entries, corpus, ids


def load_cluster_ids(cluster_path, ids):
    """Load bge_kmeans_clusters.jsonl and align labels to corpus order."""
    textid_to_cluster = {}
    with open(cluster_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            cluster_entry = json.loads(line)
            cluster_id = int(cluster_entry["cluster"])
            for member in cluster_entry.get("members", []):
                if isinstance(member, str):
                    member = json.loads(member)
                textid_to_cluster[str(member["text_id"])] = cluster_id

    missing = [identifier for identifier in ids if str(identifier) not in textid_to_cluster]
    if missing:
        preview = ", ".join(str(identifier) for identifier in missing[:5])
        raise ValueError(
            f"{len(missing)} corpus ids are missing from the cluster file; examples: {preview}"
        )

    return np.asarray([textid_to_cluster[str(identifier)] for identifier in ids], dtype=np.int32)


def collect_tfidf_pairs(corpus, tfidf_min, tfidf_max, block_size):
    """Collect upper-triangle TF-IDF pairs without materializing the full product."""
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", norm="l2")
    matrix = vectorizer.fit_transform(corpus)
    matrix_t = matrix.T.tocsc()

    rows_out = []
    cols_out = []
    values_out = []

    for start in range(0, matrix.shape[0], block_size):
        stop = min(start + block_size, matrix.shape[0])
        similarities = (matrix[start:stop] @ matrix_t).tocoo()
        global_rows = similarities.row + start
        mask = (
            (global_rows < similarities.col)
            & (similarities.data >= tfidf_min)
            & (similarities.data < tfidf_max)
        )
        if np.any(mask):
            rows_out.append(global_rows[mask].astype(np.int64, copy=False))
            cols_out.append(similarities.col[mask].astype(np.int64, copy=False))
            values_out.append(similarities.data[mask].astype(np.float32, copy=False))

    if not rows_out:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    return (
        np.concatenate(rows_out),
        np.concatenate(cols_out),
        np.concatenate(values_out),
    )


def select_candidate_pairs(
    rows,
    cols,
    tfidf_values,
    cluster_ids,
    same_cluster_only=True,
    cross_cluster_top_k=0,
):
    """Apply the cluster gate plus a deterministic cross-cluster recall backstop."""

    if cross_cluster_top_k < 0:
        raise ValueError("cross_cluster_top_k cannot be negative")
    if cross_cluster_top_k and not same_cluster_only:
        raise ValueError("cross_cluster_top_k cannot be combined with --all_clusters")

    same_mask = cluster_ids[rows] == cluster_ids[cols]
    same_indices = np.flatnonzero(same_mask)
    cross_indices = np.flatnonzero(~same_mask)
    selected_cross = set()

    if same_cluster_only and cross_cluster_top_k:
        neighbors = [[] for _ in range(len(cluster_ids))]
        for pair_index in cross_indices.tolist():
            first = int(rows[pair_index])
            second = int(cols[pair_index])
            score = float(tfidf_values[pair_index])
            neighbors[first].append((score, second, pair_index))
            neighbors[second].append((score, first, pair_index))
        for endpoint_neighbors in neighbors:
            endpoint_neighbors.sort(key=lambda item: (-item[0], item[1], item[2]))
            selected_cross.update(
                pair_index
                for _, _, pair_index in endpoint_neighbors[:cross_cluster_top_k]
            )

    if same_cluster_only:
        selected_indices = np.asarray(
            sorted(set(same_indices.tolist()) | selected_cross), dtype=np.int64
        )
    else:
        selected_indices = np.arange(rows.size, dtype=np.int64)

    selected_same = same_mask[selected_indices]
    sources = np.where(
        selected_same,
        "same_cluster",
        "cross_cluster_tfidf_topk" if same_cluster_only else "all_clusters",
    )
    stats = {
        "same_cluster_pairs": int(same_mask.sum()),
        "cross_cluster_pairs_available": int((~same_mask).sum()),
        "cross_cluster_backstop_added": int((~selected_same).sum()),
        "pairs_after_candidate_selection": int(selected_indices.size),
    }
    return (
        rows[selected_indices],
        cols[selected_indices],
        tfidf_values[selected_indices],
        sources,
        stats,
    )


def batched_cosine(embeddings, idx_a, idx_b, batch_size=200_000):
    if idx_a.numel() == 0:
        return np.empty(0, dtype=np.float32)

    scores = []
    for start in range(0, idx_a.numel(), batch_size):
        stop = min(start + batch_size, idx_a.numel())
        first = embeddings.index_select(0, idx_a[start:stop])
        second = embeddings.index_select(0, idx_b[start:stop])
        scores.append((first * second).sum(dim=1))
    return torch.cat(scores, dim=0).detach().cpu().numpy()


def metadata_fields(entry, prefix):
    fields = {
        f"{prefix}_pdf_file": entry.get("pdf_file"),
        f"{prefix}_section_number": entry.get("section_number"),
        f"{prefix}_section_title": entry.get("section_title"),
        f"{prefix}_section_level": entry.get("section_level"),
        f"{prefix}_page_number": entry.get("page_number"),
        f"{prefix}_ancestors": entry.get("ancestors"),
        f"{prefix}_kw_count": entry.get("kw_count"),
        f"{prefix}_metadata": entry.get("metadata", {}),
    }
    return fields


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine security-filtered O-RAN candidate pairs with TF-IDF and BGE."
    )
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL), help="Filtered corpus JSONL")
    parser.add_argument(
        "--cluster_path",
        default=str(DEFAULT_CLUSTER_PATH),
        help="Cluster file produced by cluster_security_segments.py",
    )
    parser.add_argument(
        "--tfidf_min",
        type=float,
        default=0.5,
        help="Lower TF-IDF cosine threshold, inclusive.",
    )
    parser.add_argument(
        "--tfidf_max",
        type=float,
        default=1.0,
        help="Upper TF-IDF cosine threshold, exclusive.",
    )
    parser.add_argument(
        "--tfidf_block_size",
        type=int,
        default=1024,
        help="Rows processed per sparse TF-IDF block.",
    )
    parser.add_argument(
        "--cross-cluster-top-k",
        type=int,
        default=1,
        help=(
            "Also retain each paragraph's top-k cross-cluster TF-IDF neighbors; "
            "the default retains one neighbor per paragraph."
        ),
    )
    parser.add_argument(
        "--sentence-encoder-checkpoint", "--bge-model",
        dest="bge_model", default=str(DEFAULT_BGE_MODEL),
        help="Trained sentence-encoder checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="BGE encoding batch size")
    parser.add_argument(
        "--device",
        default="auto",
        help="SentenceTransformer device when encoding, e.g. auto, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp16", "fp32"),
        default="auto",
        help="BGE inference dtype; must match the cache manifest when a cache is used.",
    )
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="Validated raw embedding .npy cache produced by cluster_security_segments.py.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output candidate-pair JSONL")
    parser.add_argument("--manifest", default=None, help="Output manifest; defaults to <output>.manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tfidf_block_size <= 0:
        raise ValueError("--tfidf_block_size must be positive")
    if args.cross_cluster_top_k < 0:
        raise ValueError("--cross-cluster-top-k cannot be negative")

    input_path = Path(args.jsonl).expanduser().resolve()
    cluster_path = Path(args.cluster_path).expanduser().resolve()
    model_path = Path(args.bge_model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else Path(str(output_path) + ".manifest.json")
    )
    for path in (input_path, cluster_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if output_path.resolve() in {input_path.resolve(), cluster_path.resolve()}:
        raise ValueError("Output must differ from inputs")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {', '.join(map(str, existing))}")

    print(f"Corpus: {args.jsonl}")
    print(f"Cluster file: {args.cluster_path}")
    print(f"BGE model: {args.bge_model}")
    print(f"TF-IDF range: [{args.tfidf_min}, {args.tfidf_max})")
    print("Same-cluster restriction: enabled")
    print(f"Cross-cluster TF-IDF top-k backstop: {args.cross_cluster_top_k}")

    entries, corpus, ids = load_corpus(args.jsonl)
    print(f"Loaded {len(corpus)} texts")

    rows, cols, tfidf_values = collect_tfidf_pairs(
        corpus,
        tfidf_min=args.tfidf_min,
        tfidf_max=args.tfidf_max,
        block_size=args.tfidf_block_size,
    )
    tfidf_pair_count = int(rows.size)
    print(f"Pairs after TF-IDF range: {rows.size:,}")

    cluster_ids = load_cluster_ids(args.cluster_path, ids)
    rows, cols, tfidf_values, candidate_sources, selection_stats = (
        select_candidate_pairs(
            rows,
            cols,
            tfidf_values,
            cluster_ids,
            same_cluster_only=True,
            cross_cluster_top_k=args.cross_cluster_top_k,
        )
    )
    print(
        "Pairs after candidate selection: "
        f"{rows.size:,} (same-cluster={selection_stats['same_cluster_pairs']:,}, "
        "cross-cluster backstop="
        f"{selection_stats['cross_cluster_backstop_added']:,})"
    )

    cache_manifest_path = None
    if args.embedding_cache:
        embeddings, cache_manifest_path = load_cached_embeddings(
            args.embedding_cache,
            input_path,
            ids,
            model_path,
            args.dtype,
        )
        print("Loaded validated BGE embedding cache:", args.embedding_cache)
    else:
        model_kwargs = {}
        if args.dtype == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif args.dtype == "fp32":
            model_kwargs["torch_dtype"] = torch.float32
        model = SentenceTransformer(
            str(model_path),
            device=None if args.device == "auto" else args.device,
            model_kwargs=model_kwargs,
        )
        print("Loaded model on device:", model.device)
        embeddings = model.encode(
            corpus,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    bge_values = batched_cosine(
        embeddings,
        torch.as_tensor(rows, dtype=torch.long, device=embeddings.device),
        torch.as_tensor(cols, dtype=torch.long, device=embeddings.device),
    )

    order = np.argsort(-bge_values, kind="stable")
    rows = rows[order]
    cols = cols[order]
    tfidf_values = tfidf_values[order]
    bge_values = bge_values[order]
    candidate_sources = candidate_sources[order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_output.open("w", encoding="utf-8") as out:
        for row, col, tfidf_cos, bge_cos, candidate_source in zip(
            rows, cols, tfidf_values, bge_values, candidate_sources
        ):
            first = entries[int(row)]
            second = entries[int(col)]
            first_cluster = int(cluster_ids[row])
            second_cluster = int(cluster_ids[col])
            result = {
                "id1": first["id"],
                "id2": second["id"],
                "text1": first["text"],
                "text2": second["text"],
                "tfidf_cos": float(tfidf_cos),
                "bge_cos": float(bge_cos),
                "cluster_id": first_cluster if first_cluster == second_cluster else None,
                "id1_cluster_id": first_cluster,
                "id2_cluster_id": second_cluster,
                "candidate_source": str(candidate_source),
            }
            result.update(metadata_fields(first, "id1"))
            result.update(metadata_fields(second, "id2"))
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary_output.replace(output_path)

    pair_keys = {(int(row), int(col)) for row, col in zip(rows, cols)}
    if len(pair_keys) != rows.size:
        raise RuntimeError("Candidate miner produced duplicate pairs")
    same_selected = cluster_ids[rows] == cluster_ids[cols]
    valid_sources = np.where(
        same_selected,
        candidate_sources == "same_cluster",
        np.isin(candidate_sources, ("cross_cluster_tfidf_topk", "all_clusters")),
    )
    if not np.all(valid_sources):
        raise RuntimeError("Candidate miner produced invalid candidate provenance")
    if not np.all(np.isfinite(tfidf_values)) or not np.all(np.isfinite(bge_values)):
        raise RuntimeError("Candidate miner produced non-finite scores")

    manifest = {
        "schema_version": 1,
        "script": str(Path(__file__).resolve()),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "cluster_file": str(cluster_path),
        "cluster_sha256": sha256_file(cluster_path),
        "bge_model": str(model_path),
        "bge_model_sha256": sha256_tree(model_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "corpus_records": len(corpus),
        "pairs_after_tfidf": tfidf_pair_count,
        "pairs_after_cluster_filter": selection_stats["same_cluster_pairs"],
        "pairs_after_candidate_selection": int(rows.size),
        "cross_cluster_pairs_available": selection_stats[
            "cross_cluster_pairs_available"
        ],
        "cross_cluster_backstop_added": selection_stats[
            "cross_cluster_backstop_added"
        ],
        "cross_cluster_top_k": args.cross_cluster_top_k,
        "same_cluster_only": True,
        "tfidf_min": args.tfidf_min,
        "tfidf_max_exclusive_before_float32_serialization": args.tfidf_max,
        "tfidf_scores_rounded_to_upper_bound": int(np.sum(tfidf_values == args.tfidf_max)),
        "tfidf_block_size": args.tfidf_block_size,
        "bge_batch_size": args.batch_size,
        "bge_device": args.device,
        "bge_inference_dtype": args.dtype,
        "embedding_cache": str(Path(args.embedding_cache).expanduser().resolve()) if args.embedding_cache else None,
        "embedding_cache_sha256": sha256_file(args.embedding_cache) if args.embedding_cache else None,
        "embedding_cache_manifest": str(cache_manifest_path) if cache_manifest_path else None,
        "ordered_by": "bge_cos descending stable",
        "validation": {
            "unique_pairs": len(pair_keys) == rows.size,
            "candidate_provenance": bool(np.all(valid_sources)),
            "finite_scores": bool(np.all(np.isfinite(tfidf_values)) and np.all(np.isfinite(bge_values))),
        },
    }
    atomic_json(manifest_path, manifest)

    print(f"Wrote {rows.size:,} candidate pairs to {output_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
