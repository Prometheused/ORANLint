#!/usr/bin/env python3
# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

import os, json, argparse, hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

# Optional: only needed for BGE representation
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_JSONL = str(
    PIPELINE_DIR
    / "data/generated/security_segments.jsonl"
)
DEFAULT_BGE_MODEL = str(
    PIPELINE_DIR / "models/sentence_encoder/phase_triplet"
)
DEFAULT_OUTDIR = str(
    PIPELINE_DIR
    / "data/generated/clusters"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
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


def ids_sha256(ids):
    digest = hashlib.sha256()
    for identifier in ids:
        digest.update(str(identifier).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(to_json_safe(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_or_encode_bge(
    corpus,
    ids,
    model_path,
    batch_size,
    cache_path=None,
    corpus_sha256=None,
    device="auto",
    dtype="auto",
):
    """Load a validated raw BGE embedding cache or encode and save it."""

    model_hash = sha256_tree(model_path)
    id_hash = ids_sha256(ids)
    cache_manifest_path = None
    if cache_path:
        cache_path = Path(cache_path).expanduser().resolve()
        cache_manifest_path = Path(str(cache_path) + ".manifest.json")
        if cache_path.exists() != cache_manifest_path.exists():
            raise FileNotFoundError(
                "Embedding cache and manifest must either both exist or both be absent"
            )
        if cache_path.exists():
            manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            expected = {
                "corpus_sha256": corpus_sha256,
                "ids_sha256": id_hash,
                "model_sha256": model_hash,
                "records": len(corpus),
                "inference_dtype": dtype,
            }
            mismatches = [
                key for key, value in expected.items() if manifest.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    "Embedding cache manifest mismatch: " + ", ".join(mismatches)
                )
            embeddings = np.load(cache_path, allow_pickle=False)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(corpus):
                raise ValueError("Embedding cache shape does not match the corpus")
            if not np.all(np.isfinite(embeddings)):
                raise ValueError("Embedding cache contains non-finite values")
            print(f"[cache] loaded BGE embeddings -> {cache_path}")
            return embeddings.astype(np.float32, copy=False), {
                "model_name": model_path,
                "model_sha256": model_hash,
                "embedding_cache": str(cache_path),
                "embedding_cache_sha256": sha256_file(cache_path),
                "embedding_cache_reused": True,
                "inference_dtype": dtype,
            }

    embeddings, aux = rep_bge(
        corpus,
        model_path=model_path,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    aux["model_sha256"] = model_hash
    aux["embedding_cache"] = str(cache_path) if cache_path else None
    aux["embedding_cache_reused"] = False
    aux["inference_dtype"] = dtype
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
        os.replace(temporary, cache_path)
        cache_hash = sha256_file(cache_path)
        atomic_json(
            cache_manifest_path,
            {
                "schema_version": 1,
                "corpus_sha256": corpus_sha256,
                "ids_sha256": id_hash,
                "model": str(Path(model_path).expanduser().resolve()),
                "model_sha256": model_hash,
                "records": len(corpus),
                "inference_dtype": dtype,
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "embedding_cache_sha256": cache_hash,
            },
        )
        aux["embedding_cache_sha256"] = cache_hash
        print(f"[cache] wrote BGE embeddings -> {cache_path}")
    return embeddings, aux


# -------------------- Utils --------------------

def rep_bge(
    corpus,
    model_path,
    batch_size=64,
    device="auto",
    dtype="auto",
):
    """Encode text with the trained BGE checkpoint and normalize each row."""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required for BGE mode.")
    model_kwargs = {}
    if dtype == "fp16":
        model_kwargs["torch_dtype"] = torch.float16
    elif dtype == "fp32":
        model_kwargs["torch_dtype"] = torch.float32
    model = SentenceTransformer(
        model_path,
        device=None if device == "auto" else device,
        model_kwargs=model_kwargs,
    )
    embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = embeddings.astype(np.float32)
    embeddings = normalize(embeddings, norm="l2", axis=1, copy=False)
    return embeddings, {"model_name": model_path}


def to_json_safe(x):
    # Recursively convert numpy / torch types to builtins
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
            elif isinstance(k, (np.ndarray, torch.Tensor, list, tuple, set, dict)):
                k = str(k)
            out[k] = to_json_safe(v)
        return out
    elif isinstance(x, list):
        return [to_json_safe(v) for v in x]
    elif isinstance(x, tuple) or isinstance(x, set):
        return [to_json_safe(v) for v in x]
    elif isinstance(x, np.integer):
        return int(x)
    elif isinstance(x, np.floating):
        return float(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    else:
        return x


# -------------------- I/O --------------------

def load_corpus(jsonl_path):
    ex_list, corpus, ids, metas = [], [], [], []
    pdfs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            ex_list.append(ex)
            corpus.append(ex["text"])
            ids.append(ex.get("id"))
            pdfs.append(ex.get("pdf_file"))
            metas.append(ex.get("metadata", {}))
    return ex_list, corpus, ids, metas, pdfs


def save_assignments(out_path, ids, texts, labels, pdfs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clusters = defaultdict(list)
    cluster_file_name = out_path.replace("assignments", "clusters")
    temporary = os.path.join(os.path.dirname(out_path), f".{os.path.basename(out_path)}.tmp")
    cluster_temporary = os.path.join(
        os.path.dirname(cluster_file_name), f".{os.path.basename(cluster_file_name)}.tmp"
    )
    with open(temporary, "w", encoding="utf-8") as w, open(cluster_temporary, "w", encoding="utf-8") as w_id:
        for id_, t, lab, pdf in zip(ids, texts, labels, pdfs):
            rec = json.dumps({"text_id": id_, "cluster": int(lab), "pdf_file": pdf, "text": t})
            clusters[int(lab)].append(rec)
            w.write(rec + "\n")
        for lab, recs in clusters.items():
            w_id.write(json.dumps({"cluster": lab, "members": recs}) + "\n")
    os.replace(temporary, out_path)
    os.replace(cluster_temporary, cluster_file_name)
    print(f"[write] assignments -> {out_path}")


def save_cluster_summary(out_path, summaries):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summaries = to_json_safe(summaries)
    temporary = os.path.join(os.path.dirname(out_path), f".{os.path.basename(out_path)}.tmp")
    with open(temporary, "w", encoding="utf-8") as w:
        json.dump(summaries, w, ensure_ascii=False, indent=2)
    os.replace(temporary, out_path)
    print(f"[write] cluster summary -> {out_path}")


# -------------------- Representation helpers --------------------

def pca_reduce(Z, pca_dim=100, random_state=0):
    """
    PCA + L2 normalize. Very helpful for high-dim embeddings.
    """
    if pca_dim is None or pca_dim <= 0 or pca_dim >= Z.shape[1]:
        return Z, None
    pca = PCA(n_components=pca_dim, random_state=random_state)
    Z2 = pca.fit_transform(Z)
    Z2 = normalize(Z2, norm="l2", axis=1, copy=False)
    return Z2.astype(np.float32), pca


def cluster_kmeans(Z, k=20):
    """Assign every row using the fixed K-means configuration."""
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=0,
        batch_size=4096,
        n_init="auto",
    )
    labels = model.fit_predict(Z)
    return labels, {"centroids": model.cluster_centers_, "k": k}




# -------------------- Summaries --------------------

def summarize_clusters(ids, texts, labels, aux_rep, Z=None, aux_cluster=None, n_exemplars=5):
    """
    Returns a dict: {cluster_id: {"size": int, "example_ids": [...], "example_texts": [...], ...}}

    Improvement:
      - If Z is provided:
          - For kmeans: choose exemplars closest to centroid (cosine)
          - For others: choose exemplars closest to cluster-mean vector (cosine)
      - For TF-IDF: still adds top_terms per cluster.
    """
    groups = defaultdict(list)
    for i, (id_, t, lab) in enumerate(zip(ids, texts, labels)):
        groups[int(lab)].append(i)

    summaries = {}

    # -------- exemplar selection using embeddings --------
    # We assume Z is L2-normalized row-wise. (Your code ensures this.)
    centroids = None
    if aux_cluster is not None and isinstance(aux_cluster, dict) and "centroids" in aux_cluster:
        centroids = np.asarray(aux_cluster["centroids"], dtype=np.float32)
        # Normalize centroids for cosine similarity
        centroids = normalize(centroids, norm="l2", axis=1, copy=False)

    def pick_exemplars_for_cluster(lab, idxs):
        # Fallback: first n texts
        if Z is None or len(idxs) == 0:
            return idxs[:n_exemplars], None

        # If we have kmeans centroids and the label is a valid cluster id
        if centroids is not None and lab >= 0 and lab < centroids.shape[0]:
            c = centroids[lab]  # (D,)
        else:
            # No centroids (e.g., DBSCAN): use cluster mean as proxy centroid
            sub = Z[idxs]                       # (m, D)
            c = sub.mean(axis=0, dtype=np.float32)
            c = c / (np.linalg.norm(c) + 1e-12)

        sub = Z[idxs]                           # (m, D)
        sims = sub @ c                          # cosine sims because both normalized
        # pick top n_exemplars
        top_local = np.argsort(-sims)[:n_exemplars]
        chosen = [idxs[j] for j in top_local.tolist()]
        chosen_sims = [float(sims[j]) for j in top_local.tolist()]
        return chosen, chosen_sims

    for lab, idxs in groups.items():
        ex_idxs, ex_sims = pick_exemplars_for_cluster(lab, idxs)
        summaries[int(lab)] = {
            "size": len(idxs),
            "example_ids": [ids[i] for i in ex_idxs],
            "example_texts": [texts[i] for i in ex_idxs],
        }
        if ex_sims is not None:
            summaries[int(lab)]["example_cosine_to_center"] = ex_sims

    return summaries


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cluster security-filtered segments using fixed PCA and K-means."
    )
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--pca-dim", type=int, default=100)
    parser.add_argument(
        "--sentence-encoder-checkpoint", "--bge-model",
        dest="bge_model", default=DEFAULT_BGE_MODEL,
        help="Trained sentence-encoder checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--embedding-cache")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.k <= 1:
        raise ValueError("--k must be greater than one")
    if args.pca_dim <= 0:
        raise ValueError("--pca-dim must be positive")

    input_path = Path(args.jsonl).expanduser().resolve()
    model_path = Path(args.bge_model).expanduser().resolve()
    output_dir = Path(args.outdir).expanduser().resolve()
    assignment_path = output_dir / "bge_kmeans_assignments.jsonl"
    cluster_path = output_dir / "bge_kmeans_clusters.jsonl"
    summary_path = output_dir / "bge_kmeans_summary.json"
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output_dir / "bge_kmeans_manifest.json"
    )
    existing = [
        path for path in (assignment_path, cluster_path, summary_path, manifest_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, existing)))
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    records, corpus, ids, metadata, pdfs = load_corpus(input_path)
    embeddings, representation = load_or_encode_bge(
        corpus,
        ids,
        model_path=str(model_path),
        batch_size=args.batch_size,
        cache_path=args.embedding_cache,
        corpus_sha256=sha256_file(input_path),
        device=args.device,
        dtype=args.dtype,
    )
    reduced, _ = pca_reduce(embeddings, pca_dim=args.pca_dim, random_state=0)
    representation["pca_dim"] = args.pca_dim
    labels, clustering = cluster_kmeans(reduced, k=args.k)
    counts = Counter(labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_assignments(str(assignment_path), ids, corpus, labels, pdfs)
    summaries = summarize_clusters(
        ids, corpus, labels, aux_rep=representation, Z=reduced,
        aux_cluster=clustering, n_exemplars=5,
    )
    summaries["_meta"] = {
        "records": len(corpus),
        "representation": "bge",
        "algorithm": "kmeans",
        "pca_dim": args.pca_dim,
        "k": args.k,
        "cluster_counts": dict(sorted(counts.items())),
    }
    save_cluster_summary(str(summary_path), summaries)

    manifest = {
        "schema_version": 1,
        "script": Path(__file__).name,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "sentence_encoder_checkpoint": str(model_path),
        "sentence_encoder_sha256": sha256_tree(model_path),
        "records": len(corpus),
        "pca_dim": args.pca_dim,
        "k": args.k,
        "cluster_counts": dict(sorted(counts.items())),
        "outputs": {
            "assignments": {"path": str(assignment_path), "sha256": sha256_file(assignment_path)},
            "clusters": {"path": str(cluster_path), "sha256": sha256_file(cluster_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    atomic_json(manifest_path, manifest)
    print(f"[done] clustered {len(corpus):,} records into {args.k} clusters")


if __name__ == "__main__":
    main()
