from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorly as tl
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn.cluster import KMeans, SpectralCoclustering
from sklearn.metrics import homogeneity_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorly.decomposition import non_negative_parafac


tl.set_backend("numpy")
_EPS = 1e-10
_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def _preprocess_tokens(text: str, stemmer: PorterStemmer) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [tok for tok in tokens if tok not in _STOP_WORDS]
    return [stemmer.stem(tok) for tok in tokens]


def _load_and_prepare_articles(
    csv_path: Path,
    text_col: str,
    label_col: str,
    language_col: str,
    language_filter: str | None,
    max_docs: int,
    min_words: int,
    exclude_labels: list[str],
    per_class_samples: int,
    seed: int,
) -> pd.DataFrame:
    usecols = [text_col, label_col, language_col]
    chunks = []
    total = 0
    stemmer = PorterStemmer()

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=5000, low_memory=False):
        chunk = chunk.dropna(subset=[text_col, label_col])

        if language_filter and language_col in chunk.columns:
            chunk = chunk[chunk[language_col].astype(str).str.lower() == language_filter.lower()]
        if chunk.empty:
            continue

        chunk[text_col] = chunk[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        chunk[label_col] = chunk[label_col].astype(str).str.lower().str.strip()

        if exclude_labels:
            excluded = {x.lower().strip() for x in exclude_labels}
            chunk = chunk[~chunk[label_col].isin(excluded)]
        if chunk.empty:
            continue

        chunk["tokens"] = chunk[text_col].map(lambda x: _preprocess_tokens(x, stemmer))
        chunk = chunk[chunk["tokens"].map(len) >= min_words]
        if chunk.empty:
            continue

        remaining = max_docs - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        chunks.append(chunk[[text_col, label_col, "tokens"]])
        total += len(chunk)
        if total >= max_docs:
            break

    if not chunks:
        raise ValueError("No usable articles found after filtering and preprocessing.")

    df = pd.concat(chunks, ignore_index=True)
    counts = df[label_col].value_counts()
    if len(counts) < 2:
        raise ValueError("Need at least 2 categories after filtering for unsupervised evaluation.")

    target = min(per_class_samples, int(counts.min())) if per_class_samples > 0 else int(counts.min())

    balanced = []
    for _, grp in df.groupby(label_col):
        grp = shuffle(grp, random_state=seed)
        balanced.append(grp.iloc[:target])

    out = pd.concat(balanced, ignore_index=True)
    return shuffle(out, random_state=seed).reset_index(drop=True)


def _build_tensor(tokens_per_doc: list[list[str]], vocab_size: int, window_size: int) -> tuple[np.ndarray, list[str]]:
    freq = Counter()
    for tokens in tokens_per_doc:
        freq.update(tokens)

    vocab_items = [tok for tok, _ in freq.most_common(vocab_size)]
    if not vocab_items:
        raise ValueError("Vocabulary is empty after preprocessing.")

    vocab_index = {tok: i for i, tok in enumerate(vocab_items)}
    vocab_n = len(vocab_items)

    tensor = np.zeros((len(tokens_per_doc), vocab_n, vocab_n), dtype=np.float64)

    for doc_id, tokens in enumerate(tokens_per_doc):
        token_ids = [vocab_index[tok] for tok in tokens if tok in vocab_index]
        if len(token_ids) < 2:
            continue

        for i, left in enumerate(token_ids):
            upper = min(i + window_size + 1, len(token_ids))
            for j in range(i + 1, upper):
                right = token_ids[j]
                tensor[doc_id, left, right] += 1.0
                tensor[doc_id, right, left] += 1.0

    return tensor, vocab_items


def _unfold_3way(tensor: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return tensor.reshape(tensor.shape[0], -1)
    if mode == 1:
        return np.transpose(tensor, (1, 0, 2)).reshape(tensor.shape[1], -1)
    return np.transpose(tensor, (2, 0, 1)).reshape(tensor.shape[2], -1)


def _khatri_rao(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Compute column-wise Kronecker products for aligned rank columns.
    return np.einsum("ir,jr->ijr", a, b).reshape(a.shape[0] * b.shape[0], a.shape[1])


def _cp_to_tensor(weights: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], b.shape[0], c.shape[0]), dtype=np.float64)
    for r in range(len(weights)):
        out += weights[r] * np.einsum("i,j,k->ijk", a[:, r], b[:, r], c[:, r])
    return out


def _normalize_cp_columns(weights: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for r in range(len(weights)):
        na = np.linalg.norm(a[:, r]) + _EPS
        nb = np.linalg.norm(b[:, r]) + _EPS
        nc = np.linalg.norm(c[:, r]) + _EPS
        a[:, r] /= na
        b[:, r] /= nb
        c[:, r] /= nc
        weights[r] = na * nb * nc
    return weights, a, b, c


def _cp_apr_kl(
    tensor: np.ndarray,
    rank: int,
    n_iter: int,
    seed: int,
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    rng = np.random.default_rng(seed)
    i_n, j_n, k_n = tensor.shape

    a = rng.random((i_n, rank)) + 0.1
    b = rng.random((j_n, rank)) + 0.1
    c = rng.random((k_n, rank)) + 0.1
    weights = np.ones(rank, dtype=np.float64)

    x0 = _unfold_3way(tensor, 0)
    x1 = _unfold_3way(tensor, 1)
    x2 = _unfold_3way(tensor, 2)

    prev_kl = np.inf
    for _ in range(n_iter):
        z0 = _khatri_rao(c, b)
        az = a @ z0.T + _EPS
        numer = (x0 / az) @ z0
        denom = z0.sum(axis=0, keepdims=True)
        a *= numer / (denom + _EPS)

        z1 = _khatri_rao(c, a)
        bz = b @ z1.T + _EPS
        numer = (x1 / bz) @ z1
        denom = z1.sum(axis=0, keepdims=True)
        b *= numer / (denom + _EPS)

        z2 = _khatri_rao(b, a)
        cz = c @ z2.T + _EPS
        numer = (x2 / cz) @ z2
        denom = z2.sum(axis=0, keepdims=True)
        c *= numer / (denom + _EPS)

        weights, a, b, c = _normalize_cp_columns(weights, a, b, c)

        x_hat = _cp_to_tensor(weights, a, b, c) + _EPS
        kl = float(np.sum(tensor * np.log((tensor + _EPS) / x_hat) - tensor + x_hat))

        if prev_kl < np.inf:
            rel = abs(prev_kl - kl) / max(prev_kl, _EPS)
            if rel < tol:
                prev_kl = kl
                break
        prev_kl = kl

    return weights, [a, b, c], float(prev_kl)


def _decompose_for_ranks(
    tensor: np.ndarray,
    rank_configs: list[int],
    decomposition: str,
    n_iter: int,
    seed: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    factors_list: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []

    for idx, rank in enumerate(rank_configs):
        local_seed = seed + idx
        if decomposition == "cp_apr_kl":
            weights, factors, final_kl = _cp_apr_kl(
                tensor=tensor,
                rank=rank,
                n_iter=n_iter,
                seed=local_seed,
            )
            doc_factor = factors[0] * weights
            diagnostics.append({"rank": rank, "method": decomposition, "final_kl": final_kl})
        else:
            weights, factors = non_negative_parafac(
                tensor,
                rank=rank,
                init="random",
                random_state=local_seed,
                n_iter_max=n_iter,
                tol=1e-5,
                verbose=0,
            )
            doc_factor = factors[0] * weights
            diagnostics.append({"rank": rank, "method": decomposition, "final_kl": None})

        factors_list.append(np.maximum(doc_factor, 0.0))

    return factors_list, diagnostics


def _tier1_factor_metrics(doc_factor: np.ndarray, y_true: np.ndarray, top_k_news: list[int]) -> dict[str, Any]:
    n_docs, n_factors = doc_factor.shape
    out: dict[str, Any] = {}

    for k in top_k_news:
        k_eff = max(1, min(k, n_docs))
        homs = []
        outlier_var = []
        dominant_labels = []

        for r in range(n_factors):
            idx = np.argsort(doc_factor[:, r])[::-1][:k_eff]
            labels = y_true[idx]
            counts = Counter(labels)
            dom_label, dom_count = counts.most_common(1)[0]
            homs.append(dom_count / len(labels))
            outlier_var.append(max(0, len(counts) - 1))
            dominant_labels.append(int(dom_label))

        out[str(k_eff)] = {
            "homogeneity_mean": float(np.mean(homs)),
            "homogeneity_std": float(np.std(homs)),
            "outlier_variety_mean": float(np.mean(outlier_var)),
            "categories_identified": int(len(set(dominant_labels))),
            "n_factors": int(n_factors),
        }

    return out


def _partition_factor_column(values: np.ndarray, percentiles: list[int]) -> np.ndarray:
    out = np.zeros((len(values), len(percentiles) + 1), dtype=np.float64)
    zeros = values <= 0
    out[zeros, -1] = 1.0

    nz = values[~zeros]
    if len(nz) == 0:
        return out

    thresholds = np.percentile(nz, sorted(percentiles, reverse=True))
    for i, thr in enumerate(thresholds):
        mask = (values >= thr) & (~zeros)
        out[mask, i] = values[mask]

    no_bin = (out[:, :-1].sum(axis=1) == 0) & (~zeros)
    if np.any(no_bin):
        out[no_bin, len(percentiles) - 1] = values[no_bin]

    return out


def _build_collective_matrix(doc_factors: list[np.ndarray], ecdf_percentiles: list[int]) -> sparse.csr_matrix:
    blocks = []
    for factor in doc_factors:
        cols = []
        for col_idx in range(factor.shape[1]):
            col = factor[:, col_idx]
            cols.append(_partition_factor_column(col, ecdf_percentiles))
        blocks.append(np.concatenate(cols, axis=1))
    return sparse.csr_matrix(np.concatenate(blocks, axis=1))


def _co_cluster(collective: sparse.csr_matrix, n_coclusters: int, seed: int) -> np.ndarray:
    n_rows = collective.shape[0]
    row_sums = np.asarray(collective.sum(axis=1)).ravel()
    col_sums = np.asarray(collective.sum(axis=0)).ravel()
    valid_rows = row_sums > 0
    valid_cols = col_sums > 0

    if not np.any(valid_rows):
        return np.zeros(n_rows, dtype=int)

    reduced = collective[valid_rows][:, valid_cols]
    k = max(2, min(n_coclusters, reduced.shape[0]))

    try:
        model = SpectralCoclustering(n_clusters=k, random_state=seed)
        model.fit(reduced)
        labels_valid = model.row_labels_
    except ValueError:
        dense = reduced.toarray()
        labels_valid = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(dense)

    labels = np.full(n_rows, fill_value=-1, dtype=int)
    labels[valid_rows] = labels_valid
    if np.any(labels == -1):
        labels[labels == -1] = int(labels_valid.max()) + 1 if len(labels_valid) else 0
    return labels


def _outlier_variety(y_true: np.ndarray, clusters: np.ndarray) -> dict[str, Any]:
    per_cluster = []
    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        labels = y_true[idx]
        if len(labels) == 0:
            continue
        counts = Counter(labels)
        dominant_label, dominant_count = counts.most_common(1)[0]
        outlier_labels = [lab for lab in counts.keys() if lab != dominant_label]
        per_cluster.append(
            {
                "cluster": int(c),
                "size": int(len(labels)),
                "dominant_label": int(dominant_label),
                "dominant_fraction": float(dominant_count / len(labels)),
                "outlier_count": int(len(labels) - dominant_count),
                "outlier_variety": int(len(outlier_labels)),
            }
        )

    mean_variety = float(np.mean([x["outlier_variety"] for x in per_cluster])) if per_cluster else None
    return {"mean_outlier_variety": mean_variety, "clusters": per_cluster}


def _single_run(
    df: pd.DataFrame,
    label_col: str,
    vocab_size: int,
    window_size: int,
    rank_configs: list[int],
    decomposition: str,
    n_iter: int,
    ecdf_percentiles: list[int],
    n_coclusters: int,
    top_k_news: list[int],
    seed: int,
) -> dict[str, Any]:
    tokens_per_doc = df["tokens"].tolist()
    tensor, vocab = _build_tensor(tokens_per_doc=tokens_per_doc, vocab_size=vocab_size, window_size=window_size)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[label_col].values)

    doc_factors, decomp_diag = _decompose_for_ranks(
        tensor=tensor,
        rank_configs=rank_configs,
        decomposition=decomposition,
        n_iter=n_iter,
        seed=seed,
    )

    tier1_by_rank: dict[str, Any] = {}
    for rank, factor in zip(rank_configs, doc_factors):
        tier1_by_rank[str(rank)] = _tier1_factor_metrics(factor, y_true=y, top_k_news=top_k_news)

    collective = _build_collective_matrix(doc_factors=doc_factors, ecdf_percentiles=ecdf_percentiles)
    row_labels = _co_cluster(collective=collective, n_coclusters=n_coclusters, seed=seed)

    tier2_hom = float(homogeneity_score(y, row_labels))
    tier2_out = _outlier_variety(y_true=y, clusters=row_labels)

    return {
        "classes": encoder.classes_.tolist(),
        "vocab_size_used": len(vocab),
        "decomposition_diagnostics": decomp_diag,
        "tier1": tier1_by_rank,
        "tier2": {
            "homogeneity": tier2_hom,
            "outlier": tier2_out,
            "cluster_labels": row_labels.tolist(),
        },
    }


def _aggregate_tier1(runs: list[dict[str, Any]], rank_configs: list[int], top_k_news: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rank in rank_configs:
        rk = str(rank)
        out[rk] = {}
        for k in top_k_news:
            kk = str(k)
            homs = []
            outv = []
            cats = []
            for run in runs:
                if rk in run["tier1"] and kk in run["tier1"][rk]:
                    m = run["tier1"][rk][kk]
                    homs.append(m["homogeneity_mean"])
                    outv.append(m["outlier_variety_mean"])
                    cats.append(m["categories_identified"])
            out[rk][kk] = {
                "homogeneity_median": float(np.median(homs)) if homs else None,
                "homogeneity_mean": float(np.mean(homs)) if homs else None,
                "outlier_variety_mean": float(np.mean(outv)) if outv else None,
                "categories_identified_mean": float(np.mean(cats)) if cats else None,
            }
    return out


def _write_tier_csvs(output_dir: Path, runs: list[dict[str, Any]], tier1_summary: dict[str, Any]) -> None:
    tier1_rows = []
    for rank, by_k in tier1_summary.items():
        for k, vals in by_k.items():
            tier1_rows.append({"rank": int(rank), "top_k": int(k), **vals})
    pd.DataFrame(tier1_rows).to_csv(output_dir / "tier1_summary.csv", index=False)

    tier2_rows = []
    for i, run in enumerate(runs):
        tier2_rows.append(
            {
                "run": i,
                "homogeneity": run["tier2"]["homogeneity"],
                "mean_outlier_variety": run["tier2"]["outlier"]["mean_outlier_variety"],
            }
        )
    pd.DataFrame(tier2_rows).to_csv(output_dir / "tier2_runs.csv", index=False)


def run_pipeline(
    csv_path: Path,
    text_col: str,
    label_col: str,
    language_col: str,
    language_filter: str | None,
    max_docs: int,
    min_words: int,
    exclude_labels: list[str],
    per_class_samples: int,
    vocab_size: int,
    window_size: int,
    rank_configs: list[int],
    decomposition: str,
    n_iter: int,
    ecdf_percentiles: list[int],
    n_coclusters: int,
    top_k_news: list[int],
    repeats: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    base_df = _load_and_prepare_articles(
        csv_path=csv_path,
        text_col=text_col,
        label_col=label_col,
        language_col=language_col,
        language_filter=language_filter,
        max_docs=max_docs,
        min_words=min_words,
        exclude_labels=exclude_labels,
        per_class_samples=per_class_samples,
        seed=seed,
    )

    runs = []
    for r in range(repeats):
        df = shuffle(base_df, random_state=seed + r).reset_index(drop=True)
        runs.append(
            _single_run(
                df=df,
                label_col=label_col,
                vocab_size=vocab_size,
                window_size=window_size,
                rank_configs=rank_configs,
                decomposition=decomposition,
                n_iter=n_iter,
                ecdf_percentiles=ecdf_percentiles,
                n_coclusters=n_coclusters,
                top_k_news=top_k_news,
                seed=seed + r,
            )
        )

    tier1_summary = _aggregate_tier1(runs=runs, rank_configs=rank_configs, top_k_news=top_k_news)
    tier2_homs = np.array([x["tier2"]["homogeneity"] for x in runs], dtype=np.float64)
    tier2_out = [x["tier2"]["outlier"]["mean_outlier_variety"] for x in runs]
    tier2_out = [x for x in tier2_out if x is not None]

    assignments = pd.DataFrame(
        {
            "doc_id": np.arange(len(runs[-1]["tier2"]["cluster_labels"])),
            "cluster": runs[-1]["tier2"]["cluster_labels"],
            "label": base_df[label_col].tolist(),
        }
    )
    assignments_path = output_dir / "cocluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    _write_tier_csvs(output_dir=output_dir, runs=runs, tier1_summary=tier1_summary)

    report = {
        "dataset": {
            "csv_path": str(csv_path),
            "n_docs": int(len(base_df)),
            "class_distribution": base_df[label_col].value_counts().to_dict(),
            "excluded_labels": exclude_labels,
            "min_words": min_words,
            "per_class_samples": per_class_samples,
        },
        "config": {
            "vocab_size": vocab_size,
            "window_size": window_size,
            "rank_configs": rank_configs,
            "decomposition": decomposition,
            "n_iter": n_iter,
            "ecdf_percentiles": ecdf_percentiles,
            "n_coclusters": n_coclusters,
            "top_k_news": top_k_news,
            "repeats": repeats,
            "seed": seed,
        },
        "tier1": {
            "summary": tier1_summary,
        },
        "tier2": {
            "homogeneity_median": float(np.median(tier2_homs)) if len(tier2_homs) else None,
            "homogeneity_mean": float(np.mean(tier2_homs)) if len(tier2_homs) else None,
            "homogeneity_std": float(np.std(tier2_homs)) if len(tier2_homs) else None,
            "mean_outlier_variety_median": float(np.median(tier2_out)) if tier2_out else None,
            "mean_outlier_variety_mean": float(np.mean(tier2_out)) if tier2_out else None,
            "runs": runs,
            "assignments": str(assignments_path),
        },
    }

    (output_dir / "study_config.json").write_text(json.dumps(report["config"], indent=2), encoding="utf-8")
    return report
