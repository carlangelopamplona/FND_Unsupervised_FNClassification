import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralCoclustering
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from src.fnd_tensor.pipeline import (
    _build_collective_matrix,
    _build_tensor,
    _decompose_for_ranks,
    _load_and_prepare_articles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tier-2 CP/PARAFAC figure from fake.csv")
    parser.add_argument("--csv", type=str, default="data/fake.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="type")
    parser.add_argument("--lang-col", type=str, default="language")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--max-docs", type=int, default=600)
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--exclude-labels", nargs="+", default=["fake", "bs"])
    parser.add_argument("--per-class-samples", type=int, default=75)
    parser.add_argument("--vocab-size", type=int, default=120)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--rank-configs", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--ecdf-percentiles", type=int, nargs="+", default=[90, 80, 65])
    parser.add_argument("--n-coclusters", type=int, default=8)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reports/tier2_cp_parafac_figure.png")
    return parser.parse_args()


def _fit_coclustering(collective, n_coclusters: int, seed: int):
    n_rows, n_cols = collective.shape
    row_sums = np.asarray(collective.sum(axis=1)).ravel()
    col_sums = np.asarray(collective.sum(axis=0)).ravel()
    valid_rows = row_sums > 0
    valid_cols = col_sums > 0

    if not np.any(valid_rows):
        row_labels = np.zeros(n_rows, dtype=int)
        col_labels = np.zeros(n_cols, dtype=int)
        return row_labels, col_labels

    reduced = collective[valid_rows][:, valid_cols]
    k = max(2, min(n_coclusters, reduced.shape[0]))

    try:
        model = SpectralCoclustering(n_clusters=k, random_state=seed)
        model.fit(reduced)
        row_valid = model.row_labels_
        col_valid = model.column_labels_
    except ValueError:
        dense = reduced.toarray()
        row_valid = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(dense)
        col_valid = np.zeros(reduced.shape[1], dtype=int)

    row_labels = np.full(n_rows, fill_value=-1, dtype=int)
    col_labels = np.full(n_cols, fill_value=-1, dtype=int)
    row_labels[valid_rows] = row_valid
    col_labels[valid_cols] = col_valid

    if np.any(row_labels == -1):
        row_labels[row_labels == -1] = int(row_valid.max()) + 1 if len(row_valid) else 0
    if np.any(col_labels == -1):
        col_labels[col_labels == -1] = int(col_valid.max()) + 1 if len(col_valid) else 0

    return row_labels, col_labels


def _topk_metrics_for_model(
    collective,
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    y: np.ndarray,
    top_k: int,
) -> tuple[list[float], list[float], dict[int, list[float]]]:
    n_clusters = int(max(row_labels)) + 1
    homs = []
    outlier_varieties = []
    by_category = defaultdict(list)

    for c in range(n_clusters):
        row_idx = np.where(row_labels == c)[0]
        col_idx = np.where(col_labels == c)[0]
        if len(row_idx) == 0:
            continue

        # Score documents by cumulative membership strength within each co-cluster.
        if len(col_idx) > 0:
            scores = np.asarray(collective[row_idx][:, col_idx].sum(axis=1)).ravel()
        else:
            scores = np.asarray(collective[row_idx].sum(axis=1)).ravel()

        order = np.argsort(scores)[::-1]
        k = max(1, min(top_k, len(row_idx)))
        chosen_docs = row_idx[order[:k]]

        labels = y[chosen_docs]
        counts = Counter(labels)
        dom, cnt = counts.most_common(1)[0]
        hom = cnt / len(labels)

        homs.append(hom)
        outlier_varieties.append(max(0, len(counts) - 1))
        by_category[int(dom)].append(hom)

    return homs, outlier_varieties, by_category


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_and_prepare_articles(
        csv_path=Path(args.csv),
        text_col=args.text_col,
        label_col=args.label_col,
        language_col=args.lang_col,
        language_filter=args.language,
        max_docs=args.max_docs,
        min_words=args.min_words,
        exclude_labels=args.exclude_labels,
        per_class_samples=args.per_class_samples,
        seed=args.seed,
    )

    encoder = LabelEncoder()
    encoder.fit(df[args.label_col].values)

    top_hom_values = {k: [] for k in args.top_k}
    top_outlier_values = {k: [] for k in args.top_k}
    rank_panel_data = {r: {k: [] for k in args.top_k} for r in args.rank_configs}
    category_hom = defaultdict(lambda: {k: [] for k in args.top_k})

    for rep in range(args.repeats):
        rep_df = shuffle(df, random_state=args.seed + rep).reset_index(drop=True)
        y = encoder.transform(rep_df[args.label_col].values)
        tokens = rep_df["tokens"].tolist()

        tensor, _ = _build_tensor(tokens_per_doc=tokens, vocab_size=args.vocab_size, window_size=args.window_size)

        # Build the primary Tier-2 ensemble outputs for summary and category panels.
        doc_factors, _ = _decompose_for_ranks(
            tensor=tensor,
            rank_configs=args.rank_configs,
            decomposition="cp_apr_kl",
            n_iter=args.n_iter,
            seed=args.seed + rep,
        )
        collective = _build_collective_matrix(doc_factors, args.ecdf_percentiles)
        row_labels, col_labels = _fit_coclustering(collective, n_coclusters=args.n_coclusters, seed=args.seed + rep)

        for top_k in args.top_k:
            homs, outs, by_cat = _topk_metrics_for_model(
                collective=collective,
                row_labels=row_labels,
                col_labels=col_labels,
                y=y,
                top_k=top_k,
            )
            if homs:
                top_hom_values[top_k].extend(homs)
            if outs:
                top_outlier_values[top_k].extend(outs)

            for cat_id, vals in by_cat.items():
                cat_name = encoder.inverse_transform([cat_id])[0]
                category_hom[cat_name][top_k].extend(vals)

        # Evaluate rank sensitivity by fitting Tier-2 with one rank per run.
        for rank in args.rank_configs:
            rank_factors, _ = _decompose_for_ranks(
                tensor=tensor,
                rank_configs=[rank],
                decomposition="cp_apr_kl",
                n_iter=args.n_iter,
                seed=args.seed + rep,
            )
            rank_collective = _build_collective_matrix(rank_factors, args.ecdf_percentiles)
            rank_row_labels, rank_col_labels = _fit_coclustering(
                rank_collective,
                n_coclusters=args.n_coclusters,
                seed=args.seed + rep,
            )

            for top_k in args.top_k:
                homs, _, _ = _topk_metrics_for_model(
                    collective=rank_collective,
                    row_labels=rank_row_labels,
                    col_labels=rank_col_labels,
                    y=y,
                    top_k=top_k,
                )
                if homs:
                    rank_panel_data[rank][top_k].extend(homs)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.46, wspace=0.28)
    top_labels = [f"Top {k}" for k in args.top_k]

    # Panel (a): homogeneity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.boxplot([top_hom_values[k] for k in args.top_k], patch_artist=True)
    ax1.set_xticklabels(top_labels, rotation=0)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Homogeneity")
    ax1.set_title("(a) Tier-2 homogeneity")

    # Panel (b): outlier diversity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot([top_outlier_values[k] for k in args.top_k], patch_artist=True)
    ax2.set_xticklabels(top_labels, rotation=0)
    ax2.set_ylabel("Diversity of outliers")
    ax2.set_title("(b) Tier-2 outliers diversity")

    # Panel (c): rank sensitivity with up to three inset subplots.
    outer = fig.add_subplot(gs[0, 2])
    outer.axis("off")
    rank_count = min(3, len(args.rank_configs))
    sub = gs[0, 2].subgridspec(rank_count, 1, hspace=0.6)
    for i in range(rank_count):
        rank = args.rank_configs[i]
        ax = fig.add_subplot(sub[i, 0])
        data = [rank_panel_data[rank][k] for k in args.top_k]
        ax.boxplot(data, patch_artist=True)
        ax.set_ylim(0, 1)
        ax.set_title(f"Homogeneity rank = {rank}", fontsize=8)
        ax.set_xticklabels(top_labels, fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
    outer.set_title("(c) rank sensitivity", y=1.02)

    categories = sorted(category_hom.keys())[:6]
    panel_titles = ["d", "e", "f", "g", "h", "i"]

    for i in range(6):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        if i < len(categories):
            cat = categories[i]
            vals = [
                float(np.mean(category_hom[cat][k])) if category_hom[cat][k] else np.nan
                for k in args.top_k
            ]
            ax.plot(args.top_k, vals, marker="o", linestyle="--", color="#7d5ab5", label="Tier-2 CP/PARAFAC")
            ax.set_title(f"({panel_titles[i]}) {cat}")
            ax.set_ylim(0, 1)
            ax.set_xticks(args.top_k)
            ax.set_ylabel("average homogeneity rate")
            ax.legend(loc="lower left", fontsize=8)
        else:
            ax.axis("off")

    fig.suptitle("Tier-2 CP/PARAFAC on fake.csv", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
