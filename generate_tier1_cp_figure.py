import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from src.fnd_tensor.pipeline import _build_tensor, _decompose_for_ranks, _load_and_prepare_articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tier-1 CP/PARAFAC figure from fake.csv")
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
    parser.add_argument("--rank-configs", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reports/tier1_cp_parafac_figure.png")
    return parser.parse_args()


def _factor_stats(doc_factor: np.ndarray, y: np.ndarray, top_k: int) -> dict[str, list[float] | list[int]]:
    n_docs, n_factors = doc_factor.shape
    k = max(1, min(top_k, n_docs))

    homs = []
    outlier_varieties = []
    dominant_labels = []

    for r in range(n_factors):
        idx = np.argsort(doc_factor[:, r])[::-1][:k]
        labels = y[idx]
        counts = Counter(labels)
        dom, cnt = counts.most_common(1)[0]
        homs.append(cnt / len(labels))
        outlier_varieties.append(max(0, len(counts) - 1))
        dominant_labels.append(int(dom))

    return {
        "homogeneity": homs,
        "outlier_variety": outlier_varieties,
        "dominant_labels": dominant_labels,
    }


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
    y_full = encoder.fit_transform(df[args.label_col].values)

    top_hom_values = {k: [] for k in args.top_k}
    top_outlier_values = {k: [] for k in args.top_k}
    kernel_hom_means = {kernel: {k: [] for k in args.top_k} for kernel in args.kernel_sizes}
    category_hom = defaultdict(lambda: {k: [] for k in args.top_k})

    for rep in range(args.repeats):
        rep_df = shuffle(df, random_state=args.seed + rep).reset_index(drop=True)
        y_rep = encoder.transform(rep_df[args.label_col].values)
        tokens = rep_df["tokens"].tolist()

        for kernel in args.kernel_sizes:
            tensor, _ = _build_tensor(tokens_per_doc=tokens, vocab_size=args.vocab_size, window_size=kernel)
            factors, _ = _decompose_for_ranks(
                tensor=tensor,
                rank_configs=args.rank_configs,
                decomposition="cp_apr_kl",
                n_iter=args.n_iter,
                seed=args.seed + rep,
            )

            for top_k in args.top_k:
                rank_homs = []
                rank_outliers = []

                for factor in factors:
                    stats = _factor_stats(factor, y_rep, top_k)
                    rank_homs.extend(stats["homogeneity"])
                    rank_outliers.extend(stats["outlier_variety"])

                    for hom, dom in zip(stats["homogeneity"], stats["dominant_labels"]):
                        label_name = encoder.inverse_transform([dom])[0]
                        category_hom[label_name][top_k].append(hom)

                if kernel == 10:
                    if rank_homs:
                        top_hom_values[top_k].append(float(np.mean(rank_homs)))
                    if rank_outliers:
                        top_outlier_values[top_k].append(float(np.mean(rank_outliers)))

                if rank_homs:
                    kernel_hom_means[kernel][top_k].append(float(np.mean(rank_homs)))

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28)
    top_labels = [f"Top {k}" for k in args.top_k]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.boxplot([top_hom_values[k] for k in args.top_k], patch_artist=True)
    ax1.set_xticklabels(top_labels, rotation=0)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Homogeneity")
    ax1.set_title("(a) Tier-1 homogeneity (CP/PARAFAC)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot([top_outlier_values[k] for k in args.top_k], patch_artist=True)
    ax2.set_xticklabels(top_labels, rotation=0)
    ax2.set_ylabel("Diversity of outliers")
    ax2.set_title("(b) Tier-1 outlier diversity (CP/PARAFAC)")

    ax3 = fig.add_subplot(gs[0, 2])
    for kernel in args.kernel_sizes:
        vals = [float(np.mean(kernel_hom_means[kernel][k])) if kernel_hom_means[kernel][k] else np.nan for k in args.top_k]
        ax3.plot(args.top_k, vals, marker="o", label=f"Kernel {kernel}")
    ax3.set_ylim(0, 1)
    ax3.set_xticks(args.top_k)
    ax3.set_ylabel("Homogeneity")
    ax3.set_title("(c) Kernel sensitivity (CP/PARAFAC)")
    ax3.legend(loc="lower left", fontsize=8)

    categories = sorted(category_hom.keys())[:6]
    panel_titles = ["d", "e", "f", "g", "h", "i"]

    for i in range(6):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        if i < len(categories):
            cat = categories[i]
            vals = [float(np.mean(category_hom[cat][k])) if category_hom[cat][k] else np.nan for k in args.top_k]
            ax.plot(args.top_k, vals, marker="o", linestyle="--", color="#7d5ab5", label="CP/PARAFAC")
            ax.set_title(f"({panel_titles[i]}) {cat}")
            ax.set_ylim(0, 1)
            ax.set_xticks(args.top_k)
            ax.set_ylabel("average homogeneity rate")
            ax.legend(loc="lower left", fontsize=8)
        else:
            ax.axis("off")

    fig.suptitle("Tier-1 decomposition using only CP/PARAFAC on fake.csv", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
