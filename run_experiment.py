import argparse
import json
from pathlib import Path

from src.fnd_tensor.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study-aligned two-tier fake news clustering pipeline."
    )
    parser.add_argument("--csv", type=str, default="data/fake.csv", help="Path to CSV dataset.")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name.")
    parser.add_argument("--label-col", type=str, default="type", help="Label column name.")
    parser.add_argument("--lang-col", type=str, default="language", help="Language column name.")
    parser.add_argument("--language", type=str, default="english", help="Language filter.")
    parser.add_argument("--max-docs", type=int, default=1500, help="Maximum documents to load before balancing.")
    parser.add_argument("--min-words", type=int, default=100, help="Minimum preprocessed words per document.")
    parser.add_argument(
        "--exclude-labels",
        type=str,
        nargs="+",
        default=["fake", "bs"],
        help="Labels to exclude from study-aligned protocol.",
    )
    parser.add_argument(
        "--per-class-samples",
        type=int,
        default=75,
        help="Balanced sample count per class (set <=0 to use minimum class count).",
    )
    parser.add_argument("--vocab-size", type=int, default=250, help="Vocabulary size.")
    parser.add_argument("--window-size", type=int, default=10, help="Token co-occurrence window size (delta).")
    parser.add_argument(
        "--rank-configs",
        type=int,
        nargs="+",
        default=[6, 8, 10],
        help="PARAFAC rank configurations for ensemble.",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        default="cp_apr_kl",
        choices=["cp_apr_kl", "nn_parafac"],
        help="Decomposition backend: CP-APR-like KL (default) or TensorLy non-negative PARAFAC.",
    )
    parser.add_argument("--n-iter", type=int, default=80, help="Max iterations per decomposition run.")
    parser.add_argument(
        "--ecdf-percentiles",
        type=int,
        nargs="+",
        default=[90, 80, 65],
        help="ECDF percentiles used for factor partitioning in Tier-2.",
    )
    parser.add_argument("--n-coclusters", type=int, default=8, help="Number of co-clusters in Tier-2.")
    parser.add_argument(
        "--top-k-news",
        type=int,
        nargs="+",
        default=[10, 15, 20, 25, 30],
        help="Top-k documents per latent factor for Tier-1 metrics.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Repeats on shuffled balanced instances.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=str, default="reports", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_pipeline(
        csv_path=Path(args.csv),
        text_col=args.text_col,
        label_col=args.label_col,
        language_col=args.lang_col,
        language_filter=args.language,
        max_docs=args.max_docs,
        min_words=args.min_words,
        exclude_labels=args.exclude_labels,
        per_class_samples=args.per_class_samples,
        vocab_size=args.vocab_size,
        window_size=args.window_size,
        rank_configs=args.rank_configs,
        decomposition=args.decomposition,
        n_iter=args.n_iter,
        ecdf_percentiles=args.ecdf_percentiles,
        n_coclusters=args.n_coclusters,
        top_k_news=args.top_k_news,
        repeats=args.repeats,
        seed=args.seed,
        output_dir=output_dir,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
