import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tier-2 outlier table from cocluster assignments.")
    parser.add_argument("--assignments", type=str, default="reports/cocluster_assignments.csv")
    parser.add_argument("--output-csv", type=str, default="reports/tier2_outlier_table.csv")
    parser.add_argument("--output-md", type=str, default="reports/tier2_outlier_table.md")
    return parser.parse_args()


def _cluster_dominant_labels(df: pd.DataFrame) -> dict[int, str]:
    dominant = {}
    for cluster_id, grp in df.groupby("cluster"):
        label_counts = grp["label"].value_counts()
        dominant[int(cluster_id)] = str(label_counts.index[0])
    return dominant


def _build_table(df: pd.DataFrame) -> pd.DataFrame:
    dominant_by_cluster = _cluster_dominant_labels(df)
    outlier_counts_by_category: dict[str, Counter] = defaultdict(Counter)
    outlier_totals_by_category: Counter = Counter()

    for cluster_id, grp in df.groupby("cluster"):
        dom = dominant_by_cluster[int(cluster_id)]
        for label, count in grp["label"].value_counts().items():
            if str(label) == dom:
                continue
            outlier_counts_by_category[dom][str(label)] += int(count)
            outlier_totals_by_category[dom] += int(count)

    rows = []
    categories = sorted(df["label"].astype(str).unique())
    for category in categories:
        counts = outlier_counts_by_category.get(category, Counter())
        total = outlier_totals_by_category.get(category, 0)

        if total == 0 or not counts:
            outlier_text = "none"
        else:
            ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            parts = [f"{name} ({(cnt / total) * 100:.1f}%)" for name, cnt in ranked]
            outlier_text = ", ".join(parts)

        rows.append(
            {
                "category": category,
                "outliers_descending_percentage_of_outliers": outlier_text,
                "total_outliers": int(total),
            }
        )

    return pd.DataFrame(rows)


def _to_markdown(df: pd.DataFrame) -> str:
    header = "| category | outliers (in descending order) - percentage of outliers | total outliers |\n"
    sep = "|---|---|---|\n"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['category']} | {row['outliers_descending_percentage_of_outliers']} | {int(row['total_outliers'])} |\n"
        )
    return "".join(lines)


def main() -> None:
    args = parse_args()
    assignments_path = Path(args.assignments)
    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)

    if not assignments_path.exists():
        raise FileNotFoundError(f"Assignments file not found: {assignments_path}")

    df = pd.read_csv(assignments_path)
    needed = {"cluster", "label"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Assignments file must contain columns: {sorted(needed)}")

    table_df = _build_table(df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(out_csv, index=False)
    out_md.write_text(_to_markdown(table_df), encoding="utf-8")

    print(f"Saved Tier-2 table CSV to {out_csv}")
    print(f"Saved Tier-2 table Markdown to {out_md}")


if __name__ == "__main__":
    main()
