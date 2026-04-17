import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figure outputs from Tier-2 outlier table.")
    parser.add_argument("--input-csv", type=str, default="reports/tier2_outlier_table.csv")
    parser.add_argument("--table-fig", type=str, default="reports/tier2_outlier_table_figure.png")
    parser.add_argument("--bar-fig", type=str, default="reports/tier2_outlier_stacked_bar.png")
    return parser.parse_args()


def _parse_outlier_text(text: str) -> list[tuple[str, float]]:
    text = str(text).strip()
    if text.lower() == "none" or not text:
        return []

    parts = [p.strip() for p in text.split(",") if p.strip()]
    out = []
    for p in parts:
        match = re.match(r"(.+?)\s*\((\d+(?:\.\d+)?)%\)", p)
        if not match:
            continue
        label = match.group(1).strip()
        pct = float(match.group(2))
        out.append((label, pct))
    return out


def _draw_table_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig_h = max(2.8, 1.1 + 0.55 * len(df))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")

    cell_text = []
    for _, row in df.iterrows():
        cell_text.append(
            [
                str(row["category"]),
                str(row["outliers_descending_percentage_of_outliers"]),
                str(int(row["total_outliers"])),
            ]
        )

    table = ax.table(
        cellText=cell_text,
        colLabels=[
            "Category",
            "Outliers (descending) - % of outliers",
            "Total outliers",
        ],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.16, 0.66, 0.18],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f2c14e")
            cell.set_text_props(weight="bold", color="#1f1f1f")
        else:
            cell.set_facecolor("#fff9e9" if r % 2 else "#fff2cc")
        cell.set_edgecolor("#d0d0d0")

    ax.set_title("Tier-2 Outlier Table", fontsize=14, weight="bold", pad=12)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_stacked_bar_figure(df: pd.DataFrame, out_path: Path) -> None:
    categories = df["category"].astype(str).tolist()
    parsed = [_parse_outlier_text(t) for t in df["outliers_descending_percentage_of_outliers"]]

    outlier_labels = sorted({label for row in parsed for label, _ in row})
    if not outlier_labels:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No outlier composition available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    data = np.zeros((len(categories), len(outlier_labels)), dtype=float)
    for i, row in enumerate(parsed):
        for label, pct in row:
            j = outlier_labels.index(label)
            data[i, j] = pct

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(categories), dtype=float)
    colors = ["#f39c12", "#e67e22", "#d35400", "#c0392b", "#7f8c8d", "#f1c40f"]

    for j, label in enumerate(outlier_labels):
        vals = data[:, j]
        ax.bar(categories, vals, bottom=bottom, label=label, color=colors[j % len(colors)], edgecolor="white")
        bottom += vals

    ax.set_ylim(0, 100)
    ax.set_ylabel("Outlier share (%)")
    ax.set_title("Tier-2 Outlier Composition by Dominant Category", weight="bold")
    ax.legend(title="Outlier label", loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    for i, total in enumerate(bottom):
        ax.text(i, min(total + 1.5, 99), f"{total:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_csv)
    table_fig = Path(args.table_fig)
    bar_fig = Path(args.bar_fig)

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    required = {
        "category",
        "outliers_descending_percentage_of_outliers",
        "total_outliers",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {sorted(required)}")

    table_fig.parent.mkdir(parents=True, exist_ok=True)
    _draw_table_figure(df, table_fig)
    _draw_stacked_bar_figure(df, bar_fig)

    print(f"Saved table figure to {table_fig}")
    print(f"Saved stacked bar figure to {bar_fig}")


if __name__ == "__main__":
    main()
