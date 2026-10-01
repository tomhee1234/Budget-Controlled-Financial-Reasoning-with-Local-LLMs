"""Analyse-Skript fuer die Experiment-Ergebnisse."""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV_PATHS = [
    "results/experiment_log_Q4.csv",
    "results/experiment_log_Q8.csv",
]

OUTPUT_DIR = "results/analysis"


def load_data(paths):
    frames = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Hinweis: {path} nicht gefunden, wird uebersprungen.")
            continue
        df = pd.read_csv(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "Keine der angegebenen CSV-Dateien wurde gefunden. "
            "Prueft die Pfade oder gebt sie explizit mit --files an."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["correct"] = combined["correct"].astype(str).str.strip().eq("True")
    combined["condition_group"] = combined["condition"].apply(
        lambda c: re.sub(r"_budget\d+$", "", c)
    )
    return combined


def summary_table(df):
    grouped = (
        df.groupby(["quantization", "condition"])
        .agg(
            accuracy=("correct", "mean"),
            mean_tokens=("total_tokens", "mean"),
            mean_time_s=("total_time_seconds", "mean"),
            n=("correct", "size"),
        )
        .reset_index()
    )
    grouped["accuracy_pct"] = (grouped["accuracy"] * 100).round(1)
    grouped["efficiency_acc_per_1k_tokens"] = (
        grouped["accuracy"] / (grouped["mean_tokens"] / 1000)
    ).round(3)
    return grouped.sort_values(["quantization", "condition"])


def plot_accuracy_by_condition(df, output_path):
    summary = (
        df.groupby(["quantization", "condition_group"])["correct"]
        .mean()
        .reset_index()
    )
    conditions = sorted(summary["condition_group"].unique())
    quantizations = sorted(summary["quantization"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.8 / max(len(quantizations), 1)
    x_positions = range(len(conditions))

    for i, quant in enumerate(quantizations):
        subset = summary[summary["quantization"] == quant].set_index("condition_group")
        values = [subset["correct"].get(c, 0) for c in conditions]
        offsets = [x + i * bar_width for x in x_positions]
        ax.bar(offsets, values, width=bar_width, label=quant)

    ax.set_xticks([x + bar_width * (len(quantizations) - 1) / 2 for x in x_positions])
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy pro Bedingung und Quantisierung")
    ax.legend(title="Quantisierung")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tokens_vs_accuracy(df, output_path):
    summary = (
        df.groupby(["quantization", "condition"])
        .agg(accuracy=("correct", "mean"), mean_tokens=("total_tokens", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"Q4_K_M": "o", "Q8_0": "^"}
    for quant, group in summary.groupby("quantization"):
        marker = markers.get(quant, "s")
        ax.scatter(group["mean_tokens"], group["accuracy"], label=quant, marker=marker, s=60)
        for _, row in group.iterrows():
            ax.annotate(row["condition"], (row["mean_tokens"], row["accuracy"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Durchschnittliche Tokens pro Anfrage")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Accuracy-Efficiency-Trade-off (jeder Punkt = eine Bedingung)")
    ax.legend(title="Quantisierung")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cde_over_budget(df, output_path):
    subset = df[df["condition_group"].isin(["C_fixed", "D_generic", "E_financial"])].copy()
    subset = subset.dropna(subset=["budget"])
    if subset.empty:
        return
    subset["budget"] = subset["budget"].astype(int)

    quantizations = sorted(subset["quantization"].unique())
    fig, axes = plt.subplots(1, len(quantizations), figsize=(6 * len(quantizations), 5), squeeze=False)

    for ax, quant in zip(axes[0], quantizations):
        quant_data = subset[subset["quantization"] == quant]
        for group_name, group_df in quant_data.groupby("condition_group"):
            grouped = group_df.groupby("budget")["correct"].mean().sort_index()
            ax.plot(grouped.index, grouped.values, marker="o", label=group_name)
        ax.set_title(f"Quantisierung: {quant}")
        ax.set_xlabel("Reasoning-Budget (Tokens)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

    fig.suptitle("C (kein Prompt) vs. D (generisch) vs. E (finanzspezifisch) ueber Budgets")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def to_markdown_table(df):
    cols = ["quantization", "condition", "accuracy_pct", "mean_tokens", "mean_time_s",
            "efficiency_acc_per_1k_tokens", "n"]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df[cols].iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, separator] + rows)


def main(paths):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(paths)
    print(f"{len(df)} Zeilen geladen aus {len(paths)} Datei(en).")

    summary = summary_table(df)
    summary_csv_path = os.path.join(OUTPUT_DIR, "summary_table.csv")
    summary.to_csv(summary_csv_path, index=False)
    print(f"\nZusammenfassungstabelle gespeichert: {summary_csv_path}")

    markdown = to_markdown_table(summary)
    markdown_path = os.path.join(OUTPUT_DIR, "summary_table.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Markdown-Tabelle gespeichert: {markdown_path}\n")
    print(markdown)

    acc_path = os.path.join(OUTPUT_DIR, "accuracy_by_condition.png")
    plot_accuracy_by_condition(df, acc_path)
    print(f"\nDiagramm gespeichert: {acc_path}")

    tradeoff_path = os.path.join(OUTPUT_DIR, "tokens_vs_accuracy.png")
    plot_tokens_vs_accuracy(df, tradeoff_path)
    print(f"Diagramm gespeichert: {tradeoff_path}")

    cde_path = os.path.join(OUTPUT_DIR, "cde_over_budget.png")
    if df['condition_group'].isin(['C_fixed', 'D_generic', 'E_financial']).any():
        plot_cde_over_budget(df, cde_path)
        print(f"Diagramm gespeichert: {cde_path}")

    print(f"\nFertig. Alle Ergebnisse liegen in {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=DEFAULT_CSV_PATHS)
    args = parser.parse_args()
    main(args.files)
