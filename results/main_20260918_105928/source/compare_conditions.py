"""Statistische Tests fuer gepaarte Bedingungsvergleiche (McNemar-Test)."""

from __future__ import annotations

import argparse
import math
import os

import pandas as pd

DEFAULT_CSV_PATHS = [
    "results/experiment_log_Q4.csv",
    "results/experiment_log_Q8.csv",
]

OUTPUT_PATH = "results/analysis/statistical_tests.md"

COMPARISONS = [
    ("H1: Budget 256 vs. 1024 (C, kein Zusatzprompt)", "C_fixed_budget256", "C_fixed_budget1024"),
    ("H1: Budget 256 vs. 1024 (D, generische Intervention)", "D_generic_budget256", "D_generic_budget1024"),
    ("H1: Budget 256 vs. 1024 (E, finanzspezifische Intervention)", "E_financial_budget256", "E_financial_budget1024"),
    ("H3: D vs. E bei Budget 256", "D_generic_budget256", "E_financial_budget256"),
    ("H3: D vs. E bei Budget 1024", "D_generic_budget1024", "E_financial_budget1024"),
]


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cumulative = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * cumulative)


def load_data(paths):
    frames = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Hinweis: {path} nicht gefunden, wird uebersprungen.")
            continue
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("Keine CSV-Dateien gefunden.")
    combined = pd.concat(frames, ignore_index=True)
    combined["correct"] = combined["correct"].astype(str).str.strip().eq("True")
    return combined


def run_comparison(df, quant, cond_a, cond_b):
    subset = df[df["quantization"] == quant]
    pivot = subset[subset["condition"].isin([cond_a, cond_b])].pivot_table(
        index=["task_id", "repetition"], columns="condition", values="correct", aggfunc="first"
    )
    if cond_a not in pivot.columns or cond_b not in pivot.columns:
        return None

    pivot = pivot.dropna(subset=[cond_a, cond_b])
    n_pairs = len(pivot)
    if n_pairs == 0:
        return None

    b = int(((pivot[cond_a] == True) & (pivot[cond_b] == False)).sum())
    c = int(((pivot[cond_a] == False) & (pivot[cond_b] == True)).sum())
    both_correct = int((pivot[cond_a] & pivot[cond_b]).sum())
    both_wrong = int((~pivot[cond_a] & ~pivot[cond_b]).sum())

    acc_a = pivot[cond_a].mean()
    acc_b = pivot[cond_b].mean()
    p_value = mcnemar_exact_pvalue(b, c)

    return {
        "n_pairs": n_pairs, "acc_a": acc_a, "acc_b": acc_b,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "only_a_correct": b, "only_b_correct": c,
        "p_value": p_value, "significant_05": p_value < 0.05,
    }


def main(paths):
    df = load_data(paths)
    quantizations = sorted(df["quantization"].unique())

    lines = ["# Statistische Tests (McNemar, gepaarter Vergleich)\n"]
    lines.append("Explorative Auswertung: Wiederholungen derselben Aufgabe sind nicht zwingend "
                 "unabhaengig. Die p-Werte sind nicht fuer Mehrfachtests korrigiert und "
                 "duerfen allein keine bestaetigenden Signifikanzbehauptungen begruenden.\n")
    lines.append(
        "Getestet wird, ob sich zwei Bedingungen bei denselben Aufgaben "
        "signifikant unterscheiden. p < 0.05 gilt ueblicherweise als "
        "signifikant, sollte bei kleinen Stichproben mit Vorsicht "
        "interpretiert werden.\n"
    )

    for quant in quantizations:
        lines.append(f"\n## Quantisierung: {quant}\n")
        lines.append(
            "| Vergleich | n Paare | Accuracy A | Accuracy B | nur A richtig | "
            "nur B richtig | p-Wert | signifikant (p<0.05) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

        print(f"\n=== Quantisierung: {quant} ===")
        for label, cond_a, cond_b in COMPARISONS:
            result = run_comparison(df, quant, cond_a, cond_b)
            if result is None:
                print(f"  {label}: uebersprungen (keine passenden Daten)")
                continue

            sig_marker = "JA" if result["significant_05"] else "nein"
            print(
                f"  {label}: n={result['n_pairs']}, "
                f"Acc A={result['acc_a']:.1%}, Acc B={result['acc_b']:.1%}, "
                f"p={result['p_value']:.4f}, signifikant={sig_marker}"
            )

            lines.append(
                f"| {label} | {result['n_pairs']} | {result['acc_a']:.1%} | "
                f"{result['acc_b']:.1%} | {result['only_a_correct']} | "
                f"{result['only_b_correct']} | {result['p_value']:.4f} | {sig_marker} |"
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nErgebnisse zusaetzlich gespeichert in: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=DEFAULT_CSV_PATHS)
    args = parser.parse_args()
    main(args.files)
