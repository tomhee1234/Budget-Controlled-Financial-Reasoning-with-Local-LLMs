"""Parametrisierter Aufgabengenerator fuer einfache Finanzkennzahlen.

Erzeugt beliebig viele Aufgaben mit garantiert korrekter Ground Truth,
nach demselben Prinzip wie FinChain (Xie et al., 2026): Formel-Template +
Zufallswerte + programmatisch berechnete Loesung. So vermeidet ihr
Kontamination (das Modell kann die Aufgabe nicht auswendig kennen) und
muesst die Loesungen nicht von Hand nachrechnen.

Nutzung:
    python task_generator.py
erzeugt tasks_generated.json, das run_experiment.py einliest.

Ergaenzt gerne weitere gen_*-Funktionen fuer eure "mehrstufig"-Kategorie,
z.B. eine Aufgabe, die erst die Rendite und dann daraus die Sharpe Ratio
berechnen laesst.
"""

from __future__ import annotations

import json
import random


def gen_sharpe_ratio(rng: random.Random) -> dict:
    expected_return = round(rng.uniform(0.03, 0.15), 4)
    risk_free = round(rng.uniform(0.01, 0.04), 4)
    std_dev = round(rng.uniform(0.05, 0.25), 4)
    sharpe = round((expected_return - risk_free) / std_dev, 4)
    text = (
        f"A portfolio has an expected annual return of {expected_return * 100:.2f}%, "
        f"a standard deviation of {std_dev * 100:.2f}%, and the risk-free rate is "
        f"{risk_free * 100:.2f}%. Calculate the Sharpe Ratio."
    )
    return {"text": text, "ground_truth": sharpe, "type": "sharpe_ratio"}


def gen_portfolio_return(rng: random.Random) -> dict:
    w1 = round(rng.uniform(0.2, 0.8), 2)
    w2 = round(1 - w1, 2)
    r1 = round(rng.uniform(0.02, 0.15), 4)
    r2 = round(rng.uniform(0.02, 0.15), 4)
    portfolio_return = round(w1 * r1 + w2 * r2, 4)
    text = (
        f"An investor allocates {w1 * 100:.0f}% of a portfolio to Asset A "
        f"(expected return {r1 * 100:.2f}%) and {w2 * 100:.0f}% to Asset B "
        f"(expected return {r2 * 100:.2f}%). Calculate the expected portfolio return "
        f"as a decimal (e.g. 0.05 for 5%)."
    )
    return {"text": text, "ground_truth": portfolio_return, "type": "portfolio_return"}


def gen_correlation_from_covariance(rng: random.Random) -> dict:
    std_a = round(rng.uniform(0.05, 0.30), 4)
    std_b = round(rng.uniform(0.05, 0.30), 4)
    correlation = round(rng.uniform(-0.9, 0.9), 4)
    covariance = round(correlation * std_a * std_b, 6)
    text = (
        f"Asset A has a standard deviation of {std_a * 100:.2f}% and Asset B has a "
        f"standard deviation of {std_b * 100:.2f}%. The covariance between the two "
        f"assets is {covariance}. Calculate the correlation coefficient between A and B."
    )
    return {"text": text, "ground_truth": covariance / (std_a * std_b), "type": "correlation"}


def gen_multi_step_sharpe_from_prices(rng: random.Random) -> dict:
    """Mehrstufige Aufgabe: erst Rendite aus Preisen ableiten, dann Sharpe Ratio."""
    price_start = round(rng.uniform(80, 150), 2)
    growth = round(rng.uniform(0.02, 0.20), 4)
    price_end = round(price_start * (1 + growth), 2)
    actual_return = (price_end - price_start) / price_start
    risk_free = round(rng.uniform(0.01, 0.03), 4)
    std_dev = round(rng.uniform(0.08, 0.22), 4)
    sharpe = (actual_return - risk_free) / std_dev
    text = (
        f"A stock traded at ${price_start:.2f} at the start of the year and at "
        f"${price_end:.2f} at the end of the year (no dividends paid). The "
        f"annualized standard deviation of returns is {std_dev * 100:.2f}% and the "
        f"risk-free rate is {risk_free * 100:.2f}%. First calculate the annual "
        f"return, then calculate the Sharpe Ratio."
    )
    return {"text": text, "ground_truth": sharpe, "type": "multi_step_sharpe"}


GENERATORS = {
    "einfach": [gen_sharpe_ratio, gen_portfolio_return],
    "mittel": [gen_correlation_from_covariance],
    "mehrstufig": [gen_multi_step_sharpe_from_prices],
}


def generate_dataset(n_per_type: int = 10, seed: int = 42) -> list:
    rng = random.Random(seed)
    tasks = []
    counter = 0
    for difficulty, generators in GENERATORS.items():
        for gen_fn in generators:
            for _ in range(n_per_type):
                counter += 1
                task = gen_fn(rng)
                task["id"] = f"{task['type']}_{counter:03d}"
                task["difficulty"] = difficulty
                tasks.append(task)
    return tasks


if __name__ == "__main__":
    dataset = generate_dataset(n_per_type=15)
    with open("tasks_generated.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"{len(dataset)} Aufgaben generiert -> tasks_generated.json")
