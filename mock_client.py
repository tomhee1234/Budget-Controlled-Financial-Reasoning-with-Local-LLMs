"""Mock-Version von LlamaCppClient zum Testen der gesamten Pipeline

Simuliert plausible Antworten (inkl. gelegentlichem stopped_limit=True),
damit run_experiment.py end-to-end durchlaufen lassen und die
CSV-Struktur/Auswertung schon vorbereiten. 
LlamaCppClient oder MockLlamaCppClient - das Interface ist identisch.
"""

from __future__ import annotations

import random


class MockLlamaCppClient:
    def __init__(self, server_url: str = "mock", timeout: int = 300):
        self.server_url = server_url
        self._rng = random.Random(42)

    def set_seed(self, seed):
        self._rng.seed(seed)

    def count_tokens(self, text):
        return len(text.split())

    def complete(self, prompt: str, n_predict: int, stop=None,
                 temperature: float = 0.6, top_p: float = 0.95,
                 cache_prompt: bool = True, repeat_penalty: float = 1.0) -> dict:
        # Simuliert: manchmal wird das Budget ausgeschoepft (stopped_limit),
        # manchmal endet das Modell "von selbst" vorher.
        used_tokens = self._rng.randint(max(1, n_predict // 3), n_predict)
        stopped_limit = used_tokens >= n_predict

        if "</think>" in (stop or []):
            text = self._fake_reasoning(used_tokens)
        else:
            fake_value = round(self._rng.uniform(-1.5, 1.5), 4)
            text = f"\nFinal Answer: {fake_value}"
            if prompt.endswith("<think>\n"):
                text = "Example reasoning.</think>" + text

        return {
            "text": text,
            "tokens_predicted": used_tokens,
            "stopped_limit": stopped_limit,
            "stopped_word": not stopped_limit,
            "stop_type": "limit" if stopped_limit else "word",
            "model": "mock",
            "time_seconds": round(used_tokens * 0.01, 3),
        }

    def _fake_reasoning(self, n_tokens: int) -> str:
        filler = "Let me compute step by step. " * max(1, n_tokens // 6)
        if self._rng.random() < 0.3:
            filler += "Wait, let me reconsider this calculation. "
        return filler[: n_tokens * 5]  # grobe Zeichen-statt-Token-Simulation
