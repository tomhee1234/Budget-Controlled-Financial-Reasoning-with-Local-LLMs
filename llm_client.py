"""Duenner Client fuer die native llama.cpp-Server-API (/completion).

Wir nutzen bewusst die native API statt der OpenAI-kompatiblen
/v1/chat/completions, weil wir fuer das Budget-Forcing (Bedingungen C-F)
mitten in der Generierung abbrechen und mit einem manuell weitergebauten
Prompt fortfahren muessen. Das laesst sich mit der Chat-API kaum steuern.

    curl http://localhost:8080/completion -d '{"prompt": "Hallo", "n_predict": 16}'
"""

from __future__ import annotations

import time
from typing import Optional

import requests
import config


class LlamaCppClient:
    def __init__(self, server_url: str, timeout: int = 900, repeat_penalty: float = 1.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.repeat_penalty = repeat_penalty
        self.seed = config.SEED
        self.call_index = 0

    def set_seed(self, seed: int):
        self.seed = seed
        self.call_index = 0

    def count_tokens(self, text: str) -> int:
        response = requests.post(f"{self.server_url}/tokenize",
                                 json={"content": text, "add_special": False}, timeout=self.timeout)
        response.raise_for_status()
        return len(response.json()["tokens"])

    def complete(
        self,
        prompt: str,
        n_predict: int,
        stop: Optional[list] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        cache_prompt: bool = True,
    ) -> dict:
        """Ruft den /completion-Endpoint auf.

        Rueckgabe (dict):
        - text: generierter Text (ohne den Prompt)
        - tokens_predicted: Anzahl generierter Tokens
        - stopped_limit: True, wenn n_predict erreicht wurde, OHNE dass ein
                          stop-String gefunden wurde (Trigger fuers Forcing)
        - stopped_word: True, wenn ein stop-String getroffen wurde
        - time_seconds: gemessene Laufzeit der Anfrage
        """
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "cache_prompt": cache_prompt,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": 256,
            "seed": (self.seed + self.call_index) % (2**31),
            "top_k": config.TOP_K,
            "min_p": config.MIN_P,
            "stream": False,
        }
        self.call_index += 1
        if stop:
            payload["stop"] = stop

        start = time.perf_counter()
        response = requests.post(
            f"{self.server_url}/completion",
            json=payload,
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
        if "content" not in data or "tokens_predicted" not in data:
            raise ValueError(f"Unexpected completion response fields: {sorted(data)}")
        if not any(key in data for key in ("stop_type", "stopped_limit", "stopped_word")):
            raise ValueError("Server does not expose a supported completion stop reason")
        if data.get("truncated"):
            raise ValueError("Server truncated the prompt: increase context before continuing")

        return {
            "text": data.get("content", ""),
            "tokens_predicted": data.get("tokens_predicted", 0),
            "stopped_limit": data.get("stop_type") == "limit" if "stop_type" in data else bool(data.get("stopped_limit")),
            "stopped_word": data.get("stop_type") == "word" if "stop_type" in data else bool(data.get("stopped_word")),
            "stop_type": data.get("stop_type", "limit" if data.get("stopped_limit") else "word" if data.get("stopped_word") else "eos"),
            "time_seconds": elapsed,
            "model": data.get("model", "?"),
        }
