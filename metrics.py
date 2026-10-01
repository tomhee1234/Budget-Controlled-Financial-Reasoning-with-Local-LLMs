"""Hilfsfunktionen zum Parsen der Modellantwort und zur Korrektheitspruefung."""

from __future__ import annotations

import re
import math
from typing import Optional

ANSWER_PATTERN = re.compile(
    r"Final Answer:\s*\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+|(?=\.\d))(?:\.\d*)?(?:[eE][+-]?\d+)?)(?![\d,.eE])", re.IGNORECASE
)


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extrahiert die letzte Zahl nach 'Final Answer:' aus dem Modelltext.
    """
    # Keep the explicit answer marker; accept common presentation wrappers only.
    # Never fall back to arbitrary numbers or a boxed value inside reasoning.
    normalized = text.replace('**', '').replace('$$', '').replace('−', '-')
    normalized = normalized.replace(r'\boxed{', '').replace(r'\(', '').replace(r'\)', '')
    normalized = re.sub(r'\\text\{(Final Answer:\s*)\}', r'\1', normalized, flags=re.IGNORECASE)
    matches = ANSWER_PATTERN.findall(normalized)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        value = float(raw)
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def is_correct(predicted: Optional[float], ground_truth: float, tolerance: float) -> bool:
    """Numerischer Abgleich mit relativer Toleranz (siehe config.TOLERANCE)."""
    if predicted is None:
        return False
    if ground_truth == 0:
        return abs(predicted) <= tolerance
    return abs(predicted - ground_truth) / abs(ground_truth) <= tolerance


def classify_error_placeholder(predicted: Optional[float], ground_truth: float) -> str:
    """Grobe Heuristik zur VORSORTIERUNG der Fehlerarten.
    """
    if predicted is None:
        return "keine_antwort_extrahiert"
    if ground_truth != 0 and abs(predicted + ground_truth) / abs(ground_truth) < 0.02:
        return "moeglicher_vorzeichenfehler"
    if ground_truth != 0 and abs(predicted - ground_truth * 100) / abs(ground_truth * 100) < 0.02:
        return "moeglicher_prozent_einheitenfehler"
    if ground_truth != 0 and abs(predicted - ground_truth / 100) / abs(ground_truth / 100) < 0.02:
        return "moeglicher_prozent_einheitenfehler"
    return "unklar_manuell_pruefen"
