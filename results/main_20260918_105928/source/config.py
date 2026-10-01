"""Zentrale Konfiguration fuer die Experiment-Pipeline.

Passt hier alle Werte an eure Hardware/euer Setup an, bevor ihr
run_experiment.py startet.
"""

# Eine Konfiguration pro Quantisierungsstufe, die ihr testen wollt.
# 'label' erscheint in den Ergebnis-Tabellen. 'server_url' zeigt auf den
# jeweils laufenden llama-server (siehe README.md - Schritt 2).
# Ein Worker pro eindeutigem Server; beide Karten werden gleichzeitig genutzt.
QUANTIZATIONS = [
    {"label": "Q4_K_M", "server_url": "http://127.0.0.1:8080", "repeat_penalty": 1.0},
    {"label": "Q8_0", "server_url": "http://127.0.0.1:8081", "repeat_penalty": 1.0},
]

# Reasoning-Budgets in Tokens fuer Bedingung C/D/E (Folie 7 im Expose)
FIXED_BUDGETS = [256, 512, 1024]

# Start-/Erweiterungsbudget fuer die adaptive Bedingung F
ADAPTIVE_START_BUDGET = 256
ADAPTIVE_EXTENSION = 256
ADAPTIVE_MAX_EXTENSIONS = 1

# Technische Sicherheitsgrenze fuer B; kein Budget-Forcing, natuerliches Ende.
# Grenztreffer werden getrennt ausgewiesen, nicht als freier Abschluss bezeichnet.
FREE_REASONING_BUDGET = 16384
SERVER_CONTEXT = 32768

# Wie viele Tokens die finale Antwort nach dem Forcing maximal bekommt
FINAL_ANSWER_BUDGET = 200
FINAL_ANSWER_BUDGET_INTERVENTION = 200

# Wie oft jede (Aufgabe, Bedingung, Budget, Quantisierung)-Kombination
# wiederholt wird, um Streuung/Varianz zu erfassen
REPETITIONS = 2

# Sampling-Parameter (bewusst konservativ fuer reproduzierbarere Ergebnisse)
TEMPERATURE = 0.6
TOP_P = 0.95
REPEAT_PENALTY_DEFAULT = 1.0
TOP_K = 20
MIN_P = 0.0
SEED = 42

# Toleranz fuer "numerisch korrekt" (relativ, z.B. 0.01 = 1%)
TOLERANCE = 0.01

# Nur fuer alte Skripte; der neue Runner verwendet --run-dir und eigene CSVs.
OUTPUT_CSV = "results/experiment_log.csv"
