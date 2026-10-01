"""Zentrale Konfiguration
"""

# Eine Konfiguration pro Quantisierungsstufe
# Ein Worker pro eindeutigem Server; beide Karten werden gleichzeitig genutzt.
QUANTIZATIONS = [
    {"label": "Q4_K_M", "server_url": "http://127.0.0.1:8080", "repeat_penalty": 1.0},
    {"label": "Q8_0", "server_url": "http://127.0.0.1:8081", "repeat_penalty": 1.0},
]

# Reasoning-Budgets in Tokens fuer Bedingung C/D/E
FIXED_BUDGETS = [256, 512, 1024]

# Start-/Erweiterungsbudget fuer die adaptive Bedingung F
ADAPTIVE_START_BUDGET = 256
ADAPTIVE_EXTENSION = 256
ADAPTIVE_MAX_EXTENSIONS = 1

# Technische Sicherheitsgrenze fuer B; kein Budget-Forcing, natuerliches Ende.
FREE_REASONING_BUDGET = 16384
SERVER_CONTEXT = 32768

# Wie viele Tokens die finale Antwort nach dem Forcing maximal bekommt
FINAL_ANSWER_BUDGET = 200
FINAL_ANSWER_BUDGET_INTERVENTION = 200

# Wiederholungen (Aufgabe, Bedingung, Budget, Quantisierung)-Kombination
REPETITIONS = 2

# Sampling-Parameter
TEMPERATURE = 0.6
TOP_P = 0.95
REPEAT_PENALTY_DEFAULT = 1.0
TOP_K = 20
MIN_P = 0.0
SEED = 42

# Toleranz fuer "numerisch korrekt"
TOLERANCE = 0.01

# Nur fuer alte Skripte; der neue Runner verwendet --run-dir und eigene CSVs.
OUTPUT_CSV = "results/experiment_log.csv"
