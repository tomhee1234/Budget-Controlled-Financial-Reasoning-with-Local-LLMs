# Budget-Controlled Financial Reasoning with Local LLMs in llama.cpp

Code, task data and raw results of the research-internship project *Budget-Controlled Financial
Reasoning with Local LLMs in llama.cpp* (Tom Henn, Faris Tiro, Geri Aliaj; University of Koblenz,
Institute for Management, 2026).

The study asks how the reasoning token budget, a verification instruction and the quantization level
affect the accuracy–efficiency trade-off of a locally deployed reasoning model on financial calculation
tasks. Qwen3-8B is run with llama.cpp in the quantizations Q4_K_M and Q8_0; the reasoning budget is
enforced by budget forcing (cutting the `<think>` block after a fixed number of tokens and forcing the
final answer). No model is trained; the repository contains the evaluation pipeline and the complete
records of the main run.

## Experimental design in brief

| Condition | Reasoning | Answer |
|---|---|---|
| A | direct answer, empty `<think>` block | ≤ 200 tokens |
| B | free reasoning until the model closes `</think>` (safety cap 16,384 tokens in total) | no separate budget |
| C | fixed budget of 256 / 512 / 1,024 reasoning tokens, then `Final Answer:` is forced | ≤ 200 tokens |
| D | as C, with a generic verification instruction before the think block is closed | ≤ 200 tokens |
| E | as C, with a finance-specific verification instruction (formula, inputs, signs, units, percentages, rounding) | ≤ 200 tokens |
| F | adaptive: 256 tokens, one extension by 256 tokens if the truncated text contains a reflection marker (“wait”, “hmm”, …) | ≤ 200 tokens |

* Model: Qwen3-8B, GGUF files from `Qwen/Qwen3-8B-GGUF` at revision `7c41481f57cb95916b40956ab2f0b139b296d974`
  (SHA-256 of both files in `models/manifest.json`).
* Inference: llama.cpp `llama-server` build b10299 (commit `e40bf8864`), one server per GPU
  (two NVIDIA RTX 4000 Ada), context 32,768 tokens, native `/completion` endpoint.
* Sampling: temperature 0.6, top-p 0.95, top-k 20, no min-p, no repetition penalty; seeds derived from
  `42:task:repetition:condition` (C, D and E share the seed per budget level so that they start from an
  identical reasoning text).
* Tasks: 60 synthetic tasks (`tasks_main.json`) of four types — Sharpe ratio, portfolio return,
  correlation coefficient, multi-step Sharpe ratio — with programmatically computed reference solutions;
  a trial counts as correct if the extracted number lies within ±1 % of the reference.
* Main run: 60 tasks × 2 repetitions × 12 condition variants × 2 quantizations = 2,880 trials
  (18 September 2026, 14.1 h); results in `results/main_20260918_105928/`.

## Main results (accuracy in %, mean generated tokens and seconds per trial)

| Condition | Q4 acc. | Q4 tokens | Q4 s | Q8 acc. | Q8 tokens | Q8 s |
|---|---:|---:|---:|---:|---:|---:|
| A_direct | 31.7 | 10 | 0.2 | 28.3 | 10 | 0.4 |
| B_free | 93.3 | 6,443 | 127.9 | 85.0 | 7,394 | 237.1 |
| C_fixed_budget256 | 38.3 | 280 | 5.4 | 39.2 | 276 | 8.7 |
| D_generic_budget256 | 40.0 | 273 | 5.0 | 31.7 | 273 | 8.3 |
| E_financial_budget256 | 40.0 | 275 | 5.0 | 40.0 | 282 | 8.5 |
| C_fixed_budget512 | 50.8 | 543 | 9.9 | 47.5 | 545 | 16.4 |
| D_generic_budget512 | 40.0 | 541 | 9.9 | 40.8 | 537 | 16.2 |
| E_financial_budget512 | 44.2 | 542 | 9.9 | 45.8 | 535 | 16.2 |
| C_fixed_budget1024 | 70.0 | 1,047 | 19.2 | 65.8 | 1,062 | 32.2 |
| D_generic_budget1024 | 67.5 | 1,043 | 19.2 | 65.0 | 1,058 | 32.1 |
| E_financial_budget1024 | 65.8 | 1,039 | 19.1 | 62.5 | 1,050 | 31.8 |
| F_adaptive | 46.7 | 501 | 9.2 | 45.8 | 480 | 14.5 |

The 23 trials of B without an extractable answer (6 for Q4, 17 for Q8) contain the correct value in
`\boxed{}` behind a Markdown heading; a documented re-scoring that accepts this format gives 98.3 % and
99.2 %. Bootstrap intervals, paired comparisons and McNemar tests are produced by the scripts below.

## Repository layout

```
run_experiment.py        runner: one worker per llama-server, resumable, writes JSON records + CSV
conditions.py            prompt construction and request logic for conditions A–F
llm_client.py            thin client for the llama-server /completion endpoint
metrics.py               answer extraction (“Final Answer:” marker) and tolerance check
task_generator.py        templates and reference solutions of the four task types
config.py                budgets, sampling parameters, server URLs
study_analysis.py        task-level accuracies, bootstrap intervals, paired comparisons
compare_conditions.py    exact McNemar tests for pre-specified pairs
analyze_results.py       summary tables and quick plots
mock_client.py           offline stand-in for the server (tests and dry runs)
scripts/                 model download with hash check, server start/stop, task preparation,
                         run analysis, pilot report/audit, documented re-scoring
tests/                   unit tests (parser, conditions, analysis); run without models
tasks_main.json          the 60 tasks of the main run (seed 42); tasks_pilot.json: 4 pilot tasks
models/manifest.json     repository, revision, SHA-256 and size of the two GGUF files
results/main_20260918_105928/
    manifest.json        tasks, settings, SHA-256 of the sources and model files, package versions
    servers.json         server properties reported by llama-server (build, context, defaults)
    Q4_K_M/, Q8_0/       experiment.csv (all 1,440 trials per quantization, incl. reasoning and
                         answer texts) and records/ (one JSON file per trial)
    analysis/            task_level_summary.csv, paired_task_comparisons.csv, statistical_tests.md
    source/              copy of the Python sources at the time of the run
results/pilot_protocol_v3/   the pilot run (96 trials) incl. the documented re-scoring
analysis/                scripts that reproduce the numbers and figures of the thesis’ results chapter
```

Record fields: `quantization, task_id, difficulty, task_type, condition, budget, repetition, seed,
predicted_answer, ground_truth, correct, final_text, reasoning_tokens, total_tokens, total_time_seconds,
extensions_used, raw_reasoning, reasoning_stop, final_stop, reasoning_tokens_method,
safety_limit_reached, extension_triggers, error_category_heuristic, raw_response`.

## Reproducing the analysis (no GPU required)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # requests, pandas, matplotlib
python -m unittest discover -s tests -v                  # unit tests, no model needed
python scripts/analyze_run.py results/main_20260918_105928   # re-creates results/.../analysis/
python analysis/ch7_statistics.py                        # all numbers of the results chapter -> analysis/ch7_statistics.{json,md}
python analysis/make_figs.py                             # figures -> analysis/figures/
```

Conventions of the statistics: unit of analysis is the task (repetitions averaged within tasks),
percentile bootstrap over the 60 tasks with 5,000 draws and seed 42, exact McNemar tests on the 120
paired trials of pre-specified pairs. `requirements-lock-windows-py313.txt` lists the exact package
versions of the main run (Python 3.13.15).

## Re-running the experiment (two NVIDIA GPUs, Windows)

```powershell
python scripts/download_models.py                        # pinned revision, SHA-256 verified
python scripts/start_servers.py --executable PATH\TO\llama-server.exe
python run_experiment.py --tasks tasks_pilot.json --repetitions 1 --run-dir results/pilot_new
python run_experiment.py --tasks tasks_main.json --dry-run
python run_experiment.py --tasks tasks_main.json --run-dir results/main_new
python scripts/analyze_run.py results/main_new
```

`--tasks` should always be given (the default of the runner is an early development file that is not
included here). Every run directory is new; completed trials are stored immediately as JSON records; a run can be
resumed only if tasks, settings, sources, model files and server configuration are unchanged
(`--resume`). `scripts/rescore_run.py RUN --output DIR` re-scores stored answer texts with an extended
parser without touching the original run and documents every changed score.

## Notes

* Only the completed main run and the pilot run that froze the protocol are included; preliminary,
  mock and diagnostic runs are not.
* Local file paths in `results/*/servers.json` and `results/pilot_protocol_v3/rescored/rescoring_audit.json`
  were replaced by placeholders; all other result files are unchanged. `results/main_20260918_105928/source/`
  is byte-identical to the Python files in the repository root, as recorded in `manifest.json`.
* The models (about 5 GB and 8.7 GB) are not part of the repository; `scripts/download_models.py`
  fetches and verifies them.
