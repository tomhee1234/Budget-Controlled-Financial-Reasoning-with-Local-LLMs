"""Paired, task-level summaries: sampling repeats are not independent tasks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def mean_ci(values, seed=42, draws=5000):
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return None, None  # Four-task pilots do not support useful interval claims.
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [.025, .975]))


def analyze(paths, output_dir):
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    data['correct'] = data['correct'].astype(str).eq('True').astype(float)
    keys = ['quantization', 'task_id', 'repetition', 'condition']
    if data.duplicated(keys).any():
        raise ValueError('Duplicate trials; do not pool separate runs of the same trial')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for (quant, condition), group in data.groupby(['quantization', 'condition']):
        task_accuracy = group.groupby('task_id')['correct'].mean()
        low, high = mean_ci(task_accuracy.values)
        accuracy = float(task_accuracy.mean())
        tokens = float(group['total_tokens'].mean())
        seconds = float(group['total_time_seconds'].mean())
        summaries.append({'quantization': quant, 'condition': condition,
                          'n_tasks': len(task_accuracy), 'n_trials': len(group),
                          'accuracy': accuracy, 'ci95_low': low, 'ci95_high': high,
                          'mean_tokens': tokens, 'mean_seconds': seconds,
                          'accuracy_per_1000_tokens': accuracy * 1000 / tokens if tokens else None,
                          'accuracy_per_second': accuracy / seconds if seconds else None,
                          'missing_answer_rate': float(group['predicted_answer'].isna().mean()),
                          'safety_limit_rate': float(group['safety_limit_reached'].astype(str).eq('True').mean()) if 'safety_limit_reached' in group else None})
    pd.DataFrame(summaries).to_csv(output / 'task_level_summary.csv', index=False)
    comparisons = []
    for quant, group in data.groupby('quantization'):
        # First match repeats, then average the paired differences within each task.
        pivot = group.pivot(index=['task_id', 'repetition'], columns='condition', values='correct')
        budgets = sorted({int(c.split('_budget')[1]) for c in pivot.columns if '_budget' in c})
        pairs = [(f'E_financial_budget{b}', f'D_generic_budget{b}') for b in budgets]
        pairs += [(f'E_financial_budget{b}', f'C_fixed_budget{b}') for b in budgets]
        for first, second in pairs:
            if first not in pivot or second not in pivot:
                continue
            complete = pivot[[first, second]].dropna()
            differences = (complete[first] - complete[second]).groupby('task_id').mean()
            if len(differences) == 0:
                continue
            low, high = mean_ci(differences.values)
            comparisons.append({'quantization': quant, 'first': first, 'second': second,
                                'n_tasks': len(differences), 'n_pairs': len(complete),
                                'accuracy_difference': float(differences.mean()),
                                'ci95_low': low, 'ci95_high': high})
    pd.DataFrame(comparisons, columns=['quantization', 'first', 'second', 'n_tasks', 'n_pairs',
                                     'accuracy_difference', 'ci95_low', 'ci95_high']).to_csv(output / 'paired_task_comparisons.csv', index=False)
    (output / 'statistical_method.json').write_text(json.dumps({
        'unit': 'task_id', 'repetitions': 'averaged within task after pairing',
        'bootstrap_draws': 5000, 'bootstrap_seed': 42, 'interval': 'pointwise percentile 95%',
        'minimum_tasks_for_interval': 5,
        'multiplicity': 'Intervals are descriptive pointwise intervals, not simultaneous hypothesis tests.',
        'scope': 'Inference within the supplied task templates; not all financial reasoning.'
    }, indent=2), encoding='utf-8')
