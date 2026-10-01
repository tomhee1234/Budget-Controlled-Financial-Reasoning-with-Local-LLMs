"""Check pilot completeness and estimate main-run wall time from measured requests."""
import argparse
import json
from pathlib import Path
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
folder = args.run_dir
manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
rows = [json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*/records/*.json')]
if status['state'] != 'complete' or len(rows) != manifest['expected_results']:
    raise RuntimeError('Pilot incomplete')
if any(r['mock'] for r in rows):
    raise RuntimeError('A mock run does not validate real GPU inference')
summary = {}
main_tasks = run_experiment.load_tasks(ROOT / 'tasks_main.json')
trials_per_condition = len(main_tasks) * config.REPETITIONS
required_conditions = {label for label, _, _ in run_experiment.condition_specs()}
for label in sorted({r['quantization'] for r in rows}):
    group = [r for r in rows if r['quantization'] == label]
    missing = [r for r in group if r['predicted_answer'] is None]
    summary[label] = {
        'rows': len(group), 'correct': sum(r['correct'] for r in group),
        'missing_answers': len(missing),
        'missing_by_condition': {c: sum(r['condition'] == c for r in missing) for c in sorted({r['condition'] for r in missing})},
        'final_token_limits': sum(r['final_stop'] == 'limit' for r in group),
        'adaptive_extensions': sum(r['extensions_used'] for r in group),
        'safety_limit_hits': sum(r.get('safety_limit_reached', False) for r in group),
        'request_seconds': sum(r['total_time_seconds'] for r in group),
        'generated_tokens': sum(r['total_tokens'] for r in group),
    }
    present = {r['condition'] for r in group}
    summary[label]['main_estimate_hours'] = sum(
        statistics.mean(r['total_time_seconds'] for r in group if r['condition'] == condition)
        * trials_per_condition for condition in present) / 3600
    summary[label]['estimate_covers_full_plan'] = present == required_conditions
report = {
    'pilot': str(folder), 'elapsed_seconds': status['elapsed_seconds_this_invocation'],
    'models': summary,
    'main_estimate_hours': max(s['main_estimate_hours'] for s in summary.values()),
    'note': 'Small pilot; estimate covers ONLY the observed conditions. Excludes startup; no speed or accuracy guarantee.',
    'requires_review': [f"{r['quantization']} {r['task_id']} {r['condition']}" for r in rows
                        if r['predicted_answer'] is None and r['condition'] != 'B_free'],
}
(folder / 'pilot_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
