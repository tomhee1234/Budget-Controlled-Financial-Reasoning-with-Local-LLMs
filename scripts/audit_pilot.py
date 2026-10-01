"""Check a finished protocol pilot without changing its recorded results."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import metrics

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
folder = parser.parse_args().run_dir
status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
rows = [json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*/records/*.json')]
assert status['state'] == 'complete'
assert len(rows) == manifest['expected_results']
keys = {(r['quantization'], r['task_id'], r['repetition'], r['condition']) for r in rows}
assert len(keys) == len(rows), 'Duplicate trials'
assert not any(r['mock'] for r in rows), 'Not a real GPU pilot'
parser_differences = [
    {'quantization': r['quantization'], 'task_id': r['task_id'], 'condition': r['condition'],
     'recorded': r['predicted_answer'], 'current_parser': metrics.extract_numeric_answer(r['final_text'])}
    for r in rows if metrics.extract_numeric_answer(r['final_text']) != r['predicted_answer']]
groups = defaultdict(list)
for row in rows:
    if row['condition'].startswith(('C_', 'D_', 'E_')):
        groups[(row['quantization'], row['task_id'], row['repetition'], row['budget'])].append(row)
matched = []
for key, group in sorted(groups.items()):
    assert len(group) == 3, f'Incomplete C/D/E group: {key}'
    assert len({r['seed'] for r in group}) == 1, f'Unpaired seeds: {key}'
    matched.append({'group': list(key), 'same_reasoning': len({r['raw_reasoning'] for r in group}) == 1})
report = {
    'score_basis': 'Original recorded scores; parser differences are listed separately.',
    'parser_differences': parser_differences,
    'trials': len(rows), 'unique_trials': len(keys),
    'cde_groups': len(matched), 'cde_identical_reasoning': sum(x['same_reasoning'] for x in matched),
    'cde_trace_checks': matched,
    'free_reasoning': [{'quantization': r['quantization'], 'task': r['task_id'],
                       'tokens': r['total_tokens'], 'stop': r['final_stop'], 'correct': r['correct'],
                       'answer_present': r['predicted_answer'] is not None,
                       'safety_limit': r['safety_limit_reached']} for r in rows if r['condition'] == 'B_free'],
}
(folder / 'protocol_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'cde_trace_checks'}, indent=2))
