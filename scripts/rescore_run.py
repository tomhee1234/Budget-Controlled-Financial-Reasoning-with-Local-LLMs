"""Re-score saved final answers without altering any original run artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import metrics


def rescore(folder, output):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
    if status['state'] != 'complete':
        raise ValueError('Run must be complete')
    paths = sorted(folder.glob('*/records/*.json'))
    if len(paths) != manifest['expected_results']:
        raise ValueError('Missing results')
    tolerance = manifest['identity']['settings']['TOLERANCE']
    output.mkdir(parents=True, exist_ok=False)
    groups, changes, hashes = {}, [], {}
    for path in paths:
        row = json.loads(path.read_text(encoding='utf-8'))
        hashes[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
        before = {key: row[key] for key in ('predicted_answer', 'correct', 'error_category_heuristic')}
        predicted = metrics.extract_numeric_answer(row['final_text'])
        correct = metrics.is_correct(predicted, row['ground_truth'], tolerance)
        row.update(predicted_answer=predicted, correct=correct,
                   error_category_heuristic='' if correct else metrics.classify_error_placeholder(predicted, row['ground_truth']))
        after = {key: row[key] for key in before}
        if before != after:
            changes.append({'trial': {key: row[key] for key in ('quantization', 'task_id', 'repetition', 'condition')},
                            'before': before, 'after': after})
        groups.setdefault(row['quantization'], []).append(row)
    for label, rows in groups.items():
        target = output / label
        target.mkdir()
        with (target / 'experiment.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    shutil.copy2(ROOT / 'metrics.py', output / 'metrics.py')
    report = {'original_run': str(folder.resolve()), 'tolerance': tolerance,
              'metrics_sha256': hashlib.sha256((ROOT / 'metrics.py').read_bytes()).hexdigest(),
              'original_record_sha256': hashes, 'changes': changes,
              'note': 'Only answer extraction/scoring changed. No model calls; original records are untouched.'}
    (output / 'rescoring_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'rows': len(paths), 'changes': changes, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rescore(args.run_dir, args.output)
