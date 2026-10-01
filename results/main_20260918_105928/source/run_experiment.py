"""Parallel server workers with atomic result records and strict resume checks."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import threading
import time
import requests
import config
import conditions
import metrics

ROOT = Path(__file__).resolve().parent


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def load_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding='utf-8'))
    if not tasks or len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Tasks must be nonempty with unique IDs')
    return tasks


def condition_specs():
    specs = [('A_direct', conditions.run_condition_a, ()),
             ('B_free', conditions.run_condition_b, ())]
    for budget in config.FIXED_BUDGETS:
        specs.extend([(f'C_fixed_budget{budget}', conditions.run_condition_c, (budget,)),
                      (f'D_generic_budget{budget}', conditions.run_condition_d, (budget,)),
                      (f'E_financial_budget{budget}', conditions.run_condition_e, (budget,))])
    specs.append(('F_adaptive', conditions.run_condition_f, ()))
    return specs


def get_client(quant, use_mock):
    if use_mock:
        from mock_client import MockLlamaCppClient
        return MockLlamaCppClient(quant['server_url'])
    from llm_client import LlamaCppClient
    return LlamaCppClient(quant['server_url'], repeat_penalty=quant['repeat_penalty'])


def seed_for(task_id, repetition, condition):
    # C/D/E share the same pre-intervention sampling seed at each budget.
    # Thus their deliberate difference is the intervention, not a different draw.
    if condition.startswith(('C_fixed_budget', 'D_generic_budget', 'E_financial_budget')):
        condition = 'fixed_budget' + condition.split('_budget')[1]
    key = f'{config.SEED}:{task_id}:{repetition}:{condition}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big') % (2**31)


def preflight(quant):
    health = requests.get(quant['server_url'] + '/health', timeout=10)
    health.raise_for_status()
    props = requests.get(quant['server_url'] + '/props', timeout=30)
    props.raise_for_status()
    props = props.json()
    expected_name = f"Qwen3-8B-{quant['label']}.gguf"
    if Path(props.get('model_path', '')).resolve() != (ROOT / 'models' / expected_name).resolve():
        raise ValueError(f"Wrong model on {quant['server_url']}: {props.get('model_path')}")
    client = get_client(quant, False)
    probe = client.complete('Count from one to one hundred:', n_predict=1)
    if not probe['stopped_limit']:
        raise ValueError(f"Token-limit probe failed: {quant['label']}")
    return {'model_path': props.get('model_path'), 'build_info': props.get('build_info'),
            'default_generation_settings': props.get('default_generation_settings'), 'probe': probe}


def export_csv(folder):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder / 'records').glob('*.json'))]
    if not rows:
        return
    target = folder / 'experiment.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def worker(quant, tasks, repetitions, folder, use_mock, cancel=None, selected_conditions=None):
    folder.mkdir(exist_ok=True)
    records = folder / 'records'
    records.mkdir(exist_ok=True)
    client = get_client(quant, use_mock)
    try:
        for task in tasks:
            for repetition in range(repetitions):
                for label, function, args in condition_specs():
                    if selected_conditions is not None and label not in selected_conditions:
                        continue
                    if cancel is not None and cancel.is_set():
                        return
                    key = hashlib.sha256(f"{task['id']}:{repetition}:{label}".encode()).hexdigest()
                    record = records / f'{key}.json'
                    if record.exists():
                        json.loads(record.read_text(encoding='utf-8'))
                        continue
                    seed = seed_for(task['id'], repetition, label)
                    client.set_seed(seed)
                    run = function(client, task['text'], *args)
                    predicted = metrics.extract_numeric_answer(run.final_text)
                    correct = metrics.is_correct(predicted, task['ground_truth'], config.TOLERANCE)
                    row = {
                        'quantization': quant['label'], 'task_id': task['id'],
                        'difficulty': task.get('difficulty', ''), 'task_type': task.get('type', ''),
                        'condition': label, 'budget': int(label.split('_budget')[1]) if '_budget' in label else None,
                        'repetition': repetition, 'seed': seed, 'mock': use_mock,
                        'predicted_answer': predicted, 'ground_truth': task['ground_truth'], 'correct': correct,
                        **{k: v for k, v in asdict(run).items() if k != 'condition'},
                        'error_category_heuristic': '' if correct else metrics.classify_error_placeholder(predicted, task['ground_truth']),
                        'raw_response': run.raw_reasoning + '\n---FINAL---\n' + run.final_text,
                    }
                    atomic_json(record, row)
                    print(f"[{quant['label']}] {task['id']} rep{repetition} {label}: answer={predicted} correct={correct}", flush=True)
                export_csv(folder)
    finally:
        export_csv(folder)


def run_all(tasks_path='tasks_generated.json', use_mock=False, assume_yes=False,
            run_dir=None, repetitions=None, resume=False, dry_run=False, selected_conditions=None):
    tasks = load_tasks(tasks_path)
    repetitions = config.REPETITIONS if repetitions is None else repetitions
    if repetitions < 1:
        raise ValueError('Repetitions must be positive')
    quants = config.QUANTIZATIONS
    if len({q['server_url'] for q in quants}) != len(quants):
        raise ValueError('Parallel workers require distinct server URLs')
    available = [label for label, _, _ in condition_specs()]
    selected = available if selected_conditions is None else list(selected_conditions)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(available):
        raise ValueError(f'Invalid condition selection; available: {available}')
    expected = len(tasks) * repetitions * len(selected) * len(quants)
    print(f"Plan: {len(tasks)} tasks x {repetitions} repeats x {len(selected)} conditions x {len(quants)} models = {expected} results", flush=True)
    if dry_run:
        return {'expected_results': expected}
    settings = {name: value for name, value in vars(config).items() if name.isupper() and name != 'OUTPUT_CSV'}
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    model_manifest = ROOT / 'models' / 'manifest.json'
    if not use_mock and not model_manifest.exists():
        raise ValueError('Missing model manifest; run scripts/download_models.py first')
    identity = {'schema': 3, 'tasks': tasks, 'repetitions': repetitions, 'mock': use_mock,
                'selected_conditions': selected,
                'settings': settings, 'sources': sources,
                'environment': {'python': platform.python_version(),
                                'packages': {name: importlib.metadata.version(name)
                                             for name in ('requests', 'pandas', 'numpy', 'matplotlib')}},
                'models': json.loads(model_manifest.read_text()) if not use_mock and model_manifest.exists() else None}
    folder = Path(run_dir) if run_dir else ROOT / 'results' / (('mock_' if use_mock else 'run_') + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    manifest_path = folder / 'manifest.json'
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['identity'] != identity:
            raise ValueError('Resume refused: tasks, settings, sources or models changed')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        manifest = {'identity': identity, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'python': platform.python_version(), 'expected_results': expected}
        atomic_json(manifest_path, manifest)
        (folder / 'source').mkdir()
        for source in ROOT.glob('*.py'):
            shutil.copy2(source, folder / 'source' / source.name)
        for source in ROOT.glob('requirements*.txt'):
            shutil.copy2(source, folder / 'source' / source.name)
    lock = (folder / 'runner.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if (folder / 'runner.lock').stat().st_size == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        lock.close()
        raise RuntimeError('Run is already locked by another process')
    started = time.perf_counter()
    try:
        if not use_mock:
            with ThreadPoolExecutor(max_workers=len(quants)) as executor:
                manifests = list(executor.map(preflight, quants))
            for quant, server in zip(quants, manifests):
                context = server['default_generation_settings'].get('n_ctx', 0)
                if context != config.SERVER_CONTEXT:
                    raise ValueError(f"Wrong context on {quant['label']}: {context}, expected {config.SERVER_CONTEXT}. Restart servers.")
                if 'B_free' in selected and config.FREE_REASONING_BUDGET + 2048 > context:
                    raise ValueError('Free-reasoning safety cap leaves insufficient prompt headroom')
            if resume and (folder / 'servers.json').exists():
                previous_servers = json.loads((folder / 'servers.json').read_text(encoding='utf-8'))
                for quant, current in zip(quants, manifests):
                    previous = previous_servers[quant['label']]
                    for key in ('model_path', 'build_info', 'default_generation_settings'):
                        if previous.get(key) != current.get(key):
                            raise ValueError(f"Resume refused: server {quant['label']} changed {key}")
            atomic_json(folder / 'servers.json', dict(zip([q['label'] for q in quants], manifests)))
        atomic_json(folder / 'status.json', {'state': 'running', 'expected_results': expected})
        cancel = threading.Event()
        with ThreadPoolExecutor(max_workers=len(quants)) as executor:
            futures = [executor.submit(worker, q, tasks, repetitions, folder / q['label'], use_mock, cancel, selected) for q in quants]
            try:
                from concurrent.futures import as_completed
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                cancel.set()
                raise
        completed = sum(len(list((folder / q['label'] / 'records').glob('*.json'))) for q in quants)
        if completed != expected:
            raise RuntimeError(f'Incomplete: {completed}/{expected}')
        atomic_json(folder / 'status.json', {'state': 'complete', 'completed': completed,
                    'elapsed_seconds_this_invocation': time.perf_counter() - started})
    except BaseException as exc:
        atomic_json(folder / 'status.json', {'state': 'failed', 'error': str(exc)})
        raise
    finally:
        lock.close()
    print(f'Complete: {folder}', flush=True)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='tasks_generated.json')
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--run-dir')
    parser.add_argument('--repetitions', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--conditions', nargs='+', help='Run only these exact condition labels (e.g. B_free)')
    args = parser.parse_args()
    run_all(args.tasks, args.mock, run_dir=args.run_dir, repetitions=args.repetitions,
            resume=args.resume, dry_run=args.dry_run, selected_conditions=args.conditions)
